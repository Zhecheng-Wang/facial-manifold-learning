import torch
from model import *
from utils import *
from inference import *
from blendshapes import *
from clustering import cluster_blendshapes, cluster_blendshapes_ec8ec1a
import json

def train(config: dict):
    if not os.path.exists(config["path"]):
        os.makedirs(config["path"], exist_ok=True)
    json.dump(config, open(os.path.join(
        config["path"], "config.json"), "w+"), indent=4)

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    model, loss = build_model(config)
    model.to(device)

    # load dataset
    dataset = config["training"]["dataset"]
    augment = False
    if "augment" in config["training"]:
        augment = config["training"]["augment"]
    dataset = load_dataset(dataset=dataset, augment=augment)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # tb logger
    import torch.utils.tensorboard as tb
    writer = tb.SummaryWriter(log_dir=config["path"])

    model.train()
    # noise_std = config["training"]["noise_std"]
    n_epochs = 1000
    from tqdm import tqdm
    pbar = tqdm(range(n_epochs))
    step = 0
    for epoch in pbar:
        for data in dataset:
            data = data.to(device)
            # if noise_std > 0.0:
            #     data += torch.randn_like(data).to(device) * noise_std
            #     data = torch.clip(data, 0, 1)

            pred = model(data)
            loss_val = loss(pred, data)
            writer.add_scalar("loss", loss_val.item(), step)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            step += 1

        pbar.set_description(f"loss: {loss_val.item():.4f}", refresh=True)
        save_model(model, config)
    save_model(model, config)

if __name__ == "__main__":
    from utils import *
    from inference import *
    # load the blendshape model
    import os
    PROJ_ROOT = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), os.pardir)
    blendshapes = load_blendshape(model="SP")
    cluster = cluster_blendshapes_ec8ec1a(blendshapes, cluster_threshold=0.05, activate_threshold=0.2)
    for i in range(len(cluster)):
        for j in range(len(cluster[i])):
            # convert to int32
            cluster[i][j] = int(cluster[i][j])
    # train
    n_blendshapes = len(blendshapes)
    n_hidden_features = 24
    save_path = os.path.join(PROJ_ROOT, "experiments", "attention")
    dataset = "SP"
    config = {"path": save_path,
              "network": {"type": "attention",
                          "n_features": n_blendshapes,
                          "hidden_features": n_hidden_features,
                          "input_embedding_features": 8,
                          "num_hidden_layers": 3,
                          "nonlinearity": "ReLU",
                          "attention_query_features": 8,
                          "attention_cluster_count": 4},
              "clusters": cluster,
              "training": {"dataset": dataset,
                           "augment": True,
                           "loss": {
                               "type": "cluster_local"
                           }}}
    torch.autograd.set_detect_anomaly(True)
    train(config)
