import math
import json
import numpy as np
import torch
from tqdm import tqdm
from model import *
from utils import *
from inference import *
from blendshapes import *

import torch.utils.tensorboard as tb

def train_manifold(save_path, blendshape, cluster=[], noise_std=0.25, dataset="BEAT"):
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    n_blendshapes = len(cluster) if len(cluster) > 0 else len(blendshape)
    config = {"path": save_path,\
              "cluster": list(map(int, cluster)),\
              "network": {"n_features": n_blendshapes,\
                          "hidden_features": 64,\
                          "num_encoder_layers": 4,\
                          "latent_dimension": n_blendshapes // 2,\
                          "num_decoder_layers": 4,\
                          "nonlinearity": "ReLU"},\
              "training": {"dataset": dataset,
                           "noise_std": noise_std}}
    json.dump(config, open(os.path.join(save_path, "config.json"), "w+"), indent=4)
    train(config)

def train(config:json):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    cluster = config["cluster"]
    
    model, loss = build_model(config)
    print(model)
    model.to(device)
    
    # load dataset
    dataset = load_dataset(dataset=config["training"]["dataset"])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # tb logger
    writer = tb.SummaryWriter(config["path"])

    model.train()
    noise_std = config["training"]["noise_std"]
    n_epochs = 10000
    pbar = tqdm(range(n_epochs))
    step = 0
    for epoch in pbar:
        for data in dataset:
            # if clusters are provided, train submanifold
            if len(cluster) > 0:
                data = data[:,cluster]
            data = data.to(device)
            if noise_std > 0.0:
                data += torch.randn_like(data).to(device) * noise_std
                data = torch.clip(data, 0, 1)
            
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
    PROJ_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
    blendshapes = load_blendshape(model="SP")
    
    # compute clusters
    from clustering import *
    clusters = cluster_blendshapes(blendshapes, cluster_threshold=0.05, activate_threshold=0.2)
    
    # load dataset
    dataset = load_dataset(dataset="SP")
    print(f"dataset # of samples: {len(dataset.dataset)}")
    
    manifold_path = os.path.join(PROJ_ROOT, "experiments", "manifold")
    train_manifold(manifold_path, blendshapes, noise_std=0.0, dataset="SP")
        
    manifold_path = os.path.join(PROJ_ROOT, "experiments", "dae_manifold")
    train_manifold(manifold_path, blendshapes, noise_std=0.05, dataset="SP")

    # submanifold_path = os.path.join(PROJ_ROOT, "experiments", "submanifold")
    # for i, cluster in enumerate(clusters):
    #     cluster_path = os.path.join(submanifold_path, f"cluster_{i}")
    #     if not model_exists(cluster_path):
    #         print(f"Manifold model does not exist. Constructing {cluster_path}")
    #         train_manifold(cluster_path, cluster, noise_std=0.25)