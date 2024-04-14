import torch
from model import *
from utils import *
from inference import *
from blendshapes import *
import json
import copy 
from diffusion.gaussian_diffusion import create_gaussian_diffusion
from diffusion.resample import *
def train(config: dict):
    if not os.path.exists(config["path"]):
        os.makedirs(config["path"], exist_ok=True)
    json.dump(config, open(os.path.join(
        config["path"], "config.json"), "w+"), indent=4)

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    model = build_model(config)
    model.to(device)
    print(model)
    # load dataset
    dataset = config["training"]["dataset"]
    augment = False
    if "augment" in config["training"]:
        augment = config["training"]["augment"]
    dataset = load_dataset(dataset=dataset, augment=augment)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # tb logger
    import torch.utils.tensorboard as tb
    writer = tb.SummaryWriter(log_dir=config["path"])

    model.train()
    n_blendshapes = config["network"]["n_features"]
    n_epochs = 10000
    from tqdm import tqdm
    pbar = tqdm(range(n_epochs))
    step = 0
    for epoch in pbar:
        for w in dataset:
            noise_level = 0.2
            w = w.to(device)
            # !: in theory this sampling works, but would the frequency of sampling be sufficient? Doubt that.
            # Sample alpha values from a uniform distribution between 0 and 1
            alpha = torch.rand(w.shape[0], 1).to(device)
            # clip alpha between 0.05 and 0.95
            alpha = torch.clamp(alpha, 0.05, 0.95)
            # Sample ids uniformly across the blendshape range
            id = torch.randint(0, n_blendshapes, (w.shape[0], 1)).to(device)

            # for the alpha between 0.1 nad 0.9, 
            alignment_mask = model.valid_mask(alpha, id)
            w_noisy = w + torch.randn_like(w) * alignment_mask * noise_level

            # for the alpha with value of 0
            alpha_zeros = torch.zeros_like(alpha)
            id_zeros = torch.randint(0, n_blendshapes, (w.shape[0], 1)).to(device)
            w_target_zeros = copy.deepcopy(w)
            w_noisy_zeros = w + torch.randn_like(w) * noise_level

            # for the alpha with value of 1
            alpha_ones = torch.ones_like(alpha)
            id_ones = torch.randint(0, n_blendshapes, (w.shape[0], 1)).to(device)
            # onehot vector with id
            one_hot_mask = torch.zeros(w.shape[0], n_blendshapes).to(device)

            one_hot_mask[:, id_ones[:, 0]] = 1
            w_noisy_ones = w + torch.randn_like(w) * one_hot_mask
            w_target_ones = copy.deepcopy(w)
            
            w_noisy_overall = torch.concatenate([w_noisy_zeros, w_noisy_ones, w_noisy], dim=0)
            w_noisy_overall = torch.clamp(w_noisy_overall, 0, 1)
            alpha_overall = torch.concatenate([alpha_zeros, alpha_ones, alpha], dim=0)
            id_overall = torch.concatenate([id_zeros, id_ones, id], dim=0)

            # half let the maximum index be the driving force half the times
            binary = torch.randint(0, 2, (id_overall.shape[0], 1)).to(device)
            max_id = torch.argmax(w_noisy_overall, dim=1, keepdim=True)
            id_overall = binary * max_id + (1 - binary) * id_overall



            w_pred = model(w_noisy_overall, alpha_overall, id_overall)
            # overall target
            w_target = torch.concatenate([w_target_zeros, w_target_ones, w], dim=0)
            w_target = torch.clamp(w_target, 0, 1)

            loss = torch.mean((w_pred - w_target) ** 2)
            writer.add_scalar("loss", loss.item(), step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1

        pbar.set_description(f"loss: {loss.item():.4f}", refresh=True)
        save_model(model, config)
    save_model(model, config)


def train_diffusion(config: dict):
    if not os.path.exists(config["path"]):
        os.makedirs(config["path"], exist_ok=True)
    json.dump(config, open(os.path.join(
        config["path"], "config.json"), "w+"), indent=4)

    diff = create_gaussian_diffusion(config)
    schedule_sampler = create_named_schedule_sampler("uniform", diff)
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    model = build_model(config)
    model.to(device)
    # load dataset
    dataset = config["training"]["dataset"]
    augment = False
    if "augment" in config["training"]:
        augment = config["training"]["augment"]
    dataset = load_dataset(dataset=dataset, augment=augment, batch_size=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # tb logger
    import torch.utils.tensorboard as tb
    writer = tb.SummaryWriter(log_dir=config["path"])

    model.train()
    n_blendshapes = config["network"]["n_features"]
    n_epochs = 10000
    from tqdm import tqdm
    pbar = tqdm(range(n_epochs))
    step = 0
    for epoch in pbar:
        for w in dataset:
            noise_level = 0.3
            w = w.to(device)
            # !: in theory this sampling works, but would the frequency of sampling be sufficient? Doubt that.
            # Sample alpha values from a uniform distribution between 0 and 1
            alpha = torch.rand(w.shape[0], 1).to(device)
            # clip alpha between 0.05 and 0.95
            alpha = torch.clamp(alpha, 0.05, 0.95)
            # Sample ids uniformly across the blendshape range
            id = torch.randint(0, n_blendshapes, (w.shape[0], 1)).to(device)
            # for the alpha between 0.1 nad 0.9, 
            alignment_mask = model.valid_mask(alpha, id)
            w_noisy = w + torch.randn_like(w) * alignment_mask * noise_level

            # for the alpha with value of 0
            alpha_zeros = torch.zeros_like(alpha)
            id_zeros = torch.randint(0, n_blendshapes, (w.shape[0], 1)).to(device)
            alignment_mask_0 = torch.ones(w.shape[0], n_blendshapes).to(device)
            w_target_zeros = copy.deepcopy(w)
            w_noisy_zeros = w + torch.randn_like(w) * noise_level

            # for the alpha with value of 1
            alpha_ones = torch.ones_like(alpha)
            id_ones = torch.randint(0, n_blendshapes, (w.shape[0], 1)).to(device)
            # onehot vector with id
            one_hot_mask = torch.zeros(w.shape[0], n_blendshapes).to(device)
            one_hot_mask[:, id_ones[:, 0]] = 1
            alignment_mask_1 = one_hot_mask
            w_noisy_ones = w + torch.randn_like(w) * one_hot_mask
            w_target_ones = copy.deepcopy(w)
            
            w_noisy_overall = torch.concatenate([w_noisy_zeros, w_noisy_ones, w_noisy], dim=0)
            w_noisy_overall = torch.clamp(w_noisy_overall, 0, 1)
            alpha_overall = torch.concatenate([alpha_zeros, alpha_ones, alpha], dim=0)
            alignment_mask_overall = torch.concatenate([alignment_mask_0, alignment_mask_1, alignment_mask], dim=0)

            times, __ = schedule_sampler.sample(alignment_mask_overall.shape[0], torch.device(device))
            times *= 0
            # half let the maximum index be the driving force half the times

            # diff.training_losses(

            # )

            w_pred = model(w_noisy_overall, times, alpha_overall, alignment_mask_overall)
            # overall target
            w_target = torch.concatenate([w_target_zeros, w_target_ones, w], dim=0)
            w_target = torch.clamp(w_target, 0, 1)

            loss = torch.mean((w_pred - w_target) ** 2)
            writer.add_scalar("loss", loss.item(), step)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1

        pbar.set_description(f"loss: {loss.item():.4f}", refresh=True)
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
    # train
    n_blendshapes = len(blendshapes)
    n_hidden_features = 16
    save_path = os.path.join(PROJ_ROOT, "experiments", "test_diffusion")
    dataset = "SP"
    config = {"path": save_path,
              "network": {"type": "diffusion",
                          "n_features": n_blendshapes,
                          "hidden_features": n_hidden_features,
                          "encoder_layers": 2,
                          "decoder_layers": 4,
                          "nonlinearity": "ReLU"},
              "training": {"dataset": dataset,
                           "loss": {
                               "type": "mse"
                           }},
                           "alpha_1_proportion": 1,
                           "alpha_0_proportion": 1,
                           "onehot_examples_propotion": 1,
                           "diff_steps": 10,
                           "loss_type": "MSE",
                           }
    train_diffusion(config)
