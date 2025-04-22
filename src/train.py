import os
import json
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import load_dataset, load_blendshape
from scripts.SMOTE import BalancedSMOTEDataset
from model import build_model, save_model, KL_divergence
from clustering import compute_jaccard_similarity

def train(config: dict):
    # Prepare output folder & config dump
    os.makedirs(config["path"], exist_ok=True)
    json.dump(config, open(os.path.join(config["path"], "config.json"), "w+"), indent=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(config).to(device)
    print(model)

    # DataLoader now yields (w_gt, selected_id, alpha)
    loader = load_dataset(
        batch_size=config["training"].get("batch_size", 32),
        dataset=config["training"]["dataset"],
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"].get("lr", 1e-3))
    writer = SummaryWriter(log_dir=config["path"])

    model.train()
    step = 0
    noise_std = config["training"].get("noise_std", 0.05)
    kl_weight = config["training"].get("kl_weight", 1.0)

    for epoch in range(config["training"].get("n_epochs", 200)):
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for w_gt, selected_id, alpha in pbar:
            # Move to device

            w_gt = w_gt.to(device)                 # [B, m]
            selected_id = selected_id.to(device)   # [B, 1]

            if config["network"]["type"] == "controller":
                # Rinat controller expects selected_id to be a 1D tensor
                selected_id = selected_id.squeeze(1)
                # Compute mask inside model
                mask = model.valid_mask(alpha, selected_id)  # [B, m]

                # Corrupt only masked entries
                noise = torch.randn_like(w_gt) * noise_std * mask
                w_in = torch.clamp(w_gt + noise, 0.0, 1.0)

                # Forward pass
                w_pred, mu, logvar, mask_pred = model(w_in, selected_id, alpha)

                # Reconstruction loss only on mask_pred == 1
                rec_loss = ((w_pred - w_gt) ** 2 * mask_pred).sum(dim=1).mean()

                # KL divergence
                kl_loss = KL_divergence(mu, logvar).mean()
                loss = rec_loss + kl_weight * kl_loss
                writer.add_scalar("loss/kl", kl_loss.item(), step)

                pbar.set_postfix({
                    "rec": f"{rec_loss.item():.4f}",
                    "kl":  f"{kl_loss.item():.4f}"  
                })

            elif config["network"]["type"] == "rinat_controller":

                noise = torch.randn_like(w_gt) * noise_std
                w_in = torch.clamp(w_gt + noise, 0.0, 1.0)
                w_pred = model(w_in, selected_id, alpha)
                # Reconstruction loss
                rec_loss = ((w_pred - w_gt) ** 2).sum(dim=1).mean()
                sparsity_loss = (torch.abs(w_pred)).sum(dim=1).mean()
                
                loss = rec_loss + sparsity_loss * config["training"].get("sparsity_weight", 0.1)

                pbar.set_postfix({
                    "rec": f"{rec_loss.item():.4f}",
                })


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            writer.add_scalar("loss/total", loss.item(), step)
            writer.add_scalar("loss/reconstruction", rec_loss.item(), step)
            step += 1
        # Save checkpoint each epoch
        save_model(model, config)

    # Final save
    save_model(model, config)


if __name__ == "__main__":
    from utils import PROJ_ROOT
    from clustering import compute_jaccard_similarity

    # Load blendshapes (for external similarity)
    blendshapes = load_blendshape(model="SP")
    S = compute_jaccard_similarity(blendshapes, threshold=0.1)

    # Build config
    # config = {
    #     "path": os.path.join(PROJ_ROOT, "experiments", "controller_with_SMOTE"),
    #     "network": {
    #         "type": "controller",
    #         "n_features": len(blendshapes),
    #         "hidden_features": 64,
    #         "num_encoder_layers": 5,
    #         "latent_dim": 8,
    #         "num_decoder_layers": 5,
    #         "nonlinearity": "ReLU",
    #     },
    #     "training": {
    #         "dataset": "SP_SMOTE",
    #         "similarity": S.tolist(),
    #         "batch_size": 128,
    #         "lr": 1e-5,
    #         "n_epochs": 500,
    #         "kl_weight": 1E-2,
    #         "noise_std": 0.2,
    #     }
    # }
    config = {
        "path": os.path.join(PROJ_ROOT, "experiments", "rinat_small"),
        "network": {
            "type": "rinat_controller",
            "n_features": len(blendshapes),
            "hidden_features": 12,
            "num_encoder_layers": 1,
            "latent_dim": 7,
            "num_decoder_layers": 1,
            "nonlinearity": "ReLU",
        },
        "training": {
            "dataset": "SP_SMOTE",
            "similarity": S.tolist(),
            "batch_size": 64,
            "lr": 1e-3,
            "n_epochs": 300,
            "kl_weight": 1E-2,
            "sparsity_weight": 0.1,
            "noise_std": 0.1,
        }
    }
    train(config)

