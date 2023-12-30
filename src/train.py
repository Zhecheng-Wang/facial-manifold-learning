import json
import numpy as np
import torch
from tqdm import tqdm
from model import build_model, save_model
from model import DenoisingAutoEncoder as DAE
from model import VariationalAutoEncoder as VAE
from utils import load_dataset
from blendshapes import *
import math
import torch.nn.functional as F
import torch.utils.tensorboard as tb

def train(save_path:str, clusters=[], dataset=None, network_type="dae"):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    n_blendshapes = len(clusters) if len(clusters) > 0 else 51
    latent_dim = math.floor(math.sqrt(n_blendshapes))
    # round to nearest 2's power
    latent_dim = int(2**math.ceil(math.log(latent_dim, 2)))
    encoder_hidden_features = []
    decoder_hidden_features = []
    hidden_feature = latent_dim*2
    while hidden_feature <= n_blendshapes:
        encoder_hidden_features.append(hidden_feature)
        decoder_hidden_features.append(hidden_feature)
        hidden_feature *= 2
    encoder_hidden_features.reverse()
    if network_type != "ae" and network_type != "dae" and network_type != "vae":
        raise Exception("Invalid network type")
    config = {"path": save_path,\
              "type": network_type,\
              "clusters": list(map(int, clusters)),\
              "network": {"n_features": n_blendshapes,\
                          "encoder_hidden_features": encoder_hidden_features,\
                          "latent_dim": latent_dim,\
                          "decoder_hidden_features": decoder_hidden_features,\
                          "nonlinearity": "ReLU",
                          "noise_std": 0.25}}
    
    model, loss = build_model(config)
    model.to(device)
    
    json_save_path = os.path.join(save_path, "config.json")
    json.dump(config, open(json_save_path, "w"))
    
    # load dataset if not provided
    if dataset is None:
        dataset = load_dataset()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # tb logger
    writer = tb.SummaryWriter(save_path)

    model.train()
    n_epochs = 5
    pbar = tqdm(range(n_epochs))
    step = 0
    for epoch in pbar:
        for data in dataset:
            # if clusters are provided, train submanifold
            if len(clusters) > 0:
                data = data[:,clusters]
            data = data.to(device)
            
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
