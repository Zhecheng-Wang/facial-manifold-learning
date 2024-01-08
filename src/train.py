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

def train(save_path:str, clusters=[], dataset=None, network_type="dae"):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    n_blendshapes = len(clusters) if len(clusters) > 0 else 51
    if network_type != "ae" and network_type != "dae":
        raise Exception("Invalid network type")
    
    config = {"path": save_path,\
              "type": network_type,\
              "clusters": list(map(int, clusters)),\
              "network": {"n_features": n_blendshapes,\
                          "hidden_features": 64,\
                          "num_encoder_layers": 4,\
                          "latent_dimension": n_blendshapes // 2,\
                          "num_decoder_layers": 4,\
                          "nonlinearity": "ReLU",
                          "noise_std": 0.5}}
    
    model, loss = build_model(config)
    print(model)
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
    n_epochs = 1000
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

if __name__ == "__main__":
    from utils import *
    from inference import *
    # load the blendshape model
    import os
    PROJ_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
    BLENDSHAPES_PATH = os.path.join(PROJ_ROOT, "data", "AppleAR", "OBJs")
    blendshapes = load_blendshape(BLENDSHAPES_PATH)
    
    # compute clusters
    from clustering import *
    clusters = cluster_blendshapes(blendshapes, cluster_threshold=0.05, activate_threshold=0.2)
    
    # load dataset
    dataset = load_dataset()
    print(f"dataset # of samples: {len(dataset.dataset)}")
    
    # manifold_path = os.path.join(PROJ_ROOT, "experiments", "manifold")
    # if not model_exists(manifold_path):
    #     print(f"Manifold model does not exist. Constructing {manifold_path}")
    #     manifold_construction(manifold_path, dataset=dataset, network_type="ae")
        
    manifold_path = os.path.join(PROJ_ROOT, "experiments", "dae_manifold")
    if not model_exists(manifold_path):
        print(f"Manifold model does not exist. Constructing {manifold_path}")
        manifold_construction(manifold_path, dataset=dataset, network_type="dae")

    # submanifold_path = os.path.join(PROJ_ROOT, "experiments", "submanifold")
    # for i, cluster in enumerate(clusters):
    #     cluster_path = os.path.join(submanifold_path, f"cluster_{i}")
    #     if not model_exists(cluster_path):
    #         print(f"Manifold model does not exist. Constructing {cluster_path}")
    #         manifold_construction(cluster_path, cluster, dataset=dataset, network_type="ae")
            
    # submanifold_path = os.path.join(PROJ_ROOT, "experiments", "dae_submanifold")
    # for i, cluster in enumerate(clusters):
    #     cluster_path = os.path.join(submanifold_path, f"cluster_{i}")
    #     if not model_exists(cluster_path):
    #         print(f"Manifold model does not exist. Constructing {cluster_path}")
    #         manifold_construction(cluster_path, cluster, dataset=dataset, network_type="dae")
