import numpy as np
import torch
from tqdm import tqdm
from network import MLP
from blendshapes import *

def train(coords, labels):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    model = MLP(in_features=2,\
                out_features=1,\
                num_hidden_layers=3,\
                hidden_features=64,\
                nonlinearity="ReLU").to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
    
    loss = torch.nn.MSELoss().to(device)

    coords = torch.tensor(coords).reshape(-1, 2).to(device, dtype=torch.float32)
    labels = torch.tensor(labels).reshape(-1, 1).to(device, dtype=torch.float32)

    model.train()
    pbar = tqdm(range(10000))
    for epoch in pbar:     
        labels_pred = model(coords)
        loss_val = loss(labels_pred, labels)

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        pbar.set_description(f"loss: {loss_val.item():.4f}", refresh=True)

    return model
