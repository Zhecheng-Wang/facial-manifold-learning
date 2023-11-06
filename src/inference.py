import numpy as np
import torch
from network import MLP
from blendshapes import *

def load_model(model_path):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = MLP(in_features=2,\
                out_features=1,\
                num_hidden_layers=3,\
                hidden_features=64,\
                nonlinearity="Sine").to(device)
    
    model.load_state_dict(torch.load(model_path))

    return model

def infer(model, coords):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    N = coords.shape[0]
    coords = torch.tensor(coords).reshape(-1, 2).to(device, dtype=torch.float32)
    with torch.inference_mode():
        labels_pred = model(coords).squeeze()
    return labels_pred.reshape(N, N).detach().cpu().numpy()