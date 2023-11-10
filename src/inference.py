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
                nonlinearity="ReLU").to(device)
    
    model.load_state_dict(torch.load(model_path))

    return model

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def infer(model, coords, return_grad=False):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    N = coords.shape[0]
    coords = torch.tensor(coords).reshape(-1, 2).to(device, dtype=torch.float32)
    if return_grad:
        coords = coords.requires_grad_(True)
        value_pred = model(coords)
        grad = gradient(value_pred, coords)
        return value_pred.reshape(N, N).detach().cpu().numpy(), grad.reshape(N, N, 2).detach().cpu().numpy()
    else:
        with torch.inference_mode():
            value_pred = model(coords).squeeze()
        return value_pred.reshape(N, N).detach().cpu().numpy()
