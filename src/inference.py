import os
import torch
from model import *
from utils import *
from blendshapes import *

def infer(model, w, alpha, id):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    with torch.inference_mode():
        w = torch.tensor(w).to(device, dtype=torch.float32)
        alpha = torch.tensor(alpha).to(device, dtype=torch.float32)
        id = torch.tensor(id).to(device, dtype=torch.int64)
        if len(w.shape) == 1:
            w = w.unsqueeze(0)
        if len(alpha.shape) <= 1:
            alpha = alpha.view(1, 1)
        if len(id.shape) <= 1:
            id = id.view(1, 1)
        w_pred = model.infer(w, alpha, id)
        w_pred = w_pred.detach().squeeze(0).cpu().numpy()
    return w_pred

def sample_configurations(blendshapes, weights):
    # generate # n_sample random weights
    n_samples = weights.shape[0]
    V = np.zeros((n_samples, *blendshapes.V.shape))
    for i in range(n_samples):
        V[i] = blendshapes.eval(weights[i])
    return V

def projection(weights, model, alpha, id):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    # project to the manifold
    proj_weights = infer(model, weights, alpha, id)
    return proj_weights