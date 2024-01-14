import os
import torch
from model import *
from utils import *
from blendshapes import *
from train import *
from inference import *

def infer(model, x):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    with torch.inference_mode():
        x = torch.tensor(x).to(device, dtype=torch.float32)
        y = model.infer(x)
        y = y.detach().cpu().numpy()
    return y

def sample_configurations(blendshapes, weights):
    # generate # n_sample random weights
    n_samples = weights.shape[0]
    V = np.zeros((n_samples, *blendshapes.V.shape))
    for i in range(n_samples):
        V[i] = blendshapes.eval(weights[i])
    return V

def projection(weights, model):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)

    # project to the manifold
    proj_weights = infer(model, weights)
    
    return proj_weights