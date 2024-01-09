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
        y = model(x)
        if isinstance(y, tuple):
            y = y[0]
        y = y.detach().cpu().numpy()
    return y

def sample_configurations(blendshapes, weights):
    # generate # n_sample random weights
    n_samples = weights.shape[0]
    V = np.zeros((n_samples, *blendshapes.V.shape))
    for i in range(n_samples):
        V[i] = blendshapes.eval(weights[i])
    return V

def manifold_projection(blendshapes, weights, model, return_geometry=True):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    
    n_samples = weights.shape[0]
    
    # project to the manifold
    proj_weights = infer(model, weights)
    
    if not return_geometry:
        return proj_weights
    
    # geometry of the blendshapes
    V_proj = np.zeros((n_samples, *blendshapes.V.shape))
    for i in range(n_samples):
        V_proj[i] = blendshapes.eval(proj_weights[i])
    
    return proj_weights, V_proj

def submanifolds_construction(save_path, clusters):
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    dataset = load_dataset()
    for i, cluster in enumerate(clusters):
        cluster_save_path = os.path.join(save_path, f"cluster_{i}")
        if not os.path.exists(cluster_save_path):
            os.makedirs(cluster_save_path, exist_ok=True)
        train(cluster_save_path, cluster, dataset)

def submanifolds_projection(blendshapes, weights, ensemble, return_geometry=True):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    n_samples = weights.shape[0]
    
    # project to the submanifold
    proj_weights = np.zeros((n_samples, len(blendshapes)))
    for i, (model, cluster) in enumerate(ensemble):
        model = model.to(device)
        proj_weights[:,cluster] = infer(model, weights[:,cluster])
        
    if not return_geometry:
        return proj_weights
    
    # geometry of the blendshapes
    V_proj = np.zeros((n_samples, *blendshapes.V.shape))
    for i in range(n_samples):
        V_proj[i] = blendshapes.eval(proj_weights[i])
    
    return proj_weights, V_proj