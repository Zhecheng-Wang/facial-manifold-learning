import pickle
import numpy as np
import torch 
import sys
sys.path.append("/Users/evanpan/Documents/GitHub/ManifoldExploration/src")
sys.path.append("/Users/evanpan/Documents/GitHub/ManifoldExploration")
from blendshapes import FLAMEBlendshapes, BasicBlendshapes, BasicBlendshapesTorch
from flame_utils import *
import polyscope as ps
import polyscope.imgui as psim
from scripts.polyscope_playback import MeshAnimator, MultiMeshAnimator
import copy
from sklearn.decomposition import PCA
import os   


def optimize_flame_weights(flame_torch, shape_params, pose_params, V_target, steps=100):
    exp_params.requires_grad = True
    jaw_params.requires_grad = True
    exp_params_start = torch.zeros([1, 100]).to(flame_torch.device)
    jaw_params_start = torch.zeros([1, 3]).to(flame_torch.device)
    exp_params_start.requires_grad = True
    jaw_params_start.requires_grad = True
    optimizer = torch.optim.Adam([exp_params_start, jaw_params_start], lr=0.5)
    for i in range(steps):
        vertices, landmarks2d, landmarks3d = flame_torch(shape_params, exp_params_start, pose_params=torch.concat([pose_params, jaw_params_start], dim=1))
        loss = torch.mean((vertices - V_target) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("step: ", i, "loss: ", loss.item())
    
    return exp_params_start, jaw_params_start

def solve_flame_params_direct(flame_model, V_target):
    """
    Solve for expression parameters and jaw parameters using a direct least squares solution.
    This is more efficient than the iterative approach for a purely linear model.
    
    Parameters:
    -----------
    flame_model : FLAME
        An initialized FLAME model instance
    V_target : torch.Tensor
        Target vertex configuration of shape (V, 3)
    
    Returns:
    --------
    exp_params : torch.Tensor
        Optimized expression parameters
    jaw_params : torch.Tensor
        Optimized jaw pose parameters
    """
    device = V_target.device
    
    # Extract expression blendshapes and jaw pose blendshapes
    exp_blendshapes, jaw_pose_blendshapes, mean_shape = get_flame_blendshapes(flame_model)
    
    # Ensure target vertices are properly formatted
    V_target = V_target.reshape(-1, 3)
    
    # Compute delta from mean shape
    delta_V = V_target - mean_shape
    
    # Reshape blendshapes to construct the linear system
    n_vertices = mean_shape.shape[0]
    n_exp = exp_blendshapes.shape[2]
    
    # Reshape exp_blendshapes from [n_vertices, 3, n_exp] to [n_vertices*3, n_exp]
    exp_basis = exp_blendshapes.reshape(-1, n_exp)
    
    # Reshape jaw_pose_blendshapes from [n_vertices, 3, 3] to [n_vertices*3, 3]
    jaw_basis = jaw_pose_blendshapes.reshape(-1, 3)
    
    # Concatenate bases to form the full linear system
    full_basis = torch.cat([exp_basis, jaw_basis], dim=1)
    
    # Reshape delta_V to [n_vertices*3]
    delta_V_flat = delta_V.reshape(-1)
    
    # Solve the least squares problem: min ||full_basis @ params - delta_V_flat||^2
    # Using torch.linalg.lstsq for a more stable solution
    solution, residuals, rank, singular_values = torch.linalg.lstsq(full_basis, delta_V_flat.unsqueeze(1))
    
    # Extract parameters from solution
    exp_params = solution[:n_exp].reshape(1, n_exp)
    jaw_params = solution[n_exp:].reshape(1, 3)
    
    return exp_params, jaw_params


# load surrogate model
surrogate_model_root_path = "/Users/evanpan/Documents/GitHub/ManifoldExploration/experiments/linear_surrogate_test/"
neutral_path = os.path.join(surrogate_model_root_path, "linear_surrogate_mean.npy")
blendshape_path = os.path.join(surrogate_model_root_path, "linear_surrogate.npy")
face_path = os.path.join(surrogate_model_root_path, "linear_surrogate_Face.npy")

V_neutral = np.load(neutral_path)
V_blendshapes = np.load(blendshape_path)
F = np.load(face_path)
names = np.arange(V_blendshapes.shape[0])
names = ["blendshape_" + str(i) for i in names]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
surrogate_model = BasicBlendshapesTorch(V_neutral, F, V_blendshapes, names)
surrogate_model.to(device)
surrogate_model.requires_grad_(False)

w = torch.zeros([surrogate_model.blendshapes.shape[0]]).to(device)
# output of the surrogate model
v_out_surrogate = surrogate_model.eval(w)

# load flame model
flame = FLAMEBlendshapes()
flame_torch = flame.flame
shape_params = torch.zeros([1, 100]).to(flame_torch.device)
tex_params = torch.zeros([1, 50]).to(flame_torch.device)
pose_params = torch.zeros([1, 3]).to(flame_torch.device)
eye_pose_params = torch.zeros([1, 6]).to(flame_torch.device)


# optimizing the flame parameters to match the surrogate model
# exp_params_iter, jaw_params_iter = optimize_flame_weights(flame_torch, shape_params, pose_params, v_out_surrogate[0].to(flame_torch.device), steps=200)
exp_params, jaw_params = solve_flame_params_direct(flame_torch, v_out_surrogate[0].to(flame_torch.device))

weight_flame = torch.zeros([1, 103]).to(flame_torch.device)
weight_flame[0, :100] = exp_params
weight_flame[0, 100:103] = jaw_params
weight_flame = weight_flame.detach().cpu().numpy()
v_out_flame = flame.eval(weight_flame[0])


# get a UI 


