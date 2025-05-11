import pickle
import numpy as np
import torch 
import sys
sys.path.append("/Users/evanpan/Documents/GitHub/ManifoldExploration/src")
sys.path.append("/Users/evanpan/Documents/GitHub/ManifoldExploration")
from blendshapes import FLAMEBlendshapes, BasicBlendshapes
import polyscope as ps
import polyscope.imgui as psim
from scripts.polyscope_playback import MeshAnimator, MultiMeshAnimator
from flame_utils import *
import copy
from sklearn.decomposition import PCA
import os   

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


surrogate_model_root_path = "/Users/evanpan/Documents/GitHub/ManifoldExploration/experiments/linear_surrogate_test/"

neutral_path = os.path.join(surrogate_model_root_path, "linear_surrogate_mean.npy")
blendshape_path = os.path.join(surrogate_model_root_path, "linear_surrogate.npy")
face_path = os.path.join(surrogate_model_root_path, "linear_surrogate_Face.npy")
neutral = np.load(neutral_path)
blendshapes = np.load(blendshape_path)
F = np.load(face_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
names_id = np.arange(blendshapes.shape[0])
names = ["blendshape_" + str(i) for i in names_id] 
surrogate_model = BasicBlendshapes(neutral, F, blendshapes, names=names)
V0 = surrogate_model.V
n_blendshapes  = len(surrogate_model)
weights        = np.zeros(n_blendshapes, dtype=float)


flame = FLAMEBlendshapes()
flame_weights = np.zeros([1, 103])


ps.set_verbosity(0)
ps.init()
ps.set_ground_plane_mode("none")
ps.set_view_projection_mode("orthographic")
ps.set_front_dir("z_front")
ps.set_background_color([0, 0, 0])

V0  = surrogate_model.eval(weights)
SM0 = ps.register_surface_mesh(
    "face", V0, surrogate_model.F,
    color=[0.9, 0.9, 0.9], smooth_shade=True,
    edge_width=0.25, material="normal"
)
flame.translation = np.array([0.2, 0, 0])
flame_torch = flame.flame
V_FLAME = flame.eval(flame_weights[0])
SM_FLAME = ps.register_surface_mesh(
    "face_flame", V_FLAME, flame.F,
    color=[0.9, 0.9, 0.9], smooth_shade=True,
    edge_width=0.25, material="normal"
)
# takes in V of the surrogate model and returns the FLAME weights
def run_controller(V_surrogate) -> np.ndarray:
    """Runs the VAE/MLP controller once and returns a flat numpy vector."""
    shape_params = torch.zeros([1, 100]).to(flame_torch.device)
    tex_params = torch.zeros([1, 50]).to(flame_torch.device)
    pose_params = torch.zeros([1, 3]).to(flame_torch.device)
    eye_pose_params = torch.zeros([1, 6]).to(flame_torch.device)
    # optimizing the flame parameters to match the surrogate model
    # exp_params_iter, jaw_params_iter = optimize_flame_weights(flame_torch, shape_params, pose_params, v_out_surrogate[0].to(flame_torch.device), steps=200)
    V_surrogate_torch = torch.from_numpy(V_surrogate).float().to(flame_torch.device)
    exp_params, jaw_params = solve_flame_params_direct(flame_torch, V_surrogate_torch)
    flame_weights = torch.zeros([1, 103]).to(flame_torch.device)
    flame_weights[0, :100] = exp_params
    flame_weights[0, 100:103] = jaw_params
    flame_weights = flame_weights.detach().cpu().numpy()
    return flame_weights

def gui():
    global weights, selection_threshold, current_frame, last_slider_index, flame_weights

    # ------------------------------------------------ Reset
    if psim.Button("Reset to Canonical"):
        weights[:]           = 0.0
        current_frame        = 0
        last_slider_index    = 0
        SM0.update_vertex_positions(surrogate_model.eval(weights))
        flame_weights = run_controller(surrogate_model.eval(weights))
        SM_FLAME.update_vertex_positions(flame.eval(flame_weights[0]))


    # ------------------------------------------------ Alpha cutoff


    # ------------------------------------------------ Blendshape sliders
    for i, name in enumerate(surrogate_model.names):
        changed_bs, new_val = psim.SliderFloat(name, float(weights[i]), -1, 1.0)
        if changed_bs:
            last_slider_index = i
            weights[i]        = new_val                 # keep user edit
            SM0.update_vertex_positions(surrogate_model.eval(weights))
            flame_weights = run_controller(surrogate_model.eval(weights))
            SM_FLAME.update_vertex_positions(flame.eval(flame_weights[0]))


# ---------------------------------------------------------------------
ps.set_user_callback(gui)
ps.show()


