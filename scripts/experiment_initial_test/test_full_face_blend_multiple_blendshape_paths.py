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
from scipy.interpolate import interp1d

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

def display_a_single_mesh(V, F):
    ps.remove_all_structures()
    ps.set_verbosity(0)
    ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_view_projection_mode("orthographic")
    ps.set_front_dir("z_front")
    ps.set_background_color([0, 0, 0])
    ps.register_surface_mesh(
        "mesh", V, F,
        color=[0.9, 0.9, 0.9],
        edge_width=0.25, material="normal"
        )
    ps.show()
    
controller_range = [-0.03, 0.03]

surrogate_model_root_path = "/Users/evanpan/Documents/GitHub/ManifoldExploration/experiments/full_face_bs_test/"

feature_we_care_about = ["eye_region", "lips", "nose", "forehead"]
face_path = os.path.join(surrogate_model_root_path, "linear_surrogate_Face.npy")
bs_per_feature = {}
for feature in feature_we_care_about:
    neutral_path = os.path.join(surrogate_model_root_path, f"linear_surrogate_mean_{feature}.npy")
    blendshape_path = os.path.join(surrogate_model_root_path, f"linear_surrogate_{feature}.npy")
    face_path = os.path.join(surrogate_model_root_path, "linear_surrogate_Face.npy")
    neutral = np.load(neutral_path)
    blendshapes = np.load(blendshape_path)
    F = np.load(face_path)
    bs_per_feature[feature] = {
        "neutral": neutral,
        "blendshapes": blendshapes,
        "F": F
    }
all_blendshapes = []
for feature in feature_we_care_about:
    all_blendshapes.append(bs_per_feature[feature]["blendshapes"])
blendshapes = np.concatenate(all_blendshapes, axis=0)
neutral = bs_per_feature["lips"]["neutral"]
F = bs_per_feature["lips"]["F"]



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
names_id = np.arange(blendshapes.shape[0])
names = ["blendshape_" + str(i) for i in names_id] 
surrogate_model = BasicBlendshapes(neutral, F, blendshapes, names=names)
V0 = surrogate_model.V.copy()
n_blendshapes  = len(surrogate_model)
weights        = np.zeros(n_blendshapes, dtype=float)
flame = FLAMEBlendshapes()
flame_weights = np.zeros([1, 103])
flame_torch = flame.flame.to(device)

# obtain a path in blendshape space
geometry_space_path = []
weight_range = np.linspace(controller_range[0], controller_range[1], 100)
for bs_i in range(0, n_blendshapes):
    path_for_bs_i = []
    weight_range = weight_range
    weights = np.zeros(n_blendshapes, dtype=float)
    for w in weight_range:
        weights[bs_i] = w
        V = surrogate_model.eval(weights)
        path_for_bs_i.append(V)
    geometry_space_path.append(np.array(path_for_bs_i))
geometry_space_path = np.array(geometry_space_path)

# animator = MeshAnimator(geometry_space_path[0], F)
# animator.run()



exp_params, jaw_params = solve_flame_params_direct(flame.flame, torch.from_numpy(V0))
flame_zero = np.zeros([1, 103])
flame_zero[0, :100] = exp_params.cpu().numpy()
flame_zero[0, 100:103] = jaw_params.cpu().numpy()

# obtain the path in flame space
flame_space_path = []
geometry_space_of_solve_flame_path = []
for bs_i in range(0, n_blendshapes):
    flame_path_for_bs_i = []
    geometry_space_of_solve_flame_path_for_bs_i = []
    for f_i in range(0, geometry_space_path.shape[1]):
        V_target = geometry_space_path[bs_i, f_i]
        V_target = torch.tensor(V_target, device=device)
        exp_params, jaw_params = solve_flame_params_direct(flame.flame, V_target)
        flame_weights[0, :100] = exp_params.cpu().numpy()
        flame_weights[0, 100:103] = jaw_params.cpu().numpy()
        V_flame = flame.eval(flame_weights[0])
        flame_path_for_bs_i.append(flame_weights.copy() - flame_zero)
        geometry_space_of_solve_flame_path_for_bs_i.append(V_target.cpu().numpy())
    flame_space_path.append(np.array(flame_path_for_bs_i))
    geometry_space_of_solve_flame_path.append(geometry_space_of_solve_flame_path_for_bs_i)

flame_space_path = np.array(flame_space_path)
geometry_space_of_solve_flame_path = np.array(geometry_space_of_solve_flame_path)

# create interp for flame_space_path
flame_space_path_interp = []
for i in range(n_blendshapes):
    flame_space_path_interp.append(
        interp1d(weight_range, flame_space_path[i, :, 0], axis=0, fill_value="extrapolate", bounds_error=False)
    )

# related to UI

def run_controller(flame_space_path_interp, weights) -> np.ndarray:
    """Runs the VAE/MLP controller once and returns a flat numpy vector."""
    # optimizing the flame parameters to match the surrogate model
    # exp_params_iter, jaw_params_iter = optimize_flame_weights(flame_torch, shape_params, pose_params, v_out_surrogate[0].to(flame_torch.device), steps=200)
    flame_zero_weights = flame_zero.copy()
    for i in range(len(weights)):
        flame_weights_i = flame_space_path_interp[i](weights[i])
        flame_zero_weights += flame_weights_i
    return flame_zero_weights

def gui():
    global weights, selection_threshold, flame_zero, current_frame, last_slider_index, flame_weights, flame_space_path_interp

    # ------------------------------------------------ Reset
    if psim.Button("Reset to Canonical"):
        weights[:]           = 0.0
        current_frame        = 0
        last_slider_index    = 0
        SM0.update_vertex_positions(surrogate_model.eval(weights))
        flame_weights = run_controller(flame_space_path_interp, weights)
        SM_FLAME.update_vertex_positions(flame.eval(flame_weights[0]))


    # ------------------------------------------------ Alpha cutoff


    # ------------------------------------------------ Blendshape sliders
    for i, name in enumerate(surrogate_model.names):
        changed_bs, new_val = psim.SliderFloat(name, float(weights[i]), controller_range[0], controller_range[1])
        if changed_bs:
            last_slider_index = i
            weights[i]        = new_val                 # keep user edit
            SM0.update_vertex_positions(surrogate_model.eval(weights))
            flame_weights = run_controller(flame_space_path_interp, weights)
            SM_FLAME.update_vertex_positions(flame.eval(flame_weights[0]))
        # print(weights)


flame = FLAMEBlendshapes()
flame_weights = np.zeros([1, 103])

ps.remove_all_structures()
ps.init()
ps.set_verbosity(0)
ps.set_ground_plane_mode("none")
ps.set_view_projection_mode("orthographic")
ps.set_front_dir("z_front")
ps.set_background_color([0, 0, 0])

V0  = surrogate_model.eval(weights)
SM0 = ps.register_surface_mesh(
    "face", V0, surrogate_model.F,
    color=[0.4, 0.4, 0.4], smooth_shade=False,
    edge_width=0.25, material="normal"
)
flame.translation = np.array([0.2, 0, 0])
flame_torch = flame.flame
V_FLAME = flame.eval(flame_zero[0])
SM_FLAME = ps.register_surface_mesh(
    "face_flame", V_FLAME, flame.F,
    color=[0.9, 0.9, 0.9], smooth_shade=True,
    edge_width=0.25, material="normal"
)
# ---------------------------------------------------------------------
ps.set_user_callback(gui)
ps.show()

