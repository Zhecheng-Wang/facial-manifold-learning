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
import time

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
        

flame = FLAMEBlendshapes()

# load the pickle_dataset:
dataset_path = "/Users/evanpan/Documents/GitHub/ManifoldExploration/data/MeadRavdess/val_mead_ravdess_0.1.pickle"
with open(dataset_path, "rb") as f:
    data = pickle.load(f)

# load flame mask
mask_path = "/Users/evanpan/Documents/GitHub/ManifoldExploration/data/flame_model/FLAME_masks/FLAME_masks.pkl"
with open(mask_path, "rb") as f:
    mask = pickle.load(f, encoding="latin1")

# get the keys of the data
data_keys = list(data.keys())

# get 200 sample
np.random.seed(42)  # for reproducibility
random_indices = np.random.choice(len(data_keys), 100, replace=False).tolist()
samples = []
exp = []
jaw = []
for i in range(len(random_indices)):
    sample = data[data_keys[random_indices[i]]]
    exp.append(sample["exp"][0])
    jaw.append(sample["jaw"][0])
exp = np.concatenate(exp, axis=0)
jaw = np.concatenate(jaw, axis=0)
weight = np.concatenate([exp, jaw], axis=1, dtype=np.float32)

# animate it in polyscope 

# convert to vertex space
V_sample_i_original = flame.V
V_neutral = flame.V
V_sample_i_original = np.expand_dims(V_sample_i_original, axis=0)
V_sample_i_original = [V_sample_i_original]
for i in range(0, len(weight)):
    V_sample_i_original.append(np.expand_dims(flame.eval(weight[i]), axis=0))
V_sample_i_original = np.concatenate(V_sample_i_original, axis=0)

# freeze all but lips
for feature in mask.keys():
    V_sample_i_altered = copy.deepcopy(V_sample_i_original)
    feature_vertices = mask[feature]
    not_feature_vertices = np.delete(np.arange(V_sample_i_altered.shape[1]), feature_vertices)
    V_sample_i_altered[:, not_feature_vertices, :] = V_neutral[not_feature_vertices, :]
    V_sample_i_altered = torch.from_numpy(V_sample_i_altered).float().to(flame.flame.device)
    V_sample_i_altered.shape
    # optimize the flame weight to fit the frozen sample
    flame_torch = flame.flame
    shape_params = torch.zeros([1, 100]).to(flame_torch.device)
    exp_params = torch.zeros([1, 100]).to(flame_torch.device)
    tex_params = torch.zeros([1, 50]).to(flame_torch.device)
    pose_params = torch.zeros([1, 3]).to(flame_torch.device)
    jaw_params = torch.zeros([1, 3]).to(flame_torch.device)
    eye_pose_params = torch.zeros([1, 6]).to(flame_torch.device)
    optimized_weight = torch.zeros(weight.shape).to(flame_torch.device)

    for frame_i in range(weight.shape[0]):
        start_time = time.time()
        # for i in range(100):
        #     vertices, landmarks2d, landmarks3d = flame_torch(shape_params, exp_params, pose_params=torch.concat([pose_params, jaw_params], dim=1))
        #     loss_local = torch.mean((vertices[0, feature_vertices, :] - V_sample_i_altered[frame_i, feature_vertices, :])**2)
        #     loss_non_local = torch.mean((vertices[0, not_feature_vertices, :] - V_sample_i_altered[frame_i, not_feature_vertices, :])**2)
        #     loss = loss_local + loss_non_local
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        exp_params, jaw_params = solve_flame_params_direct(flame.flame, V_sample_i_altered[frame_i])
        
        optimized_weight[frame_i, :100] = exp_params.data
        optimized_weight[frame_i, 100:103] = jaw_params.data

        end_time = time.time()
        print("frame: ", frame_i, "time per frame: ", end_time - start_time)
    optimized_weight = optimized_weight.detach().numpy()

    # animate the optimized weight
    v_sample_i_optimized = flame.V
    V_neutral = flame.V
    v_sample_i_optimized = np.expand_dims(v_sample_i_optimized, axis=0)
    v_sample_i_optimized = [v_sample_i_optimized]
    v_sample_i_optimized[0].shape
    for i in range(0, len(optimized_weight)):
        v_sample_i_optimized.append(np.expand_dims(flame.eval(optimized_weight[i]), axis=0))
    v_sample_i_optimized = np.concatenate(v_sample_i_optimized, axis=0)

    # save the optimized weight
    partially_frozened_model_weights = "/Users/evanpan/Documents/GitHub/ManifoldExploration/experiments/full_face_bs_test/bs_for_" + feature + ".npy"
    np.save(partially_frozened_model_weights, optimized_weight)

masks_we_care_about = ["eye_region", "lips", "nose", "forehead"]
for feature in masks_we_care_about:
    partially_frozened_model_weights = "/Users/evanpan/Documents/GitHub/ManifoldExploration/experiments/full_face_bs_test/bs_for_" + feature + ".npy"
    # animator = MultiMeshAnimator([(V_sample_i_original, F), (V_sample_i_altered, F), (v_sample_i_optimized, F)], offset_distance=0.2)
    # animator.run()
    optimized_weight_for_feature_i = np.load(partially_frozened_model_weights)

    V_sample = []
    for i in range(0, len(optimized_weight_for_feature_i)):
        V_sample.append(np.expand_dims(flame.eval(optimized_weight_for_feature_i[i]), axis=0))
    V_sample = np.concatenate(V_sample, axis=0)
    V_sample = V_sample - V_neutral

    # generate a linear surrogate of the blendshapes (i.e. local blendshapes)
    pca = PCA(n_components=10)
    V_sample = V_sample.reshape(V_sample.shape[0], -1)  # flatten the vertices

    pca = pca.fit(V_sample)
    pca.mean_ = V_neutral
    # # take the first K components
    K = 0
    explained_variance_counter = 0
    for i in range(0, pca.components_.shape[0]):
        explained_variance_counter += pca.explained_variance_ratio_[i]
        print(explained_variance_counter)
        if explained_variance_counter >= 0.99:
            K = i + 1
            break
    print("K: ", K, "explained variance: ", explained_variance_counter)

    linear_surrogate = pca.components_[:K, :]
    print("linear surrogate shape: ", linear_surrogate.shape)
    linear_surrogate.shape
    linear_surrogate_mean = pca.mean_
    linear_surrogate.shape
    linear_surrogate = linear_surrogate.reshape(K, -1, 3)
    linear_surrogate_mean = linear_surrogate_mean.reshape(-1, 3)
    # scale the components to ensure when blend weight is 1, the surrogate is (3*std) times the original
    singular_values = pca.singular_values_[:K]
    linear_surrogate = linear_surrogate * singular_values[:K].reshape(K, 1, 1) * 3
    # save these as a blendshape
    surrogate_model_root_path = "/Users/evanpan/Documents/GitHub/ManifoldExploration/experiments/full_face_bs_test/"
    # os.makedirs(surrogate_model_root_path, exist_ok=True)

    np.save(surrogate_model_root_path + f"linear_surrogate_{feature}.npy", linear_surrogate)
    np.save(surrogate_model_root_path + f"linear_surrogate_mean_{feature}.npy", linear_surrogate_mean)
    np.save(surrogate_model_root_path + f"linear_surrogate_Face.npy", flame.F)

