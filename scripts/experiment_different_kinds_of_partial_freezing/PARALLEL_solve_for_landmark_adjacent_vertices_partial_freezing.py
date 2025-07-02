import pickle
import numpy as np
import torch 
import sys
sys.path.append("/Users/evanpan/Documents/GitHub/ManifoldExploration/src")
sys.path.append("/Users/evanpan/Documents/GitHub/ManifoldExploration")
sys.path.append("/scratch/ondemand29/evanpan/facial-manifold-learning")
sys.path.append("/scratch/ondemand29/evanpan/facial-manifold-learning/src")
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
from flame_utils import vertices2landmarks
from naive_autosegmentation import compute_vertex_assignments, build_adjacency_list, compute_geodesic_distances

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

def optimize_batched_flame_weights(flame_torch, V_sample_original, V_neutral, 
                                 feature_related_indices, feature_unrelated_vertices, 
                                 batch_size=8, lr=0.1, num_iterations=100):
    """
    Optimize FLAME weights for multiple frames in parallel batches.
    
    Parameters:
    -----------
    flame_torch : FLAME model
    V_sample_original : np.ndarray
        Original vertex samples [num_frames, num_vertices, 3]
    V_neutral : np.ndarray
        Neutral face vertices [num_vertices, 3]
    feature_related_indices : list
        Indices of vertices related to the current feature
    feature_unrelated_vertices : list
        Indices of vertices unrelated to the current feature
    batch_size : int
        Number of frames to process in parallel
    lr : float
        Learning rate
    num_iterations : int
        Number of optimization iterations
        
    Returns:
    --------
    optimized_weights : np.ndarray
        Optimized weights [num_frames, 103]
    """
    num_frames = V_sample_original.shape[0]
    device = flame_torch.device
    
    # Pre-convert targets to tensors and move to device
    local_goals = torch.from_numpy(V_sample_original[:, feature_related_indices, :]).to(device).float()
    non_local_goal = torch.from_numpy(V_neutral[feature_unrelated_vertices, :]).to(device).float()
    
    # Initialize output array
    optimized_weights = np.zeros((num_frames, 103), dtype=np.float32)
    
    # Process in batches
    for batch_start in range(0, num_frames, batch_size):
        batch_end = min(batch_start + batch_size, num_frames)
        current_batch_size = batch_end - batch_start
        
        print(f"Processing batch {batch_start//batch_size + 1}/{(num_frames + batch_size - 1)//batch_size}")
        
        # Initialize parameters for the batch
        shape_params = torch.zeros([current_batch_size, 100]).to(device)
        exp_params = torch.zeros([current_batch_size, 100]).to(device)
        tex_params = torch.zeros([current_batch_size, 50]).to(device)
        pose_params = torch.zeros([current_batch_size, 3]).to(device)
        jaw_params = torch.zeros([current_batch_size, 3]).to(device)
        eye_pose_params = torch.zeros([current_batch_size, 6]).to(device)
        
        exp_params.requires_grad = True
        jaw_params.requires_grad = True
        
        # Get targets for current batch
        batch_local_goals = local_goals[batch_start:batch_end]  # [batch_size, num_local_vertices, 3]
        batch_non_local_goal = non_local_goal.unsqueeze(0).expand(current_batch_size, -1, -1)  # [batch_size, num_non_local_vertices, 3]
        
        optimizer = torch.optim.Adam([exp_params, jaw_params], lr=lr)
        
        start_time = time.time()
        
        for iteration in range(num_iterations):
            # Forward pass for entire batch
            pose_combined = torch.cat([pose_params, jaw_params], dim=1)  # [batch_size, 6]
            vertices_batch, _, _ = flame_torch(shape_params, exp_params, pose_params=pose_combined)
            # vertices_batch shape: [batch_size, num_vertices, 3]
            
            # Compute losses for the batch
            vertices_local = vertices_batch[:, feature_related_indices]  # [batch_size, num_local_vertices, 3]
            vertices_non_local = vertices_batch[:, feature_unrelated_vertices]  # [batch_size, num_non_local_vertices, 3]
            
            loss_local = torch.mean((vertices_local - batch_local_goals)**2)
            loss_non_local = torch.mean((vertices_non_local - batch_non_local_goal)**2) * 10
            loss = loss_local + loss_non_local
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Optional: print progress for first batch or every few iterations
            if batch_start == 0 and (iteration % 20 == 0 or iteration == num_iterations - 1):
                print(f"  Iteration {iteration}, Loss: {loss.item():.6f}")
        
        # Store optimized weights
        optimized_weights[batch_start:batch_end, :100] = exp_params.detach().cpu().numpy()
        optimized_weights[batch_start:batch_end, 100:103] = jaw_params.detach().cpu().numpy()
        
        end_time = time.time()
        print(f"  Batch completed in {end_time - start_time:.2f} seconds, Final loss: {loss.item():.6f}")
    
    return optimized_weights

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

# Configuration parameters
BATCH_SIZE = 128  # Adjust based on your GPU memory
LEARNING_RATE = 0.1
NUM_ITERATIONS = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
print(f"Using device: {device}")

flame = FLAMEBlendshapes()
lmk_indices = flame.F.shape
flame.flame.to(device)
ROOT = "/scratch/ondemand29/evanpan/facial-manifold-learning"
K = 5
facial_landmark_groups = {
    "jaw": list(range(0, 17)),  # 0-16: jawline points
    
    "right_eyebrow": list(range(17, 22)),  # 17-21: right eyebrow
    "left_eyebrow": list(range(22, 27)),   # 22-26: left eyebrow
    
    "nose_bridge": list(range(27, 31)),    # 27-30: nose bridge
    "nose_tip": list(range(31, 36)),       # 31-35: nose tip and nostrils
    
    "right_eye": list(range(36, 42)),      # 36-41: right eye
    "left_eye": list(range(42, 48)),       # 42-47: left eye
    
    "lip": list(range(48, 68)),      # 48-67: outer lip contour
}

# load the pickle_dataset:
dataset_path = os.path.join(ROOT, "data/MeadRavdess/val_mead_ravdess_0.1.pickle")
with open(dataset_path, "rb") as f:
    data = pickle.load(f)

# load flame masks
mask_path = os.path.join(ROOT, "data/flame_model/FLAME_masks/FLAME_masks.pkl")
with open(mask_path, "rb") as f:
    mask = pickle.load(f, encoding="latin1")

# get the keys of the data
data_keys = list(data.keys())
 
# get 200 sample
np.random.seed(42)  # for reproducibility
random_indices = np.random.choice(len(data_keys), 200, replace=False).tolist()
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

# evaluate the blendshapes at the sampled weights 
V_sample_i_original = flame.V
V_neutral = flame.V
__, lm_neutral = flame.eval(return_landmarks=True)
V_sample_i_original = np.expand_dims(V_sample_i_original, axis=0)
V_sample_i_original = [V_sample_i_original]
lm_sample_i_original = [np.expand_dims(lm_neutral, axis=0)]
for i in range(0, len(weight)):
    V_frame_i, lm_frame_i = flame.eval(weight[i], return_landmarks=True)
    V_sample_i_original.append(np.expand_dims(V_frame_i, axis=0))
    lm_sample_i_original.append(np.expand_dims(lm_frame_i, axis=0))

V_sample_i_original = np.concatenate(V_sample_i_original, axis=0)
lm_sample_i_original = np.concatenate(lm_sample_i_original, axis=0)

# obtain the landmarks group names
facial_landmark_groups_keys = list(facial_landmark_groups.keys())

# partically freeze the vertices based on landmark groups
for feature in facial_landmark_groups_keys:
    print(f"\n=== Processing feature: {feature} ===")
    
    # compute vertices that are involved with the feature
    local_features = facial_landmark_groups[feature]
    landmark_face_indices = flame.flame.full_lmk_faces_idx  # the indices of the faces that contain the landmarks
    locally_involved_vertices = []
    for ldmk_i in range(len(local_features)):
        points = flame.F[landmark_face_indices[0, local_features[ldmk_i]]].tolist()
        locally_involved_vertices += points
    locally_involved_vertices = np.array(list(set(locally_involved_vertices)))
    # get the non_loaclly involved vertices
    non_local_features = list(set(range(0, 68)) - set(local_features))
    non_locally_involved_vertices = []
    for ldmk_i in range(len(non_local_features)):
        points = flame.F[landmark_face_indices[0, non_local_features[ldmk_i]]].tolist()
        non_locally_involved_vertices += points
    non_locally_involved_vertices = np.array(list(set(non_locally_involved_vertices)))

    # compute the vertex assignments
    feature_related_indices, feature_unrelated_vertices = compute_vertex_assignments(
        flame.V, flame.F, locally_involved_vertices, non_locally_involved_vertices, K=K)
    feature_related_indices = list(feature_related_indices)
    feature_unrelated_vertices = list(feature_unrelated_vertices)

    # optimize the flame weight to fit the frozen sample using batched optimization
    flame_torch = flame.flame
    
    print(f"Optimizing {weight.shape[0]} frames with batch size {BATCH_SIZE}")
    optimized_weight = optimize_batched_flame_weights(
        flame_torch, V_sample_i_original, V_neutral,
        feature_related_indices, feature_unrelated_vertices,
        batch_size=BATCH_SIZE, lr=LEARNING_RATE, num_iterations=NUM_ITERATIONS
    )

    # animate the optimized weight
    v_sample_i_optimized = flame.V
    V_neutral = flame.V
    v_sample_i_optimized = np.expand_dims(v_sample_i_optimized, axis=0)
    v_sample_i_optimized = [v_sample_i_optimized]
    for i in range(0, len(optimized_weight)):
        v_sample_i_optimized.append(np.expand_dims(flame.eval(optimized_weight[i]), axis=0))
    v_sample_i_optimized = np.concatenate(v_sample_i_optimized, axis=0)

    # save the optimized weight
    save_dir = os.path.join(ROOT, f"experiments/full_face_bs_test_freeze_landmarks_and_{K}_ajacent_200_videos/")
    os.makedirs(save_dir, exist_ok=True)
    partially_frozened_model_weights = os.path.join(save_dir, "bs_for_{}".format(feature + ".npy"))
    np.save(partially_frozened_model_weights, optimized_weight)
    
    print(f"Completed feature: {feature}")