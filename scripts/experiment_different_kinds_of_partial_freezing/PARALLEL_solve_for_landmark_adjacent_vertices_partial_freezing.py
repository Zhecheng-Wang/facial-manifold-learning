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

class BatchedManualAdam:
    """Manual Adam implementation that handles batched parameters with independent momentum states."""
    
    def __init__(self, batch_size, param_shapes, lr=0.001, betas=(0.9, 0.999), eps=1e-8, device='cuda', dtype=torch.float32):
        """
        Initialize BatchedManualAdam with configurable precision.
        
        Args:
            batch_size: Number of batches
            param_shapes: List of parameter shapes
            lr: Learning rate
            betas: Adam beta parameters
            eps: Epsilon for numerical stability
            device: Device to use
            dtype: torch.dtype, either torch.float32 or torch.float64 (double)
        """
        self.batch_size = batch_size
        self.param_shapes = param_shapes  # List of shapes, e.g. [(100,), (3,)]
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.t = 0
        
        # Initialize moment estimates for each frame independently
        self.m = []
        self.v = []
        
        for shape in param_shapes:
            # Create momentum tensors for all frames in batch
            m_shape = (batch_size,) + shape
            self.m.append(torch.zeros(m_shape, device=device, dtype=dtype))
            self.v.append(torch.zeros(m_shape, device=device, dtype=dtype))
    
    def step(self, params, grads):
        """
        params: list of tensors with shape [batch_size, ...]
        grads: list of tensors with shape [batch_size, ...]
        """
        self.t += 1
        
        with torch.no_grad():
            for i, (param, grad) in enumerate(zip(params, grads)):
                # Update biased first moment estimate
                self.m[i].mul_(self.betas[0]).add_(grad, alpha=1 - self.betas[0])
                # Update biased second raw moment estimate
                self.v[i].mul_(self.betas[1]).addcmul_(grad, grad, value=1 - self.betas[1])
                
                # Compute bias-corrected moment estimates
                m_hat = self.m[i] / (1 - self.betas[0]**self.t)
                v_hat = self.v[i] / (1 - self.betas[1]**self.t)
                
                # Update parameters
                param.add_(m_hat / (torch.sqrt(v_hat) + self.eps), alpha=-self.lr)

def optimize_batched_flame_weights_manual_adam(flame_torch, V_sample_original, V_neutral, 
                                             feature_related_indices, feature_unrelated_vertices, 
                                             batch_size=8, lr=0.001, num_iterations=100):
    """
    Optimize FLAME weights with truly independent Adam states per frame using manual implementation.
    """
    num_frames = V_sample_original.shape[0]
    device = flame_torch.device
    
    # Pre-convert targets to tensors and move to device
    local_goals = torch.from_numpy(V_sample_original[:, feature_related_indices, :]).to(device).double()
    non_local_goal = torch.from_numpy(V_neutral[feature_unrelated_vertices, :]).to(device).double()
    # Initialize output array
    optimized_weights = np.zeros((num_frames, 103), dtype=np.float64)
    
    # Process in batches
    for batch_start in range(0, num_frames, batch_size):
        batch_end = min(batch_start + batch_size, num_frames)
        current_batch_size = batch_end - batch_start
        
        print(f"Processing batch {batch_start//batch_size + 1}/{(num_frames + batch_size - 1)//batch_size}")
        
        # Initialize parameters for the batch
        exp_params = torch.zeros([current_batch_size, 100], device=device, requires_grad=True, dtype=torch.double)
        jaw_params = torch.zeros([current_batch_size, 3], device=device, requires_grad=True, dtype=torch.double)
        # Create manual Adam optimizer with independent states for each frame
        param_shapes = [(100,), (3,)]
        optimizer = BatchedManualAdam(current_batch_size, param_shapes, lr=lr, device=device, dtype=torch.double)
        
        # Get targets for current batch
        batch_local_goals = local_goals[batch_start:batch_end]
        batch_non_local_goal = non_local_goal.unsqueeze(0).expand(current_batch_size, -1, -1)
        
        start_time = time.time()
        
        for iteration in range(num_iterations):
            # Create other required parameters
            shape_params = torch.zeros([current_batch_size, 100], device=device).double()
            pose_params = torch.zeros([current_batch_size, 3], device=device).double()
            
            # Forward pass for entire batch
            pose_combined = torch.cat([pose_params, jaw_params], dim=1)
            vertices_batch, _, _ = flame_torch(shape_params, exp_params, pose_params=pose_combined)
            
            # Compute losses for the batch
            vertices_local = vertices_batch[:, feature_related_indices]
            vertices_non_local = vertices_batch[:, feature_unrelated_vertices]
            


            # Compute per-frame losses
            # loss_local = torch.mean((vertices_local - batch_local_goals)**2, dim=[1,2])
            # loss_non_local = torch.mean((vertices_non_local - batch_non_local_goal)**2, dim=[1,2]) * 10

            loss_local = torch.mean((vertices_local - batch_local_goals)**2, dim=[1,2]).sum()
            loss_non_local = torch.mean((vertices_non_local - non_local_goal.unsqueeze(0).expand(current_batch_size, -1, -1))**2, dim=[1,2]).sum() * 10
            
            losses = loss_local + loss_non_local
            total_loss = losses
            
            # Backward pass
            total_loss.backward()
            # Manual Adam step with independent momentum for each frame
            optimizer.step([exp_params, jaw_params], [exp_params.grad, jaw_params.grad])
            # Zero gradients
            exp_params.grad.zero_()
            jaw_params.grad.zero_()
            
            if batch_start == 0 and (iteration % 20 == 0 or iteration == num_iterations - 1):
                print(f"  Iteration {iteration}, Loss: {total_loss.item():.6f}")
        
        # Store optimized weights
        optimized_weights[batch_start:batch_end, :100] = exp_params.float().detach().cpu().numpy()
        optimized_weights[batch_start:batch_end, 100:103] = jaw_params.float().detach().cpu().numpy()
        
        end_time = time.time()
        print(f"  Batch completed in {end_time - start_time:.2f} seconds, Final loss: {total_loss.item():.6f}")
    
    return optimized_weights


# Configuration parameters
BATCH_SIZE = 512  # Adjust based on your GPU memory
LEARNING_RATE = 0.5
NUM_ITERATIONS = 150
NUM_FRAMES = 200 # -1 means all frames
K = 10

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")        
print(f"Using device: {device}")

flame = FLAMEBlendshapes(torch.double, device=device)
lmk_indices = flame.F.shape
flame.flame.to(device)
ROOT = "/scratch/ondemand29/evanpan/facial-manifold-learning"
# ROOT = "/Users/evanpan/Documents/GitHub/ManifoldExploration"

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
if NUM_FRAMES > 0:
    random_indices = np.random.choice(len(data_keys), NUM_FRAMES, replace=False).tolist()
else:
    random_indices = np.arange(len(data_keys)).tolist()  # use all samples
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
    # feature = facial_landmark_groups_keys[0]
    print(f"\n=== Processing feature: {feature} ===")
    
    # compute vertices that are involved with the feature
    local_features = facial_landmark_groups[feature]
    landmark_face_indices = flame.flame.full_lmk_faces_idx  # the indices of the faces that contain the landmarks
    locally_involved_vertices = []
    for ldmk_i in range(len(local_features)):
        points = flame.F[landmark_face_indices[0, local_features[ldmk_i]]].tolist()
        locally_involved_vertices += points
    locally_involved_vertices = np.array(list(set(locally_involved_vertices))) # verteice indices that are involved with the feature
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
    
    V_sample_i_original = V_sample_i_original.astype(np.float64)
    V_neutral = V_neutral.astype(np.float64)
    
    optimized_weight = optimize_batched_flame_weights_manual_adam(
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
    save_dir = os.path.join(ROOT, f"experiments/full_face_bs_test_freeze_landmarks_and_{K}_ajacent_200_video/")
    os.makedirs(save_dir, exist_ok=True)
    partially_frozened_model_weights = os.path.join(save_dir, "bs_for_{}".format(feature + ".npy"))
    np.save(partially_frozened_model_weights, optimized_weight)
    
    print(f"Completed feature: {feature}")