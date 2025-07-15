import numpy as np
import os, sys
sys.path.append("/Users/evanpan/Documents/GitHub/ManifoldExploration/src")
from utils import load_ARKit_blendshape
from blendshapes import FLAMEBlendshapes, BasicBlendshapes
import torch
import polyscope as ps
import polyscope.imgui as psim
from scripts.experiment_different_kinds_of_partial_freezing.naive_autosegmentation import compute_vertex_assignments
import pickle

def get_lm_indices_and_bary_weights_from_FLAME():
    """
    Returns the indices of the FLAME landmarks and their barycentric weights.
    """
    FLAME_facial_landmark_groups = {
        "jaw": list(range(7, 10)),  # 0-16: jawline points
        
        "right_eyebrow": list(range(17, 22)),  # 17-21: right eyebrow
        "left_eyebrow": list(range(22, 27)),   # 22-26: left eyebrow
        
        "nose_bridge": list(range(27, 31)),    # 27-30: nose bridge
        "nose_tip": list(range(31, 36)),       # 31-35: nose tip and nostrils
        
        "right_eye": list(range(36, 42)),      # 36-41: right eye
        "left_eye": list(range(42, 48)),       # 42-47: left eye
        
        "lip": list(range(48, 68)),      # 48-67: outer lip contour
    }
    lms_we_care_about = []
    for key in FLAME_facial_landmark_groups.keys():
         # these are the row indices after we use barycentric coordinates to map to the landmarks
        lms_we_care_about.extend(FLAME_facial_landmark_groups[key])
    lms_we_care_about = np.array(lms_we_care_about, dtype=np.int64)
    flameBS = FLAMEBlendshapes()
    # barcycentric weights, lms_we_care_about_can 
    flame_bary_weights = flameBS.flame.full_lmk_bary_coords[0, lms_we_care_about]
    flame_mesh_face_indices = flameBS.flame.full_lmk_faces_idx[0, lms_we_care_about]

    full_bary_weights = torch.zeros((flame_bary_weights.shape[0], flameBS.V.shape[0],), dtype=torch.float32)
    for i in range(0, flame_mesh_face_indices.shape[0]):
        triangles = flameBS.F[flame_mesh_face_indices[i]]
        full_bary_weights[i, triangles] = flame_bary_weights[i]

    grouped_bary_weights_dict = {}
    for key in list(FLAME_facial_landmark_groups.keys()):
        # key = list(FLAME_facial_landmark_groups.keys())[0]
        grouped_flame_bary_weights = flameBS.flame.full_lmk_bary_coords[0, FLAME_facial_landmark_groups[key]]
        grouped_mesh_face_indices = flameBS.flame.full_lmk_faces_idx[0, FLAME_facial_landmark_groups[key]] # get the indices of the face

        grouped_bary_weights = torch.zeros((grouped_flame_bary_weights.shape[0], flameBS.V.shape[0],), dtype=torch.float32)
        for i in range(0, grouped_mesh_face_indices.shape[0]):
            triangles = flameBS.F[grouped_mesh_face_indices[i]]
            grouped_bary_weights[i, triangles] = grouped_flame_bary_weights[i]
        grouped_bary_weights_dict[key] = grouped_bary_weights

    return flame_mesh_face_indices, full_bary_weights, grouped_bary_weights_dict

def get_lm_indices_from_ARKit():
    ARkitBS = load_ARKit_blendshape()
    
    ARKIT_LM_DICT = {
    "jaw": [915, 1047, 912], # I'm only going to use the center 3 lol
    "right_eyebrow": [197, 335, 327, 328, 348],
    "left_eyebrow": [781, 763, 762, 768, 646],
    "nose_bridge": [36, 13, 10, 8, ],
    "nose_tip": [308, 76, 4, 525, 743],
    "right_eye": [1101, 1096, 1093, 1089, 1085, 1107],
    "left_eye": [1081, 1077, 1074, 1069, 1064, 1061],
    "lip": [244, 239, 237, 22, 671, 673, 678, 723, 698, 27, 263, 288, 
            394, 256, 24, 691, 684, 700, 25, 265],
    }
    lm_indices = [ARKIT_LM_DICT[key] for key in ARKIT_LM_DICT.keys()]
    lm_indices = np.concatenate(lm_indices, axis=0)
    full_bary_weights = torch.zeros((lm_indices.shape[0], ARkitBS.V.shape[0]), dtype=torch.float32)

    for i in range(0, lm_indices.shape[0]):
        full_bary_weights[i, lm_indices[i]] = 1.0  # Set the barycentric weight to 1 for the landmark vertex    
    
    grouped_ARKit_bary_weights_dict = {}
    for key in list(ARKIT_LM_DICT.keys()):
        grouped_ARKit_bary_weights = torch.zeros((len(ARKIT_LM_DICT[key]), ARkitBS.V.shape[0]), dtype=torch.float32)
        for i, lm_index in enumerate(ARKIT_LM_DICT[key]):
            grouped_ARKit_bary_weights[i, lm_index] = 1.0
        grouped_ARKit_bary_weights_dict[key] = grouped_ARKit_bary_weights


    return lm_indices.tolist(), full_bary_weights, grouped_ARKit_bary_weights_dict

def display_a_single_mesh(V, F, points=None):
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
    if points is not None:
        ps.register_point_cloud(
            "points", points,
            radius=0.01,
            color=[1.0, 0.0, 0.0],material="normal"
        )
    ps.reset_camera_to_home_view()
    ps.show()

def display_pairs_of_meshes(Vs1, Fs1, Vs2, Fs2, offset=0.3):
    ps.remove_all_structures()
    ps.set_verbosity(0)
    ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_view_projection_mode("orthographic")
    ps.set_front_dir("z_front")
    ps.set_background_color([0, 0, 0])
    
    for i in range(len(Vs1)):
        V1 = Vs1[i] + np.array([-i*offset, 0, 0], dtype=np.float32)
        F1 = Fs1[i]
        ps.register_surface_mesh(
            f"mesh_{i}_1", V1, F1,
            color=[0.9, 0.9, 0.9],
            edge_width=0.25
        )
        
        V2 = Vs2[i] + np.array([-i*offset, offset, 0], dtype=np.float32)
        F2 = Fs2[i]
        ps.register_surface_mesh(
            f"mesh_{i}_2", V2, F2,
            color=[0.5, 0.5, 0.5],
            edge_width=0.25
        )    
    ps.reset_camera_to_home_view()
    ps.show()

def compute_landmark_groups_of_blendshape(V_0, V_bs, landmark_groups):
    # V_0 = ARkitBS.V
    # V_bs = ARkitBS.blendshapes[0] + ARkitBS.V
    # landmark_groups = ARkit_lm_groups
    
    involved_lm_groups = []
    for key in list(landmark_groups.keys()):
        # key = list(landmark_groups.keys())[0]
        lms_0 = landmark_groups[key] @ V_0
        lms_bs = landmark_groups[key] @ V_bs

        diff = lms_bs - lms_0
        diff_mag = torch.norm(diff, dim=-1).mean()

        if diff_mag >= 1E-4:
            involved_lm_groups.append(key)
    return involved_lm_groups
        
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

 
# Parameters for optimizing mesh deformations 
LEARNING_RATE = 0.03
ITERATIONS = 5000
K=5
CLIPS_OF_DATA = 1

# em parameters
EM_LEARNING_RATE = 0.00003
EM_WEIGHT_LEARN_RATE = 0.01
W_FROZEN = 0.0005
EM_ITERATIONS = 10
EM_FACS_WEIGHT_ITERATIONS = 1000
EM_FACS_DIRE_ITERATIONS = 100
W_REG = 1000
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############### loading models and landmarks ##################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
flameBS = FLAMEBlendshapes(device=device)
flame_lm_indices, flame_full_bary_weights, flame_lm_groups = get_lm_indices_and_bary_weights_from_FLAME()
flame_LM = flame_full_bary_weights @ flameBS.V
# display_a_single_mesh(flameBS.V, flameBS.F, flame_LM.detach().numpy())

ARkitBS = load_ARKit_blendshape()
ARkit_lm_indices, ARkit_full_bary_weights, ARkit_lm_groups = get_lm_indices_from_ARKit()
ARkit_LM = ARkit_full_bary_weights @ ARkitBS.V
# display_a_single_mesh(ARkitBS.V, ARkitBS.F, ARkit_LM)

all_lm_groups = list(flame_lm_groups.keys())

frozen_LM_mask = []
for bs_i in range(0, ARkitBS.blendshapes.shape[0]):
    print(f"Processing blendshape {bs_i} of {ARkitBS.blendshapes.shape[0]}")
    involved_lm_groups = compute_landmark_groups_of_blendshape(ARkitBS.V, ARkitBS.blendshapes[bs_i] + ARkitBS.V, ARkit_lm_groups)
    # generate the indices of the landmarks we care about
    involved_lm_indices = []
    for key in involved_lm_groups:
        barycentric_coord_matrix = flame_lm_groups[key]
        for lm_i in range(barycentric_coord_matrix.shape[0]):
            for v_i in range(barycentric_coord_matrix.shape[1]):
                if barycentric_coord_matrix[lm_i, v_i] > 0.0:
                    involved_lm_indices.append(v_i)

    # these are the ones we want frozen
    non_involved_lm_indices = []
    for key in all_lm_groups:
        if key not in involved_lm_groups:
            barycentric_coord_matrix = flame_lm_groups[key]
            for lm_i in range(barycentric_coord_matrix.shape[0]):
                for v_i in range(barycentric_coord_matrix.shape[1]):
                    if barycentric_coord_matrix[lm_i, v_i] > 0.0:
                        non_involved_lm_indices.append(v_i)

    non_frozen_set, frozen_set = compute_vertex_assignments(flameBS.V, flameBS.F,
        key_point_set_A=involved_lm_indices,
        key_point_set_B=non_involved_lm_indices,
        K=K)
    non_frozen_set = list(non_frozen_set)
    frozen_set = list(frozen_set)
    frozen_set_mat = torch.zeros(flameBS.V.shape, dtype=torch.double, device=device)
    frozen_set_mat[frozen_set, :] = 1.0  # set the frozen set to 1.0
    frozen_LM_mask.append(frozen_set_mat)
frozen_LM_mask = torch.stack(frozen_LM_mask, dim=0)  # (blendshapes, vertices, 3)
frozen_LM_mask = frozen_LM_mask[:, :, 0]

############### loading initialized weights ##################
surrogate_model_root_path = "/Users/evanpan/Documents/GitHub/ManifoldExploration/experiments/FACS_Based_flame_sliders_with_L1_frozen_LM_W_frozen_0p002"
# surrogate_model_root_path = "/Users/evanpan/Documents/GitHub/ManifoldExploration/experiments/full_face_bs_test/"
flame = FLAMEBlendshapes()
FACS_directions = []
for i in range(0, 51):
    exp_path = os.path.join(surrogate_model_root_path, f"exp_params_{i}.npy")
    jaw_path = os.path.join(surrogate_model_root_path, f"jaw_params_{i}.npy")
    exp_params = np.load(exp_path)
    jaw_params = np.load(jaw_path)
    FACS_directions.append(np.concatenate([exp_params, jaw_params], axis=1))

# load weights to torch tensor
FACS_directions = np.concatenate(FACS_directions, dtype=np.double, axis=0)
FACS_directions = torch.from_numpy(FACS_directions).to(device)
FACS_directions = FACS_directions.double()
FACS_directions.requires_grad = True  # we will optimize this
############### load reference data ##################
# ROOT = "/scratch/ondemand29/evanpan/facial-manifold-learning"
ROOT = "/Users/evanpan/Documents/GitHub/ManifoldExploration"
# load the pickle_dataset:
dataset_path = os.path.join(ROOT, "data/MeadRavdess/val_mead_ravdess_0.1.pickle")
with open(dataset_path, "rb") as f:
    data = pickle.load(f)
    data_keys = list(data.keys())
# randomly select CLIPS_OF_DATA samples from the dataset
if CLIPS_OF_DATA > 0:
    random_indices = np.random.choice(len(data_keys), CLIPS_OF_DATA, replace=False).tolist()
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
weight = np.concatenate([exp, jaw], axis=1, dtype=np.double)
# this is the target
weight = torch.from_numpy(weight).to(device) # (Frames, 103)


################ EM optimization ##################
weight = weight[0:10]
current_batch_size = weight.shape[0]

shape_params_FACS = torch.zeros([51, 100]).to(device).double()
tex_params_FACS = torch.zeros([51, 50]).to(device).double()
pose_params_FACS = torch.zeros([51, 3]).to(device).double()
flame_full_bary_weights = flame_full_bary_weights.double().to(device)  # (Landmarks, Vertices)
ARkit_full_bary_weights = ARkit_full_bary_weights.double().to(device)

shape_params_frames = torch.zeros((weight.shape[0], 100), device=device, dtype=torch.double)  # (Frames, 100)
tex_params_frames = torch.zeros((weight.shape[0], 50), device=device, dtype=torch.double)  # (Frames, 50)
pose_params_frames = torch.zeros((weight.shape[0], 3), device=device, dtype=torch.double)  # (Frames, 3)
# losses are weighted differently

flame_model = FLAMEBlendshapes(device=device, dtype=torch.double)
flame_module = flame_model.flame
flame_neutral = torch.from_numpy(flame_model.V).to(device).double()  # (Vertices, 3)
flame_neutral = torch.unsqueeze(flame_neutral, dim=0)  # (1, Vertices, 3)

losses_weights_recon = []
losses_directions_recond = []
for i in range(0, EM_ITERATIONS):
# for i in range(0, 1):
    # initialize ADAM optimizers since we don't want interference between EM iterations
    direction_optimizer =  BatchedManualAdam(51, [(103,) ,], lr=EM_LEARNING_RATE, device=device, dtype=torch.double)
    weight_optimizer = BatchedManualAdam(current_batch_size, [(51, )], lr=EM_WEIGHT_LEARN_RATE, device=device, dtype=torch.double)
    # optimize for facs_weights
    FACS_weights = torch.abs(torch.randn((weight.shape[0], FACS_directions.shape[0]), device=device, dtype=torch.double)*0.01)  # initialize with small random values
    # fix the FACS directions
    FACS_directions.requires_grad = False  # we will optimize this
    FACS_weights.requires_grad = True  # we will optimize this
<<<<<<< HEAD
    for fitting_iter in range(0, 100):
        FACS_based_weights = FACS_weights @ FACS_directions  # (Frames, 103)
        recon_loss_latent = torch.norm(FACS_based_weights - weight, p=2, dim=-1).mean()
        
=======
    # update the shape and pose parameters iteratively
    for fitting_iter in range(0, EM_FACS_WEIGHT_ITERATIONS):
        FACS_based_weights = (FACS_weights ** 2) @ FACS_directions  # (Frames, 103)
        # latent based reconstruction loss (we ignore these for now)        
>>>>>>> refs/remotes/origin/evan
        V_bs, _, _ = flame_module(shape_params_frames, FACS_based_weights[:, :100], pose_params=torch.concat([pose_params_frames, FACS_based_weights[:, 100:103]], dim=1))
        V_gt, _, _ = flame_module(shape_params_frames, weight[:, :100], pose_params=torch.concat([pose_params_frames, weight[:, 100:]], dim=1))
        recon_loss_geometry = torch.norm(V_bs - V_gt, p=2, dim=-1).mean()  # (Frames, Vertices)
        l1_loss = torch.norm(FACS_weights, p=1, dim=-1).mean() # L1 regularization
        loss = recon_loss_geometry + 0.000001 * l1_loss  # add the L1 regularization term
        loss.backward()
        weight_optimizer.step([FACS_weights], [FACS_weights.grad])
        FACS_weights.grad.zero_() # reset grad
        # if fitting_iter % 10 == 0:
        #     print(f"EM iteration {i}, fitting iteration {fitting_iter}: recon loss geometry: {recon_loss_geometry.item()}, l1 loss: {l1_loss.item()}")
    losses_weights_recon.append(recon_loss_geometry.item())        
    print(f"weights -> EM iter: {i}, recon loss: {recon_loss_geometry.item()}, l1 loss: {l1_loss.item()}")


    # FACS_based_weights.shape
    # from matplotlib import pyplot as plt
    # plt.bar(np.arange(0, FACS_based_weights[0].shape[0]), FACS_based_weights[0].detach().cpu().numpy())
    # plt.bar(np.arange(0, weight[0].shape[0]), weight[0].detach().cpu().numpy(), alpha=0.5)

    # show the geometry
    V_bs, _, _ = flame_module(shape_params_frames, FACS_based_weights[:, :100], pose_params=torch.concat([pose_params_frames, FACS_based_weights[:, 100:103]], dim=1))
    V_gt, _, _ = flame_module(shape_params_frames, weight[:, :100], pose_params=torch.concat([pose_params_frames, weight[:, 100:]], dim=1))
    display_pairs_of_meshes(
        [V_bs[0].detach().cpu().numpy()],
        [flame_model.F],
        [V_gt[0].detach().cpu().numpy()],
        [flame_model.F],
        offset=0.3
    )


    # optimize for FACS_directions
    FACS_weights.requires_grad = False  # we will optimize this
    FACS_directions.requires_grad = True  # we will optimize this
    FACS_weights_sq = FACS_weights ** 2  # (Frames, 51)
    for fitting_iter in range(0, EM_FACS_DIRE_ITERATIONS):
        FACS_based_weights = FACS_weights_sq @ FACS_directions  # (Frames, 103)
        # compute geometry loss on each frame
        V_bs, _, _ = flame_module(shape_params_frames, FACS_based_weights[:, :100], pose_params=torch.concat([pose_params_frames, FACS_based_weights[:, 100:103]], dim=1))
        V_gt, _, _ = flame_module(shape_params_frames, weight[:, :100], pose_params=torch.concat([pose_params_frames, weight[:, 100:]], dim=1))
        recon_loss_geometry = torch.norm(V_bs - V_gt, p=2, dim=-1).mean()  # (Frames, Vertices)
        # compute the frozen loss
        exp_params = FACS_directions[:, :100]
        jaw_params = FACS_directions[:, 100:103]
        # compute frozen loss
        V_bs, _, _ = flame_module(shape_params_FACS, exp_params, pose_params=torch.concat([pose_params_FACS, jaw_params], dim=1))
        frozen_loss = torch.norm(V_bs - flame_neutral, p=2, dim=-1) * torch.unsqueeze(frozen_LM_mask, dim=0)  # (Frames, Vertices)
        frozen_loss = frozen_loss.mean()
        # compute LM loss
        flame_LM = flame_full_bary_weights @ V_bs  # (Frames, Vert
        AR_kit_LM = ARkit_full_bary_weights @ ARkitBS.V  # (Frames, Vertices, 3)
        lm_loss = torch.norm(flame_LM - AR_kit_LM, p=2, dim=-1).mean()
        loss = recon_loss_geometry + W_REG * (lm_loss + frozen_loss * W_FROZEN)  # add the frozen loss
        loss.backward()
        direction_optimizer.step([FACS_directions], [FACS_directions.grad])
        FACS_directions.grad.zero_()  # reset grad
        # if fitting_iter % 10 == 0:
        #     print(f"dires -> fitting_iter: {fitting_iter}, reconstruction loss: {recon_loss_geometry.item()}, frozen loss: {frozen_loss.item()}, lm loss: {lm_loss.item()}")
    losses_directions_recond.append(recon_loss_geometry.item())

    print(f"dires -> EM iter: {i}, reconstruction loss: {recon_loss_geometry.item()}, frozen loss: {frozen_loss.item()}, lm loss: {lm_loss.item()}")
    if i % 10 == 0:
        # implement a stopping condition (stop if losses_directions_recond[-1] is greater than the average of last 5 losses)
        if len(losses_directions_recond) > 5 and losses_directions_recond[-1] > np.mean(losses_directions_recond[-5:]):
            print(f"Stopping early at iteration {i} due to increasing loss.")
            break
        # save the FACS_directions
        save_folder = f"EM_refinement_{CLIPS_OF_DATA}_clips_data"
        save_folder = os.path.join(surrogate_model_root_path, save_folder)
        os.mkdir(save_folder) if not os.path.exists(save_folder) else None
        np.save(os.path.join(save_folder, f"FACS_directions_{i}.npy"), FACS_directions.detach().cpu().numpy())  
        
    
    # load the 
save_root = os.path.join(surrogate_model_root_path, "EM_optimized_FACS_directions_geometry_based")

if not os.path.exists(save_root):
    os.makedirs(save_root)
save_root
FACS_directions_np = FACS_directions.detach().cpu().numpy()

for i in range(FACS_directions_np.shape[0]):
    exp_params = FACS_directions_np[i][:100]
    jaw_params = FACS_directions_np[i][100:]
    np.save(os.path.join(save_root, f"exp_params_{i}.npy"), exp_params)
    np.save(os.path.join(save_root, f"jaw_params_{i}.npy"), jaw_params)



    


