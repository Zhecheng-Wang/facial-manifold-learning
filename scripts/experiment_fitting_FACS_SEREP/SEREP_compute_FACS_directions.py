import os
from omegaconf import OmegaConf
import sys
sys.path.append("/code/speech2face_baselines/")
sys.path.append("/code/expressive-speech2face/")
sys.path.append("/code/facial-manifold-learning/src")
from speech2face.mesh.utils import loadObj
from speech2face.models.spiral.get_model import get_model
from speech2face.scripts.id_exp_convert_tests import replace_on_cfg
import torch 
import numpy as np
from tqdm import trange
import polyscope as ps
from matplotlib import pyplot as plt
from web_visualizer_server import *
import pickle 
from blendshapes import FLAMEBlendshapes, BasicBlendshapes
from utils import load_ARKit_blendshape
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from scripts.experiment_different_kinds_of_partial_freezing.naive_autosegmentation import *


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

def get_lm_indices_and_bary_weights_from_SEREP():
    """
    Returns the indices of the FLAME landmarks and their barycentric weights.
    """
    SEREP_facial_landmark_groups = {
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
    for key in SEREP_facial_landmark_groups.keys():
         # these are the row indices after we use barycentric coordinates to map to the landmarks
        lms_we_care_about.extend(SEREP_facial_landmark_groups[key])
    lms_we_care_about = np.array(lms_we_care_about, dtype=np.int64)
    # barcycentric weights, lms_we_care_about_can
    full_bary_weights = np.load("/code/facial-manifold-learning/scripts/experiment_fitting_FACS_SEREP/ubi_topo_lmk_barycentric_matrix.npy")
    full_bary_weights = torch.from_numpy(full_bary_weights).float()
    full_bary_we_care = full_bary_weights[lms_we_care_about]
    grouped_bary_weights_dict = {}
    for key in list(SEREP_facial_landmark_groups.keys()):
        # key = list(FLAME_facial_landmark_groups.keys())[0]
        grouped_flame_bary_weights = full_bary_weights[SEREP_facial_landmark_groups[key]]
        grouped_mesh_face_indices = full_bary_weights[SEREP_facial_landmark_groups[key]] # get the indices of the face
        grouped_bary_weights_dict[key] = grouped_flame_bary_weights
        
    return None, full_bary_we_care, grouped_bary_weights_dict

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

class DummyArgs:
    def __init__(self, input, output):
        self.config = "/code/models/id_exp_apply_model/config.yaml"
        self.checkpoint = "/code/models/id_exp_apply_model/checkpoint_epoch41.pth"
        self.neutral = "/code/models/S077_HSP_M_20/Head/S077_HSP_M_20_Head.obj"
        self.output = output
        self.input = input        
        self.scale = 0.01
        self.shift = (0, 169.44, 5.2)

neighborhood_distance=0.03
LEARNING_RATE = 0.01
ITERATIONS = 5000
W_FROZEN = 0.001

SEREP_latent_dire_root = "/code/facial-manifold-learning/experiments/SEREP_fACS_latents/"
SEREP_latent_save_dir = os.path.join(SEREP_latent_dire_root, "alternative_freeze_latent_for_all_FACS_AUs.npy")
faces_path = "/mnt/e/Projects/Ubi_Speech2face/models/flame_retopo/faces.pickle" 
FLAME_TINGS_ROOT = "/code/models/flame2ubi"
FLAME_IN_UBI_fname = "generic_model_ubito_flame_v2.pkl"
flame_lmk_path =  "landmark_embedding.npy"
FLAME_IN_UBI_fname = os.path.join(FLAME_TINGS_ROOT, FLAME_IN_UBI_fname)
flame_lmk_path = os.path.join(FLAME_TINGS_ROOT, flame_lmk_path)
flame_in_ubi_path = FLAME_IN_UBI_fname
args = DummyArgs(input=None, output=None)

config = OmegaConf.load(args.config)
replace_on_cfg(config)
config.model.data_root = r"/expnet_root"
device = torch.device('cuda')

# load model
model = get_model(config.model, device)
checkpoint = torch.load(args.checkpoint)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.to(device)

# load neutral mesh
batch_size = 1
neutral = loadObj(args.neutral)['verts']
neutral = (neutral - np.array(args.shift)) * args.scale
neutral = neutral.astype(np.float32)
neutral = torch.from_numpy(neutral).to(device)
mean, std = checkpoint['meanstd']
neutral = (neutral - mean) / std
SEREP_neutral = neutral.tile(batch_size, 1, 1) * std + mean
SEREP_neutral_np = SEREP_neutral.detach().cpu().numpy()[0]
SEREP_F = loadObj(args.neutral)["tris"]

# load landmarks and geometry related to ARkit and SEREP
ARKIT_bs = load_ARKit_blendshape()
lm_indices_ARKIT, full_bary_weightsARKIT, grouped_bary_weights_dict_ARKit = get_lm_indices_from_ARKit()
__, full_bary_weights_SEREP, grouped_bary_weights_dict_SEREP = get_lm_indices_and_bary_weights_from_SEREP()

full_bary_weights_SEREP = full_bary_weights_SEREP.to(device)
for group in grouped_bary_weights_dict_SEREP:
    grouped_bary_weights_dict_SEREP[group] = grouped_bary_weights_dict_SEREP[group].to(device)
all_lm_groups = list(grouped_bary_weights_dict_ARKit.keys())
all_losses = []
latent_for_all_FACS_AUs = []
for bs_i in range(0, ARKIT_bs.blendshapes.shape[0]):
    # bs_i = 0
    involved_lm_groups = compute_landmark_groups_of_blendshape(ARKIT_bs.V, ARKIT_bs.blendshapes[bs_i] + ARKIT_bs.V, grouped_bary_weights_dict_ARKit)
    # generate the indices of the landmarks we care about
    involved_lm_indices = []
    for key in involved_lm_groups:
        barycentric_coord_matrix = grouped_bary_weights_dict_SEREP[key]
        # Find all indices where values > 0
        nonzero_indices = torch.nonzero(barycentric_coord_matrix > 0.0, as_tuple=False)
        # Extract the v_i indices (column indices)
        v_indices = nonzero_indices[:, 1].tolist()
        involved_lm_indices.extend(v_indices)
        
    # these are the ones we want frozen
    non_involved_lm_indices = []
    for key in all_lm_groups:
        if key not in involved_lm_groups:
            barycentric_coord_matrix = grouped_bary_weights_dict_SEREP[key]
            # Find all indices where values > 0
            nonzero_indices = torch.nonzero(barycentric_coord_matrix > 0.0, as_tuple=False)
            # Extract the v_i indices (column indices)
            v_indices = nonzero_indices[:, 1].tolist()
            non_involved_lm_indices.extend(v_indices)
            
    non_frozen_set, frozen_set = compute_weighted_vertex_assignments(SEREP_neutral_np, SEREP_F,
        key_point_set_A=involved_lm_indices,
        key_point_set_B=non_involved_lm_indices,
        max_distance=neighborhood_distance)
    all_vertices = set(range(0, SEREP_neutral.shape[1]))
    frozen_set = all_vertices - non_frozen_set
    non_frozen_set = list(non_frozen_set)
    frozen_set = list(frozen_set)
    

    # get the target landmark positions
    target_V_i = ARKIT_bs.blendshapes[bs_i]
    target_lm_i = full_bary_weightsARKIT @ target_V_i
    target_lm_i = torch.tensor(target_lm_i, dtype=torch.float32, device=device)
    
    # initialize the SEREP latent code for optimization
    SEREP_latent = torch.zeros([1, 64], dtype=torch.float32).to(device)
    SEREP_latent.requires_grad = True
    optimizer = BatchedManualAdam(batch_size=1, param_shapes=[(64, ),], lr=0.02, device=device, dtype=torch.float32)
    loss_for_bs_i = []
    for i in range(0, ITERATIONS):
        # compute the FLAME mesh
        V_optimized = model.id_encoder(neutral, SEREP_latent) # unnormalized mesh
        V_optimized = V_optimized * std + mean  # denormalize
        frozen_loss = V_optimized[0, frozen_set, :] - SEREP_neutral[0, frozen_set, :] # both are denormalized
        frozen_loss = torch.mean(torch.abs(frozen_loss))
        V_optimized = V_optimized - SEREP_neutral
        # compute the landmark positions
        SEREP_LM = full_bary_weights_SEREP @ V_optimized
        # compute the loss
        lm_loss = torch.mean((SEREP_LM - target_lm_i) ** 2)
        loss = lm_loss + frozen_loss * W_FROZEN
        # backpropagate
        loss.backward()
        optimizer.step([SEREP_latent], [SEREP_latent.grad])
        SEREP_latent.grad.zero_() # reset grad
        if i % 50 == 0:
            print(f"Iteration {i}, lm loss {lm_loss.item()}, frozen loss {frozen_loss.item()}")
            # add a stop condition
            past_3_mean = np.mean(loss_for_bs_i[-3:]) if len(loss_for_bs_i) >= 3 else None
            if past_3_mean is not None and  loss.item() - past_3_mean > 0:
                print(f"Stopping early at iteration {i} with lm loss {lm_loss.item()}, frozen loss {frozen_loss.item()}")
                break
            
        loss_for_bs_i.append(loss.item())
    latent_for_all_FACS_AUs.append(SEREP_latent.detach().cpu().numpy())
    all_losses.append({"lm_loss": lm_loss.item(), "frozen_loss": frozen_loss.item(), "total_loss": loss.item()})

latent_for_all_FACS_AUs = np.array(latent_for_all_FACS_AUs, dtype=np.float32)[:, 0, :]

if os.path.exists(SEREP_latent_dire_root):
    print("SEREP FACS latents directory already exists, not creating it again.")
else:
    os.mkdir(SEREP_latent_dire_root)
np.save(SEREP_latent_save_dir, latent_for_all_FACS_AUs)


#######################
####################### debug #######################
# input code 
if False:
    code = torch.zeros([batch_size, 64], device=device)
    mesh =  model.id_encoder(neutral, code) # [1, 13473, 3]
    mesh = mesh * std + mean

    # run visualizer
    visualizer, task = run_visualizer()
    V = mesh.detach().cpu().numpy()[0]
    visualizer.add_mesh(f"neutral", V, SEREP_F)
    pt_colors = np.array([[1, 0, 0]])
    pt_colors = np.tile(pt_colors, [len(frozen_set), 1])
    visualizer.add_point_cloud("frozen", V[frozen_set], radius=0.005, colors=pt_colors)



    # for i in range(len(latent_for_all_FACS_AUs)):
    for i in range(50, len(latent_for_all_FACS_AUs)):
        SEREP_latent = torch.tensor(latent_for_all_FACS_AUs[i], dtype=torch.float32, device=device)
        name = ARKIT_bs.names[i]
        mesh =  model.id_encoder(neutral, SEREP_latent) # [1, 13473, 3]
        mesh = mesh * std + mean
        V = mesh.detach().cpu().numpy()[0]
        constant = 50
        visualizer.add_mesh(f"SEREP_{name}", V + (i-constant)*np.array([0, 0.3, 0]), SEREP_F)
    visualizer.clear_all()
        

    V_neutral = V_optimized.detach().cpu().numpy()[0]
    visualizer.add_mesh("neutral", V_neutral, SEREP_F)
    full_bary_weights_SEREPnp = full_bary_weights_SEREP.detach().cpu().numpy()
    new_lmk = full_bary_weights_SEREPnp @ (V)
