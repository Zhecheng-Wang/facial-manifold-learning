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



def get_heatmap_colors(scalar_values, min_val=None, max_val=None):
    """Convert scalar values to RGB colors using a heat map."""
    if scalar_values is None:
        return None
    
    # Normalize scalar values to [0, 1]
    scalar_values = np.array(scalar_values)
    if min_val is None:
        min_val = np.min(scalar_values)
    if max_val is None:
        max_val = np.max(scalar_values)
        
    if max_val == min_val:
        # If all values are the same, use middle color
        normalized = np.full_like(scalar_values, 0.5)
    else:
        normalized = (scalar_values - min_val) / (max_val - min_val)
    
    # Create colormap (blue to red heat map)
    colors = plt.cm.coolwarm(normalized)[:, :3]  # Take only RGB, drop alpha
    return colors
    
def load_SEREP_blendshape_model():
    class DummyArgs:
        def __init__(self, input, output):
            self.config = "/code/models/id_exp_apply_model/config.yaml"
            self.checkpoint = "/code/models/id_exp_apply_model/checkpoint_epoch41.pth"
            self.neutral = "/code/models/S077_HSP_M_20/Head/S077_HSP_M_20_Head.obj"
            self.output = output
            self.input = input        
            self.scale = 0.01
            self.shift = (0, 169.44, 5.2)

    SEREP_zero = np.zeros([1, 64])
    # load the SEREP model
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
    SEREP_F = loadObj(args.neutral)["tris"]
    
    # compute the neutral mesh for the blendshape model 
    neutral_for_blendshape = model.id_encoder(neutral, torch.from_numpy(SEREP_zero).to(device).float())
    neutral_for_blendshape = neutral_for_blendshape * std + mean
    neutral_for_blendshape = neutral_for_blendshape.detach().cpu().numpy()[0]

    n_blendshapes = 51
    # load the solved latent directions
    surrogate_model_root_path = "/code/facial-manifold-learning/experiments/SEREP_fACS_latents"
    SEREP_latent_directions = os.path.join(surrogate_model_root_path, "alternative_freeze_latent_for_all_FACS_AUs.npy")
    SEREP_latent_directions = np.load(SEREP_latent_directions)
    SEREP_latent_directions.dtype = np.float32
    surrogate_Vs = []
    for i in range(n_blendshapes):
        # i=0
        SEREP_latent_i = SEREP_latent_directions[i]
        SEREP_latent_i = torch.from_numpy(SEREP_latent_i).to(device).unsqueeze(0).float()
        bs_mesh_i = model.id_encoder(neutral, SEREP_latent_i) # [1, 13473, 3]
        bs_mesh_i = bs_mesh_i * std + mean
        surrogate_Vs.append(bs_mesh_i.detach().cpu().numpy()[0] - neutral_for_blendshape)
    
    # use the zero latent code to get the neutral mesh
    surrogate_model = BasicBlendshapes(
        names=[f"blendshape_{i}" for i in range(n_blendshapes)],
        V=neutral_for_blendshape.copy(),
        blenshapes=np.array(surrogate_Vs),
        F=SEREP_F,
    )
    
    return model, surrogate_model, SEREP_latent_directions, [std, mean], neutral, neutral_for_blendshape


SEREP_model, SEREP_surrogate, SEREP_latent_directions, [std, mean], encoder_input_neutral, blendshape_input_neutral = load_SEREP_blendshape_model()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# evaluate the linearity of each blendshape, i.e. f(a*x) = a*f(x)
all_results_linear = []
for bs_i in range(51):
    all_results_bs_i = []
    for iter_i in range(11):
        random_bs_weights = np.zeros((51))
        random_val = iter_i*0.1
        random_bs_weights[bs_i] = random_val
        facs_latent_code = random_bs_weights @ SEREP_latent_directions
        facs_latent_code = torch.from_numpy(facs_latent_code).to(device).unsqueeze(0).float()   
        V_through_latent = SEREP_model.id_encoder(encoder_input_neutral, facs_latent_code)
        V_through_latent = V_through_latent * std + mean
        V_through_latent = V_through_latent.detach().cpu().numpy()[0]
        V_through_latent = V_through_latent - blendshape_input_neutral.mean(axis=0)
        V_through_surrogate = SEREP_surrogate.eval(random_bs_weights)
        error = np.linalg.norm(V_through_latent - V_through_surrogate, axis=1)
        all_results_bs_i.append({
            "V_through_latent": V_through_latent,
            "V_through_surrogate": V_through_surrogate,
            "error": error,
            "mean_error": error.mean(),
            "bs_i": bs_i,
            "random_val": random_val
        })
    all_results_linear.append(all_results_bs_i)    
# evaluate multi-linearity, defined as f(ax + by) = af(x) + bf(y).
all_results_multilinear = []
for i in range(0, 200):
    random_bs_weights = np.random.uniform(0, 1, size=(51,))
    facs_latent_code = random_bs_weights @ SEREP_latent_directions
    facs_latent_code = torch.from_numpy(facs_latent_code).to(device).unsqueeze(0).float()
    V_through_latent = SEREP_model.id_encoder(encoder_input_neutral, facs_latent_code)
    V_through_latent = V_through_latent * std + mean
    V_through_latent = V_through_latent.detach().cpu().numpy()[0]
    V_through_latent = V_through_latent - blendshape_input_neutral.mean(axis=0)
    V_through_surrogate = SEREP_surrogate.eval(random_bs_weights)
    error = np.linalg.norm(V_through_latent - V_through_surrogate, axis=1)
    result_i = {
        "V_through_latent": V_through_latent,
        "V_through_surrogate": V_through_surrogate,
        "error": error,
        "mean_error": error.mean()
    }
    all_results_multilinear.append(result_i)

# neutralll = SEREP_model.id_encoder(encoder_input_neutral, torch.zeros([1, 64], device=device).float())
# neutralll = neutralll * std + mean
# neutralll = neutralll.detach().cpu().numpy()[0]

# print(neutralll - blendshape_input_neutral)


# let's the plot the difference between the surrogate and the latent mesh 
for bs_i in range(51):
    mean_errors = []
    max_errors = []
    min_errors = []
    for res in all_results_linear[bs_i]:
        mean_errors.append(res["mean_error"])
        max_errors.append(res["error"].max())
        min_errors.append(res["error"].min())
    x_axis = np.arange(len(mean_errors)) * 0.1
    # plt.plot(x_axis, mean_errors, label=f"bs_{bs_i}")
    plt.plot(x_axis, mean_errors, label=f"bs_{bs_i}")
    
plt.xlabel("Blendshape weight")
plt.ylabel("Mean error MSE")
plt.title("Mean error between a*f(x) and f(a*x)")


# display the results in our visualizer
show_local_linearity=False
show_global_linearity=True

visualizer, task = run_visualizer() # note this can only be called once per interactive session

if show_local_linearity:    
    
    bs_i = 2
    
    visualizer.clear_all()
    min_error_result = min(all_results_linear[bs_i], key=lambda x: x["mean_error"])
    max_error_result = max(all_results_linear[bs_i], key=lambda x: x["mean_error"])
    # sample 2 meshes
    sampled_results = np.random.choice(all_results_linear[bs_i], size=2, replace=False)
    mean_error = np.mean([res["mean_error"] for res in all_results_linear[bs_i]])
    min_error_heatmap = get_heatmap_colors(min_error_result["error"])
    visualizer.add_mesh(f"min_error_through_latent", min_error_result["V_through_latent"] + np.array([[0.2, 0, 0]]), SEREP_surrogate.F, colors=min_error_heatmap)
    visualizer.add_mesh(f"min_error_through_surrogate", min_error_result["V_through_surrogate"], SEREP_surrogate.F)

    max_error_heatmap = get_heatmap_colors(max_error_result["error"])
    visualizer.add_mesh(f"max_error_through_latent", max_error_result["V_through_latent"] + np.array([[0.2, 0.2, 0]]), SEREP_surrogate.F, colors=max_error_heatmap)
    visualizer.add_mesh(f"max_error_through_surrogate", max_error_result["V_through_surrogate"] + np.array([[0, 0.2, 0]]), SEREP_surrogate.F)

    sample_i_heatmap = get_heatmap_colors(sampled_results[0]["error"])
    visualizer.add_mesh(f"sample_i_through_latent", sampled_results[0]["V_through_latent"] + np.array([[0.2, 0.4, 0]]), SEREP_surrogate.F, colors=sample_i_heatmap)
    visualizer.add_mesh(f"sample_i_through_surrogate", sampled_results[0]["V_through_surrogate"] + np.array([[0.0, 0.4, 0]]), SEREP_surrogate.F)

    sample_j_heatmap = get_heatmap_colors(sampled_results[1]["error"])
    visualizer.add_mesh(f"sample_j_through_latent", sampled_results[1]["V_through_latent"] + np.array([[0.2, 0.6, 0]]), SEREP_surrogate.F, colors=sample_j_heatmap)
    visualizer.add_mesh(f"sample_j_through_surrogate", sampled_results[1]["V_through_surrogate"] + np.array([[0.0, 0.6, 0]]), SEREP_surrogate.F)

if show_global_linearity:
    visualizer.clear_all()
    # get the results with the least error, highest error and compute the mean error
    min_error_result = min(all_results_multilinear, key=lambda x: x["mean_error"])
    max_error_result = max(all_results_multilinear, key=lambda x: x["mean_error"])
    # sample 2 meshes
    sampled_results = np.random.choice(all_results_multilinear, size=2, replace=False)
    mean_error = np.mean([res["mean_error"] for res in all_results_multilinear])    

    min_error_heatmap = get_heatmap_colors(min_error_result["error"])
    visualizer.add_mesh(f"min_error_through_latent", min_error_result["V_through_latent"] + np.array([[0.2, 0, 0]]), SEREP_surrogate.F, colors=min_error_heatmap)
    visualizer.add_mesh(f"min_error_through_surrogate", min_error_result["V_through_surrogate"], SEREP_surrogate.F)

    max_error_heatmap = get_heatmap_colors(max_error_result["error"])
    visualizer.add_mesh(f"max_error_through_latent", max_error_result["V_through_latent"] + np.array([[0.2, 0.2, 0]]), SEREP_surrogate.F, colors=max_error_heatmap)
    visualizer.add_mesh(f"max_error_through_surrogate", max_error_result["V_through_surrogate"] + np.array([[0, 0.2, 0]]), SEREP_surrogate.F)

    sample_i_heatmap = get_heatmap_colors(sampled_results[0]["error"])
    visualizer.add_mesh(f"sample_i_through_latent", sampled_results[0]["V_through_latent"] + np.array([[0.2, 0.4, 0]]), SEREP_surrogate.F, colors=sample_i_heatmap)
    visualizer.add_mesh(f"sample_i_through_surrogate", sampled_results[0]["V_through_surrogate"] + np.array([[0.0, 0.4, 0]]), SEREP_surrogate.F)

    sample_j_heatmap = get_heatmap_colors(sampled_results[1]["error"])
    visualizer.add_mesh(f"sample_j_through_latent", sampled_results[1]["V_through_latent"] + np.array([[0.2, 0.6, 0]]), SEREP_surrogate.F, colors=sample_j_heatmap)
    visualizer.add_mesh(f"sample_j_through_surrogate", sampled_results[1]["V_through_surrogate"] + np.array([[0.0, 0.6, 0]]), SEREP_surrogate.F)
    
    