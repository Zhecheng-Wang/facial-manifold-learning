import numpy as np
import os, sys
sys.path.append("/Users/evanpan/Documents/GitHub/ManifoldExploration/src")
sys.path.append("/code/facial-manifold-learning/src")
from utils import load_ARKit_blendshape
import torch
import polyscope as ps
import polyscope.imgui as psim
from web_visualizer_server import *
from matplotlib import pyplot as plt

def get_heatmap_colors(scalar_values):
    """Convert scalar values to RGB colors using a heat map."""
    if scalar_values is None:
        return None
    
    # Normalize scalar values to [0, 1]
    scalar_values = np.array(scalar_values)
    min_val = np.min(scalar_values)
    max_val = np.max(scalar_values)
    
    if max_val == min_val:
        # If all values are the same, use middle color
        normalized = np.full_like(scalar_values, 0.5)
    else:
        normalized = (scalar_values - min_val) / (max_val - min_val)
    
    # Create colormap (blue to red heat map)
    colors = plt.cm.coolwarm(normalized)[:, :3]  # Take only RGB, drop alpha
    return colors
    
def load_flame_blendshape_model():
    controller_range = [0, 1]
    n_blendshapes = 51
    # surrogate_model_root_path = "/Users/evanpan/Documents/GitHub/ManifoldExploration/experiments/FACS_Based_flame_sliders_with_L1_correctly_frozen_K=5/"
    surrogate_model_root_path = "/code/facial-manifold-learning/experiments/FACS_Based_flame_sliders_with_L1_weighted_geodesic_frozen_ND=0p02/"
    flame = FLAMEBlendshapes()

    weights = []
    for i in range(0, n_blendshapes):
        exp_path = os.path.join(surrogate_model_root_path, f"exp_params_{i}.npy")
        jaw_path = os.path.join(surrogate_model_root_path, f"jaw_params_{i}.npy")
        exp_params = np.load(exp_path)
        jaw_params = np.load(jaw_path)
        if exp_params.ndim == 1:
            exp_params = exp_params.reshape(1, -1)
        if jaw_params.ndim == 1:
            jaw_params = jaw_params.reshape(1, -1)
        weights.append(np.concatenate([exp_params, jaw_params], axis=1))
    weights = np.concatenate(weights, axis=0)
    flame_zero = np.zeros([103])
    surrogate_Vs = []
    for i in range(n_blendshapes):
        flame_weights_i = weights[i]
        surrogate_Vs.append(flame.eval(flame_weights_i) - flame.eval(flame_zero))
    surrogate_model = BasicBlendshapes(
        names=[f"blendshape_{i}" for i in range(n_blendshapes)],
        V=flame.V,
        blenshapes=np.array(surrogate_Vs),
        F=flame.F,
    )


    return flame, surrogate_model, flame_zero, weights

flame_model, flame_surrogate, flame_zero, FACS_directions = load_flame_blendshape_model()

# linear is defined as f(ax + by) = af(x) + bf(y). 

significant_meshes = {}

all_results = []
for i in range(0, 200):
    random_bs_weights = np.random.uniform(0, 1, size=(51,))
    facs_latent_code = random_bs_weights @ FACS_directions
    V_through_latent = flame_model.eval(facs_latent_code)
    V_through_surrogate = flame_surrogate.eval(random_bs_weights)
    error = np.linalg.norm(V_through_latent - V_through_surrogate, axis=1)
    result_i = {
        "V_through_latent": V_through_latent,
        "V_through_surrogate": V_through_surrogate,
        "error": error,
        "mean_error": error.mean()
    }
    all_results.append(result_i)



visualizer, task = run_visualizer()

# get the results with the least error, highest error and compute the mean error
min_error_result = min(all_results, key=lambda x: x["mean_error"])
max_error_result = max(all_results, key=lambda x: x["mean_error"])
# sample 2 meshes
sampled_results = np.random.choice(all_results, size=2, replace=False)
mean_error = np.mean([res["mean_error"] for res in all_results])    

min_error_heatmap = get_heatmap_colors(min_error_result["error"])
visualizer.add_mesh(f"min_error_through_latent", min_error_result["V_through_latent"] + np.array([[0.2, 0, 0]]), flame_surrogate.F, colors=min_error_heatmap)
visualizer.add_mesh(f"min_error_through_surrogate", min_error_result["V_through_surrogate"], flame_surrogate.F)

max_error_heatmap = get_heatmap_colors(max_error_result["error"])
visualizer.add_mesh(f"max_error_through_latent", max_error_result["V_through_latent"] + np.array([[0.2, 0.2, 0]]), flame_surrogate.F, colors=max_error_heatmap)
visualizer.add_mesh(f"max_error_through_surrogate", max_error_result["V_through_surrogate"] + np.array([[0, 0.2, 0]]), flame_surrogate.F)

sample_i_heatmap = get_heatmap_colors(sampled_results[0]["error"])
visualizer.add_mesh(f"sample_i_through_latent", sampled_results[0]["V_through_latent"] + np.array([[0.2, 0.4, 0]]), flame_surrogate.F, colors=sample_i_heatmap)
visualizer.add_mesh(f"sample_i_through_surrogate", sampled_results[0]["V_through_surrogate"] + np.array([[0.0, 0.4, 0]]), flame_surrogate.F)

sample_j_heatmap = get_heatmap_colors(sampled_results[1]["error"])
visualizer.add_mesh(f"sample_j_through_latent", sampled_results[1]["V_through_latent"] + np.array([[0.2, 0.6, 0]]), flame_surrogate.F, colors=sample_j_heatmap)
visualizer.add_mesh(f"sample_j_through_surrogate", sampled_results[1]["V_through_surrogate"] + np.array([[0.0, 0.6, 0]]), flame_surrogate.F)