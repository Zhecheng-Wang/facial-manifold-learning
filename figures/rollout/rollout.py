import os
import sys
import numpy as np
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))
src_path = os.path.join(PROJ_ROOT, "src")
sys.path.append(src_path)
FIGURES_FOLDER = os.path.join(PROJ_ROOT, "figures")
FIGURE_NAME = os.path.dirname(os.path.abspath(__file__))
from blendshapes import load_blendshape
from utils import *
from submanifold import *

def compute_error(weights, weights_gt):
    return np.linalg.norm(weights - weights_gt, axis=1).mean()

if __name__ == "__main__":
    # load the blendshape model
    import os
    BLENDSHAPES_PATH = os.path.join(PROJ_ROOT, "data", "AppleAR", "OBJs")
    blendshapes = load_blendshape(BLENDSHAPES_PATH)
    
    # compute clusters
    from clustering import *
    clusters = cluster_blendshapes(blendshapes, cluster_threshold=0.05, activate_threshold=0.2)
    
    # project to the manifold
    n_frames = 3600
    # generate # of n_sample random weights
    weights_gt = parse_BEAT_json(os.path.join(PROJ_ROOT, "data", "BEAT", "1", "1_wayne_0_9_16.json"))
    weights_gt = weights_gt[:n_frames, :]
    weights = weights_gt + np.random.normal(loc=0, scale=0.25, size=weights_gt.shape)
    weights = np.clip(weights, 0, 1)
    print("Error:", compute_error(weights, weights_gt))
    V = sample_configurations(blendshapes, weights)
    V_gt = np.zeros((n_frames, *blendshapes.V.shape))
    for i in range(n_frames):
        V_gt[i] = blendshapes.eval(weights_gt[i,:])
    
    dae_manifold_path = os.path.join(PROJ_ROOT, "experiments", "dae_manifold")
    config = load_config(dae_manifold_path)
    model = load_model(config)
    proj_weights_dm, V_proj_dm = manifold_projection(blendshapes, weights, model)
    print("DAE M Projection Error:", compute_error(proj_weights_dm, weights_gt))
    
    vae_submanifold_path = os.path.join(PROJ_ROOT, "experiments", "vae_submanifold")
    vae_ensemble = []
    for i, cluster in enumerate(clusters):
        cluster_path = os.path.join(vae_submanifold_path, f"cluster_{i}")
        if not model_exists(cluster_path):
            print(f"Manifold model does not exist. Constructing {cluster_path}")
            manifold_construction(cluster_path, cluster, network_type="vae")
        config = load_config(cluster_path)
        model = load_model(config)
        vae_ensemble.append((model, config["clusters"]))
    proj_weights_vsm, V_proj_vsm = submanifolds_projection(blendshapes, weights, vae_ensemble)
    print("VAE SM Projection Error:", compute_error(proj_weights_vsm, weights_gt))

    n_digits = len(str(n_frames))
    os.makedirs(os.path.join(FIGURES_FOLDER, FIGURE_NAME, "ground_truth"), exist_ok=True)
    os.makedirs(os.path.join(FIGURES_FOLDER, FIGURE_NAME, "manifold"), exist_ok=True)
    os.makedirs(os.path.join(FIGURES_FOLDER, FIGURE_NAME, "submanifold"), exist_ok=True)
    for i in range(n_frames):
        obj_path = os.path.join(FIGURES_FOLDER, FIGURE_NAME, "ground_truth", f"{str(i).zfill(n_digits)}.obj")
        igl.write_triangle_mesh(obj_path, V_gt[i], blendshapes.F)
        obj_path = os.path.join(FIGURES_FOLDER, FIGURE_NAME, "manifold", f"{str(i).zfill(n_digits)}.obj")
        igl.write_triangle_mesh(obj_path, V_proj_dm[i], blendshapes.F)
        obj_path = os.path.join(FIGURES_FOLDER, FIGURE_NAME, "submanifold", f"{str(i).zfill(n_digits)}.obj")
        igl.write_triangle_mesh(obj_path, V_proj_dm[i], blendshapes.F)