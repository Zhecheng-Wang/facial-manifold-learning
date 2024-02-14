import os
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import polyscope as ps
import PIL
import json

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
    n_samples = 1000
    # generate # of n_sample random weights
    dataset = load_dataset()
    weights_gt = random_sample(dataset, n_samples)
    weights = weights_gt + np.random.normal(loc=0, scale=0.25, size=weights_gt.shape)
    weights = np.clip(weights, 0, 1)
    print("Error:", compute_error(weights, weights_gt))
    V = sample_configurations(blendshapes, weights)
    manifold_path = os.path.join(PROJ_ROOT, "experiments", "manifold")
    if not model_exists(manifold_path):
        print(f"Manifold model does not exist. Constructing {manifold_path}")
        manifold_construction(manifold_path)
    config = load_config(manifold_path)
    model = load_model(config)
    proj_weights_m, V_proj_m = manifold_projection(blendshapes, weights, model)
    print("M Projection Error:", compute_error(proj_weights_m, weights_gt))
    
    dae_manifold_path = os.path.join(PROJ_ROOT, "experiments", "dae_manifold")
    if not model_exists(dae_manifold_path):
        print(f"Manifold model does not exist. Constructing {dae_manifold_path}")
        manifold_construction(dae_manifold_path)
    config = load_config(dae_manifold_path)
    model = load_model(config)
    proj_weights_dm, V_proj_m = manifold_projection(blendshapes, weights, model)
    print("DAE M Projection Error:", compute_error(proj_weights_dm, weights_gt))
    
    vae_manifold_path = os.path.join(PROJ_ROOT, "experiments", "vae_manifold")
    if not model_exists(vae_manifold_path):
        print(f"Manifold model does not exist. Constructing {vae_manifold_path}")
        manifold_construction(vae_manifold_path)
    config = load_config(vae_manifold_path)
    model = load_model(config)
    proj_weights_vm, V_proj_m = manifold_projection(blendshapes, weights, model)
    print("VAE M Projection Error:", compute_error(proj_weights_vm, weights_gt))
    
    # project to the submanifold
    submanifold_path = os.path.join(PROJ_ROOT, "experiments", "submanifold")
    ensemble = []
    for i, cluster in enumerate(clusters):
        cluster_path = os.path.join(submanifold_path, f"cluster_{i}")
        if not model_exists(cluster_path):
            print(f"Manifold model does not exist. Constructing {cluster_path}")
            manifold_construction(cluster_path, cluster, network_type="ae")
        config = load_config(cluster_path)
        model = load_model(config)
        ensemble.append((model, config["clusters"]))
    proj_weights_sm, V_proj_sm = submanifolds_projection(blendshapes, weights, ensemble)
    print("SM Projection Error:", compute_error(proj_weights_sm, weights_gt))
    
    dae_submanifold_path = os.path.join(PROJ_ROOT, "experiments", "dae_submanifold")
    dae_ensemble = []
    for i, cluster in enumerate(clusters):
        cluster_path = os.path.join(dae_submanifold_path, f"cluster_{i}")
        if not model_exists(cluster_path):
            print(f"Manifold model does not exist. Constructing {cluster_path}")
            manifold_construction(cluster_path, cluster, network_type="dae")
        config = load_config(cluster_path)
        model = load_model(config)
        dae_ensemble.append((model, config["clusters"]))
    proj_weights_dsm, V_proj_sm = submanifolds_projection(blendshapes, weights, dae_ensemble)
    print("DAE SM Projection Error:", compute_error(proj_weights_dsm, weights_gt))
    
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
    proj_weights_vsm, V_proj_sm = submanifolds_projection(blendshapes, weights, vae_ensemble)
    print("VAE SM Projection Error:", compute_error(proj_weights_vsm, weights_gt))
    
    # ps.init()
    # ps.set_program_name("Clustering Visualization")
    # ps.set_verbosity(0)
    # ps.set_SSAA_factor(3)
    # ps.set_max_fps(60)
    # ps.set_ground_plane_mode("none")
    # ps.set_view_projection_mode("orthographic")
    # import matplotlib.pyplot as plt
    # import PIL
    # # fig, axes = plt.subplots(n_samples, 3, figsize=(3*2, n_samples*3))
    # fig, axes = plt.subplots(3, n_samples, figsize=(n_samples*2, 3*2))
    # for i in range(n_samples):
    #     ps_V = ps.register_surface_mesh(f"V", V[i], blendshapes.F, smooth_shade=True, enabled=True)
    #     ps.screenshot(f"temp.png")
    #     img = PIL.Image.open("temp.png")
    #     w, h = img.size
    #     diff = 0.35 * w
    #     img = img.crop((diff, 100, w-diff, h-100))
    #     # axes[i,0].imshow(img)
    #     # axes[i,0].axis("off")
    #     axes[0,i].imshow(img)
    #     axes[0,i].axis("off")
    #     ps_V.set_enabled(False)
    #     ps_V_proj_m = ps.register_surface_mesh(f"V_proj_manifold", V_proj_m[i], blendshapes.F, smooth_shade=True, enabled=True)
    #     ps.screenshot(f"temp.png")
    #     img = PIL.Image.open("temp.png")
    #     w, h = img.size
    #     diff = 0.35 * w
    #     img = img.crop((diff, 100, w-diff, h-100))
    #     # axes[i,1].imshow(img)
    #     # axes[i,1].axis("off")
    #     axes[1,i].imshow(img)
    #     axes[1,i].axis("off")
    #     ps_V_proj_m.set_enabled(False)
    #     ps_V_proj_sm = ps.register_surface_mesh(f"V_proj_submanifold", V_proj_sm[i], blendshapes.F, smooth_shade=True, enabled=True)
    #     ps.screenshot(f"temp.png")
    #     img = PIL.Image.open("temp.png")
    #     w, h = img.size
    #     diff = 0.35 * w
    #     img = img.crop((diff, 100, w-diff, h-100))
    #     # axes[i,2].imshow(img)
    #     # axes[i,2].axis("off")
    #     axes[2,i].imshow(img)
    #     axes[2,i].axis("off")
    #     ps_V_proj_sm.set_enabled(False)
    #     # ps.show()
    # os.remove("temp.png")
    # fig.tight_layout()
    # fig.savefig("compare2.png", dpi=300, bbox_inches="tight", transparent=True)
    # fig.savefig("compare2.pdf", dpi=300, bbox_inches="tight", transparent=True)