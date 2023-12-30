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
    n_frames = 1000
    # generate # of n_sample random weights
    dataset = load_dataset()
    weights_gt = random_sample(dataset, n_frames)
    
    dae_manifold_path = os.path.join(PROJ_ROOT, "experiments", "dae_manifold")
    config = load_config(dae_manifold_path)
    dae_model = load_model(config)
    
    vae_submanifold_path = os.path.join(PROJ_ROOT, "experiments", "vae_submanifold")
    vae_ensemble = []
    for i, cluster in enumerate(clusters):
        cluster_path = os.path.join(vae_submanifold_path, f"cluster_{i}")
        config = load_config(cluster_path)
        model = load_model(config)
        vae_ensemble.append((model, config["clusters"]))
    
    n_trials = 20
    sv_samples = np.linspace(0, 1, n_trials)
    trials = np.zeros(n_frames*n_trials*3)
    events = np.zeros(n_frames*n_trials*3, dtype=str)
    errors = np.zeros(n_frames*n_trials*3)
    for i in range(n_trials):
        sv = sv_samples[i]
        weights = weights_gt + np.random.normal(loc=0, scale=sv, size=weights_gt.shape)
        weights = np.clip(weights, 0, 1)
        proj_weights_dm, V_proj_dm = manifold_projection(blendshapes, weights, dae_model)
        proj_weights_vsm, V_proj_vsm = submanifolds_projection(blendshapes, weights, vae_ensemble)
        # proj_vsm_error = compute_error(proj_weights_vsm, weights_gt)
        # print(i, proj_vsm_error)
        trials[i*n_frames*3:(i+1)*n_frames*3] = sv
        events[i*n_frames*3:(i+1)*n_frames*3] = np.concatenate([["Error"]*n_frames, ["DAE Projection Error"]*n_frames, ["VAE  Projection Error"]*n_frames])
        errors[i*n_frames*3:(i+1)*n_frames*3] = np.concatenate([\
            np.linalg.norm(weights - weights_gt, axis=1),\
            np.linalg.norm(proj_weights_dm - weights_gt, axis=1),\
            np.linalg.norm(proj_weights_vsm - weights_gt, axis=1)])
    
    import pandas as pd
    df = pd.DataFrame({"trials": trials, "models": events, "errors": errors})
    sns.set_theme()
    path = '/home/zhecheng/.local/share/fonts/LinBiolinum_R.ttf'
    biolinum_font = fm.FontProperties(fname=path)
    sns.set(font=biolinum_font.get_name())
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x="trials", y="errors", hue="models", style="models",\
        errorbar="sd",\
        markers=True, dashes=False, legend="full")
    plt.xlabel("Noise Level ($\sigma$)", fontsize=20)
    plt.ylabel("$\mathcal{C}$-space Distance ($L^2$)", rotation='horizontal', fontsize=20)
    # move y-label to top
    ax = plt.gca()
    ax.yaxis.set_label_coords(0.2, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_FOLDER, FIGURE_NAME, "noise_test.png"), dpi=300)
    plt.savefig(os.path.join(FIGURES_FOLDER, FIGURE_NAME, "noise_test.pdf"), dpi=300)