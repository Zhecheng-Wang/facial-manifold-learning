import os
import sys
import numpy as np
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
PREVIEW_FOLDER = os.path.join(PROJ_ROOT, "preview")
from blendshapes import *
from utils import *
from submanifold import *
import polyscope as ps

def preview(blendshape:BasicBlendshapes, weights, save_path, color=[0.8, 0.8, 0.8]):
    V = blendshape.eval(weights)
    ps.register_surface_mesh("preview", V, blendshape.F, color=color, smooth_shade=True)
    ps.screenshot(save_path)
    ps.remove_all_structures()

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
    n_frames = 100
    # generate # of n_sample random weights
    weights_gt = parse_BEAT_json(os.path.join(PROJ_ROOT, "data", "BEAT", "1", "1_wayne_0_9_16.json"))
    weights_gt = weights_gt[:n_frames, :]
    print(weights_gt)
    weights = weights_gt + np.random.normal(loc=0, scale=0.25, size=weights_gt.shape)
    weights = np.clip(weights, 0, 1)
    print("Error:", compute_error(weights, weights_gt))
    
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

    ps.init()
    ps.set_verbosity(0)
    ps.set_SSAA_factor(3)
    ps.set_ground_plane_mode("none")
    ps.set_view_projection_mode("orthographic")
    ps.set_autocenter_structures(False)
    ps.set_autoscale_structures(False)
    n_digits = len(str(n_frames))
    os.makedirs(os.path.join(PREVIEW_FOLDER, "ground_truth"), exist_ok=True)
    os.makedirs(os.path.join(PREVIEW_FOLDER, "manifold"), exist_ok=True)
    os.makedirs(os.path.join(PREVIEW_FOLDER, "submanifold"), exist_ok=True)
    for i in range(n_frames):
        save_path = os.path.join(PREVIEW_FOLDER, "ground_truth", f"{str(i).zfill(n_digits)}.png")
        preview(blendshapes, weights_gt[i], save_path)
        save_path = os.path.join(PREVIEW_FOLDER, "manifold", f"{str(i).zfill(n_digits)}.png")
        preview(blendshapes, proj_weights_dm[i], save_path)
        save_path = os.path.join(PREVIEW_FOLDER, "submanifold", f"{str(i).zfill(n_digits)}.png")
        preview(blendshapes, proj_weights_vsm[i], save_path)
        
    # convert image sequence to video (mp4)
    import cv2
    for folder in ["ground_truth", "manifold", "submanifold"]:
        image_folder = os.path.join(PREVIEW_FOLDER, folder)
        video_name = f"{folder}.mp4"
        images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
        images = sorted(images, key=lambda x: int(x.split(".")[0]))
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape
        # write mp4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(os.path.join(PREVIEW_FOLDER, video_name), fourcc, 30, (width,height))
        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))
        cv2.destroyAllWindows()
        video.release()