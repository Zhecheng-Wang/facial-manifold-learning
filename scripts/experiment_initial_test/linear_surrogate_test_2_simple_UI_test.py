import pickle
import numpy as np
import torch 
import sys
sys.path.append("/Users/evanpan/Documents/GitHub/ManifoldExploration/src")
sys.path.append("/Users/evanpan/Documents/GitHub/ManifoldExploration")
from blendshapes import FLAMEBlendshapes, BasicBlendshapes
import polyscope as ps
import polyscope.imgui as psim
from scripts.polyscope_playback import MeshAnimator, MultiMeshAnimator
import copy
from sklearn.decomposition import PCA
import os   

surrogate_model_root_path = "/Users/evanpan/Documents/GitHub/ManifoldExploration/experiments/linear_surrogate_test/"

neutral_path = os.path.join(surrogate_model_root_path, "linear_surrogate_mean.npy")
blendshape_path = os.path.join(surrogate_model_root_path, "linear_surrogate.npy")
face_path = os.path.join(surrogate_model_root_path, "linear_surrogate_Face.npy")
neutral = np.load(neutral_path)
blendshapes = np.load(blendshape_path)
F = np.load(face_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
names_id = np.arange(blendshapes.shape[0])
names = ["blendshape_" + str(i) for i in names_id] 
blendshapes = BasicBlendshapes(neutral, F, blendshapes, names=names)
V0 = blendshapes.V
n_blendshapes  = len(blendshapes)
weights        = np.zeros(n_blendshapes, dtype=float)

ps.set_verbosity(0)
ps.init()
ps.set_ground_plane_mode("none")
ps.set_view_projection_mode("orthographic")
ps.set_front_dir("z_front")
ps.set_background_color([0, 0, 0])

V0  = blendshapes.eval(weights)
SM0 = ps.register_surface_mesh(
    "face", V0, blendshapes.F,
    color=[0.9, 0.9, 0.9], smooth_shade=True,
    edge_width=0.25, material="normal"
)

def run_controller(w_vec: np.ndarray, sel_id: int) -> np.ndarray:
    """Runs the VAE/MLP controller once and returns a flat numpy vector."""
    w_in  = torch.from_numpy(w_vec).unsqueeze(0).to(device).float()
    return w_in

def gui():
    global weights, selection_threshold, current_frame, last_slider_index

    # ------------------------------------------------ Reset
    if psim.Button("Reset to Canonical"):
        weights[:]           = 0.0
        current_frame        = 0
        last_slider_index    = 0
        SM0.update_vertex_positions(blendshapes.eval(weights))


    # ------------------------------------------------ Alpha cutoff


    # ------------------------------------------------ Blendshape sliders
    for i, name in enumerate(blendshapes.names):
        changed_bs, new_val = psim.SliderFloat(name, float(weights[i]), -1, 1.0)
        if changed_bs:
            last_slider_index = i
            weights[i]        = new_val                 # keep user edit
            w_pred            = run_controller(weights.copy(), sel_id=i)
            # w_pred[i]         = new_val                 # protect edited coef
            weights[:]        = w_pred
            SM0.update_vertex_positions(blendshapes.eval(weights))

# ---------------------------------------------------------------------
ps.set_user_callback(gui)
ps.show()


