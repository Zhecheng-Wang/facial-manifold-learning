import os
import numpy as np
import torch
from utils import load_blendshape, SPDataset, load_config
from model import load_model
import polyscope as ps
import polyscope.imgui as psim

# ---------------------------------------------------------------------
# I/O & initialisation
# ---------------------------------------------------------------------
PROJ_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
)

config = load_config(os.path.join(PROJ_ROOT, "experiments", "rinat_small"))
model = load_model(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

blendshapes = load_blendshape(model="SP")

dataset = SPDataset()
frame_weights = dataset.data.numpy()  # [N, m]
n_frames = len(dataset)
n_blendshapes = len(blendshapes)

# ---------------------------------------------------------------------
# Global GUI state
# ---------------------------------------------------------------------
weights = np.zeros(n_blendshapes, dtype=float)
selection_threshold = 0.5
current_frame = 0
last_slider_index = 0  # track “active” BS id

# ---------------------------------------------------------------------
# Polyscope viewer setup
# ---------------------------------------------------------------------
ps.set_verbosity(0)
ps.init()
ps.set_ground_plane_mode("none")
ps.set_view_projection_mode("orthographic")
ps.set_front_dir("z_front")
ps.set_background_color([0, 0, 0])

V0 = blendshapes.eval(weights)
SM0 = ps.register_surface_mesh(
    "face",
    V0,
    blendshapes.F,
    color=[0.9, 0.9, 0.9],
    smooth_shade=True,
    edge_width=0.25,
    material="normal",
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def run_controller(w_vec: np.ndarray, sel_id: int) -> np.ndarray:
    """Runs the VAE/MLP controller once and returns a flat numpy vector."""
    w_in = torch.from_numpy(w_vec).unsqueeze(0).to(device).float()
    alpha = torch.tensor([[selection_threshold]], device=device)
    sid = torch.tensor([[sel_id]], dtype=torch.long, device=device)

    with torch.no_grad():
        w_pred, *_ = model(w_in, sid, alpha)
    w_pred = torch.clamp(w_pred, 0.0, 1.0)  # [1, m]
    w_pred = w_pred.squeeze(0).cpu().numpy()
    print(f"Predicted weights: {w_pred}")
    print(f"Difference with input: {np.abs(w_pred - w_vec)}")
    return w_pred


# ---------------------------------------------------------------------
# GUI callback
# ---------------------------------------------------------------------
def gui():
    global weights, selection_threshold, current_frame, last_slider_index

    # ------------------------------------------------ Reset
    if psim.Button("Reset to Canonical"):
        weights[:] = 0.0
        current_frame = 0
        last_slider_index = 0
        SM0.update_vertex_positions(blendshapes.eval(weights))

    psim.SameLine()
    if psim.Button("Reset to Frame"):
        weights[:] = frame_weights[current_frame]
        last_slider_index = 0
        SM0.update_vertex_positions(blendshapes.eval(weights))

    # ------------------------------------------------ Frame selector
    psim.Text(f"Frame: {current_frame + 1}/{n_frames}")
    changed_frame, new_frame = psim.SliderInt(
        "Go to Frame", current_frame, 0, n_frames - 1
    )
    if changed_frame:
        current_frame = new_frame
        weights[:] = frame_weights[current_frame]
        SM0.update_vertex_positions(blendshapes.eval(weights))

    psim.Separator()

    # ------------------------------------------------ Alpha cutoff
    changed_alpha, new_alpha = psim.SliderFloat(
        "alpha cutoff", selection_threshold, 0.0, 1.0
    )
    if changed_alpha:
        selection_threshold = new_alpha

    psim.Separator()

    # ------------------------------------------------ Blendshape sliders
    for i, name in enumerate(blendshapes.names):
        changed_bs, new_val = psim.SliderFloat(name, float(weights[i]), 0.0, 1.0)
        if changed_bs:
            last_slider_index = i
            weights[i] = new_val  # keep user edit
            w_pred = run_controller(weights.copy(), sel_id=i)
            # w_pred[i]         = new_val                 # protect edited coef
            weights[:] = w_pred
            SM0.update_vertex_positions(blendshapes.eval(weights))


# ---------------------------------------------------------------------
ps.set_user_callback(gui)
ps.show()
