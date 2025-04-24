import os
import numpy as np
import torch
from utils import load_blendshape, SPDataset, load_config
from clustering import compute_jaccard_similarity
from model import load_model
import polyscope as ps
import polyscope.imgui as psim

# ---------------------------------------------------------------------
# I/O & initialisation
# ---------------------------------------------------------------------
PROJ_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
)
# PROJ_ROOT = "/Users/evanpan/Documents/GitHub/ManifoldExploration"
config = load_config(os.path.join(PROJ_ROOT, "experiments", "10-clusters"))
model  = load_model(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

blendshapes = load_blendshape(model="SP")
dataset        = SPDataset()
frame_weights  = dataset.data.numpy()                 # [N, m]
frame_weights_std = np.std(frame_weights, axis=0)

n_frames       = len(dataset)
n_blendshapes  = len(blendshapes)

# ---------------------------------------------------------------------
# computer dataset stats to optimize UI
# ---------------------------------------------------------------------
# the standard deviation of the weights

# ---------------------------------------------------------------------
# Global GUI state
# ---------------------------------------------------------------------
weights              = np.zeros(n_blendshapes, dtype=float)
selection_threshold  = 0.5
current_frame        = 0
last_slider_index    = 0                              # track “active” BS id
weights_hidden       = np.zeros(n_blendshapes, dtype=float)
# ---------------------------------------------------------------------
# Polyscope viewer setup
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def generate_colors(k):
    """
    Generate a list of K visually distinct colors as RGBA tuples.
    Each color is represented as (r, g, b, a) where each value is in [0.0, 1.0]
    
    Args:
        k (int): Number of colors to generate
        
    Returns:
        list: List of K color tuples in (r, g, b, a) format
    """
    import colorsys
    
    colors = []
    
    # Generate colors with good spacing in HSV color space
    for i in range(k):
        # Use golden ratio for even distribution around the color wheel
        h = i * 0.618033988749895 % 1.0  # golden ratio conjugate
        s = 0.7 + 0.3 * ((i // 3) % 3) / 2.0  # vary the saturation
        v = 0.85 + 0.15 * ((i // 9) % 2)  # vary the value/brightness
        
        # Convert HSV to RGB
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        
        # Add alpha of 1.0
        colors.append([r, g, b, 1.0])
    
    return colors

def run_controller(w_vec: np.ndarray, sel_id: int) -> np.ndarray:
    """Runs the VAE/MLP controller once and returns a flat numpy vector."""
    w_in  = torch.from_numpy(w_vec).unsqueeze(0).to(device).float()
    alpha = torch.tensor([[selection_threshold]], device=device)
    sid   = torch.tensor([[sel_id]], dtype=torch.long, device=device)

    with torch.no_grad():
        w_pred, *_ = model(w_in, sid, alpha)
    w_pred = torch.clamp(w_pred, 0, 1.0)          # [1, m]
    w_pred = w_pred.squeeze(0).cpu().numpy()
    # print(f"Predicted weights: {w_pred}")
    # print(f"Difference with input: {np.abs(w_pred - w_vec)}")
    return w_pred

# ---------------------------------------------------------------------
# GUI callback
# ---------------------------------------------------------------------
def gui():
    global weights, selection_threshold, current_frame, last_slider_index, weights_hidden

    # ------------------------------------------------ Reset
    if psim.Button("Reset to Canonical"):
        weights_hidden[:]           = 0.0
        weights[:]               = 0.0
        current_frame        = 0
        last_slider_index    = 0
        SM0.update_vertex_positions(blendshapes.eval(weights_hidden))
    
    psim.SameLine()
    if psim.Button("Reset to Frame"):
        weights_hidden[:]    = frame_weights[current_frame]
        weights[:]           = frame_weights[current_frame]
        last_slider_index    = 0
        SM0.update_vertex_positions(blendshapes.eval(weights_hidden))

    # ------------------------------------------------ Frame selector
    psim.Text(f"Frame: {current_frame + 1}/{n_frames}")
    changed_frame, new_frame = psim.SliderInt(
        "Go to Frame", current_frame, 0, n_frames - 1
    )
    if changed_frame:
        current_frame = new_frame
        weights[:]    = frame_weights[current_frame]
        weights_hidden = frame_weights[current_frame]
        SM0.update_vertex_positions(blendshapes.eval(weights_hidden))

    psim.Separator()

    # ------------------------------------------------ Alpha cutoff
    changed_alpha, new_alpha = psim.SliderFloat(
        "alpha cutoff", selection_threshold, 0.0, 1.0
    )
    if changed_alpha:
        selection_threshold = new_alpha

    psim.Separator()

    # ------------------------------------------------ Blendshape sliders
    slider_width = 100

    clusters_names = model.cluster_names
    cluster_asignment = model.clustering
    
    for cluster_name in clusters_names:
        cluster_ids = cluster_asignment[cluster_name]
        cluster_stds = frame_weights_std[cluster_ids].tolist()
        # sort cluster_ids by cluster_stds
        cluster_ids = [x for _, x in sorted(zip(cluster_stds, cluster_ids), reverse=True)]    
        cluster_asignment[cluster_name][:] = cluster_ids

    cluster_colors = generate_colors(len(clusters_names))
    for cluster_i, cluster_name in enumerate(clusters_names):
        cluster_ids = cluster_asignment[cluster_name]
        cluster_color = cluster_colors[cluster_i]
        cluster_stds = frame_weights_std[cluster_ids]
        # normalized cluster_stds between 0 and 1
        cluster_stds = (cluster_stds - np.min(cluster_stds)) / (np.max(cluster_stds) - np.min(cluster_stds))
        cluster_stds += 0.2
        cluster_stds = np.minimum(cluster_stds, 1)
        for i, id in enumerate(cluster_ids):
            name = blendshapes.names[id]
            psim.SetNextItemWidth(slider_width)
            changed_bs, new_val = psim.SliderFloat("##hidden"+str(id), float(weights[id]), 0.0, 1.0)
            psim.SameLine()
            psim.SetNextItemWidth(slider_width)
            alpha_color = cluster_stds[i]
            cluster_color[-1] = alpha_color
            psim.PushStyleColor(psim.ImGuiCol_SliderGrab, cluster_color) 
            psim.BeginDisabled(True)
            __, __ = psim.SliderFloat(name, float(weights_hidden[id]), 0.0, 1.0)
            psim.EndDisabled()
            psim.PopStyleColor(1)
            if changed_bs:
                last_slider_index = id
                weights[id]        = new_val                 # keep user edit
                w_pred            = run_controller(weights.copy(), sel_id=id)
                # w_pred[i]         = new_val                 # protect edited coef
                # weights[:]        = w_pred
                weights_hidden[:] = w_pred
                SM0.update_vertex_positions(blendshapes.eval(weights_hidden))
        psim.Separator()



# ---------------------------------------------------------------------
ps.set_user_callback(gui)
ps.show()
