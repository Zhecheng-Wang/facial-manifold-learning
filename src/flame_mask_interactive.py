import os
import numpy as np
import torch
import pickle
from typing import Callable
from numpy.typing import NDArray
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy import linalg

import polyscope as ps
import polyscope.imgui as psim

from model import load_model
from solver import (
    LeastSquaresSolverVanilla,
    GradientSolverSciPy,
    LeastSquaresSolver,
)
from utils import (
    load_blendshape,
    MeadRavdessDataset,
    load_config,
    explore,
    inspect_attrs,
)


# ---------------------------------------------------------------------
# I/O & model data
# ---------------------------------------------------------------------
print("Loading model data...")
PROJ_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
)

config = load_config(os.path.join(PROJ_ROOT, "experiments", "rinat_small"))
model = load_model(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

blendshapes = load_blendshape(model="FLAME")
n_blendshapes = len(blendshapes)
inspect_attrs(blendshapes)

# ---------------------------------------------------------------------
# Animation Data
# ---------------------------------------------------------------------
print("Loading animation data...")

# 01-01-05-02-02-02-07
# W037_contempt_level_3_033
# M042_angry_level_1_010
# 01-02-04-02-02-02-20
dataset = MeadRavdessDataset("01-02-04-02-02-02-07")
n_frames = len(dataset)

print("Number of frames:", n_frames)
frame_weights = np.zeros((n_frames, n_blendshapes), dtype=float)
for i in range(n_frames):
    # frame_w = np.concat([dataset[i]["exp"], dataset[i]["jaw"]], axis=0)
    frame_w = np.concat([dataset[i]["exp"], np.zeros(dataset[i]["jaw"].shape)], axis=0)
    frame_weights[i] = frame_w
print("Frame weights shape:", frame_weights.shape)

# ---------------------------------------------------------------------
# Global GUI state
# ---------------------------------------------------------------------
weights = np.zeros(n_blendshapes, dtype=float)
selection_threshold = 0.5
current_frame = 0
last_slider_index = 0  # track “active” BS id

# -----------------------------------------------------
# Load flame_masks.pkl
# -----------------------------------------------------
print("Loading masks...")
mask_local_path = os.path.join("data", "flame_model", "FLAME_masks", "flame_masks.pkl")
flame_mask_path = os.path.join(PROJ_ROOT, mask_local_path)

# ['eye_region', 'neck', 'left_eyeball', 'right_eyeball', 'right_ear',
# 'right_eye_region', 'forehead', 'lips', 'nose', 'scalp', 'boundary',
# 'face', 'left_ear', 'left_eye_region']
with open(flame_mask_path, "rb") as f:
    mask_data = pickle.load(f, encoding="latin1")

# -----------------------------------------------------
# Create masked blendshapes
# -----------------------------------------------------
cluster_name = "lips"
print("Clustering:", cluster_name)
mask_cluster = mask_data[cluster_name]


def mask_vertices(
    source_mesh: NDArray[np.float32],
    target_mesh: NDArray[np.float32],
    mask_vertices: NDArray[np.int64],
) -> NDArray[np.float32]:
    """Apply mask_vertices from source_mesh to target_mesh."""
    target_mesh[mask_vertices] = source_mesh[mask_vertices]
    return target_mesh


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
    blendshapes.F.tolist(),
    color=[0.9, 0.9, 0.9],
    smooth_shade=True,
    edge_width=0.25,
    material="normal",
)

SM_MASK = ps.register_surface_mesh(
    "face_mask",
    V0,
    blendshapes.F.tolist(),
    color=[0.9, 0.9, 0.9],
    smooth_shade=True,
    edge_width=0.25,
    material="clay",
)

SM_PRED = ps.register_surface_mesh(
    "face_pred",
    V0,
    blendshapes.F.tolist(),
    color=[0.9, 0.9, 0.9],
    smooth_shade=True,
    edge_width=0.25,
    material="normal",
)

ps.get_surface_mesh("face").translate([-0.2, 0, 0])
ps.get_surface_mesh("face_pred").translate([0.2, 0, 0])


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def run_controller(w_vec: np.ndarray, sel_id: int) -> np.ndarray:
    """Runs the VAE/MLP controller once and returns a flat numpy vector."""
    w_in = torch.from_numpy(w_vec).unsqueeze(0).to(device).float()
    return w_in


# ---------------------------------------------------------------------
# Least Squares Solver in Parallel
# ---------------------------------------------------------------------
def m_pred_fn_torch_factory(blendshapes, base, mask):
    def fn(z: torch.Tensor) -> torch.Tensor:
        pred = blendshapes.eval(z)
        targ = base.clone()
        targ[mask] = pred[mask]
        return targ

    return fn


# worker function must also be top‐level
def solve_one(i, z_shape, blendshapes, frame_weights):
    # print(f"Solving frame {i+1}/{len(frame_weights)}")
    # solver = LeastSquaresSolver(z_shape, lr=1e-1, steps=300, opt_name="gd")
    solver = GradientSolverSciPy(z_shape)
    mfn = m_pred_fn_torch_factory(
        blendshapes, V0, mask_cluster
    )  # closes only picklable objects
    target = blendshapes.eval(frame_weights[i])
    return solver.solve(m_pred_fn=mfn, target=target, verbose=True)


results = Parallel(n_jobs=-1)(
    delayed(solve_one)(i, n_blendshapes, blendshapes, frame_weights)
    for i in tqdm(range(n_frames), desc="Solving frames")
)
frame_pred_weights = np.stack(results, axis=0)

# frame_pred_weights = np.zeros_like(frame_weights)
# for i in tqdm(range(n_frames), desc="Solving frames"):
#     frame_pred_weights[i] = solve_one(i, n_blendshapes, blendshapes, frame_weights)


pred_weights = np.zeros_like(weights)


# ---------------------------------------------------------------------
# GUI callback
# ---------------------------------------------------------------------
def gui():
    global weights, selection_threshold, current_frame, last_slider_index, pred_weights

    # ------------------------------------------------ Reset
    if psim.Button("Reset to Canonical"):
        weights[:] = 0.0
        current_frame = 0
        last_slider_index = 0

        SM0.update_vertex_positions(V0)
        SM_MASK.update_vertex_positions(V0)
        SM_PRED.update_vertex_positions(V0)

    psim.SameLine()
    if psim.Button("Reset to Frame"):
        weights[:] = frame_weights[current_frame]
        last_slider_index = 0

        SM0.update_vertex_positions(blendshapes.eval(weights))
        masked_V0 = mask_vertices(blendshapes.eval(weights), V0, mask_cluster)
        SM_MASK.update_vertex_positions(masked_V0)

        # pred_weights = solver.solve(m_pred_fn(), masked_V0)
        pred_weights[:] = frame_pred_weights[current_frame]
        SM_PRED.update_vertex_positions(blendshapes.eval(pred_weights))

    # ------------------------------------------------ Frame selector
    psim.Text(f"Frame: {current_frame + 1}/{n_frames}")
    changed_frame, new_frame = psim.SliderInt(
        "Go to Frame", current_frame, 0, n_frames - 1
    )
    if changed_frame:
        current_frame = new_frame
        weights[:] = frame_weights[current_frame]

        SM0.update_vertex_positions(blendshapes.eval(weights))
        masked_V0 = mask_vertices(blendshapes.eval(weights), V0, mask_cluster)
        SM_MASK.update_vertex_positions(masked_V0)

        # pred_weights = solver.solve(m_pred_fn(), masked_V0)
        pred_weights[:] = frame_pred_weights[current_frame]
        SM_PRED.update_vertex_positions(blendshapes.eval(pred_weights))

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
        changed_bs, new_val = psim.SliderFloat(name, float(weights[i]), -3.0, 3.0)
        if changed_bs:
            last_slider_index = i
            weights[i] = new_val  # keep user edit
            w_pred = run_controller(weights.copy(), sel_id=i)
            # w_pred[i]         = new_val                 # protect edited coef
            weights[:] = w_pred

            SM0.update_vertex_positions(blendshapes.eval(weights))
            masked_V0 = mask_vertices(blendshapes.eval(weights), V0, mask_cluster)
            SM_MASK.update_vertex_positions(masked_V0)

            # pred_weights = solver.solve(m_pred_fn(), masked_V0)
            pred_weights[i] = new_val
            w_pred = run_controller(pred_weights.copy(), sel_id=i)
            pred_weights[:] = w_pred

            SM_PRED.update_vertex_positions(blendshapes.eval(pred_weights))


# ---------------------------------------------------------------------
ps.set_user_callback(gui)
ps.show()
