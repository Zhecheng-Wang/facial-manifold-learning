import os
from blendshapes import *
from clustering import compute_ruzicka_similarity, compute_jaccard_similarity
from utils import *
from inference import *
import polyscope as ps
import polyscope.imgui as psim
PROJ_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir))

config = load_config(os.path.join(PROJ_ROOT, "experiments", "controller"))
model = load_model(config)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

blendshapes = load_blendshape(model="SP")
weights = np.zeros(len(blendshapes))

# similarity = compute_ruzicka_similarity(blendshapes)
similarity = compute_jaccard_similarity(blendshapes)

selection_threshold = 0.5
# if contextual = 0, then will be inferenced under the current weights
# if contextual = 1, then will be inferenced under the global weights
changed_index = 0

weights_history = []

def update_mesh():
    global SM0, blendshapes
    V0 = blendshapes.eval(weights)
    # V1 = blendshapes.eval(proj_weights)
    # V1[:, 0] += 20
    SM0.update_vertex_positions(V0)
    # SM1.update_vertex_positions(V1)

def gui():
    global weights, blendshapes, \
        selection_threshold, changed_index, weights_history
    # save snapshot
    if psim.Button("snapshot") or (psim.IsKeyPressed(psim.GetKeyIndex(psim.ImGuiKey_Space))):
        weights_history.append(weights.copy())
        print(f"snapshot saved: {len(weights_history)}")
    psim.SameLine()
    # undo
    if psim.Button("undo") or (psim.IsKeyPressed(psim.GetKeyIndex(psim.ImGuiKey_Backspace))):
        if len(weights_history) > 0:
            weights = weights_history.pop()
            print(f"undo: {len(weights_history)}")
            update_mesh()
    psim.Separator()
    selection_changed, selection_threshold = psim.SliderFloat(
        "selection threshold", selection_threshold, v_min=0, v_max=1)
    psim.Separator()
    # make sliders thinner
    changed = np.zeros(len(blendshapes), dtype=bool)
    for i in range(len(blendshapes)):
        changed[i], weights[i] = psim.SliderFloat(
            f"{blendshapes.names[i]}", weights[i], v_min=0, v_max=1)
    if changed.any():
        changed_index = np.where(changed)[0]
        proj_weights = weights.copy()
        with torch.inference_mode():
            y = model(torch.from_numpy(proj_weights).to(torch.float32).unsqueeze(0), torch.tensor([selection_threshold]).reshape(1,1), torch.tensor([changed_index]).reshape(1,1))
            proj_weights = y.detach().cpu().numpy()[0]
        weights = proj_weights
        update_mesh()

ps.set_verbosity(0)
ps.set_SSAA_factor(4)
ps.set_program_name("Interactive Viewer")
ps.set_ground_plane_mode("none")
ps.set_view_projection_mode("orthographic")
ps.set_autocenter_structures(False)
ps.set_autoscale_structures(False)
ps.set_front_dir("z_front")
ps.set_background_color([0, 0, 0])
ps.init()
ps.set_user_callback(gui)
V0 = blendshapes.eval(weights)
# V1 = blendshapes.eval(global_proj_weights)
# V1[:, 0] += 20
SM0 = ps.register_surface_mesh("original", V0, blendshapes.F, color=[
                               0.9, 0.9, 0.9], smooth_shade=True, edge_width=0.25, material="normal")
# SM1 = ps.register_surface_mesh("projected", V1, blendshapes.F, color=[0.9,0.9,0.9], smooth_shade=True, edge_width=0.25, material="normal")
ps.show()
