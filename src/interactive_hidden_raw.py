import os
from blendshapes import *
from clustering import compute_ruzicka_similarity, compute_jaccard_similarity
from utils import *
from inference import *
import polyscope as ps
import polyscope.imgui as psim
PROJ_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir))

config = load_config(os.path.join(PROJ_ROOT, "experiments", "lipmlp_sparse_test"))
model = load_model(config)

blendshapes = load_blendshape(model="SP")
weights = np.zeros(len(blendshapes))
local_proj_weights = projection(np.zeros(len(blendshapes)), model)
global_proj_weights = projection(np.zeros(len(blendshapes)), model)
# used to keep a history of control
raw_weights = np.zeros(len(blendshapes))

# similarity = compute_ruzicka_similarity(blendshapes)
similarity = compute_jaccard_similarity(blendshapes)

selection_threshold = 0.5
# if contextual = 0, then will be inferenced under the current weights
# if contextual = 1, then will be inferenced under the global weights
contextual_weight = 1.0

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
    global config, weights, local_proj_weights, global_proj_weights, blendshapes, \
        selection_threshold, contextual_weight, changed_index, weights_history, raw_weights
    # save snapshot
    if psim.Button("snapshot") or (psim.IsKeyPressed(psim.GetKeyIndex(psim.ImGuiKey_Space))):
        weights_history.append(weights.copy())
        print(f"snapshot saved: {len(weights_history)}")
    psim.SameLine()
    if psim.Button("undo") or (psim.IsKeyPressed(psim.GetKeyIndex(psim.ImGuiKey_Backspace))):
        if len(weights_history) > 0:
            weights = weights_history.pop()
            print(f"undo: {len(weights_history)}")
            update_mesh()
    psim.SameLine()
    if psim.Button("reset"):
        raw_weights = np.zeros(len(blendshapes))
        weights = np.zeros(len(blendshapes))
        update_mesh()
    psim.Separator()
    selection_changed, selection_threshold = psim.SliderFloat(
        "selection threshold", selection_threshold, v_min=0, v_max=1)
    contextual_changed, contextual_weight = psim.SliderFloat(
        "contextual weight", contextual_weight, v_min=0, v_max=1)
    psim.Separator()
    # make sliders thinner
    changed = np.zeros(len(blendshapes), dtype=bool)
    for i in range(len(blendshapes)):
        changed[i], weights[i] = psim.SliderFloat(
            f"{blendshapes.names[i]}", weights[i], v_min=0, v_max=1)

    if changed.any():
        changed_index = np.where(changed)[0]
        if config["training"]["dataset"] == "SPDeltaWeight":
            delta_weights = np.zeros(len(blendshapes))
            delta_weights[changed_index] = weights[changed_index] - raw_weights[changed_index]
            local_proj_weights = projection(delta_weights, model)
            weights += local_proj_weights
            raw_weights[changed_index] = weights[changed_index]

        else:
            raw_weights[changed_index] = weights[changed_index]
            activated = (similarity[changed_index] >= np.clip(
                (1-selection_threshold)+1e-8, 0, 1)).squeeze() 
            weights[activated] = projection(raw_weights, model)[activated]# during this projection the activate
            weights[changed_index] = raw_weights[changed_index]
        # weights[changed_index] = raw_weights[changed_index]
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
