import os
from blendshapes import *
from clustering import compute_ruzicka_similarity, compute_jaccard_similarity
from utils import *
from inference import *
import polyscope as ps
import polyscope.imgui as psim
import imgui

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
prev_weight = np.zeros(len(blendshapes))

def update_mesh():
    global SM0, blendshapes
    V0 = blendshapes.eval(weights)
    # V1 = blendshapes.eval(proj_weights)
    # V1[:, 0] += 20
    SM0.update_vertex_positions(V0)
    # SM1.update_vertex_positions(V1)
def custom_slider_with_colored_background(label, value, min_value, max_value, size=(0, 0)):
    # Get the window draw list to add custom drawing commands
    # Calculate the slider's position and size
    cursor_pos_x, cursor_pos_y = psim.get_cursor_screen_pos()
    slider_width = 200  # Set the desired width of your slider
    slider_height = 20  # Set the desired height of your slider

    # Calculate the midpoint of the slider for the two-colored background
    midpoint = cursor_pos_x + (slider_width * value / (max_value - min_value))

    # Draw the left half of the background (red)
    draw_list.add_rect_filled(cursor_pos_x, cursor_pos_y, midpoint, cursor_pos_y + slider_height, psim.get_color_u32_rgba(1, 0, 0, 1))

    # Draw the right half of the background (black)
    draw_list.add_rect_filled(midpoint, cursor_pos_y, cursor_pos_x + slider_width, cursor_pos_y + slider_height, psim.get_color_u32_rgba(0, 0, 0, 1))

    # Overlay the standard ImGui slider on top
    changed, new_value = psim.slider_float(label, value, min_value, max_value, size=size)
    psim.end()  # End the window

    return changed, new_value
def gui():
    global config, weights, local_proj_weights, global_proj_weights, blendshapes, \
        selection_threshold, contextual_weight, changed_index, weights_history, raw_weights, prev_weight
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
    __, __ = custom_slider_with_colored_background("selection threshold", 0.5, 0, 1)
    psim.Separator()

    selection_changed, selection_threshold = psim.SliderFloat(
        "selection threshold", selection_threshold, v_min=0, v_max=1)
    contextual_changed, contextual_weight = psim.SliderFloat(
        "contextual weight", contextual_weight, v_min=0, v_max=1)
    psim.Separator()
    # make sliders thinner
    changed = np.zeros(len(blendshapes), dtype=bool)
    for i in range(len(blendshapes)):
        changed[i], raw_weights[i] = psim.SliderFloat(
            f"{blendshapes.names[i]}", raw_weights[i], v_min=0, v_max=1)
        

    if changed.any():
        changed_index = np.where(changed)[0]
        projected_prev_weight = projection(prev_weight, model)
        projected_new_weight = projection(raw_weights, model)
        weights = weights + (projected_new_weight - projected_prev_weight)
        weights[changed_index] = raw_weights[changed_index]
        # weights[changed_index] = raw_weights[changed_index]
        update_mesh()
        prev_weight = raw_weights.copy()
    

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
