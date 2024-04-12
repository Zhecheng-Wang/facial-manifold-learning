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
print(len(blendshapes))
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
    SM0.update_vertex_positions(V0)
    
    # SM1.update_vertex_positions(V1)
def update_mesh_color_affinity_map(model, changed_index):
    return
    # a sequence of 10 colors for 10 different clusters, starting from red
    colors = [(1, 0, 0), (1, 0.5, 0), (1, 1, 0), (0.5, 1, 0), (0, 1, 0), (0, 1, 0.5), (0, 1, 1), (0, 0.5, 1), (0, 0, 1), (0.5, 0, 1)]
    global SM0, blendshapes, ACTIVATED_VERTICES
    relavant_blendshapes = model.affinity_matrix[changed_index]
    # number of vertices
    num_vert = blendshapes.V.shape[0]

    # get the top 10 indices from relavant_blendshapes
    top_10_indices = np.argsort(relavant_blendshapes)[-10:][0].tolist()
    # print(top_10_indices)
    # top_10_indices = top_10_indices[::-1]
    # initialze vertex colors
    vertex_colors = np.ones((num_vert, 3)) * 0.9
    for i, idx in enumerate(top_10_indices):
        if i >= len(colors):
            break
        for v in ACTIVATED_VERTICES[top_10_indices[i]]:
            vertex_colors[v] = colors[i]
    SM0.add_color_quantity("activated", vertex_colors, enabled=True)




def gui():
    global weights, blendshapes, \
        selection_threshold, changed_index, weights_history
    selection_changed, selection_threshold = psim.SliderFloat(
        "selection threshold", selection_threshold, v_min=0, v_max=1)
    # update_mesh_color_affinity_map(model, 1)
    psim.Separator()
    # make sliders thinner
    changed = np.zeros(len(blendshapes), dtype=bool)
    for i in range(len(blendshapes)):
        changed[i], weights[i] = psim.SliderFloat(
            f"{blendshapes.names[i]}", weights[i], v_min=0, v_max=1)
    if changed.any():
        changed_index = np.where(changed)[0]
        proj_weights = projection(weights, model, selection_threshold, changed_index)
        weights = proj_weights
        update_mesh()
        update_mesh_color_affinity_map(model, changed_index)

# get the activated vertice indices for each blendshape
ACTIVATED_VERTICES = []
for b in blendshapes.blendshapes:
    # get the vertices of the top 100 activated vertices
    vertices_of_blendshape_b = np.argsort(b)[-100:][0].tolist()
    print(vertices_of_blendshape_b)

    ACTIVATED_VERTICES.append(vertices_of_blendshape_b.copy())
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
                               0.9, 0.9, 0.9], smooth_shade=True, edge_width=0.25, material="wax")
# SM1 = ps.register_surface_mesh("projected", V1, blendshapes.F, color=[0.9,0.9,0.9], smooth_shade=True, edge_width=0.25, material="normal")s

ps.show()
