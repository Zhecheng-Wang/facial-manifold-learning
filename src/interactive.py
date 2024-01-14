import os
from blendshapes import *
from clustering import compute_ruzicka_similarity, compute_jaccard_similarity
from utils import *
from inference import *
import polyscope as ps
import polyscope.imgui as psim
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

config = load_config(os.path.join(PROJ_ROOT, "experiments", "lipmlp"))
model = load_model(config)

blendshapes = load_blendshape(model="SP")
weights = np.zeros(len(blendshapes))
local_proj_weights = projection(np.zeros(len(blendshapes)), model)
global_proj_weights = projection(np.zeros(len(blendshapes)), model)

# similarity = compute_ruzicka_similarity(blendshapes)
similarity = compute_jaccard_similarity(blendshapes)

def update_mesh():
    global SM0, SM1, blendshapes
    V0 = blendshapes.eval(weights)
    V1 = blendshapes.eval(global_proj_weights)
    V1[:,0] += 20
    SM0.update_vertex_positions(V0)
    SM1.update_vertex_positions(V1)

def gui():
    global weights, local_proj_weights, global_proj_weights, blendshapes
    changed = np.zeros(len(blendshapes), dtype=bool)
    for i in range(len(blendshapes)):
        changed[i], weights[i] = psim.SliderFloat(f"{blendshapes.names[i]}", weights[i], v_min=0, v_max=1)
    if changed.any():
        changed_index = np.where(changed)[0]
        # print(changed_index)
        # global_proj_weights = projection(weights, model)
        activated = (similarity[changed_index] > 1e-2).squeeze()
        print(similarity[changed_index])
        global_proj_weights[activated] = projection(weights, model)[activated]
        update_mesh()

ps.set_verbosity(0)
ps.set_SSAA_factor(4)
ps.set_program_name("Interactive Viewer")
ps.set_ground_plane_mode("none")
ps.set_view_projection_mode("orthographic")
ps.set_autocenter_structures(False)
ps.set_autoscale_structures(False)
ps.set_front_dir("z_front")
ps.set_background_color([0,0,0])
ps.init()
ps.set_user_callback(gui)
V0 = blendshapes.eval(weights)
V1 = blendshapes.eval(global_proj_weights)
V1[:,0] += 20
SM0 = ps.register_surface_mesh("original", V0, blendshapes.F, color=[0.9,0.9,0.9], smooth_shade=True, edge_width=0.25, material="normal")
SM1 = ps.register_surface_mesh("projected", V1, blendshapes.F, color=[0.9,0.9,0.9], smooth_shade=True, edge_width=0.25, material="normal")
ps.show()