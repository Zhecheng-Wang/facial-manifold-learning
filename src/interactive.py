import os
from blendshapes import *
from utils import *
from inference import *
import polyscope as ps
import polyscope.imgui as psim
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

blendshapes = load_blendshape(model="SP")
weights = np.zeros(len(blendshapes))
local_proj_weights = np.zeros(len(blendshapes))
global_proj_weights = np.zeros(len(blendshapes))

mode = "local"
config = load_config(os.path.join(PROJ_ROOT, "experiments", "hae"))
clusters = config["clusters"]
model = load_model(config)
alpha = 0.0

cluster_lookup = np.zeros(len(blendshapes), dtype=int)
for i, cluster in enumerate(clusters):
    cluster_lookup[cluster] = i
    
print(clusters)

def proj_weights(alpha):
    return (1-alpha) * local_proj_weights + alpha * global_proj_weights

def update_mesh():
    global SM0, SM1, blendshapes, alpha
    V0 = blendshapes.eval(weights)
    V1 = blendshapes.eval(proj_weights(alpha))
    V1[:,0] += 20
    SM0.update_vertex_positions(V0)
    SM1.update_vertex_positions(V1)

def gui():
    global weights, local_proj_weights, global_proj_weights, blendshapes, alpha, cluster_lookup
    alpha_changed, alpha = psim.SliderFloat("alpha", alpha, v_min=0, v_max=1)
    changed = np.zeros(len(blendshapes), dtype=bool)
    for i in range(len(blendshapes)):
        changed[i], weights[i] = psim.SliderFloat(f"{blendshapes.names[i]}", weights[i], v_min=0, v_max=1)
    if changed.any() or alpha_changed:
        changed_cluster = None
        if changed.any():
            changed_weight_index = np.where(changed)[0][0]
            print(f"Changed weights: {changed_weight_index}")
            changed_cluster_index = cluster_lookup[changed_weight_index]
            print(f"Changed clusters: {changed_cluster_index}")
            changed_cluster = clusters[changed_cluster_index]
            print(f"Changed cluster: {changed_cluster}")
        if mode == "local" and changed_cluster is not None:
            local_proj_weights[...,changed_cluster] = projection(weights, model, alpha=0.0)[...,changed_cluster]
        global_proj_weights = projection(weights, model, alpha=1.0)
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
V1 = blendshapes.eval(proj_weights(alpha))
V1[:,0] += 20
SM0 = ps.register_surface_mesh("original", V0, blendshapes.F, color=[0.9,0.9,0.9], smooth_shade=True, edge_width=0.25, material="normal")
SM1 = ps.register_surface_mesh("projected", V1, blendshapes.F, color=[0.9,0.9,0.9], smooth_shade=True, edge_width=0.25, material="normal")
ps.show()