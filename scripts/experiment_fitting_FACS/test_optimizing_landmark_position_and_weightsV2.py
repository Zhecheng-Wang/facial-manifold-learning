import numpy as np
import os, sys
sys.path.append("/Users/evanpan/Documents/GitHub/ManifoldExploration/src")
from utils import load_ARKit_blendshape
from blendshapes import FLAMEBlendshapes, BasicBlendshapes
import torch
import polyscope as ps
import polyscope.imgui as psim
from scripts.experiment_different_kinds_of_partial_freezing.naive_autosegmentation import compute_vertex_assignments


def get_lm_indices_and_bary_weights_from_FLAME():
    """
    Returns the indices of the FLAME landmarks and their barycentric weights.
    """
    FLAME_facial_landmark_groups = {
        "jaw": list(range(7, 10)),  # 0-16: jawline points
        
        "right_eyebrow": list(range(17, 22)),  # 17-21: right eyebrow
        "left_eyebrow": list(range(22, 27)),   # 22-26: left eyebrow
        
        "nose_bridge": list(range(27, 31)),    # 27-30: nose bridge
        "nose_tip": list(range(31, 36)),       # 31-35: nose tip and nostrils
        
        "right_eye": list(range(36, 42)),      # 36-41: right eye
        "left_eye": list(range(42, 48)),       # 42-47: left eye
        
        "lip": list(range(48, 68)),      # 48-67: outer lip contour
    }
    lms_we_care_about = []
    for key in FLAME_facial_landmark_groups.keys():
         # these are the row indices after we use barycentric coordinates to map to the landmarks
        lms_we_care_about.extend(FLAME_facial_landmark_groups[key])
    lms_we_care_about = np.array(lms_we_care_about, dtype=np.int64)
    flameBS = FLAMEBlendshapes()
    # barcycentric weights, lms_we_care_about_can 
    flame_bary_weights = flameBS.flame.full_lmk_bary_coords[0, lms_we_care_about]
    flame_mesh_face_indices = flameBS.flame.full_lmk_faces_idx[0, lms_we_care_about]

    full_bary_weights = torch.zeros((flame_bary_weights.shape[0], flameBS.V.shape[0],), dtype=torch.float32)
    for i in range(0, flame_mesh_face_indices.shape[0]):
        triangles = flameBS.F[flame_mesh_face_indices[i]]
        full_bary_weights[i, triangles] = flame_bary_weights[i]

    grouped_bary_weights_dict = {}
    for key in list(FLAME_facial_landmark_groups.keys()):
        # key = list(FLAME_facial_landmark_groups.keys())[0]
        grouped_flame_bary_weights = flameBS.flame.full_lmk_bary_coords[0, FLAME_facial_landmark_groups[key]]
        grouped_mesh_face_indices = flameBS.flame.full_lmk_faces_idx[0, FLAME_facial_landmark_groups[key]] # get the indices of the face

        grouped_bary_weights = torch.zeros((grouped_flame_bary_weights.shape[0], flameBS.V.shape[0],), dtype=torch.float32)
        for i in range(0, grouped_mesh_face_indices.shape[0]):
            triangles = flameBS.F[grouped_mesh_face_indices[i]]
            grouped_bary_weights[i, triangles] = grouped_flame_bary_weights[i]
        grouped_bary_weights_dict[key] = grouped_bary_weights

    return flame_mesh_face_indices, full_bary_weights, grouped_bary_weights_dict

def get_lm_indices_from_ARKit():
    ARkitBS = load_ARKit_blendshape()
    
    ARKIT_LM_DICT = {
    "jaw": [915, 1047, 912], # I'm only going to use the center 3 lol
    "right_eyebrow": [197, 335, 327, 328, 348],
    "left_eyebrow": [781, 763, 762, 768, 646],
    "nose_bridge": [36, 13, 10, 8, ],
    "nose_tip": [308, 76, 4, 525, 743],
    "right_eye": [1101, 1096, 1093, 1089, 1085, 1107],
    "left_eye": [1081, 1077, 1074, 1069, 1064, 1061],
    "lip": [244, 239, 237, 22, 671, 673, 678, 723, 698, 27, 263, 288, 
            394, 256, 24, 691, 684, 700, 25, 265],
    }
    lm_indices = [ARKIT_LM_DICT[key] for key in ARKIT_LM_DICT.keys()]
    lm_indices = np.concatenate(lm_indices, axis=0)
    full_bary_weights = torch.zeros((lm_indices.shape[0], ARkitBS.V.shape[0]), dtype=torch.float32)

    for i in range(0, lm_indices.shape[0]):
        full_bary_weights[i, lm_indices[i]] = 1.0  # Set the barycentric weight to 1 for the landmark vertex    
    
    grouped_ARKit_bary_weights_dict = {}
    for key in list(ARKIT_LM_DICT.keys()):
        grouped_ARKit_bary_weights = torch.zeros((len(ARKIT_LM_DICT[key]), ARkitBS.V.shape[0]), dtype=torch.float32)
        for i, lm_index in enumerate(ARKIT_LM_DICT[key]):
            grouped_ARKit_bary_weights[i, lm_index] = 1.0
        grouped_ARKit_bary_weights_dict[key] = grouped_ARKit_bary_weights


    return lm_indices.tolist(), full_bary_weights, grouped_ARKit_bary_weights_dict

def display_a_single_mesh(V, F, points=None):
    ps.remove_all_structures()
    ps.set_verbosity(0)
    ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_view_projection_mode("orthographic")
    ps.set_front_dir("z_front")
    ps.set_background_color([0, 0, 0])
    ps.register_surface_mesh(
        "mesh", V, F,
        color=[0.9, 0.9, 0.9],
        edge_width=0.25, material="normal"
        )
    if points is not None:
        ps.register_point_cloud(
            "points", points,
            radius=0.01,
            color=[1.0, 0.0, 0.0],material="normal"
        )
    ps.reset_camera_to_home_view()
    ps.show()

def display_pairs_of_meshes(Vs1, Fs1, Vs2, Fs2, offset=0.3):
    ps.remove_all_structures()
    ps.set_verbosity(0)
    ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_view_projection_mode("orthographic")
    ps.set_front_dir("z_front")
    ps.set_background_color([0, 0, 0])
    
    for i in range(len(Vs1)):
        V1 = Vs1[i] + np.array([-i*offset, 0, 0], dtype=np.float32)
        F1 = Fs1[i]
        ps.register_surface_mesh(
            f"mesh_{i}_1", V1, F1,
            color=[0.9, 0.9, 0.9],
            edge_width=0.25
        )
        
        V2 = Vs2[i] + np.array([-i*offset, offset, 0], dtype=np.float32)
        F2 = Fs2[i]
        ps.register_surface_mesh(
            f"mesh_{i}_2", V2, F2,
            color=[0.5, 0.5, 0.5],
            edge_width=0.25
        )    
    ps.reset_camera_to_home_view()
    ps.show()

def compute_landmark_groups_of_blendshape(V_0, V_bs, landmark_groups):
    # V_0 = ARkitBS.V
    # V_bs = ARkitBS.blendshapes[0] + ARkitBS.V
    # landmark_groups = ARkit_lm_groups
    
    involved_lm_groups = []
    for key in list(landmark_groups.keys()):
        # key = list(landmark_groups.keys())[0]
        lms_0 = landmark_groups[key] @ V_0
        lms_bs = landmark_groups[key] @ V_bs

        diff = lms_bs - lms_0
        diff_mag = torch.norm(diff, dim=-1).mean()

        if diff_mag >= 1E-4:
            involved_lm_groups.append(key)
    return involved_lm_groups
        

 
# compute_landmark_groups_of_blendshape(ARkitBS.V, ARkitBS.blendshapes[0], ARkit_lm_groups)

LEARNING_RATE = 0.03
ITERATIONS = 5000
W_FROZEN = 0.0005
K=5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
flameBS = FLAMEBlendshapes(device=device)
flame_lm_indices, flame_full_bary_weights, flame_lm_groups = get_lm_indices_and_bary_weights_from_FLAME()
flame_LM = flame_full_bary_weights @ flameBS.V
# display_a_single_mesh(flameBS.V, flameBS.F, flame_LM.detach().numpy())

ARkitBS = load_ARKit_blendshape()
ARkit_lm_indices, ARkit_full_bary_weights, ARkit_lm_groups = get_lm_indices_from_ARKit()
ARkit_LM = ARkit_full_bary_weights @ ARkitBS.V
# display_a_single_mesh(ARkitBS.V, ARkitBS.F, ARkit_LM)

all_lm_groups = list(flame_lm_groups.keys())
# optimize for the FLAME latent code for each arkit blendshapes
flame_param_dires = []
for bs_i in range(0, ARkitBS.blendshapes.shape[0]):
    involved_lm_groups = compute_landmark_groups_of_blendshape(ARkitBS.V, ARkitBS.blendshapes[bs_i] + ARkitBS.V, ARkit_lm_groups)
    # generate the indices of the landmarks we care about
    involved_lm_indices = []
    for key in involved_lm_groups:
        barycentric_coord_matrix = flame_lm_groups[key]
        for lm_i in range(barycentric_coord_matrix.shape[0]):
            for v_i in range(barycentric_coord_matrix.shape[1]):
                if barycentric_coord_matrix[lm_i, v_i] > 0.0:
                    involved_lm_indices.append(v_i)

    # these are the ones we want frozen
    non_involved_lm_indices = []
    for key in all_lm_groups:
        if key not in involved_lm_groups:
            barycentric_coord_matrix = flame_lm_groups[key]
            for lm_i in range(barycentric_coord_matrix.shape[0]):
                for v_i in range(barycentric_coord_matrix.shape[1]):
                    if barycentric_coord_matrix[lm_i, v_i] > 0.0:
                        non_involved_lm_indices.append(v_i)

    non_frozen_set, frozen_set = compute_vertex_assignments(flameBS.V, flameBS.F,
        key_point_set_A=involved_lm_indices,
        key_point_set_B=non_involved_lm_indices,
        K=K)
    non_frozen_set = list(non_frozen_set)
    frozen_set = list(frozen_set)
    
    # get the target landmark positions
    target_V_i = ARkitBS.blendshapes[bs_i]
    target_lm_i = target_V_i[ARkit_lm_indices, :]
    target_lm_i = torch.tensor(target_lm_i, dtype=torch.float32, device=device)

    # initialize the flame latent code for optimization
    flame_latent = torch.zeros(flameBS.weights.shape, dtype=torch.float32, requires_grad=True)
    flame_model = flameBS.flame

    # get the params
    shape_params = torch.zeros([1, 100]).to(device)
    exp_params = torch.zeros([1, 100]).to(device)
    tex_params = torch.zeros([1, 50]).to(device)
    pose_params = torch.zeros([1, 3]).to(device)
    jaw_params = torch.zeros([1, 3]).to(device)
    eye_pose_params = torch.zeros([1, 6]).to(device)

    exp_params.requires_grad = True
    jaw_params.requires_grad = True
    optimizer = torch.optim.Adam([exp_params, jaw_params], lr=LEARNING_RATE)
    neutral_flame = torch.from_numpy(flameBS.V).to(device)
    loss_for_bs_i = []
    for i in range(0, ITERATIONS):
        optimizer.zero_grad()
        
        # compute the FLAME mesh
        V_optimized, _, _ = flame_model(shape_params, exp_params, pose_params=torch.concat([pose_params, jaw_params], dim=1))
        frozen_loss = V_optimized[0, frozen_set, :] - neutral_flame[frozen_set, :]
        frozen_loss = torch.mean(torch.abs(frozen_loss))
        V_optimized = V_optimized - neutral_flame
        # compute the landmark positions
        flame_LM = flame_full_bary_weights @ V_optimized
    
        # compute the loss
        loss = torch.mean((flame_LM - target_lm_i) ** 2) + frozen_loss * W_FROZEN
        
        # backpropagate
        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            print(f"Iteration {i}, Loss: {loss.item()}")
            # add a stop condition
            past_3_mean = np.mean(loss_for_bs_i[-3:]) if len(loss_for_bs_i) >= 3 else None
            if past_3_mean is not None and  loss.item() - past_3_mean > 0:
                print(f"Stopping early at iteration {i} with loss {loss.item()}")
                break

        loss_for_bs_i.append(loss.item())

    flame_param_dires.append([exp_params, jaw_params])

flame_param_dires = [[x[0].detach().cpu().numpy(), x[1].detach().cpu().numpy()] for x in flame_param_dires]
# save these
save_root = "/Users/evanpan/Documents/GitHub/ManifoldExploration/experiments/FACS_Based_flame_sliders_with_L1_frozen_LM_w_frozen_0p002_K=5"
if not os.path.exists(save_root):
    os.makedirs(save_root)
for i in range(len(flame_param_dires)):
    exp_params = flame_param_dires[i][0]
    jaw_params = flame_param_dires[i][1]
    np.save(os.path.join(save_root, f"exp_params_{i}.npy"), exp_params)
    np.save(os.path.join(save_root, f"jaw_params_{i}.npy"), jaw_params)

visualize = True
if visualize:
    flame_mesh_bs = []
    flame_F = []
    for i in range(len(flame_param_dires)):
        exp_params = flame_param_dires[i][0]
        jaw_params = flame_param_dires[i][1]
        weights = np.concatenate([exp_params, jaw_params], axis=1)
        verts = flameBS.eval(weights[0])
        flame_mesh_bs.append(verts)
        flame_F.append(flameBS.F)

    AR_kit_mesh_bs = []
    AR_kit_F = []
    for i in range(0, ARkitBS.blendshapes.shape[0]):
        AR_kit_mesh_bs.append(ARkitBS.blendshapes[i] + ARkitBS.V)
        AR_kit_F.append(ARkitBS.F)

    display_pairs_of_meshes(flame_mesh_bs, flame_F, AR_kit_mesh_bs, AR_kit_F, offset=0.2)

