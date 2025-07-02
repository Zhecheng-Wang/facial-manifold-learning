import pickle
import numpy as np
import torch 
import sys
sys.path.append("/Users/evanpan/Documents/GitHub/ManifoldExploration/src")
sys.path.append("/Users/evanpan/Documents/GitHub/ManifoldExploration")
from blendshapes import FLAMEBlendshapes, BasicBlendshapes
import polyscope as ps
import polyscope.imgui as psim
from scripts.polyscope_playback import MeshAnimator, MultiMeshAnimator
from flame_utils import *
import copy
from sklearn.decomposition import PCA
import os   
from scipy.interpolate import interp1d

ps.set_view_projection_mode("perspective")
ps.remove_all_structures()
ps.init()
ps.set_verbosity(0)
ps.set_ground_plane_mode("none")
ps.set_front_dir("z_front")
ps.set_background_color([0, 0, 0])

RADIUS = 0.01
TEST_CAMERA_RAYS = False
mouse_x, mouse_y = 0, 0
intersects_point = False

# place a point in world space with known image space 


def screen_to_world_space(screen_coord, depth=-3.0):
    width, height = ps.get_window_size()
    # map from screen space to normalized device coordinates (NDC)
    ndc_x = screen_coord[0] / width
    ndc_y = 1 - screen_coord[1] / height

    # from NDC to camera space
    camera_params = ps.get_view_camera_parameters()
    aspect_ratio = camera_params.get_aspect()
    fov_vert = camera_params.get_fov_vertical_deg()
    
    h_image_space = 2 * np.tan(np.radians(fov_vert) / 2)
    w_image_space = h_image_space * aspect_ratio

    # screen space coordinates
    x_prime = ndc_x * w_image_space - w_image_space / 2
    y_prime = ndc_y * h_image_space - h_image_space / 2
    # convert to camera space
    px = x_prime * depth
    py = y_prime * depth
    pz = depth
    point_camera_space = np.array([px, py, pz])
    # convert to world space
    M_world_to_camera = camera_params.get_view_mat()
    M_camera_to_world = np.linalg.inv(M_world_to_camera)
    position_world = M_camera_to_world @ np.append(point_camera_space, 1)
    position_world = position_world[:3]  # discard the homogeneous coordinate
    return position_world

    


def place_points_in_world_space():
    global COUNTER
    width, height = ps.get_window_size()
    # do 8 three-quarter points
    three_quarters = [
        np.array([width / 4, height / 4]),
        np.array([width / 2, height / 4]),
        np.array([width / 2, height / 2]),
        np.array([3 * width / 4, height / 4]),
        np.array([width / 4, height / 2]),
        np.array([3 * width / 4, height / 2]),
        np.array([width / 4, 3 * height / 4]),
        np.array([width / 2, 3 * height / 4]),
        np.array([3 * width / 4, 3 * height / 4]),
    ]
    for i, screen_coord in enumerate(three_quarters):
        point_world_space = screen_to_world_space(screen_coord, depth=-3.0)
        ps.register_point_cloud(f"three_quarters_{i}_{COUNTER}",
                                np.array([point_world_space]), 
                                radius=0.01, color=[0, 1, 0])
    COUNTER += 1

def ui_callback():
    if psim.Button("Place points based on current camera view"):
        place_points_in_world_space()

COUNTER = 0
ps.set_user_callback(ui_callback)
ps.show()