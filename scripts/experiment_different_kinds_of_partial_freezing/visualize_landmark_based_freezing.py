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
import time
facial_landmark_groups = {
    "jaw": list(range(0, 17)),  # 0-16: jawline points
    
    "right_eyebrow": list(range(17, 22)),  # 17-21: right eyebrow
    "left_eyebrow": list(range(22, 27)),   # 22-26: left eyebrow
    
    "nose_bridge": list(range(27, 31)),    # 27-30: nose bridge
    "nose_tip": list(range(31, 36)),       # 31-35: nose tip and nostrils
    
    "right_eye": list(range(36, 42)),      # 36-41: right eye
    "left_eye": list(range(42, 48)),       # 42-47: left eye
    
    "lip": list(range(48, 68)),      # 48-67: outer lip contour
}

masks_we_care_about = list(facial_landmark_groups.keys())
Vs = []
flame = FLAMEBlendshapes()
for feature in masks_we_care_about:
    print("feature: ", feature)
    V_for_feature_i = []
    partially_frozened_model_weights = "/Users/evanpan/Documents/GitHub/ManifoldExploration/experiments/full_face_bs_test_freeze_landmarks/bs_for_" + feature + ".npy"
    try:
        optimized_weight_for_feature_i = np.load(partially_frozened_model_weights)[:1000]
    except:
        print(f"Could not load weights for {feature}. Skipping...")
        break
    for i in range(0, len(optimized_weight_for_feature_i)):
        V_for_feature_i.append(np.expand_dims(flame.eval(optimized_weight_for_feature_i[i]), axis=0))
    V_for_feature_i = np.concatenate(V_for_feature_i, axis=0)
    Vs.append(V_for_feature_i)

input_list = []
for i in range(len(Vs)):
    V_i = Vs[i]
    F_i = flame.F
    # Create a polyscope mesh for each feature
    input_list.append((V_i, F_i))
animator = MultiMeshAnimator(input_list, offset_distance=0.2)
animator.run()

