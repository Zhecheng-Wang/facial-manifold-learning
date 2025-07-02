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

masks_we_care_about = ["eye_region", "lips", "nose", "forehead"]
Vs = []
flame = FLAMEBlendshapes()
for feature in masks_we_care_about:
    print("feature: ", feature)
    V_for_feature_i = []
    partially_frozened_model_weights = "/Users/evanpan/Documents/GitHub/ManifoldExploration/experiments/full_face_bs_test/bs_for_" + feature + ".npy"
    # animator = MultiMeshAnimator([(V_sample_i_original, F), (V_sample_i_altered, F), (v_sample_i_optimized, F)], offset_distance=0.2)
    # animator.run()
    optimized_weight_for_feature_i = np.load(partially_frozened_model_weights)[:1000]
    for i in range(0, len(optimized_weight_for_feature_i)):
        V_for_feature_i.append(np.expand_dims(flame.eval(optimized_weight_for_feature_i[i]), axis=0))
    V_for_feature_i = np.concatenate(V_for_feature_i, axis=0)
    Vs.append(V_for_feature_i)

animator = MultiMeshAnimator([(Vs[0], flame.F), (Vs[1], flame.F), (Vs[2], flame.F), (Vs[3], flame.F)], offset_distance=0.2)
animator.run()
