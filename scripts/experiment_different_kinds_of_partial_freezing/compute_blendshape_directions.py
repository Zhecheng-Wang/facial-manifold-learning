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
from sklearn.decomposition import PCA, FastICA
import os   
from scipy.interpolate import interp1d
import time
        

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
flame = FLAMEBlendshapes()
lmk_indices = flame.F.shape
flame.flame.to(device)
ROOT = "/scratch/ondemand29/evanpan/facial-manifold-learning"
ROOT = "/Users/evanpan/Documents/GitHub/ManifoldExploration"
K = 5
# facial_landmark_groups = {
#     "jaw": list(range(0, 17)),  # 0-16: jawline points
    
#     "right_eyebrow": list(range(17, 22)),  # 17-21: right eyebrow
#     "left_eyebrow": list(range(22, 27)),   # 22-26: left eyebrow
    
#     "nose_bridge": list(range(27, 31)),    # 27-30: nose bridge
#     "nose_tip": list(range(31, 36)),       # 31-35: nose tip and nostrils
    
#     "right_eye": list(range(36, 42)),      # 36-41: right eye
#     "left_eye": list(range(42, 48)),       # 42-47: left eye
    
#     "lip": list(range(48, 68)),      # 48-67: outer lip contour
# }
facial_landmark_groups = {    
    "right_eyebrow": list(range(17, 22)),  # 17-21: right eyebrow
    "left_eyebrow": list(range(22, 27)),   # 22-26: left eyebrow
    
    "nose": list(range(27, 36)),       # 31-35: nose tip and nostrils
    
    "right_eye": list(range(36, 42)),      # 36-41: right eye
    "left_eye": list(range(42, 48)),       # 42-47: left eye
    
    "lip+jaw": list(range(48, 68)) + list(range(0, 17)),      # 48-67: outer lip contour
}

V_neutral = flame.V
masks_we_care_about = list(facial_landmark_groups.keys())
for feature in masks_we_care_about:
    partially_frozened_model_weights = "/Users/evanpan/Documents/GitHub/ManifoldExploration/experiments/full_face_bs_test_freeze_landmarks_and_5_ajacent_200_video_aggregated_LM/bs_for_" + feature + ".npy"
    # animator = MultiMeshAnimator([(V_sample_i_original, F), (V_sample_i_altered, F), (v_sample_i_optimized, F)], offset_distance=0.2)
    # animator.run()
    optimized_weight_for_feature_i = np.load(partially_frozened_model_weights)

    V_sample = []
    for i in range(0, len(optimized_weight_for_feature_i)):
        V_sample.append(np.expand_dims(flame.eval(optimized_weight_for_feature_i[i]), axis=0))
    V_sample = np.concatenate(V_sample, axis=0)
    V_sample = V_sample - V_neutral

    # generate a linear surrogate of the blendshapes (i.e. local blendshapes)
    pca = PCA(n_components=10)
    V_sample = V_sample.reshape(V_sample.shape[0], -1)  # flatten the vertices

    pca = pca.fit(V_sample)
    pca.mean_ = V_neutral
    # # take the first K components
    K = 0
    explained_variance_counter = 0
    for i in range(0, pca.components_.shape[0]):
        explained_variance_counter += pca.explained_variance_ratio_[i]
        print(explained_variance_counter)
        if explained_variance_counter >= 0.99:
            K = i + 1
            break
    print("K: ", K, "explained variance: ", explained_variance_counter)

    linear_surrogate = pca.components_[:K, :]
    print("linear surrogate shape: ", linear_surrogate.shape)
    linear_surrogate.shape
    linear_surrogate_mean = pca.mean_
    linear_surrogate.shape
    linear_surrogate = linear_surrogate.reshape(K, -1, 3)
    linear_surrogate_mean = linear_surrogate_mean.reshape(-1, 3)
    # scale the components to ensure when blend weight is 1, the surrogate is (3*std) times the original
    singular_values = pca.singular_values_[:K]
    linear_surrogate = linear_surrogate * singular_values[:K].reshape(K, 1, 1) * 3
    # save these as a blendshape
    surrogate_model_root_path = "/Users/evanpan/Documents/GitHub/ManifoldExploration/experiments/full_face_bs_test_freeze_landmarks_and_5_ajacent_200_video_aggregated_LM/"
    # os.makedirs(surrogate_model_root_path, exist_ok=True)

    np.save(surrogate_model_root_path + f"linear_surrogate_{feature}.npy", linear_surrogate)
    np.save(surrogate_model_root_path + f"linear_surrogate_mean_{feature}.npy", linear_surrogate_mean)
    np.save(surrogate_model_root_path + f"linear_surrogate_Face.npy", flame.F)

