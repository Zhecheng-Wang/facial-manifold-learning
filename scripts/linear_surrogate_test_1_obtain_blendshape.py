import pickle
import numpy as np
import torch 
import sys
sys.path.append("/Users/evanpan/Documents/GitHub/ManifoldExploration/src")
sys.path.append("/Users/evanpan/Documents/GitHub/ManifoldExploration")
from blendshapes import FLAMEBlendshapes
import polyscope as ps
import polyscope.imgui as psim
from scripts.polyscope_playback import MeshAnimator, MultiMeshAnimator
import copy
from sklearn.decomposition import PCA

flame = FLAMEBlendshapes()

# load the pickle_dataset:
dataset_path = "/Users/evanpan/Documents/GitHub/ManifoldExploration/data/MeadRavdess/val_mead_ravdess_0.1.pickle"
with open(dataset_path, "rb") as f:
    data = pickle.load(f)

# load flame mask
mask_path = "/Users/evanpan/Documents/GitHub/ManifoldExploration/data/flame_model/FLAME_masks/FLAME_masks.pkl"
with open(mask_path, "rb") as f:
    mask = pickle.load(f, encoding="latin1")

# get the keys of the data
data_keys = list(data.keys())

# get one sample
sample = data[data_keys[0]]
exp = sample["exp"][0]
jaw = sample["jaw"][0]
weight = np.concatenate([exp, jaw], axis=1)
# animate it in polyscope 

# convert to vertex space
V_sample_i_original = flame.V
V_neutral = flame.V
V_sample_i_original = np.expand_dims(V_sample_i_original, axis=0)
V_sample_i_original = [V_sample_i_original]
for i in range(0, len(weight)):
    V_sample_i_original.append(np.expand_dims(flame.eval(weight[i]), axis=0))
V_sample_i_original = np.concatenate(V_sample_i_original, axis=0)
V_sample_i_altered = copy.deepcopy(V_sample_i_original)

F = flame.F

# freeze all but lips
lip_vertices = mask["lips"]
not_lip_vertices = np.delete(np.arange(V_sample_i_altered.shape[1]), lip_vertices)
V_sample_i_altered[:, not_lip_vertices, :] = V_neutral[not_lip_vertices, :]
V_sample_i_altered = torch.from_numpy(V_sample_i_altered).float().to(flame.flame.device)

# optimize the flame weight to fit the frozen sample
flame_torch = flame.flame
shape_params = torch.zeros([1, 100]).to(flame_torch.device)
exp_params = torch.zeros([1, 100]).to(flame_torch.device)
tex_params = torch.zeros([1, 50]).to(flame_torch.device)
pose_params = torch.zeros([1, 3]).to(flame_torch.device)
jaw_params = torch.zeros([1, 3]).to(flame_torch.device)
eye_pose_params = torch.zeros([1, 6]).to(flame_torch.device)
optimized_weight = torch.zeros(weight.shape).to(flame_torch.device)

V_neutral = torch.from_numpy(V_neutral).float().to(flame_torch.device)
for frame_i in range(weight.shape[0]):
    # frame_i = 0
    exp_params.data = torch.from_numpy(weight[frame_i:frame_i+1, :100])
    jaw_params.data = torch.from_numpy(weight[frame_i:frame_i+1, 100:103])
    exp_params.requires_grad = True
    jaw_params.requires_grad = True
    optimizer = torch.optim.Adam([exp_params, jaw_params], lr=0.1)
    for i in range(100):
        vertices, landmarks2d, landmarks3d = flame_torch(shape_params, exp_params, pose_params=torch.concat([pose_params, jaw_params], dim=1))
        loss_local = torch.mean((vertices[0, lip_vertices, :] - V_sample_i_altered[frame_i, lip_vertices, :])**2)
        loss_non_local = torch.mean((vertices[0, not_lip_vertices, :] - V_sample_i_altered[frame_i, not_lip_vertices, :])**2)
        loss = loss_local + loss_non_local
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    optimized_weight[frame_i, :100] = exp_params.data
    optimized_weight[frame_i, 100:103] = jaw_params.data
    print("frame: ", frame_i, "loss: ", loss.item())
optimized_weight = optimized_weight.detach().numpy()

# animate the optimized weight
v_sample_i_optimized = flame.V
V_neutral = flame.V
v_sample_i_optimized = np.expand_dims(v_sample_i_optimized, axis=0)
v_sample_i_optimized = [v_sample_i_optimized]
v_sample_i_optimized[0].shape
for i in range(0, len(optimized_weight)):
    v_sample_i_optimized.append(np.expand_dims(flame.eval(optimized_weight[i]), axis=0))
v_sample_i_optimized = np.concatenate(v_sample_i_optimized, axis=0)

# animator = MultiMeshAnimator([(V_sample_i_original, F), (V_sample_i_altered, F), (v_sample_i_optimized, F)], offset_distance=0.2)
# animator.run()

# generate a linear surrogate of the blendshapes (i.e. local blendshapes)
pca = PCA(n_components=10)
v_sample_i_optimized_2D = v_sample_i_optimized.reshape(v_sample_i_optimized.shape[0], -1)
pca = pca.fit(v_sample_i_optimized_2D)
pca.explained_variance_ratio_

# take the first K components
K = 3
linear_surrogate = pca.components_[:K, :]
linear_surrogate_mean = pca.mean_
linear_surrogate = linear_surrogate.reshape(K, -1, 3)
linear_surrogate_mean = linear_surrogate_mean.reshape(-1, 3)
# scale the components to ensure when blend weight is 1, the surrogate is (3*std) times the original
singular_values = pca.singular_values_[:K]
linear_surrogate = linear_surrogate * singular_values[:K].reshape(K, 1, 1) * 3
# save these as a blendshape
surrogate_model_root_path = "/Users/evanpan/Documents/GitHub/ManifoldExploration/experiments/linear_surrogate_test/"
# os.makedirs(surrogate_model_root_path, exist_ok=True)

np.save(surrogate_model_root_path + "linear_surrogate.npy", linear_surrogate)
np.save(surrogate_model_root_path + "linear_surrogate_mean.npy", linear_surrogate_mean)
np.save(surrogate_model_root_path + "linear_surrogate_Face.npy", F)

