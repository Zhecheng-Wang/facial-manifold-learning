import pickle
import numpy as np
import torch 
import sys
sys.path.append("/Users/evanpan/Documents/GitHub/ManifoldExploration/src")
from blendshapes import FLAMEBlendshapes
import polyscope as ps
import polyscope.imgui as psim
from scripts.polyscope_playback import MeshAnimator, MultiMeshAnimator

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
v_sample_i = flame.V
V_neutral = flame.V
v_sample_i = np.expand_dims(v_sample_i, axis=0)
v_sample_i = [v_sample_i]
v_sample_i[0].shape
for i in range(0, len(weight)):
    v_sample_i.append(np.expand_dims(flame.eval(weight[i]), axis=0))
V_sample_i = np.concatenate(v_sample_i, axis=0)

F = flame.F

# freeze all but lips
lip_vertices = mask["lips"]
not_lip_vertices = np.delete(np.arange(V_sample_i.shape[1]), lip_vertices)
V_sample_i[:, not_lip_vertices, :] = V_neutral[not_lip_vertices, :]
V_sample_i = torch.from_numpy(V_sample_i).float().to(flame.flame.device)
# optimize the flame weight to fit the frozen sample

flame_torch = flame.flame
shape_params = torch.zeros([1, 100]).to(flame_torch.device)
exp_params = torch.zeros([1, 100]).to(flame_torch.device)
tex_params = torch.zeros([1, 50]).to(flame_torch.device)
pose_params = torch.zeros([1, 3]).to(flame_torch.device)
jaw_params = torch.zeros([1, 3]).to(flame_torch.device)
eye_pose_params = torch.zeros([1, 6]).to(flame_torch.device)
optimized_weight = torch.zeros(weight.shape).to(flame_torch.device)

for frame_i in range(weight.shape[0]):
    # frame_i = 0
    exp_params.data = torch.from_numpy(weight[frame_i:frame_i+1, :100])
    jaw_params.data = torch.from_numpy(weight[frame_i:frame_i+1, 100:103])
    exp_params.requires_grad = True
    jaw_params.requires_grad = True
    optimizer = torch.optim.Adam([exp_params, jaw_params], lr=0.1)
    for i in range(100):
        vertices, landmarks2d, landmarks3d = flame_torch(shape_params, exp_params, pose_params=torch.concat([pose_params, jaw_params], dim=1))
        loss = torch.mean((vertices[0, lip_vertices, :] - V_sample_i[frame_i, lip_vertices, :])**2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    optimized_weight[frame_i, :100] = exp_params.data
    optimized_weight[frame_i, 100:103] = jaw_params.data
    print("frame: ", frame_i, "loss: ", loss.item())
optimized_weight = optimized_weight.cpu().detach().numpy()

# animate the optimized weight
v_sample_i_optimized = flame.V
V_neutral = flame.V
v_sample_i_optimized = np.expand_dims(v_sample_i_optimized, axis=0)
v_sample_i_optimized = [v_sample_i_optimized]
v_sample_i_optimized[0].shape
for i in range(0, len(optimized_weight)):
    v_sample_i_optimized.append(np.expand_dims(flame.eval(optimized_weight[i]), axis=0))
v_sample_i_optimized = np.concatenate(v_sample_i_optimized, axis=0)

animator = MultiMeshAnimator([(v_sample_i_optimized, F), (V_sample_i, F)], offset_distance=0.2)
animator.run()


