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
import time
from scripts.visualize_utils import plot_timeseries_bounds

# ========== Configuration ==========
timing_test = True
# ===================================
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

# get 200 sample
np.random.seed(42)  # for reproducibility
random_indices = np.random.choice(len(data_keys), 200, replace=False).tolist()
samples = []
exp = []
jaw = []
for i in range(len(random_indices)):
    sample = data[data_keys[random_indices[i]]]
    exp.append(sample["exp"][0])
    jaw.append(sample["jaw"][0])
exp = np.concatenate(exp, axis=0)
jaw = np.concatenate(jaw, axis=0)
weight = np.concatenate([exp, jaw], axis=1, dtype=np.float32)
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
shape_params = torch.zeros([1, 100]).to(flame_torch.device).float()
exp_params = torch.zeros([1, 100]).to(flame_torch.device).float()
tex_params = torch.zeros([1, 50]).to(flame_torch.device).float()
pose_params = torch.zeros([1, 3]).to(flame_torch.device).float()
jaw_params = torch.zeros([1, 3]).to(flame_torch.device).float()
eye_pose_params = torch.zeros([1, 6]).to(flame_torch.device).float()
optimized_weight = torch.zeros(weight.shape).to(flame_torch.device).float()

V_neutral = torch.from_numpy(V_neutral).float().to(flame_torch.device)



if timing_test == True:
    losses_over_time = []
    time_per_iteration = []
    time_per_frame = []
    iterations_total = 10
    for frame_i in range(iterations_total):
        # frame_i = 0
        exp_params.data = torch.from_numpy(weight[frame_i:frame_i+1, :100])
        jaw_params.data = torch.from_numpy(weight[frame_i:frame_i+1, 100:103])
        exp_params.requires_grad = True
        jaw_params.requires_grad = True
        optimizer = torch.optim.Adam([exp_params, jaw_params], lr=0.1) # originally 0.1
        frame_t_start = time.time()
        loss_curve_over_time_per_frame = []
        for i in range(100):
            iter_t_start = time.time()
            vertices, landmarks2d, landmarks3d = flame_torch(shape_params, exp_params, pose_params=torch.concat([pose_params, jaw_params], dim=1))
            loss_local = torch.mean((vertices[0, lip_vertices, :] - V_sample_i_altered[frame_i, lip_vertices, :])**2)
            loss_non_local = torch.mean((vertices[0, not_lip_vertices, :] - V_sample_i_altered[frame_i, not_lip_vertices, :])**2)
            loss = loss_local + loss_non_local
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_t_end = time.time()
            time_per_iteration.append(iter_t_end - iter_t_start)
            loss_curve_over_time_per_frame.append(loss.item())
        frame_t_end = time.time()
        losses_over_time.append(loss_curve_over_time_per_frame)
        time_per_frame.append(frame_t_end - frame_t_start)
        optimized_weight[frame_i, :100] = exp_params.data
        optimized_weight[frame_i, 100:103] = jaw_params.data
        print("frame: ", frame_i, "loss: ", loss.item())
    optimized_weight = optimized_weight.detach().numpy()
print("Average time per iteration: ", np.mean(time_per_iteration))
print("Average time per frame: ", np.mean(time_per_frame))
plot_timeseries_bounds(losses_over_time, log_scale=True, )
plt.show()

