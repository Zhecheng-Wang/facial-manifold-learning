import os
import numpy as np
import igl
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
import pickle
from flame_utils import FLAME, get_flame_blendshapes

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# PROJ_ROOT = "/Users/evanpan/Documents/GitHub/ManifoldExploration"
class BasicBlendshapes:
    def __init__(self, V, F, blenshapes, names):
        # V = (# of vertices, 3)
        # F = (# of faces, 3)
        # blenshapes = (# of blendshapes, # of vertices, 3)
        self.V = V # this is also the neutral pose
        self.V -= np.mean(self.V, axis=0)
        self.F = F
        self.blendshapes = blenshapes
        self.names = names
        self.delta = np.zeros((self.blendshapes.shape[0], self.V.shape[0]))
        for i in range(self.blendshapes.shape[0]):
            self.delta[i] = np.linalg.norm(self.blendshapes[i], axis=1)
        self.weights = np.zeros(self.blendshapes.shape[0])
        self.translation = np.array([0, 0, 0])
        self.scale_factor = 1
        N = igl.per_vertex_normals(self.V, self.F)
        self.facing_dir = np.mean(N, axis=0)
        self.facing_dir /= np.linalg.norm(self.facing_dir)

    def translate(self, delta_pos):
        self.translation += np.array(delta_pos)

    def scale(self, scaling=1):
        self.scale_factor = scaling
    
    def displacement(self, weights=None):
        if weights is None:
            weights = self.weights
        return (weights @ self.blendshapes.reshape(weights.shape[0], -1)).reshape(self.V.shape[0], 3)

    def eval(self, weights=None):
        V = self.V.copy()
        if weights is None:
            weights = self.weights
        V += self.displacement(weights)
        V *= self.scale_factor
        V += self.translation
        return V

    def facing_dire(self):
        return self.facing_dir
    
    def __len__(self):
        return self.blendshapes.shape[0]
    
    def __getitem__(self, idx):
        return self.blendshapes[idx]

class FLAMEBlendshapes:
    def __init__(self):
        FLAMEConfig = SimpleNamespace(
                # flame_model_path=str(__dir__ / 'data/FLAME2020/generic_model.pkl'),
                flame_model_path=os.path.join(PROJ_ROOT, "data/flame_model/FLAME2020/generic_model.pkl"),
                n_shape=100,
                n_exp=100,
                n_tex=50,
                tex_type='BFM',
                tex_path=os.path.join(PROJ_ROOT, "data/flame_model/FLAME2020/FLAME_albedo_from_BFM.npz"),
                flame_lmk_embedding_path=os.path.join(PROJ_ROOT, "data/flame_model/landmark_embedding.npy"),
            )
        self.FLAMEConfig = FLAMEConfig
        self.flame = FLAME(FLAMEConfig)
        F_flame_path = os.path.join(PROJ_ROOT, "data/flame_model/faces_flame.pickle")
        with open(F_flame_path, "rb") as f:
            F_flame = pickle.load(f)["faces"]
        
        self.shape_params = torch.zeros([1, 100]).to(self.flame.device)
        self.exp_params = torch.zeros([1, FLAMEConfig.n_exp]).to(self.flame.device)
        self.tex_params = torch.zeros([1, 50]).to(self.flame.device)
        self.pose_params = torch.zeros([1, 3]).to(self.flame.device)
        self.jaw_params = torch.zeros([1, 3]).to(self.flame.device)
        self.eye_pose_params = torch.zeros([1, 6]).to(self.flame.device)

        vertices, landmarks2d, landmarks3d = self.flame(self.shape_params, self.exp_params, pose_params=torch.concat([self.pose_params, self.jaw_params], dim=1))

        
        self.F = F_flame

        # names of the blendshapes + jaw
        self.names = [f"bs_{i}" for i in range(self.FLAMEConfig.n_exp)]
        self.names += [f"jaw_{i}" for i in range(3)]

        blendshape_exp, blendshapes_jaw, blendshape_mean = get_flame_blendshapes(self.flame)
        self.blendshape_exp = blendshape_exp.float().cpu().numpy().transpose(2, 0, 1)
        self.blendshapes_jaw = blendshapes_jaw.float().cpu().numpy().transpose(2, 0, 1)
        self.blendshape_mean = blendshape_mean.float().cpu().numpy()

        self.V = self.blendshape_mean
        self.V = self.V - np.mean(self.V, axis=0)

        # get all the blendshapes    
        self.blendshapes = np.concatenate([self.blendshape_exp + self.blendshape_mean, self.blendshapes_jaw + self.blendshape_mean], axis=0)

        self.delta = np.zeros((self.blendshapes.shape[0], self.V.shape[0]))
        for i in range(self.blendshapes.shape[0]):
            self.delta[i] = np.linalg.norm(self.blendshapes[i], axis=1)
        self.weights = np.zeros(self.blendshapes.shape[0])
        self.translation = np.array([0, 0, 0])
        self.scale_factor = 1
        N = igl.per_vertex_normals(self.V, self.F)
        self.facing_dir = np.mean(N, axis=0)
        self.facing_dir /= np.linalg.norm(self.facing_dir)

    def translate(self, delta_pos):
        self.translation += np.array(delta_pos)

    def scale(self, scaling=1):
        self.scale_factor = scaling
    
    def displacement(self, weights=None):
        if weights is None:
            weights = self.weights
        
        self.exp_params = torch.from_numpy(weights[:self.FLAMEConfig.n_exp]).float().unsqueeze(0).to(self.flame.device)
        self.jaw_params = torch.from_numpy(weights[self.FLAMEConfig.n_exp:]).float().unsqueeze(0).to(self.flame.device)
        return
    
    def eval(self, weights=None):
        self.displacement(weights) # update the parameter if it changes        
        vertices, landmarks2d, landmarks3d = self.flame(self.shape_params, self.exp_params, pose_params=torch.concat([self.pose_params, self.jaw_params], dim=1))
        V = vertices[0].cpu().numpy()
        V = V - np.mean(self.V, axis=0)
        V *= self.scale_factor
        V += self.translation
        return V
    
    def facing_dire(self):
        return self.facing_dir
    def __len__(self):
        return self.blendshapes.shape[0]
    def __getitem__(self, idx):
        return self.blendshapes[idx]
    

if __name__ == "__main__":
    flameBS = FLAMEBlendshapes()
    V = flameBS.eval()   
    # Load blendshapes
    
    
    std = get_flame_pca_std_from_pkl("/Users/evanpan/Documents/GitHub/ManifoldExploration/data/flame_model/FLAME2020/generic_model.pkl")
    
