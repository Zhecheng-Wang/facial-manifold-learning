import os
import numpy as np
import igl

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

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
