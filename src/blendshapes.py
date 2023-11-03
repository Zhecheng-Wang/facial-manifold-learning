import os
import numpy as np
import igl
import ipctk

class BasicBlendshapeModel:
    def __init__(self, V, F, blenshapes):
        # V = (# of vertices, 3)
        # F = (# of faces, 3)
        # blenshapes = (# of blendshapes, # of vertices, 3)
        self.V = V
        self.F = F
        self.blendshapes = blenshapes
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
    
class CollisionBlendshapeModel(BasicBlendshapeModel):
    def __init__(self, V, F, blenshapes):
        super().__init__(V, F, blenshapes)
        E = ipctk.edges(self.F)
        self.collison_mesh = ipctk.CollisionMesh(self.V, E, self.F)
    
    def has_intersections(self, weights=None):
        if weights is None:
            weights = self.weights
        V = self.eval(weights)
        is_intersecting = ipctk.has_intersections(self.collison_mesh, V)
        return is_intersecting
    
class IntersectionMetricBlendshapeModel(BasicBlendshapeModel):
    def __init__(self, V, F, blenshapes):
        super().__init__(V, F, blenshapes)
    
    def intersection_metric(self, weights=None):
        pass
    
def load_blendshape_model(path):
    folder_content = os.listdir(path)
    blendshape_paths = []
    count = 0
    for f in folder_content:
        file_name, file_ext = os.path.splitext(f)
        print(count, file_name, file_ext)
        if file_ext == ".obj" and file_name != "Neutral":
            blendshape_paths.append(os.path.join(path, f))
            count += 1
    N_BLENDSHAPES = len(blendshape_paths)
    print(f"Found {N_BLENDSHAPES} blendshapes.")
    neutral_path = os.path.join(path, "Neutral.obj")
    V, F = igl.read_triangle_mesh(neutral_path)
    blendshapes = np.zeros((N_BLENDSHAPES, *V.shape))
    for i in range(N_BLENDSHAPES):
        VB, _ = igl.read_triangle_mesh(blendshape_paths[i])
        blendshapes[i] = VB - V
    model = CollisionBlendshapeModel(V, F, blendshapes)
    return model