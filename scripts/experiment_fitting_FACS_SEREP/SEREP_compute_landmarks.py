import os
from omegaconf import OmegaConf
import sys
sys.path.append("/code/speech2face_baselines/")
sys.path.append("/code/expressive-speech2face/")
sys.path.append("/code/facial-manifold-learning/src")
from speech2face.mesh.utils import loadObj
from speech2face.models.spiral.get_model import get_model
from speech2face.scripts.id_exp_convert_tests import replace_on_cfg
import torch 
import numpy as np
from tqdm import trange
from matplotlib import pyplot as plt
from web_visualizer_server import *
import pickle 
from blendshapes import FLAMEBlendshapes, BasicBlendshapes
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist

def compute_barycentric_matrix(V, F, points):
    """
    Compute sparse barycentric coordinate matrix for points on a mesh surface.
    
    Parameters:
    -----------
    V : numpy.ndarray, shape (N, 3)
        Vertex positions of the mesh
    F : numpy.ndarray, shape (M, 3)
        Face indices (triangles) of the mesh
    points : numpy.ndarray, shape (K, 3)
        Points on the surface of the mesh
        
    Returns:
    --------
    scipy.sparse.csr_matrix, shape (K, N)
        Sparse barycentric coordinate matrix where each row contains
        the barycentric coordinates for one point
    """
    
    K = points.shape[0]
    N = V.shape[0]
    M = F.shape[0]
    
    # Compute face centers for initial distance estimation
    face_centers = np.mean(V[F], axis=1)  # Shape: (M, 3)
    
    # For each point, find the closest face and compute barycentric coordinates
    row_indices = []
    col_indices = []
    data = []
    
    for k in range(K):
        point = points[k]
        
        # Find closest face by center distance (rough approximation)
        distances_to_centers = np.linalg.norm(face_centers - point, axis=1)
        closest_faces = np.argsort(distances_to_centers)[:10]  # Check top 10 closest
        
        min_distance = float('inf')
        best_face_idx = None
        best_barycentric = None
        
        # Check the closest faces more carefully
        for face_idx in closest_faces:
            face = F[face_idx]
            triangle_vertices = V[face]  # Shape: (3, 3)
            
            # Project point onto triangle plane and compute barycentric coordinates
            bary_coords, distance = project_point_to_triangle(point, triangle_vertices)
            
            if distance < min_distance:
                min_distance = distance
                best_face_idx = face_idx
                best_barycentric = bary_coords
        
        # Store the barycentric coordinates in sparse matrix format
        face_vertices = F[best_face_idx]
        for i, vertex_idx in enumerate(face_vertices):
            if best_barycentric[i] > 1e-10:  # Only store non-zero coordinates
                row_indices.append(k)
                col_indices.append(vertex_idx)
                data.append(best_barycentric[i])
    
    # Create sparse matrix
    barycentric_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(K, N))
    
    return barycentric_matrix


def project_point_to_triangle(point, triangle_vertices):
    """
    Project a point onto a triangle and compute barycentric coordinates.
    
    Parameters:
    -----------
    point : numpy.ndarray, shape (3,)
        Point to project
    triangle_vertices : numpy.ndarray, shape (3, 3)
        Vertices of the triangle
        
    Returns:
    --------
    barycentric_coords : numpy.ndarray, shape (3,)
        Barycentric coordinates of the projected point
    distance : float
        Distance from point to triangle
    """
    
    v0, v1, v2 = triangle_vertices
    
    # Compute triangle edges
    edge1 = v1 - v0
    edge2 = v2 - v0
    
    # Compute normal vector
    normal = np.cross(edge1, edge2)
    normal_length = np.linalg.norm(normal)
    
    if normal_length < 1e-10:
        # Degenerate triangle
        return np.array([1.0, 0.0, 0.0]), float('inf')
    
    normal = normal / normal_length
    
    # Project point onto triangle plane
    to_point = point - v0
    distance_to_plane = np.dot(to_point, normal)
    projected_point = point - distance_to_plane * normal
    
    # Compute barycentric coordinates using the projected point
    barycentric_coords = compute_barycentric_coordinates(projected_point, triangle_vertices)
    
    # Clamp barycentric coordinates to ensure they're within the triangle
    barycentric_coords = np.clip(barycentric_coords, 0.0, 1.0)
    barycentric_coords = barycentric_coords / np.sum(barycentric_coords)
    
    # Compute actual distance from point to triangle
    closest_point_on_triangle = (barycentric_coords[0] * v0 + 
                                barycentric_coords[1] * v1 + 
                                barycentric_coords[2] * v2)
    
    distance = np.linalg.norm(point - closest_point_on_triangle)
    
    return barycentric_coords, distance


def compute_barycentric_coordinates(point, triangle_vertices):
    """
    Compute barycentric coordinates of a point with respect to a triangle.
    
    Parameters:
    -----------
    point : numpy.ndarray, shape (3,)
        Point for which to compute barycentric coordinates
    triangle_vertices : numpy.ndarray, shape (3, 3)
        Vertices of the triangle
        
    Returns:
    --------
    numpy.ndarray, shape (3,)
        Barycentric coordinates [w0, w1, w2] where point = w0*v0 + w1*v1 + w2*v2
    """
    
    v0, v1, v2 = triangle_vertices
    
    # Compute vectors
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p = point - v0
    
    # Compute dot products
    dot00 = np.dot(v0v2, v0v2)
    dot01 = np.dot(v0v2, v0v1)
    dot02 = np.dot(v0v2, v0p)
    dot11 = np.dot(v0v1, v0v1)
    dot12 = np.dot(v0v1, v0p)
    
    # Compute barycentric coordinates
    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    w = 1.0 - u - v
    
    return np.array([w, v, u])



class DummyArgs:
    def __init__(self, input, output):
        self.config = "/code/models/id_exp_apply_model/config.yaml"
        self.checkpoint = "/code/models/id_exp_apply_model/checkpoint_epoch41.pth"
        self.neutral = "/code/models/S077_HSP_M_20/Head/S077_HSP_M_20_Head.obj"
        self.output = output
        self.input = input        
        self.scale = 0.01
        self.shift = (0, 169.44, 5.2)

faces_path = "/mnt/e/Projects/Ubi_Speech2face/models/flame_retopo/faces.pickle" 
FLAME_TINGS_ROOT = "/code/models/flame2ubi"
FLAME_IN_UBI_fname = "generic_model_ubito_flame_v2.pkl"
flame_lmk_path =  "landmark_embedding.npy"
FLAME_IN_UBI_fname = os.path.join(FLAME_TINGS_ROOT, FLAME_IN_UBI_fname)
flame_lmk_path = os.path.join(FLAME_TINGS_ROOT, flame_lmk_path)
flame_in_ubi_path = FLAME_IN_UBI_fname
args = DummyArgs(input=None, output=None)

config = OmegaConf.load(args.config)
replace_on_cfg(config)
config.model.data_root = r"/expnet_root"
device = torch.device('cuda')

# load model
model = get_model(config.model, device)
checkpoint = torch.load(args.checkpoint)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.to(device)

# load landmarks
with open(flame_in_ubi_path, 'rb') as f:
    flame_in_ubi = pickle.load(f, encoding="latin1")

flame_in_ubi["f"] # faces
flame_in_ubi["weights"].shape # vertices

flame_BS = FLAMEBlendshapes()
neutral_flame = flame_BS.V
__, lmks = flame_BS.eval(return_landmarks=True) 

# for each landmark find the closest triangle on 

F = loadObj(args.neutral)["tris"]
flame_in_ubi.keys()
# load neutral mesh
batch_size = 3
neutral = loadObj(args.neutral)['verts']
neutral = (neutral - np.array(args.shift)) * args.scale
neutral = neutral.astype(np.float32)
neutral = torch.from_numpy(neutral).to(device)
mean, std = checkpoint['meanstd']
neutral = (neutral - mean) / std
neutral = neutral.tile(batch_size, 1, 1)
F = loadObj(args.neutral)["tris"]
# input code
code = torch.zeros([batch_size, 64], device=device)
code = torch.randn([batch_size, 64], device=device) * 0.1
mesh =  model.id_encoder(neutral, code) # [1, 13473, 3]
mesh = mesh * std + mean
# visualize the mesh in polyscope:
V = mesh.detach().cpu().numpy()[0]


lmks.shape
# load flame model
# in jupyter
visualizer, task = run_visualizer()
visualizer.add_mesh("flame",flame_BS.V, flame_BS.F)
visualizer.add_point_cloud("flame_lmks", lmks, colors=np.zeros((lmks.shape[0], 3)), radius=0.005)
visualizer.add_mesh("ubi", V+np.array([[0, -0.3, 0]]), F)
barycentric_matrix = compute_barycentric_matrix(flame_in_ubi["v_template"], F, lmks)
barycentric_matrix.shape
# save the barycentric matrix
np.save("/code/facial-manifold-learning/scripts/experiment_fitting_FACS_SEREP/ubi_topo_lmk_barycentric_matrix.npy", barycentric_matrix.toarray())
new_lmk = barycentric_matrix @ (V + np.array([[0, -0.3, 0]]))
visualizer.add_point_cloud("ubi_lmks", new_lmk, colors=np.zeros((new_lmk.shape[0], 3)), radius=0.005)