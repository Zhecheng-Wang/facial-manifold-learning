import asyncio
import websockets
import json
import numpy as np
import struct
from typing import Dict, Optional

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

class BlendshapeServer:
    def __init__(self, host="localhost", port=8765):
        self.host = host
        self.port = port
        self.faces_cache = {}  # Cache face data per model
        self.blendshape_config = None  # Will be set by get_blendshape_config()
        self.current_weights = {}  # Add this line to track current weights
        
    async def handle_client(self, websocket):
        """Handle incoming WebSocket connections"""
        print(f"Client connected: {websocket.remote_address}")
        
        try:
            async for message in websocket:
                try:
                    # Parse incoming message
                    data = json.loads(message)
                    
                    if data.get("type") == "get_faces":
                        # Client requesting face data
                        await self.send_faces(websocket, data.get("index_i", 0))
                    
                    elif data.get("type") == "get_blendshape_config":
                        # Client requesting blendshape configuration
                        index_i = data.get("index_i", 0)
                        use_dynamic_mode = data.get("use_dynamic_slider_position_mode", False)
                        
                        # Generate config based on the mode
                        fresh_config = get_blendshape_config(index_i, use_dynamic_mode)
                        await self.send_blendshape_config(websocket, index_i, fresh_config)

                    
                    elif data.get("type") == "update_weights":
                        # Client sending blendshape weights
                        weights = data.get("weights", {})
                        index_i = data.get("index_i", 0)
                        
                        # Track current weights for dynamic positioning
                        self.current_weights = weights.copy()  # Add this line
                        
                        # Get updated vertices
                        vertices = get_vertices(weights, index_i)
                        
                        # Send binary vertex data
                        await self.send_vertices_binary(websocket, vertices)
                        
                except json.JSONDecodeError:
                    print("Invalid JSON received")
                except Exception as e:
                    print(f"Error processing message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            print(f"Client disconnected: {websocket.remote_address}")
        except Exception as e:
            print(f"Connection error: {e}")
    
    async def send_blendshape_config(self, websocket, index_i: int, config=None):
        """Send blendshape configuration to client"""
        try:
            # Use provided config or get fresh config
            if config is None:
                config = get_blendshape_config(index_i)
            
            response = {
                "type": "blendshape_config",
                "index_i": index_i,
                "blendshapes": config
            }
            
            await websocket.send(json.dumps(response))
            # Only log on initial setup to avoid spam
            if self.blendshape_config is None:
                print(f"Sent initial blendshape config for model {index_i}: {len(config)} blendshapes")
            
        except Exception as e:
            print(f"Error sending blendshape config: {e}")
    
    async def send_faces(self, websocket, index_i: int):
        """Send face data to client (called once per model)"""
        try:
            if index_i not in self.faces_cache:
                # Get face data and cache it
                faces = get_face()  # Your function that returns face info
                self.faces_cache[index_i] = faces
            
            faces = self.faces_cache[index_i]
            
            # Send face data as JSON
            response = {
                "type": "faces_data",
                "index_i": index_i,
                "faces": faces.tolist() if isinstance(faces, np.ndarray) else faces
            }
            
            await websocket.send(json.dumps(response))
            print(f"Sent face data for model {index_i}")
            
        except Exception as e:
            print(f"Error sending faces: {e}")
    
    async def send_vertices_binary(self, websocket, vertices: np.ndarray):
        """Send vertex data as binary for efficiency"""
        if len(vertices.shape) != 2 or vertices.shape[1] != 3:
            raise ValueError(f"Vertices must be Nx3 array, got shape {vertices.shape}")

        try:
            # Ensure vertices are float32 for consistent size
            vertices_f32 = vertices.astype(np.float32)
            
            # Create binary message: [header][vertex_data]
            # Header: 4 bytes for number of vertices
            num_vertices = vertices_f32.shape[0]
            header = struct.pack('>I', num_vertices)  # Little-endian unsigned int
            
            # Vertex data: flatten to 1D array
            vertex_data = vertices_f32.flatten().tobytes()
            
            # Combine header and data
            binary_message = header + vertex_data
            
            # Send binary data
            await websocket.send(binary_message)
            
        except Exception as e:
            print(f"Error sending vertices: {e}")
    
    def start_server(self):
        """Start the WebSocket server"""
        print(f"Starting blendshape server on ws://{self.host}:{self.port}")
        
        start_server = websockets.serve(
            self.handle_client, 
            self.host, 
            self.port,
            max_size=10**7,  # 10MB max message size for large vertex data
            ping_interval=20,
            ping_timeout=10
        )
        
        return start_server

# Placeholder functions - replace with your actual implementations


def get_vertices(weights: Dict, index_i: int) -> np.ndarray:
    """
    Compute blendshape operation and return vertex positions
    
    Args:
        weights: Dictionary of blendshape weights {"blendshape_name": value}
        index_i: Model index
    
    Returns:
        np.ndarray: Nx3 array of vertex positions
    """
    global model, surrogate_model, flame_space_path_interp
    # PLACEHOLDER - Replace with your actual blendshape computation
    # This example creates a simple deformed mesh based on weights
    # Example: Create a simple mesh (replace with your actual data)
    weights_array = np.array([weights[name] for name in surrogate_model.names])
    
    deformed_vertices = model.eval(run_controller(flame_space_path_interp, weights_array))
    return deformed_vertices

def get_face():
    global model
    """
    Return face information of the mesh (triangles/indices)
    Called once per model to get topology
    
    Returns:
        Face data (format depends on your needs - could be indices, triangles, etc.)
    """
    # PLACEHOLDER - Replace with your actual face data
    # Example: Return triangle indices for a simple mesh
    
    return model.F

def get_blendshape_config(index_i: int, use_dynamic_slider_position_mode: bool = False):
    global blendshape_config, surrogate_model, model, flame_space_path_interp
    """
    Return blendshape configuration for the specified model
    
    Args:
        index_i: Model index
        use_dynamic_slider_position_mode: If True, compute positions on current mesh state,
                                        if False, use neutral mesh positions
    
    Returns:
        List of dictionaries with keys: "name", "min", "max", "default", "position_of_min", etc.
    """
    
    # Create a copy of the base config to avoid modifying the original
    config = []
    for i, base_config in enumerate(blendshape_config):
        config_copy = base_config.copy()
        config.append(config_copy)
    
    # Compute positions based on the mode
    if use_dynamic_slider_position_mode:
        # Use current mesh state (existing logic)
        V_current = model.V_cache
        base_mesh = V_current
        print("Using dynamic slider positions based on current mesh")
    else:
        # Use neutral mesh
        base_mesh = surrogate_model.V  # This should be your neutral mesh
        print("Using neutral slider positions")
    
    # Recompute slider positions
    blendshapes = surrogate_model.blendshapes
    deformation = np.linalg.norm(blendshapes, axis=2)
    K = 10
    top_k_indices = np.argsort(-deformation, axis=1)[:, :K]
    off_set = surrogate_model.facing_dir * 0.01
    
    for i, name in enumerate(surrogate_model.names):
        vertex_of_interest_i = top_k_indices[i, 0]
        config[i]["position_of_default"] = (base_mesh[vertex_of_interest_i, :] + off_set).tolist()
        config[i]["position_of_max"] = (blendshapes[i, vertex_of_interest_i, :] * config[i]["max"] + base_mesh[vertex_of_interest_i, :] + off_set).tolist()
        config[i]["position_of_min"] = (blendshapes[i, vertex_of_interest_i, :] * config[i]["min"] + base_mesh[vertex_of_interest_i, :] + off_set).tolist()
    
    return config

def run_controller(flame_space_path_interp, weights) -> np.ndarray:
    global flame_zero
    """Runs the VAE/MLP controller once and returns a flat numpy vector."""
    # optimizing the flame parameters to match the surrogate model
    # exp_params_iter, jaw_params_iter = optimize_flame_weights(flame_torch, shape_params, pose_params, v_out_surrogate[0].to(flame_torch.device), steps=200)
    flame_zero_weights = flame_zero.copy()
    for i in range(len(weights)):
        flame_weights_i = flame_space_path_interp[i](weights[i])
        flame_zero_weights += flame_weights_i
    return flame_zero_weights[0]  

def solve_flame_params_direct(flame_model, V_target):
    """
    Solve for expression parameters and jaw parameters using a direct least squares solution.
    This is more efficient than the iterative approach for a purely linear model.
    
    Parameters:
    -----------
    flame_model : FLAME
        An initialized FLAME model instance
    V_target : torch.Tensor
        Target vertex configuration of shape (V, 3)
    
    Returns:
    --------
    exp_params : torch.Tensor
        Optimized expression parameters
    jaw_params : torch.Tensor
        Optimized jaw pose parameters
    """
    device = V_target.device
    
    # Extract expression blendshapes and jaw pose blendshapes
    exp_blendshapes, jaw_pose_blendshapes, mean_shape = get_flame_blendshapes(flame_model)
    
    # Ensure target vertices are properly formatted
    V_target = V_target.reshape(-1, 3)
    
    # Compute delta from mean shape
    delta_V = V_target - mean_shape
    
    # Reshape blendshapes to construct the linear system
    n_vertices = mean_shape.shape[0]
    n_exp = exp_blendshapes.shape[2]
    
    # Reshape exp_blendshapes from [n_vertices, 3, n_exp] to [n_vertices*3, n_exp]
    exp_basis = exp_blendshapes.reshape(-1, n_exp)
    
    # Reshape jaw_pose_blendshapes from [n_vertices, 3, 3] to [n_vertices*3, 3]
    jaw_basis = jaw_pose_blendshapes.reshape(-1, 3)
    
    # Concatenate bases to form the full linear system
    full_basis = torch.cat([exp_basis, jaw_basis], dim=1)
    
    # Reshape delta_V to [n_vertices*3]
    delta_V_flat = delta_V.reshape(-1)
    
    # Solve the least squares problem: min ||full_basis @ params - delta_V_flat||^2
    # Using torch.linalg.lstsq for a more stable solution
    solution, residuals, rank, singular_values = torch.linalg.lstsq(full_basis, delta_V_flat.unsqueeze(1))
    
    # Extract parameters from solution
    exp_params = solution[:n_exp].reshape(1, n_exp)
    jaw_params = solution[n_exp:].reshape(1, 3)
    
    return exp_params, jaw_params

def load_flame_blendshape_model():
    facial_landmark_groups = {    
        "right_eyebrow": list(range(17, 22)),  # 17-21: right eyebrow
        "left_eyebrow": list(range(22, 27)),   # 22-26: left eyebrow
        
        "nose": list(range(27, 36)),       # 31-35: nose tip and nostrils
        
        "right_eye": list(range(36, 42)),      # 36-41: right eye
        "left_eye": list(range(42, 48)),       # 42-47: left eye
        
        "lip+jaw": list(range(48, 68)) + list(range(0, 17)),      # 48-67: outer lip contour
    }

    feature_we_care_about = list(facial_landmark_groups.keys())
    controller_range = [-0.05, 0.05]
    surrogate_model_root_path = "/Users/evanpan/Documents/GitHub/ManifoldExploration/experiments/full_face_bs_test_freeze_landmarks_and_10_ajacent_200_video_aggregated_LM"
    # surrogate_model_root_path = "/Users/evanpan/Documents/GitHub/ManifoldExploration/experiments/full_face_bs_test/"

    face_path = os.path.join(surrogate_model_root_path, "linear_surrogate_Face.npy")
    bs_per_feature = {}
    for feature in feature_we_care_about:
        neutral_path = os.path.join(surrogate_model_root_path, f"linear_surrogate_mean_{feature}.npy")
        blendshape_path = os.path.join(surrogate_model_root_path, f"linear_surrogate_{feature}.npy")
        face_path = os.path.join(surrogate_model_root_path, "linear_surrogate_Face.npy")
        neutral = np.load(neutral_path)
        blendshapes = np.load(blendshape_path)
        F = np.load(face_path)
        bs_per_feature[feature] = {
            "neutral": neutral,
            "blendshapes": blendshapes,
            "F": F
        }
    all_blendshapes = []
    for feature in feature_we_care_about:
        all_blendshapes.append(bs_per_feature[feature]["blendshapes"])
    blendshapes = np.concatenate(all_blendshapes, axis=0)
    neutral = bs_per_feature["right_eyebrow"]["neutral"]
    F = bs_per_feature["right_eyebrow"]["F"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names_id = np.arange(blendshapes.shape[0])
    names = ["blendshape_" + str(i) for i in names_id] 
    surrogate_model = BasicBlendshapes(neutral, F, blendshapes, names=names)
    V0 = surrogate_model.V.copy()
    n_blendshapes  = len(surrogate_model)
    weights        = np.zeros(n_blendshapes, dtype=float)
    flame = FLAMEBlendshapes()
    flame_weights = np.zeros([1, 103])
    flame_torch = flame.flame.to(device)

    # obtain a path in blendshape space
    geometry_space_path = []
    weight_range = np.linspace(controller_range[0], controller_range[1], 100)
    for bs_i in range(0, n_blendshapes):
        path_for_bs_i = []
        weight_range = weight_range
        weights = np.zeros(n_blendshapes, dtype=float)
        for w in weight_range:
            weights[bs_i] = w
            V = surrogate_model.eval(weights)
            path_for_bs_i.append(V)
        geometry_space_path.append(np.array(path_for_bs_i))
    geometry_space_path = np.array(geometry_space_path)

    # animator = MeshAnimator(geometry_space_path[0], F)
    # animator.run()



    exp_params, jaw_params = solve_flame_params_direct(flame.flame, torch.from_numpy(V0))
    flame_zero = np.zeros([1, 103])
    flame_zero[0, :100] = exp_params.cpu().numpy()
    flame_zero[0, 100:103] = jaw_params.cpu().numpy()

    # obtain the path in flame space
    flame_space_path = []
    geometry_space_of_solve_flame_path = []
    for bs_i in range(0, n_blendshapes):
        flame_path_for_bs_i = []
        geometry_space_of_solve_flame_path_for_bs_i = []
        for f_i in range(0, geometry_space_path.shape[1]):
            V_target = geometry_space_path[bs_i, f_i]
            V_target = torch.tensor(V_target, device=device)
            exp_params, jaw_params = solve_flame_params_direct(flame.flame, V_target)
            flame_weights[0, :100] = exp_params.cpu().numpy()
            flame_weights[0, 100:103] = jaw_params.cpu().numpy()
            V_flame = flame.eval(flame_weights[0])
            flame_path_for_bs_i.append(flame_weights.copy() - flame_zero)
            geometry_space_of_solve_flame_path_for_bs_i.append(V_target.cpu().numpy())
        flame_space_path.append(np.array(flame_path_for_bs_i))
        geometry_space_of_solve_flame_path.append(geometry_space_of_solve_flame_path_for_bs_i)

    flame_space_path = np.array(flame_space_path)
    geometry_space_of_solve_flame_path = np.array(geometry_space_of_solve_flame_path)

    # create interp for flame_space_path
    flame_space_path_interp = []
    for i in range(n_blendshapes):
        flame_space_path_interp.append(
            interp1d(weight_range, flame_space_path[i, :, 0], axis=0, fill_value="extrapolate", bounds_error=False)
        )


    # V0  = surrogate_model.eval(weights)
    # V_FLAME = flame.eval(flame_weights[0])

    return flame_space_path_interp, flame, surrogate_model, flame_zero

async def main():
    server = BlendshapeServer()
    # Start the server
    start_server = server.start_server()
    await start_server
    print("Server started, waiting for clients...")
    await asyncio.Future()  # Keep the server running indefinitely

# Server startup
if __name__ == "__main__":
    
    # load the model 
    flame_space_path_interp, model, surrogate_model, flame_zero = load_flame_blendshape_model()

    # get the configuration for the blendshapes
    blendshape_config = []
    for i, name in enumerate(surrogate_model.names):
        weight_dicts_i = {}
        weight_dicts_i["name"] = name
        weight_dicts_i["min"] = -0.05
        weight_dicts_i["max"] = 0.05
        weight_dicts_i["default"] = 0.0
        blendshape_config.append(weight_dicts_i)


    # compute the UI positions of the sliders
    # compute compute position of the sliders

    zero_weights = np.zeros(len(surrogate_model.names), dtype=float)
    V=model.eval(run_controller(flame_space_path_interp, zero_weights))
    # Run the event loop
    asyncio.run(main())
    loop = asyncio.get_event_loop()