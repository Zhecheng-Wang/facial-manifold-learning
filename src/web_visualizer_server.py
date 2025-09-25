import asyncio
import websockets
import json
import numpy as np
import struct
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ObjectType(Enum):
    MESH = "mesh"
    POINT_CLOUD = "point_cloud"

@dataclass
class MeshObject:
    id: str
    vertices: np.ndarray  # Nx3
    faces: np.ndarray     # Mx3 (triangle indices)
    colors: Optional[np.ndarray] = None  # Nx3 RGB colors per vertex
    visible: bool = True
    wireframe: bool = False
    opacity: float = 1.0
    
    def __post_init__(self):
        # Validate shapes
        if len(self.vertices.shape) != 2 or self.vertices.shape[1] != 3:
            raise ValueError(f"Vertices must be Nx3, got {self.vertices.shape}")
        if len(self.faces.shape) != 2 or self.faces.shape[1] != 3:
            raise ValueError(f"Faces must be Mx3, got {self.faces.shape}")
        if self.colors is not None:
            if len(self.colors.shape) != 2 or self.colors.shape[1] != 3:
                raise ValueError(f"Colors must be Nx3, got {self.colors.shape}")
            if self.colors.shape[0] != self.vertices.shape[0]:
                raise ValueError("Colors must have same number of rows as vertices")

@dataclass
class PointCloudObject:
    id: str
    points: np.ndarray    # Nx3
    colors: Optional[np.ndarray] = None  # Nx3 RGB colors per point
    sizes: Optional[np.ndarray] = None   # N point sizes (radii)
    radius: float = 0.01  # Default radius for all points if sizes not provided
    visible: bool = True
    
    def __post_init__(self):
        # Validate shapes
        if len(self.points.shape) != 2 or self.points.shape[1] != 3:
            raise ValueError(f"Points must be Nx3, got {self.points.shape}")
        if self.colors is not None:
            if len(self.colors.shape) != 2 or self.colors.shape[1] != 3:
                raise ValueError(f"Colors must be Nx3, got {self.colors.shape}")
            if self.colors.shape[0] != self.points.shape[0]:
                raise ValueError("Colors must have same number of rows as points")
        if self.sizes is not None:
            if len(self.sizes.shape) != 1 or self.sizes.shape[0] != self.points.shape[0]:
                raise ValueError("Sizes must be N-dimensional matching point count")

class MeshVisualizer:
    def __init__(self, host="localhost", port=8766):
        self.host = host
        self.port = port
        self.meshes: Dict[str, MeshObject] = {}
        self.point_clouds: Dict[str, PointCloudObject] = {}
        self.clients = set()
        
    def add_mesh(self, mesh_id: str, vertices: np.ndarray, faces: np.ndarray, 
                 colors: Optional[np.ndarray] = None, **kwargs) -> None:
        """Add or update a mesh object"""
        try:
            mesh = MeshObject(
                id=mesh_id,
                vertices=vertices.astype(np.float32),
                faces=faces.astype(np.uint32),
                colors=colors.astype(np.float32) if colors is not None else None,
                **kwargs
            )
            self.meshes[mesh_id] = mesh
            logger.info(f"Added mesh '{mesh_id}' with {len(vertices)} vertices, {len(faces)} faces")
            
            # Notify all connected clients
            asyncio.create_task(self._broadcast_mesh_update(mesh_id))
            
        except Exception as e:
            logger.error(f"Error adding mesh '{mesh_id}': {e}")
            raise
    
    def add_point_cloud(self, pc_id: str, points: np.ndarray, 
                       colors: Optional[np.ndarray] = None, 
                       sizes: Optional[np.ndarray] = None,
                       radius: float = 0.01, **kwargs) -> None:
        """Add or update a point cloud object"""
        try:
            point_cloud = PointCloudObject(
                id=pc_id,
                points=points.astype(np.float32),
                colors=colors.astype(np.float32) if colors is not None else None,
                sizes=sizes.astype(np.float32) if sizes is not None else None,
                radius=radius,
                **kwargs
            )
            self.point_clouds[pc_id] = point_cloud
            logger.info(f"Added point cloud '{pc_id}' with {len(points)} points (radius: {radius})")
            
            # Notify all connected clients
            asyncio.create_task(self._broadcast_point_cloud_update(pc_id))
            
        except Exception as e:
            logger.error(f"Error adding point cloud '{pc_id}': {e}")
            raise
    
    def remove_object(self, object_id: str) -> None:
        """Remove a mesh or point cloud"""
        removed = False
        if object_id in self.meshes:
            del self.meshes[object_id]
            removed = True
        if object_id in self.point_clouds:
            del self.point_clouds[object_id]
            removed = True
        
        if removed:
            logger.info(f"Removed object '{object_id}'")
            asyncio.create_task(self._broadcast_object_removal(object_id))
        else:
            logger.warning(f"Object '{object_id}' not found")
    
    def set_object_visibility(self, object_id: str, visible: bool) -> None:
        """Set visibility of an object"""
        updated = False
        if object_id in self.meshes:
            self.meshes[object_id].visible = visible
            updated = True
        if object_id in self.point_clouds:
            self.point_clouds[object_id].visible = visible
            updated = True
        
        if updated:
            asyncio.create_task(self._broadcast_visibility_update(object_id, visible))
    
    def clear_all(self) -> None:
        """Clear all objects"""
        self.meshes.clear()
        self.point_clouds.clear()
        asyncio.create_task(self._broadcast_clear_all())
        logger.info("Cleared all objects")
    
    def get_scene_info(self) -> Dict:
        """Get information about current scene"""
        return {
            "meshes": {mid: {
                "id": mid,
                "vertex_count": len(mesh.vertices),
                "face_count": len(mesh.faces),
                "has_colors": mesh.colors is not None,
                "visible": mesh.visible,
                "wireframe": mesh.wireframe,
                "opacity": mesh.opacity
            } for mid, mesh in self.meshes.items()},
            "point_clouds": {pcid: {
                "id": pcid,
                "point_count": len(pc.points),
                "has_colors": pc.colors is not None,
                "has_sizes": pc.sizes is not None,
                "radius": pc.radius,
                "visible": pc.visible
            } for pcid, pc in self.point_clouds.items()}
        }
    
    async def _broadcast_mesh_update(self, mesh_id: str):
        """Broadcast mesh update to all clients"""
        if not self.clients or mesh_id not in self.meshes:
            return
        
        mesh = self.meshes[mesh_id]
        message = {
            "type": "mesh_update",
            "object_id": mesh_id,
            "data": await self._serialize_mesh(mesh)
        }
        
        await self._broadcast_message(json.dumps(message))
    
    async def _broadcast_point_cloud_update(self, pc_id: str):
        """Broadcast point cloud update to all clients"""
        if not self.clients or pc_id not in self.point_clouds:
            return
        
        pc = self.point_clouds[pc_id]
        message = {
            "type": "point_cloud_update", 
            "object_id": pc_id,
            "data": await self._serialize_point_cloud(pc)
        }
        
        await self._broadcast_message(json.dumps(message))
    
    async def _broadcast_object_removal(self, object_id: str):
        """Broadcast object removal to all clients"""
        message = {
            "type": "object_removal",
            "object_id": object_id
        }
        await self._broadcast_message(json.dumps(message))
    
    async def _broadcast_visibility_update(self, object_id: str, visible: bool):
        """Broadcast visibility update to all clients"""
        message = {
            "type": "visibility_update",
            "object_id": object_id,
            "visible": visible
        }
        await self._broadcast_message(json.dumps(message))
    
    async def _broadcast_clear_all(self):
        """Broadcast clear all command to all clients"""
        message = {"type": "clear_all"}
        await self._broadcast_message(json.dumps(message))
    
    async def _broadcast_message(self, message: str):
        """Broadcast message to all connected clients"""
        if not self.clients:
            return
        
        disconnected = set()
        for client in self.clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.add(client)
        
        # Remove disconnected clients
        self.clients -= disconnected
    
    async def _serialize_mesh(self, mesh: MeshObject) -> Dict:
        """Serialize mesh data for transmission"""
        return {
            "vertices": mesh.vertices.tolist(),
            "faces": mesh.faces.tolist(),
            "colors": mesh.colors.tolist() if mesh.colors is not None else None,
            "visible": mesh.visible,
            "wireframe": mesh.wireframe,
            "opacity": mesh.opacity
        }
    
    async def _serialize_point_cloud(self, pc: PointCloudObject) -> Dict:
        """Serialize point cloud data for transmission"""
        return {
            "points": pc.points.tolist(),
            "colors": pc.colors.tolist() if pc.colors is not None else None,
            "sizes": pc.sizes.tolist() if pc.sizes is not None else None,
            "radius": pc.radius,
            "visible": pc.visible
        }
    
    async def handle_client(self, websocket):
        """Handle incoming WebSocket connections"""
        self.clients.add(websocket)
        logger.info(f"Client connected: {websocket.remote_address}")
        
        try:
            # Send current scene state to new client
            await self._send_full_scene(websocket)
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_client_message(websocket, data)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON received from client")
                except Exception as e:
                    logger.error(f"Error processing client message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {websocket.remote_address}")
        except Exception as e:
            logger.error(f"Client connection error: {e}")
        finally:
            self.clients.discard(websocket)
    
    async def _send_full_scene(self, websocket):
        """Send the complete current scene to a client"""
        try:
            # Send all meshes
            for mesh_id, mesh in self.meshes.items():
                message = {
                    "type": "mesh_update",
                    "object_id": mesh_id,
                    "data": await self._serialize_mesh(mesh)
                }
                await websocket.send(json.dumps(message))
            
            # Send all point clouds
            for pc_id, pc in self.point_clouds.items():
                message = {
                    "type": "point_cloud_update",
                    "object_id": pc_id,
                    "data": await self._serialize_point_cloud(pc)
                }
                await websocket.send(json.dumps(message))
                
        except Exception as e:
            logger.error(f"Error sending full scene: {e}")
    
    async def _handle_client_message(self, websocket, data: Dict):
        """Handle messages from clients"""
        message_type = data.get("type")
        
        if message_type == "get_scene_info":
            # Send scene information
            scene_info = self.get_scene_info()
            response = {
                "type": "scene_info",
                "data": scene_info
            }
            await websocket.send(json.dumps(response))
            
        elif message_type == "set_visibility":
            # Handle visibility change request
            object_id = data.get("object_id")
            visible = data.get("visible", True)
            if object_id:
                self.set_object_visibility(object_id, visible)
                
        elif message_type == "remove_object":
            # Handle object removal request
            object_id = data.get("object_id")
            if object_id:
                self.remove_object(object_id)
                
        elif message_type == "clear_all":
            # Handle clear all request
            self.clear_all()
        
        else:
            logger.warning(f"Unknown message type: {message_type}")
    
    def start_server(self):
        """Start the WebSocket server"""
        logger.info(f"Starting mesh visualizer server on ws://{self.host}:{self.port}")
        
        start_server = websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            max_size=50*1024*1024,  # 50MB max message size
            ping_interval=20,
            ping_timeout=10
        )
        
        return start_server

# Convenience functions for common mesh generation
def create_test_cube(size=1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Create a simple cube mesh for testing"""
    s = size / 2
    vertices = np.array([
        [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],  # bottom
        [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s]       # top
    ], dtype=np.float32)
    
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 7, 6], [4, 6, 5],  # top
        [0, 4, 5], [0, 5, 1],  # front
        [2, 6, 7], [2, 7, 3],  # back
        [0, 3, 7], [0, 7, 4],  # left
        [1, 5, 6], [1, 6, 2]   # right
    ], dtype=np.uint32)
    
    return vertices, faces

def create_test_sphere(radius=1.0, resolution=20) -> Tuple[np.ndarray, np.ndarray]:
    """Create a sphere mesh for testing"""
    phi = np.linspace(0, np.pi, resolution)
    theta = np.linspace(0, 2*np.pi, resolution)
    
    vertices = []
    for p in phi:
        for t in theta:
            x = radius * np.sin(p) * np.cos(t)
            y = radius * np.sin(p) * np.sin(t)
            z = radius * np.cos(p)
            vertices.append([x, y, z])
    
    vertices = np.array(vertices, dtype=np.float32)
    
    # Generate faces
    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            # Current quad vertices
            v1 = i * resolution + j
            v2 = i * resolution + (j + 1)
            v3 = (i + 1) * resolution + j
            v4 = (i + 1) * resolution + (j + 1)
            
            # Two triangles per quad
            faces.append([v1, v2, v3])
            faces.append([v2, v4, v3])
    
    faces = np.array(faces, dtype=np.uint32)
    return vertices, faces

# Jupyter-friendly server startup
async def start_visualizer_server(visualizer, add_test_data=True):
    """Start the visualizer server - Jupyter friendly"""
    if add_test_data:
        # Add some test objects
        cube_verts, cube_faces = create_test_cube(size=2.0)
        cube_colors = np.random.rand(len(cube_verts), 3).astype(np.float32)
        visualizer.add_mesh("test_cube", cube_verts, cube_faces, colors=cube_colors)
        
        # Add a sphere
        sphere_verts, sphere_faces = create_test_sphere(radius=1.5, resolution=15)
        sphere_verts[:, 0] += 3  # Offset to the right
        visualizer.add_mesh("test_sphere", sphere_verts, sphere_faces, wireframe=True)
        
        # Add a point cloud
        points = np.random.randn(1000, 3).astype(np.float32) * 2
        point_colors = np.random.rand(1000, 3).astype(np.float32)
        visualizer.add_point_cloud("random_points", points, colors=point_colors)
    
    # Start the server
    start_server = visualizer.start_server()
    await start_server
    logger.info("Server started, waiting for clients...")
    
    # Keep the server running
    try:
        await asyncio.Future()  # Run forevers
    except KeyboardInterrupt:
        logger.info("Server stopped")

def run_visualizer(host="localhost", port=8766, add_test_data=True):
    """
    Convenience function to run the visualizer server.
    Works in both Jupyter and regular Python environments.
    """
    visualizer = MeshVisualizer(host=host, port=port)
    
    try:
        # Try to get the current event loop (Jupyter case)
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're in Jupyter, create a task
            import threading
            import nest_asyncio
            
            # Install nest_asyncio to allow nested event loops
            try:
                nest_asyncio.apply()
                task = asyncio.create_task(start_visualizer_server(visualizer, add_test_data))
                logger.info(f"Server starting on ws://{host}:{port}")
                logger.info("Use visualizer.add_mesh() and visualizer.add_point_cloud() to add objects")
                return visualizer, task
            except ImportError:
                logger.error("Please install nest_asyncio: pip install nest_asyncio")
                return None, None
        else:
            # Regular Python environment
            asyncio.run(start_visualizer_server(visualizer, add_test_data))
    except RuntimeError:
        # No event loop running, start one
        asyncio.run(start_visualizer_server(visualizer, add_test_data))
    
    return visualizer, None

# Example usage and server startup
async def main():
    visualizer = MeshVisualizer()
    await start_visualizer_server(visualizer, add_test_data=True)

if __name__ == "__main__":
    run_visualizer()