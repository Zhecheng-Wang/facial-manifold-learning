import numpy as np
import polyscope as ps
import polyscope.imgui as psim

class MeshAnimator:
    def __init__(self, vertices, faces):
        """
        Initialize the mesh animator.
        
        Args:
            vertices (np.ndarray): Vertex array of shape (frames, vertices, 3)
            faces (np.ndarray): Face connectivity array of shape (F, 3)
        """
        self.vertices = vertices
        self.faces = faces
        self.n_frames = vertices.shape[0]
        self.current_frame = 0
        self.playing = False
        self.frame_rate = 30.0  # FPS
        self.last_time = 0.0
        
        # Initialize polyscope
        ps.set_verbosity(0)
        ps.init()
        ps.set_ground_plane_mode("none")
        ps.set_view_projection_mode("orthographic")
        ps.set_front_dir("z_front")
        ps.set_background_color([0.1, 0.1, 0.1])
        
        # Register the mesh with the first frame
        self.mesh = ps.register_surface_mesh(
            "animated_mesh",
            self.vertices[0],
            self.faces,
            color=[0.8, 0.8, 0.9],
            smooth_shade=True,
            edge_width=0.25,
            material="normal"
        )
        
        # Set up camera
        
    def update_frame(self):
        """Update the mesh to show the current frame."""
        if self.n_frames > 0:
            self.mesh.update_vertex_positions(self.vertices[self.current_frame])
    def gui_callback(self):
        """GUI callback function for controls."""
        import time
        
        # Play/Pause button
        if self.playing:
            if psim.Button("Pause"):
                self.playing = False
        else:
            if psim.Button("Play"):
                self.playing = True
        
        psim.SameLine()
        if psim.Button("Reset"):
            self.current_frame = 0
            self.playing = False
            self.update_frame()
        
        # Frame slider
        psim.Text(f"Frame: {self.current_frame + 1}/{self.n_frames}")
        changed_frame, new_frame = psim.SliderInt(
            "Frame", self.current_frame, 0, self.n_frames - 1
        )
        if changed_frame:
            self.current_frame = new_frame
            self.update_frame()
        
        # Frame rate slider
        changed_rate, new_rate = psim.SliderFloat(
            "FPS", self.frame_rate, 1.0, 120.0
        )
        if changed_rate:
            self.frame_rate = new_rate
        
        # Auto-advance frames when playing
        if self.playing:
            current_time = time.time()
            if current_time - self.last_time > 1.0 / self.frame_rate:
                self.current_frame = (self.current_frame + 1) % self.n_frames
                self.update_frame()
                self.last_time = current_time
        
        psim.Separator()
        
        # Info
        psim.Text(f"Total Frames: {self.n_frames}")
        psim.Text(f"Vertices: {self.vertices.shape[1]}")
        psim.Text(f"Faces: {self.faces.shape[0]}")
    
    def run(self):
        """Start the visualization."""
        ps.set_user_callback(self.gui_callback)
        ps.show()

class MultiMeshAnimator:
    def __init__(self, mesh_data, offset_distance=3.0):
        """
        Initialize the multi-mesh animator.
        
        Args:
            mesh_data (list): List of tuples (vertices, faces) where:
                - vertices (np.ndarray): Vertex array of shape (frames, vertices, 3)
                - faces (np.ndarray): Face connectivity array of shape (F, 3)
            offset_distance (float): Distance between meshes
        """
        self.mesh_data = mesh_data
        self.offset_distance = offset_distance
        self.n_meshes = len(mesh_data)
        self.n_frames = mesh_data[0][0].shape[0] if mesh_data else 0
        self.current_frame = 0
        self.playing = False
        self.frame_rate = 30.0  # FPS
        self.last_time = 0.0
        self.colors = []
        
        # Generate colors for different meshes
        for i in range(self.n_meshes):
            hue = i / self.n_meshes
            color = self._hsv_to_rgb(hue, 0.7, 0.9)
            self.colors.append(color)
        
        # Initialize polyscope
        ps.set_verbosity(0)
        ps.init()
        ps.set_ground_plane_mode("none")
        ps.set_view_projection_mode("orthographic")
        ps.set_front_dir("z_front")
        ps.set_background_color([0.1, 0.1, 0.1])
        
        # Register meshes
        self.meshes = []
        for i, (vertices, faces) in enumerate(mesh_data):
            # Apply offset to position meshes side by side
            offset = np.array([i * self.offset_distance, 0, 0])
            offset_vertices = vertices[0] + offset
            
            mesh = ps.register_surface_mesh(
                f"mesh_{i}",
                offset_vertices,
                faces,
                color=self.colors[i],
                smooth_shade=True,
                edge_width=0.25,
                material="normal"
            )
            self.meshes.append((mesh, offset))
        
        # Set up camera to show all meshes
        self._setup_camera()
    
    def _hsv_to_rgb(self, h, s, v):
        """Convert HSV to RGB color."""
        import colorsys
        return colorsys.hsv_to_rgb(h, s, v)
    
    def _setup_camera(self):
        """Set up camera to view all meshes."""
        # Calculate bounding box for all meshes
        all_vertices = []
        for (vertices, faces), offset in zip(self.mesh_data, [np.array([i * self.offset_distance, 0, 0]) for i in range(self.n_meshes)]):
            all_vertices.append(vertices[0] + offset)
        
        if all_vertices:
            all_vertices = np.concatenate(all_vertices, axis=0)
            center = np.mean(all_vertices, axis=0)
            extent = np.max(np.abs(all_vertices - center)) * 1.5
            
            ps.reset_camera_to_home_view()
            ps.look_at(center, center + np.array([0, -1, 0]))
    
    def update_frame(self):
        """Update all meshes to show the current frame."""
        for i, ((mesh, offset), (vertices, faces)) in enumerate(zip(self.meshes, self.mesh_data)):
            if self.current_frame < vertices.shape[0]:
                mesh.update_vertex_positions(vertices[self.current_frame] + offset)
    
    def gui_callback(self):
        """GUI callback function for controls."""
        import time
        
        # Play/Pause button
        if self.playing:
            if psim.Button("Pause"):
                self.playing = False
        else:
            if psim.Button("Play"):
                self.playing = True
        
        psim.SameLine()
        if psim.Button("Reset"):
            self.current_frame = 0
            self.playing = False
            self.update_frame()
        
        # Frame slider
        psim.Text(f"Frame: {self.current_frame + 1}/{self.n_frames}")
        changed_frame, new_frame = psim.SliderInt(
            "Frame", self.current_frame, 0, self.n_frames - 1
        )
        if changed_frame:
            self.current_frame = new_frame
            self.update_frame()
        
        # Frame rate slider
        changed_rate, new_rate = psim.SliderFloat(
            "FPS", self.frame_rate, 1.0, 120.0
        )
        if changed_rate:
            self.frame_rate = new_rate
        
        # Offset distance slider
        changed_offset, new_offset = psim.SliderFloat(
            "Offset Distance", self.offset_distance, 0.5, 10.0
        )
        if changed_offset:
            self.offset_distance = new_offset
            # Update mesh positions
            for i, (mesh, _) in enumerate(self.meshes):
                offset = np.array([i * self.offset_distance, 0, 0])
                self.meshes[i] = (mesh, offset)
            self.update_frame()
            self._setup_camera()
        
        # Auto-advance frames when playing
        if self.playing:
            current_time = time.time()
            if current_time - self.last_time > 1.0 / self.frame_rate:
                self.current_frame = (self.current_frame + 1) % self.n_frames
                self.update_frame()
                self.last_time = current_time
        
        psim.Separator()
        
        # Info
        psim.Text(f"Total Meshes: {self.n_meshes}")
        psim.Text(f"Total Frames: {self.n_frames}")
        if self.mesh_data:
            psim.Text(f"Vertices per mesh: {[v[0].shape[1] for v, f in self.mesh_data]}")
            psim.Text(f"Faces per mesh: {[f.shape[0] for v, f in self.mesh_data]}")
    
    def run(self):
        """Start the visualization."""
        ps.set_user_callback(self.gui_callback)
        ps.show()

# Example usage
