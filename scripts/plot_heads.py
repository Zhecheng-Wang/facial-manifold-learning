import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import math

def plot_multiple_faces(blendshapes, weights_list, grid_size=None, spacing=1.5, colors=None):
    """
    Plot multiple faces in a grid layout based on a list of blendweights.
    
    Parameters:
    -----------
    blendshapes : object
        The blendshape object with an eval method that takes weights and returns vertices.
    weights_list : list or ndarray
        List of weight vectors, where each vector contains blendshape coefficients.
    grid_size : tuple, optional
        (rows, cols) for the grid layout. If None, a square-ish grid is calculated.
    spacing : float, optional
        Distance between faces in the grid.
    colors : list, optional
        List of colors for each face. If None, a default color is used for all faces.
        Each color should be [r, g, b] with values between 0 and 1.
    
    Returns:
    --------
    dict
        Dictionary of registered mesh handles, keyed by their index in weights_list.
    """
    # Initialize Polyscope if not already done
    ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_view_projection_mode("orthographic")
    ps.set_front_dir("z_front")
    ps.set_background_color([0, 0, 0])

    # Determine grid dimensions if not provided
    num_faces = len(weights_list)
    if grid_size is None:
        cols = math.ceil(math.sqrt(num_faces))
        rows = math.ceil(num_faces / cols)
        grid_size = (rows, cols)
    else:
        rows, cols = grid_size
    
    # Use default color if not provided
    if colors is None:
        colors = [[0.9, 0.9, 0.9]] * num_faces
    elif len(colors) < num_faces:
        colors.extend([[0.9, 0.9, 0.9]] * (num_faces - len(colors)))
    
    # Calculate size of a single face (approximate bounding box)
    base_vertices = blendshapes.eval(np.zeros(len(blendshapes.names)))
    min_coords = np.min(base_vertices, axis=0)
    max_coords = np.max(base_vertices, axis=0)
    size = max_coords - min_coords
    
    # Compute translation offsets for grid layout
    mesh_handles = {}
    
    for i, weights in enumerate(weights_list):
        # Calculate grid position
        row = i // cols
        col = i % cols
        
        # Position offset
        offset = np.array([
            col * (size[0] * spacing),
            -row * (size[1] * spacing),
            0.0
        ])
        
        # Generate mesh for this face with the current weights
        vertices = blendshapes.eval(weights)
        
        # Apply offset
        translated_vertices = vertices + offset
        
        # Register the mesh with Polyscope
        face_name = f"face_{i}"
        mesh = ps.register_surface_mesh(
            face_name, 
            translated_vertices, 
            blendshapes.F,
            color=colors[i], 
            smooth_shade=True,
            edge_width=0.25, 
            material="normal"
        )
        
        mesh_handles[i] = mesh
    
    # Define simple GUI callback to show face indices
    def faces_gui():
        psim.Text(f"Displaying {num_faces} faces in a {rows}x{cols} grid")
        
        # Add sliders or other controls if needed
        
    ps.set_user_callback(faces_gui)
    
    # Set camera to view the entire grid
    center_x = (cols - 1) * size[0] * spacing / 2
    center_y = -(rows - 1) * size[1] * spacing / 2
    ps.look_at([center_x, center_y, 10.0], [center_x, center_y, 0.0])
    
    return mesh_handles


# Example usage:
if __name__ == "__main__":
    # Example data (would be replaced with your actual data)
    import os
    from utils import load_blendshape, SPDataset
    
    PROJ_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
    )
    
    # Load blendshapes
    blendshapes = load_blendshape(model="SP")
    
    # Get sample weights from dataset
    dataset = SPDataset()
    sample_weights = dataset.data.numpy()[:9]  # First 9 frames
    
    # Set up some example colors for variation
    colors = [
        [0.9, 0.7, 0.7],  # Light red
        [0.7, 0.9, 0.7],  # Light green
        [0.7, 0.7, 0.9],  # Light blue
        [0.9, 0.9, 0.7],  # Light yellow
        [0.9, 0.7, 0.9],  # Light magenta
        [0.7, 0.9, 0.9],  # Light cyan
        [0.8, 0.8, 0.8],  # Light gray
        [0.9, 0.8, 0.7],  # Light orange
        [0.8, 0.7, 0.9],  # Light purple
    ]
    
    # Plot multiple faces
    mesh_handles = plot_multiple_faces(blendshapes, sample_weights, colors=colors)
    
    # Show the viewer
    ps.show()