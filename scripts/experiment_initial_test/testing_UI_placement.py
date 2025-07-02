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

def display_a_single_mesh(V, F):
    ps.remove_all_structures()
    ps.set_verbosity(0)
    ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_view_projection_mode("orthographic")
    ps.set_front_dir("z_front")
    ps.set_background_color([0, 0, 0])
    ps.register_surface_mesh(
        "mesh", V, F,
        color=[0.9, 0.9, 0.9],
        edge_width=0.25, material="normal"
        )
    ps.show()

class DraggablePointOnPolyline:
    def __init__(self, name, slider_dict, initial_t=0.0):
        """
        Create a draggable point that moves along a polyline path.
        
        Args:
            name (str): Name for the UI element and point cloud
            slider_dict (dict): Dictionary with keys:
                - 'slider_pose_neutral': np.array of shape (3) - the neutral position
                - 'slider_pose_zero_to_one': np.array of shape (num_points, 3) - path from neutral to +1 (inclusive)
                - 'slider_pose_neg_one_to_zero': np.array of shape (num_points, 3) - path from -1 to neutral (inclusive)
            initial_t (float): Initial parameter value [-1, 1] along the path
        """
        self.name = name
        self.slider_dict = slider_dict
        self.current_t = initial_t
        self.is_dragging = False
        self.last_mouse_pos = None
        
        # Extract and validate input
        self.neutral_pos = slider_dict['slider_pose_neutral']
        self.zero_to_one_path = slider_dict['slider_pose_zero_to_one']
        self.neg_one_to_zero_path = slider_dict['slider_pose_neg_one_to_zero']
        
        # Validate that paths start/end at neutral position
        assert np.allclose(self.zero_to_one_path[0], self.neutral_pos), "zero_to_one path must start at neutral position"
        assert np.allclose(self.neg_one_to_zero_path[-1], self.neutral_pos), "neg_one_to_zero path must end at neutral position"
        
        # Create complete polyline by combining paths
        # neg_one_to_zero (excluding last point to avoid duplication) + zero_to_one
        self.complete_path = np.vstack([
            self.neg_one_to_zero_path[:-1],  # Exclude last point (neutral)
            self.zero_to_one_path           # Include all points (neutral to one)
        ])
        
        # Create parameter mapping
        self.neg_path_length = len(self.neg_one_to_zero_path) - 1  # Number of segments in negative path
        self.pos_path_length = len(self.zero_to_one_path) - 1      # Number of segments in positive path
        self.total_segments = self.neg_path_length + self.pos_path_length
        
        # Precompute cumulative distances for more accurate parameter mapping
        self._compute_arc_length_parameterization()
        
        # Create edges for polyline visualization
        edges = []
        for i in range(len(self.complete_path) - 1):
            edges.append([i, i + 1])
        
        # Register the polyline path
        self.polyline = ps.register_curve_network(
            f"{name}_path", 
            self.complete_path, 
            np.array(edges)
        )
        self.polyline.set_color([0.7, 0.7, 0.7])
        self.polyline.set_radius(0.01)
        
        # Create initial draggable point
        self.current_pos = self._parameter_to_position(self.current_t)
        self.point_cloud = ps.register_point_cloud(f"{name}_point", np.array([self.current_pos]))
        self.point_cloud.set_color([1.0, 0.2, 0.2])
        self.point_cloud.set_radius(0.01)
        
    def _compute_arc_length_parameterization(self):
        """Compute cumulative arc lengths for more accurate parameter mapping."""
        # Compute segment lengths
        segment_lengths = []
        for i in range(len(self.complete_path) - 1):
            length = np.linalg.norm(self.complete_path[i + 1] - self.complete_path[i])
            segment_lengths.append(length)
        
        # Cumulative lengths
        self.cumulative_lengths = np.cumsum([0] + segment_lengths)
        self.total_arc_length = self.cumulative_lengths[-1]
        
        # Find arc length up to neutral position (end of negative path)
        self.neutral_arc_length = self.cumulative_lengths[self.neg_path_length]
        
    def _parameter_to_position(self, t):
        """Convert parameter t in [-1, 1] to 3D position on polyline using arc length."""
        t = np.clip(t, -1.0, 1.0)
        
        if t <= 0:
            # Map t from [-1, 0] to arc length [0, neutral_arc_length]
            target_arc_length = (t + 1.0) * 0.5 * self.neutral_arc_length
        else:
            # Map t from [0, 1] to arc length [neutral_arc_length, total_arc_length]
            remaining_length = self.total_arc_length - self.neutral_arc_length
            target_arc_length = self.neutral_arc_length + t * remaining_length
        
        # Find the segment containing this arc length
        segment_idx = np.searchsorted(self.cumulative_lengths[1:], target_arc_length)
        segment_idx = min(segment_idx, len(self.complete_path) - 2)
        
        # Interpolate within the segment
        start_arc = self.cumulative_lengths[segment_idx]
        end_arc = self.cumulative_lengths[segment_idx + 1]
        
        if end_arc > start_arc:
            local_t = (target_arc_length - start_arc) / (end_arc - start_arc)
        else:
            local_t = 0.0
            
        local_t = np.clip(local_t, 0.0, 1.0)
        
        # Linear interpolation between segment endpoints
        start_pos = self.complete_path[segment_idx]
        end_pos = self.complete_path[segment_idx + 1]
        
        return start_pos + local_t * (end_pos - start_pos)
    
    def _position_to_parameter(self, pos):
        """Convert 3D position to parameter t in [-1, 1] by finding closest point on polyline."""
        pos = np.array(pos)
        min_dist_sq = float('inf')
        best_t = 0.0
        
        # Check each segment of the complete path
        for i in range(len(self.complete_path) - 1):
            seg_start = self.complete_path[i]
            seg_end = self.complete_path[i + 1]
            seg_vec = seg_end - seg_start
            seg_len_sq = np.dot(seg_vec, seg_vec)
            
            if seg_len_sq > 1e-10:
                # Project point onto line segment
                local_t = np.clip(np.dot(pos - seg_start, seg_vec) / seg_len_sq, 0, 1)
                closest_point = seg_start + local_t * seg_vec
                dist_sq = np.sum((pos - closest_point) ** 2)
                
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    
                    # Convert segment index and local_t to global parameter
                    start_arc = self.cumulative_lengths[i]
                    end_arc = self.cumulative_lengths[i + 1]
                    point_arc_length = start_arc + local_t * (end_arc - start_arc)
                    
                    # Convert arc length to parameter t
                    if point_arc_length <= self.neutral_arc_length:
                        # In negative section
                        if self.neutral_arc_length > 0:
                            best_t = -1.0 + 2.0 * (point_arc_length / self.neutral_arc_length)
                        else:
                            best_t = 0.0
                    else:
                        # In positive section
                        remaining_length = self.total_arc_length - self.neutral_arc_length
                        if remaining_length > 0:
                            best_t = (point_arc_length - self.neutral_arc_length) / remaining_length
                        else:
                            best_t = 0.0
        
        return np.clip(best_t, -1.0, 1.0)
    
    def _screen_to_world_space(self, screen_coord, depth=-3.0):
        """
        Convert screen coordinates to world space coordinates.
        Based on the corrected implementation provided.
        
        Args:
            screen_coord: [x, y] screen coordinates
            depth: Depth in camera space (negative for in front of camera)
        
        Returns:
            3D world space coordinates
        """
        width, height = ps.get_window_size()
        
        # Map from screen space to normalized device coordinates (NDC)
        ndc_x = screen_coord[0] / width
        ndc_y = 1 - screen_coord[1] / height
        
        # From NDC to camera space
        camera_params = ps.get_view_camera_parameters()
        aspect_ratio = camera_params.get_aspect()
        fov_vert = camera_params.get_fov_vertical_deg()
        
        h_image_space = 2 * np.tan(np.radians(fov_vert) / 2)
        w_image_space = h_image_space * aspect_ratio
        
        # Image space coordinates
        x_prime = ndc_x * w_image_space - w_image_space / 2
        y_prime = ndc_y * h_image_space - h_image_space / 2
        
        # Convert to camera space
        px = x_prime * depth
        py = y_prime * depth
        pz = depth
        point_camera_space = np.array([px, py, pz])
        
        # Convert to world space
        M_world_to_camera = camera_params.get_view_mat()
        M_camera_to_world = np.linalg.inv(M_world_to_camera)
        position_world = M_camera_to_world @ np.append(point_camera_space, 1)
        position_world = position_world[:3]  # discard the homogeneous coordinate
        
        return position_world
    
    def _world_to_screen_approx(self, world_pos):
        """
        Convert world coordinates to screen coordinates.
        Inverse of the screen_to_world_space function.
        
        Args:
            world_pos: 3D world coordinates
            
        Returns:
            [x, y] screen coordinates or None if behind camera
        """
        width, height = ps.get_window_size()
        
        # Convert to homogeneous coordinates
        world_pos_homo = np.append(world_pos, 1.0)
        
        # Transform to camera space
        camera_params = ps.get_view_camera_parameters()
        M_world_to_camera = camera_params.get_view_mat()
        camera_pos = M_world_to_camera @ world_pos_homo
        
        # Check if point is behind camera
        if camera_pos[2] >= 0:
            return None
        
        # Convert to image space
        aspect_ratio = camera_params.get_aspect()
        fov_vert = camera_params.get_fov_vertical_deg()
        
        h_image_space = 2 * np.tan(np.radians(fov_vert) / 2)
        w_image_space = h_image_space * aspect_ratio
        
        # From camera space to image space
        x_prime = camera_pos[0] / camera_pos[2]
        y_prime = camera_pos[1] / camera_pos[2]
        
        # From image space to NDC
        ndc_x = (x_prime + w_image_space / 2) / w_image_space
        ndc_y = (y_prime + h_image_space / 2) / h_image_space
        
        # From NDC to screen coordinates
        screen_x = ndc_x * width
        screen_y = (1 - ndc_y) * height
        
        return np.array([screen_x, screen_y])
    
    def _project_ray_to_polyline_plane(self, ray_origin, ray_dir):
        """Project mouse ray onto the best-fit plane of the polyline."""
        # Compute polyline plane normal using PCA for better fitting
        centered_points = self.complete_path - np.mean(self.complete_path, axis=0)
        _, _, vh = np.linalg.svd(centered_points)
        normal = vh[-1]  # Last component is the normal to the best-fit plane
        
        # Intersect ray with plane
        plane_point = np.mean(self.complete_path, axis=0)  # Centroid as plane reference
        denom = np.dot(ray_dir, normal)
        
        if abs(denom) > 1e-10:
            t = np.dot(plane_point - ray_origin, normal) / denom
            intersection = ray_origin + t * ray_dir
            return intersection
        else:
            # Ray parallel to plane, return current position
            return self.current_pos
    
    def _update_callback(self):
        """Main update callback for handling mouse interaction."""
        io = psim.GetIO()
        
        # Only allow dragging when A key is held
        a_held = psim.IsKeyDown(psim.ImGuiKey_A)
        if a_held:
            ps.set_navigation_style("none")
            mouse_x, mouse_y = psim.GetMousePos()
            
            # Check if mouse is over the draggable point
            if io.MouseClicked[0]:
                print("clicked")
                current_screen_pos = self._world_to_screen_approx(self.current_pos)
                
                if current_screen_pos is not None:
                    mouse_screen = np.array([mouse_x, mouse_y])
                    print(f"Mouse screen pos: {mouse_screen}, Current screen pos: {current_screen_pos}")
                    if np.linalg.norm(mouse_screen - current_screen_pos) < 15:
                        self.is_dragging = True
                        self.last_mouse_pos = np.array([mouse_x, mouse_y])
                        io.WantCaptureMouse = True
            
            # Handle dragging
            if self.is_dragging and self.last_mouse_pos is not None:
                psim.Text(f"Dragging {self.name} at ({mouse_x:.2f}, {mouse_y:.2f})")
                
                # Convert mouse position to world space
                # Use the depth of the current point to maintain consistent depth
                camera_params = ps.get_view_camera_parameters()
                M_world_to_camera = camera_params.get_view_mat()
                current_camera_pos = M_world_to_camera @ np.append(self.current_pos, 1)
                current_depth = current_camera_pos[2]
                
                # Get world position at current depth
                world_pos = self._screen_to_world_space([mouse_x, mouse_y], current_depth)
                
                # Find closest point on polyline and update parameter
                new_t = self._position_to_parameter(world_pos)
                self.set_parameter(new_t)
                
                self.last_mouse_pos = np.array([mouse_x, mouse_y])

            if io.MouseReleased[0]:
                self.is_dragging = False
                io.WantCaptureMouse = False
        else:
            ps.set_navigation_style("turntable")
    
    def set_parameter(self, t):
        """Set the parameter value and update point position."""
        self.current_t = np.clip(t, -1.0, 1.0)
        self.current_pos = self._parameter_to_position(self.current_t)
        
        # Update point cloud
        self.point_cloud.update_point_positions(np.array([self.current_pos]))
    
    def get_parameter(self):
        """Get current parameter value."""
        return self.current_t
    
    def get_position(self):
        """Get current 3D position."""
        return self.current_pos.copy()
    
    def show_ui(self):
        """Show ImGui controls for the slider."""
        psim.Text(f"{self.name} Controls:")
        
        changed, new_t = psim.SliderFloat(f"Parameter##{self.name}", self.current_t, -1.0, 1.0)
        if changed:
            self.set_parameter(new_t)
        
        psim.Text(f"Position: ({self.current_pos[0]:.3f}, {self.current_pos[1]:.3f}, {self.current_pos[2]:.3f})")
        psim.Text("Hold 'A' and click/drag the red point to move it along the path")
        
        # Show path information
        psim.Text(f"Path segments: {self.total_segments} (neg: {self.neg_path_length}, pos: {self.pos_path_length})")
        psim.Text(f"Total arc length: {self.total_arc_length:.3f}")
# draggable_slider = DraggablePointOnPolyline("my_slider", UI_element_positions[0], initial_t=0.0)
# draggable_slider._world_to_screen_approx(draggable_slider.current_pos)

# print(draggable_slider.view_width)
# draggable_slider._get_actual_viewport_size()

weight_interval_count = 3
mask_path = "/Users/evanpan/Documents/GitHub/ManifoldExploration/data/flame_model/FLAME_masks/FLAME_masks.pkl"
with open(mask_path, "rb") as f:
    mask = pickle.load(f, encoding="latin1")
# load blendshapes computed from linear surrogate model
controller_range = [-0.03, 0.03]
surrogate_model_root_path = "/Users/evanpan/Documents/GitHub/ManifoldExploration/experiments/full_face_bs_test/"
feature_we_care_about = ["eye_region", "lips", "nose", "forehead"]
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
neutral = bs_per_feature["lips"]["neutral"]
F = bs_per_feature["lips"]["F"]

# compute some kind of slider position + path for each of the blendshapes
displacement_of_vertices = np.linalg.norm(blendshapes, axis=2)[:, mask["face"]]  # shape (n_blendshapes, n_vertices)
# find the top 30 vertices in term of displacement
top_indices = np.argsort(displacement_of_vertices)[:, 0]  # indices
UI_element_positions = []
for i in range(blendshapes.shape[0]):
    top_vertex_zero = neutral[mask["face"]][top_indices[i], :]
    top_vertex_one = neutral[mask["face"]][top_indices[i], :] + blendshapes[:, mask["face"]][i, top_indices[i], :] * controller_range[1]
    top_vertex_neg_one = neutral[mask["face"]][top_indices[i], :] + blendshapes[:, mask["face"]][i, top_indices[i], :] * controller_range[0]
    UI_element_positions.append({"slider_pose_neutral": top_vertex_zero,
                                 "slider_pose_zero_to_one": np.array([top_vertex_zero, top_vertex_one]),
                                 "slider_pose_neg_one_to_zero": np.array([top_vertex_neg_one, top_vertex_zero])})
    


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
weight_range = np.linspace(controller_range[0], controller_range[1], weight_interval_count)
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

def run_controller(flame_space_path_interp, weights) -> np.ndarray:
    """Runs the VAE/MLP controller once and returns a flat numpy vector."""
    # optimizing the flame parameters to match the surrogate model
    # exp_params_iter, jaw_params_iter = optimize_flame_weights(flame_torch, shape_params, pose_params, v_out_surrogate[0].to(flame_torch.device), steps=200)
    flame_zero_weights = flame_zero.copy()
    for i in range(len(weights)):
        flame_weights_i = flame_space_path_interp[i](weights[i])
        flame_zero_weights += flame_weights_i
    return flame_zero_weights

def gui():
    global weights, selection_threshold, flame_zero, current_frame, last_slider_index, flame_weights, flame_space_path_interp

    # ------------------------------------------------ Reset
    if psim.Button("Reset to Canonical"):
        weights[:]           = 0.0
        current_frame        = 0
        last_slider_index    = 0
        SM0.update_vertex_positions(surrogate_model.eval(weights))
        flame_weights = run_controller(flame_space_path_interp, weights)
        SM_FLAME.update_vertex_positions(flame.eval(flame_weights[0]))

    # ------------------------------------------------ Alpha cutoff


    # ------------------------------------------------ Blendshape sliders
    for i, name in enumerate(surrogate_model.names):
        changed_bs, new_val = psim.SliderFloat(name, float(weights[i]), controller_range[0], controller_range[1])
        if changed_bs:
            last_slider_index = i
            weights[i]        = new_val                 # keep user edit
            SM0.update_vertex_positions(surrogate_model.eval(weights))
            flame_weights = run_controller(flame_space_path_interp, weights)
            SM_FLAME.update_vertex_positions(flame.eval(flame_weights[0]))
        # print(weights)

    # the draggable slider
    draggable_slider.show_ui()
    
    # You can get the current values like this:
    current_t = draggable_slider.get_parameter()
    current_pos = draggable_slider.get_position()
    
    # psim.text(f"Current t: {current_t:.3f}")
    print(current_t)


flame = FLAMEBlendshapes()
flame_weights = np.zeros([1, 103])

ps.remove_all_structures()
ps.init()
ps.set_verbosity(0)
ps.set_ground_plane_mode("none")
ps.set_view_projection_mode("orthographic")
ps.set_front_dir("z_front")
ps.set_background_color([0, 0, 0])

surrogate_model.translation = np.array([0.2, 0, 0])
V0  = surrogate_model.eval(weights)
SM0 = ps.register_surface_mesh(
    "face", V0, surrogate_model.F,
    color=[0.4, 0.4, 0.4], smooth_shade=False,
    edge_width=0.25, material="normal"
)

flame_torch = flame.flame
V_FLAME = flame.eval(flame_zero[0])
SM_FLAME = ps.register_surface_mesh(
    "face_flame", V_FLAME, flame.F,
    color=[0.9, 0.9, 0.9], smooth_shade=True,
    edge_width=0.25, material="normal"
)
draggable_slider = DraggablePointOnPolyline("my_slider", UI_element_positions[0], initial_t=0.0)
# ---------------------------------------------------------------------
# ps.set_user_callback(gui)
ps.set_user_callback(draggable_slider._update_callback)


ps.show()

