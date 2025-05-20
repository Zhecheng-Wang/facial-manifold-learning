import os
import glob
import pickle
from typing import List

import numpy as np
import torch
from numpy.typing import NDArray
from scipy import linalg

import polyscope as ps
import polyscope.imgui as psim

from model import load_model
from utils import load_blendshape, MeadRavdessDataset, load_config, inspect_attrs


class PCAAnimationViewer:
    def __init__(self, mask_region: str = "lips", pca_components: int = 3):
        # Project root
        self.proj_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir)
        )
        # Load model and blendshapes
        config = load_config(os.path.join(self.proj_root, "experiments", "rinat_small"))
        model = load_model(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(self.device).eval()
        self.blendshapes = load_blendshape(model="FLAME")
        self.n_bs = len(self.blendshapes)
        inspect_attrs(self.blendshapes)

        # Load mask indices
        mask_path = os.path.join(
            self.proj_root, "data", "flame_model", "FLAME_masks", "flame_masks.pkl"
        )
        with open(mask_path, "rb") as f:
            masks = pickle.load(f, encoding="latin1")
        self.mask_indices = np.array(masks[mask_region], dtype=int)
        self.M = len(self.mask_indices)

        # Discover animations
        ravdess_dir = os.path.join(self.proj_root, "data", "mead_ravdess")
        pkl_file = glob.glob(os.path.join(ravdess_dir, "*.pickle"))[0]
        with open(pkl_file, "rb") as f:
            self.full_ravdess = pickle.load(f)
        self.available_animations: List[str] = list(self.full_ravdess.keys())

        # PCA settings
        self.k = pca_components

        # State
        self.selected_idx = 0
        self.current_frame = 0
        self.n_frames = 0
        self.frame_weights: NDArray[np.float64] = np.empty((0, self.n_bs))
        self.mask_data: NDArray[np.float64] = np.empty((0, 3 * self.M))
        self.mean_vec: NDArray[np.float64] = np.zeros(3 * self.M)
        self.modes: NDArray[np.float64] = np.zeros((self.k, 3 * self.M))
        self.proj_weights: NDArray[np.float64] = np.empty((0, self.k))

        # Polyscope meshes and template
        self.mesh_orig = None
        self.mesh_masked = None
        self.mesh_pca = None
        self.V0 = None  # base template

    def load_animation(self, anim_id: str) -> NDArray[np.float64]:
        ds = MeadRavdessDataset(anim_id)
        T = len(ds)
        w = np.zeros((T, self.n_bs), dtype=float)
        for t in range(T):
            exp = ds[t]["exp"]
            jaw = ds[t]["jaw"]
            w[t] = np.concatenate([exp, np.zeros(jaw.shape)], axis=0)
        return w

    def compute_mask_data(self):
        arr = np.zeros((self.n_frames, 3 * self.M), dtype=float)
        for t in range(self.n_frames):
            V = self.blendshapes.eval(self.frame_weights[t]).cpu().numpy()
            arr[t] = V[self.mask_indices].reshape(-1)
        self.mask_data = arr

    def compute_pca(self):
        arr = self.mask_data
        mean = arr.mean(axis=0)
        U, S, Vt = linalg.svd(arr - mean, full_matrices=False)
        comps = Vt[: self.k]
        self.mean_vec = mean
        self.modes = comps
        self.proj_weights = (arr - mean) @ comps.T

    def init_polyscope(self):
        ps.set_verbosity(0)
        ps.init()
        ps.set_ground_plane_mode("none")
        ps.set_view_projection_mode("orthographic")
        ps.set_front_dir("z_front")
        ps.set_background_color([0, 0, 0])
        # Base template
        self.V0 = self.blendshapes.eval(np.zeros(self.n_bs)).cpu().numpy()
        # Three views
        self.mesh_orig = ps.register_surface_mesh(
            "Original", self.V0, self.blendshapes.F.tolist(), smooth_shade=True
        )
        self.mesh_orig.translate([-0.2, 0, 0])
        self.mesh_masked = ps.register_surface_mesh(
            "MaskedOnly", self.V0, self.blendshapes.F.tolist(), smooth_shade=True
        )
        self.mesh_masked.translate([0.0, 0, 0])
        self.mesh_pca = ps.register_surface_mesh(
            "PCARecon", self.V0, self.blendshapes.F.tolist(), smooth_shade=True
        )
        self.mesh_pca.translate([0.2, 0, 0])

    def update_meshes(self):
        w = self.frame_weights[self.current_frame]
        V = self.blendshapes.eval(w).cpu().numpy()

        # Original
        self.mesh_orig.update_vertex_positions(V)

        # Masked only: freeze outside mask
        V_mask = V.copy()
        outside = np.ones(len(V), dtype=bool)
        outside[self.mask_indices] = False
        V_mask[outside] = self.V0[outside]
        self.mesh_masked.update_vertex_positions(V_mask)

        # PCA reconstruct
        recon = self.mean_vec + self.proj_weights[self.current_frame] @ self.modes
        V_pca = V.copy()
        V_pca[self.mask_indices] = recon.reshape(self.M, 3)
        self.mesh_pca.update_vertex_positions(V_pca)

    def gui(self):
        # Animation dropdown
        changed, idx = psim.Combo(
            "Animation", self.selected_idx, self.available_animations
        )
        if changed and idx != self.selected_idx:
            self.selected_idx = idx
            self.frame_weights = self.load_animation(self.available_animations[idx])
            self.n_frames = len(self.frame_weights)
            self.compute_mask_data()
            self.compute_pca()
            self.current_frame = 0
            self.update_meshes()
        psim.Separator()
        # Frame slider
        psim.Text(f"Frame: {self.current_frame+1}/{self.n_frames}")
        ch, f = psim.SliderInt("Frame", self.current_frame, 0, self.n_frames - 1)
        if ch:
            self.current_frame = f
            self.update_meshes()

    def run(self):
        # initial load
        self.frame_weights = self.load_animation(
            self.available_animations[self.selected_idx]
        )
        self.n_frames = len(self.frame_weights)
        self.compute_mask_data()
        self.compute_pca()
        self.init_polyscope()
        self.update_meshes()
        ps.set_user_callback(lambda: self.gui())
        ps.show()


if __name__ == "__main__":
    viewer = PCAAnimationViewer()
    viewer.run()
