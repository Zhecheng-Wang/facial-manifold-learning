from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from numpy.typing import NDArray

from blendshapes import FLAMEBlendshapes
from data.loader import (
    AnimationLoader,
    apply_mask,
    discover_animations,
)
from data.utils import inspect_attrs, load_config
from model import load_model
from optim.mesh_diff import optimize_masked_mesh_weights
from pca.utils import compute_pca
from ui.polyscope_setup import init_view

# ——— Type aliases ———
NDArrayfp = NDArray[np.float32]
IntArray = NDArray[np.intp]
Mesh = Any
MeshTriple = tuple[Mesh, Mesh, Mesh]


class PCAAnimationViewer:
    proj_root: Path
    n_bs: int
    mask_vids: IntArray
    anim_list: list[str]
    loader: AnimationLoader
    k: int

    blendshapes: FLAMEBlendshapes

    anim_idx: int
    current_frame: int
    n_frames: int

    frame_w_orig: NDArrayfp
    mask_data: NDArrayfp
    pca_mean: NDArrayfp
    pca_modes: NDArrayfp
    pca_weights: NDArrayfp
    frame_w_pred: NDArrayfp

    meshes: list[list[Mesh]]
    error_meshes: list[list[NDArrayfp]]
    mask_meshes: NDArrayfp
    v0: NDArrayfp

    def __init__(self, mask_region: str = "lips", pca_components: int = 5) -> None:
        # project root
        self.proj_root = Path(__file__).resolve().parent.parent

        # load FLAME model & blendshapes
        cfg = load_config(self.proj_root / "experiments" / "rinat_small")
        _model = load_model(cfg)
        self.blendshapes = FLAMEBlendshapes()
        inspect_attrs(self.blendshapes)
        self.n_bs = len(self.blendshapes)

        # load mask indices
        mask_file = (
            self.proj_root / "data" / "flame_model" / "FLAME_masks" / "flame_masks.pkl"
        )
        with open(mask_file, "rb") as f:
            masks: dict[str, list[int]] = pickle.load(f, encoding="latin1")
        self.mask_vids = np.array(masks[mask_region], dtype=int)

        # animation loader & list
        self.loader = AnimationLoader(self.n_bs)
        self.anim_list = discover_animations(self.proj_root)

        # PCA settings
        self.k = pca_components

        # runtime state
        self.anim_idx = 0
        self.current_frame = 0
        self.n_frames = 0

        # placeholders for data
        self.frame_w_orig = np.empty((0, self.n_bs), dtype=np.float32)
        self.frame_w_pred = np.empty((0, self.n_bs), dtype=np.float32)

        self.mask_data = np.empty((0,), dtype=np.float32)

        self.pca_mean = np.empty((0,), dtype=np.float32)
        self.pca_modes = np.empty((0,), dtype=np.float32)
        self.pca_weights = np.empty((0,), dtype=np.float32)

        # Polyscope meshes: Orig, Masked, PCA
        self.meshes = []
        self.mask_meshes = np.empty((0,), dtype=np.float32)
        self.error_meshes = []
        zero_w = np.zeros(self.n_bs, dtype=np.float32)
        self.v0 = self.blendshapes.eval(zero_w).cpu().numpy().astype(np.float32)

    def setup(self) -> None:
        # Polyscope expects lists here, so convert; stub doesn’t know NDArray
        self.meshes = init_view(self.v0, self.blendshapes.F.tolist())
        self._update_data()
        self._update_meshes()

    def _update_data(self) -> None:
        # Original weights
        self.frame_w_orig = self.loader.load(self.anim_list[self.anim_idx])
        self.n_frames = self.frame_w_orig.shape[0]

        # Get masked meshes for cluster
        self.mask_meshes = apply_mask(
            self.frame_w_orig, self.blendshapes, self.v0, self.mask_vids
        )

        # Optimize between masked mesh and original to get w
        self.frame_w_pred = optimize_masked_mesh_weights(
            self.blendshapes, self.frame_w_orig, self.mask_meshes
        )

        pred_local_meshes = np.array(
            [self.blendshapes.eval(z).flatten() for z in self.frame_w_pred]
        )

        self.pca_mean, self.pca_modes, self.pca_weights = compute_pca(
            pred_local_meshes, self.k
        )

        self._calculate_error()

    def _calculate_error(self) -> None:
        anim_shape = [self.n_frames, self.v0.shape[0]]
        grid_width = len(self.meshes) - 1
        self.error_meshes = [
            [np.zeros(anim_shape, dtype=np.float32) for _ in range(grid_width)]
            for _ in range(grid_width)
        ]

        for f in range(self.n_frames):
            V_pred = self.blendshapes.eval(self.frame_w_pred[f]).cpu().numpy()
            V_orig = self.blendshapes.eval(self.frame_w_orig[f]).cpu().numpy()
            V_mask = self.mask_meshes[f]
            recon = (self.pca_mean + self.pca_weights[f] @ self.pca_modes).reshape(
                -1, 3
            )

            frame_meshes = [V_pred, V_orig, V_mask, recon]
            for r in range(len(frame_meshes)):
                for c in range(len(frame_meshes)):
                    r_mesh = frame_meshes[r]
                    c_mesh = frame_meshes[c]
                    err = np.linalg.norm(r_mesh - c_mesh, axis=1)
                    self.error_meshes[r][c][f] = err

    def _update_meshes(self) -> None:
        """1x4 faces
            Still   Orig    Masked  PCA
        Pred
        """

        # Optimized on Masked
        pred_w_mask = self.frame_w_pred[self.current_frame]
        pred = self.blendshapes.eval(pred_w_mask)
        self.meshes[0][1].update_vertex_positions(pred)
        self.meshes[1][0].update_vertex_positions(pred)

        # fetch current frame full vertices
        W = self.frame_w_orig[self.current_frame, :]
        V_full = self.blendshapes.eval(W).cpu().numpy()

        # original mesh
        self.meshes[0][2].update_vertex_positions(V_full)
        self.meshes[2][0].update_vertex_positions(V_full)

        # masked region mesh
        mask_mesh = self.mask_meshes[self.current_frame]
        self.meshes[0][3].update_vertex_positions(mask_mesh)
        self.meshes[3][0].update_vertex_positions(mask_mesh)

        # PCA reconstruction mesh
        recon = self.pca_mean + self.pca_weights[self.current_frame, :] @ self.pca_modes
        V_pca = recon.reshape(-1, 3)
        # V_pca[self.mask_vids] = recon.reshape(-1, 3)
        self.meshes[0][4].update_vertex_positions(V_pca)
        self.meshes[4][0].update_vertex_positions(V_pca)

        for r in range(len(self.error_meshes)):
            for c in range(len(self.error_meshes)):
                err_frame = self.error_meshes[r][c][self.current_frame]
                # self.meshes[r][c].add_col
                colors_vert = np.zeros(3)
                colors_vert += (r * len(self.error_meshes) + c) / len(
                    self.error_meshes
                ) ** 2
                self.meshes[r + 1][c + 1].set_color(colors_vert)
                self.meshes[r + 1][c + 1].add_scalar_quantity(
                    "norm error",
                    err_frame,
                    defined_on="vertices",
                    cmap="viridis",
                    enabled=True,
                )

    def gui(self) -> None:
        # animation selector
        changed, new_idx = psim.Combo("Animation", self.anim_idx, self.anim_list)
        if changed:
            self.anim_idx = new_idx
            self._update_data()
            self._update_meshes()

        psim.Separator()

        # frame slider
        f_changed, new_frame = psim.SliderInt(
            "Frame", self.current_frame, 0, self.n_frames - 1
        )
        if f_changed:
            self.current_frame = new_frame
            self._update_meshes()

    def run(self) -> None:
        self.setup()
        ps.set_user_callback(self.gui)
        ps.show()


if __name__ == "__main__":
    PCAAnimationViewer(mask_region="lips", pca_components=3).run()
