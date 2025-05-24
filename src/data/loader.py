from __future__ import annotations

import re
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from data.utils import MeadRavdessDataset


def discover_animations(proj_root: Path) -> list[str]:
    """
    Find the first MEAD-Ravdess pickle under proj_root/data/mead_ravdess
    and return its animation keys.
    """
    ravdess_dir = proj_root / "data" / "mead_ravdess"
    pkl_files = list(ravdess_dir.glob("*.pickle"))
    if not pkl_files:
        raise FileNotFoundError(f"No MEAD-Ravdess pickle in {ravdess_dir}")

    # Load and cast to a dict of str→Any
    raw = pickle.loads(pkl_files[0].read_bytes())
    full_data: dict[str, Any] = raw

    def get_filtered_keys(full_data: dict[str, any]) -> list[str]:
        pattern = re.compile(r"^[\d-]+$")  # filter on digits and -
        return list(filter(pattern.fullmatch, full_data.keys()))

    return get_filtered_keys(full_data)


class AnimationLoader:
    """Load per-frame FLAME [exp, jaw] weights from a MEAD-Ravdess animation."""

    def __init__(self, n_bs: int) -> None:
        self.n_bs = n_bs

    def load(self, anim_id: str) -> NDArray[np.float32]:
        """
        Parameters
        ----------
        anim_id : str
            Key into the MEAD-Ravdess dataset.

        Returns
        -------
        weights : (T, n_bs) float32 array of FLAME expression+jaw coefficients.
        """
        ds = MeadRavdessDataset(anim_id)
        T = len(ds)
        weights: NDArray[np.float32] = np.zeros((T, self.n_bs), dtype=np.float32)
        for t in range(T):
            exp: NDArray[np.float32] = ds[t]["exp"]
            jaw: NDArray[np.float32] = ds[t]["jaw"]
            # stack exp + zeros for jaw region
            weights[t] = np.concatenate([exp, np.zeros_like(jaw)], axis=0)
        return weights


def apply_mask(
    ws: NDArray[np.float32],
    blendshapes: Any,
    base: NDArray[np.float32],
    mask_vids: NDArray[np.intp],
) -> NDArray[np.float32]:
    """
    Applies mask_indices on the blendshape weights to get frozen faces except
    for vertices in mask_indices.
    """
    T = ws.shape[0]
    V, dim = base.shape
    arr: NDArray[np.float32] = np.repeat(base[np.newaxis, ...], T, axis=0)
    for i in range(T):
        orig = blendshapes.eval(ws[i]).cpu().numpy()
        arr[i][mask_vids] = orig[mask_vids]
    return arr


def compute_mask_data(
    weights: NDArray[np.float32],
    blendshapes: Any,
    mask_indices: NDArray[np.intp],
) -> NDArray[np.float32]:
    """
    Extract only the masked-region vertices from each FLAME mesh and flatten.

    Parameters
    ----------
    weights : (T, n_bs) FLAME parameters per frame
    blendshapes : FLAME model with .eval(weights) → (V,3) array
    mask_indices : (M,) integer array of vertex indices to keep

    Returns
    -------
    arr : (T, 3*M)float32 array of masked XYZ coordinates per frame
    """
    T = weights.shape[0]
    M = mask_indices.shape[0]
    D = 3 * M

    arr: NDArray[np.float32] = np.zeros((T, D), dtype=np.float32)
    for t in range(T):
        # eval returns a torch.Tensor; .cpu().numpy() → np.ndarray
        Vt: NDArray[np.float32] = blendshapes.eval(weights[t]).cpu().numpy()
        arr[t] = Vt[mask_indices].reshape(-1)
    return arr
