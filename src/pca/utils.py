from numpy.typing import NDArray
import numpy as np
from numpy.linalg import svd


def compute_pca(arr: NDArray[np.float64], k: int) -> tuple[
    NDArray[np.float64],  # mean vector (D,)
    NDArray[np.float64],  # top-k principal directions (k×D)
    NDArray[np.float64],  # projections of each row onto those directions (T×k)
]:
    """Return (mean, top-k modes, projections) for PCA on arr (T×D)."""
    # use numpy’s mean (has type stubs)
    mean: NDArray[np.float64] = np.mean(arr, axis=0)
    X: NDArray[np.float64] = arr - mean

    # numpy.linalg.svd is fully typed (u, s, vt)
    _, _, vt = svd(X, full_matrices=False)

    modes: NDArray[np.float64] = vt[:k]  # (k, D)
    projections: NDArray[np.float64] = X @ modes.T  # (T, k)

    return mean, modes, projections
