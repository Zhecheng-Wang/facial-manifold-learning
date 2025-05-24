import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import Callable
import torch

from solver import GradientSolverSciPy, LeastSquaresSolver
import numpy.typing as npt

NDArrayfp = npt.NDArray[np.float32]


def masked_pred_fn_factory(blendshapes):
    def fn(z: torch.Tensor) -> torch.Tensor:
        return blendshapes.eval(z)

    return fn


def solve_one(blendshapes, mask_v: NDArrayfp):
    z_shape = len(blendshapes)
    # solver = GradientSolverSciPy(z_shape)
    solver = LeastSquaresSolver(z_shape)
    m_fn = masked_pred_fn_factory(blendshapes)
    return solver.solve(m_pred_fn=m_fn, target=mask_v, verbose=True)


def optimize_masked_mesh_weights(
    blendshapes,
    frame_ws: NDArrayfp,
    mask_meshes: NDArrayfp,
) -> NDArrayfp:
    n_frames = frame_ws.shape[0]
    results = Parallel(n_jobs=-1)(
        delayed(solve_one)(blendshapes, mask_meshes[i])
        for i in tqdm(range(n_frames), desc="Solving masked pred frames")
    )

    return np.stack(results, axis=0)
