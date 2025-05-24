from sys import dont_write_bytecode
import numpy as np
import polyscope as ps
from numpy.typing import NDArray
from typing import Any

Mesh = Any


def init_view(V0: NDArray[np.float64], faces: list[float]) -> list[list[Mesh]]:
    """
    Initialize Polyscope with three side-by-side surface meshes.

    Parameters
    ----------
    V0 : list of (x, y, z)
        Base vertex positions.
    faces : list of (i, j, k)
        Triangle connectivity.
    translations : list of (x, y, z)
        Offsets for the three view placements.

    Returns
    -------
    meshes : list of SurfaceMesh
        [mesh_orig, mesh_masked, mesh_pca]
    """
    ps.set_verbosity(0)
    ps.init()

    ps.set_ground_plane_mode("none")
    ps.set_view_projection_mode("orthographic")
    ps.set_background_color([0, 0, 0])
    # intr = ps.CameraIntrinsics(fov_vertical_deg=40.0, aspect=2.0)
    # extr = ps.CameraExtrinsics(
    #     root=(0.0, 0.0, 5.0), look_dir=(0.0, 0.0, 0.0), up_dir=(0.0, 1.0, 0.0)
    # )
    # new_params = ps.CameraParameters(intr, extr)
    # ps.set_view_camera_parameters(new_params)
    # ps_cam = ps.register_camera_view("cam1", new_params)

    drow = 0.4
    dcol = 0.25

    names = ["Predicted", "Original", "MaskedOnly", "PCARecon"]
    width = len(names) + 1

    meshes = [[None] * width for _ in range(width)]

    def trans(row: int, col: int):
        return np.array([dcol * col, drow * row, 0.0])

    # rows, cols
    for i in range(1, width):
        color_mesh = np.random.rand(3)
        name = names[i - 1]
        mesh = ps.register_surface_mesh(
            name + "_row", V0, faces, smooth_shade=True, color=color_mesh
        )
        mesh.translate(trans(i, 0))
        meshes[i][0] = mesh

        mesh = ps.register_surface_mesh(
            name + "_col", V0, faces, smooth_shade=True, color=color_mesh
        )
        mesh.translate(trans(0, i))
        meshes[0][i] = mesh

    for r in range(len(names)):
        for c in range(len(names)):
            r_name = names[r]
            c_name = names[c]
            name = r_name + "x" + c_name

            mesh = ps.register_surface_mesh(name, V0, faces, smooth_shade=True)
            mesh.translate(trans(r + 1, c + 1))
            meshes[r + 1][c + 1] = mesh

    return meshes
