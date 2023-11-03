import numpy as np
from blendshapes import BasicBlendshapeModel, load_blendshape_model

def cluster_blendshapes(model:BasicBlendshapeModel, cluster_threshold=0.5, activate_threshold=0.1):
    """cluster blendshapes based on IoU between regions of influence
    Args:
        model (BasicBlendshapeModel): blendshape model
        cluster_threshold (float, optional): IoU threshold for clustering. Defaults to 0.5.
        activate_threshold (float, optional): threshold for activating a vertex. Defaults to 0.1.
    """

    # compute region of influence for each blendshape
    regions = np.zeros((model.blendshapes.shape[0], model.V.shape[0]), dtype=bool)
    for i in range(model.blendshapes.shape[0]):
        blendshape_norm = np.linalg.norm(model.blendshapes[i], axis=1)
        blendshape_max_norm = np.max(blendshape_norm)
        regions[i] = (blendshape_norm / blendshape_max_norm) > activate_threshold

    # compute IoU between blendshapes
    IoU = np.zeros((regions.shape[0], regions.shape[0]))
    for i in range(regions.shape[0]):
        for j in range(i+1, regions.shape[0]):
            IoU[i, j] = np.sum(regions[i] & regions[j]) / np.sum(regions[i] | regions[j])
            IoU[j, i] = IoU[i, j]
    
    # cluster blendshapes, sort by IoU
    clusters = []
    visited = np.zeros(regions.shape[0], dtype=bool)
    for i in range(regions.shape[0]):
        if visited[i]:
            continue
        visited[i] = True
        cluster = [i]
        for j in range(i+1, regions.shape[0]):
            if visited[j]:
                continue
            if IoU[i, j] >= cluster_threshold:
                cluster.append(j)
                visited[j] = True
        clusters.append(cluster)

    return clusters

if __name__ == "__main__":
    # load the blendshape model
    import os
    PROJ_PATH = os.path.dirname(os.path.abspath(__file__))
    BLENDSHAPES_PATH = os.path.join(os.pardir, "data", "Apple blendshapes51 OBJs", "OBJs")
    model = load_blendshape_model(BLENDSHAPES_PATH)
    # compute clusters
    clusters = cluster_blendshapes(model, cluster_threshold=0.3)
    print(f"Found {len(clusters)} clusters.")

    # visualize clusters
    import polyscope as ps
    ps.init()
    ps.set_program_name("Clustering Visualization")
    ps.set_verbosity(0)
    ps.set_SSAA_factor(3)
    ps.set_max_fps(60)
    ps.set_ground_plane_mode("none")
    ps.set_view_projection_mode("orthographic")
    vis_clusters = []
    # compute the bounding box of the model
    bounding_box = (np.max(model.V, axis=0) - np.min(model.V, axis=0))*1.2

    for cluster_id, cluster in enumerate(clusters):
        N = len(cluster)
        NV = model.V.shape[0]
        NF = model.F.shape[0]
        V = np.zeros((N*NV, 3))
        F = np.zeros((N*NF, 3), dtype=np.int32)
        D = np.zeros(N*NV)
        for i, blendshape in enumerate(cluster):
            D[i*NV:(i+1)*NV] = np.linalg.norm(model.blendshapes[blendshape], axis=1)
            V[i*NV:(i+1)*NV] = model.V + model.blendshapes[blendshape]
            V[i*NV:(i+1)*NV,0] += i * bounding_box[0]
            F[i*NF:(i+1)*NF] = model.F + i*NV
        V[:,1] += cluster_id * bounding_box[1]
        vis_cluster = ps.register_surface_mesh(f"cluster {cluster_id}", V, F, smooth_shade=True)
        vis_cluster.add_scalar_quantity("displacement", D, enabled=True, cmap="turbo")
        vis_clusters.append(vis_cluster)
    ps.show()
