from collections import defaultdict  
import numpy as np
import igl
from blendshapes import BasicBlendshapes, load_blendshape

def cluster_blendshapes(blendshapes:BasicBlendshapes, cluster_threshold=0.25, activate_threshold=0.1):
    """cluster blendshapes based on IoU between regions of influence, only return clusters without mirror symmetry
    Args:
        blendshapes (BasicBlendshapes): blendshape model
        cluster_threshold (float, optional): IoU threshold for clustering. Defaults to 0.25.
        activate_threshold (float, optional): threshold for activating a vertex. Defaults to 0.1.
    Returns:
        clusters (list): list of clusters, each cluster is a list of blendshape indices
        symmetric_blendshapes (dict): symmetric blendshapes, key is the blendshape index, value is a list of symmetric blendshape indices
    """
    # number of blendshapes
    n_blendshapes = len(blendshapes)

    # compute displacement magnitude for each blendshape
    blendshapes_norm = np.linalg.norm(blendshapes.blendshapes, axis=2)

    # # find symmetric blendshapes
    # symmetric_blendshapes = defaultdict(list)
    # # precompute blendshape norm sum
    # blendshapes_norm_sum = np.sum(blendshapes_norm, axis=1)
    # for i in range(n_blendshapes):
    #     for j in range(i+1, n_blendshapes):
    #         # if the sum of the blendshape norm is the same, they might be symmetric
    #         if np.isclose(blendshapes_norm_sum[i], blendshapes_norm_sum[j]):
    #             # if the displacement is the same, they are symmetric
    #             mirror_blendshape = blendshapes.copy()
    #             mirror_blendshape[:,0] *= -1
    #             max_diff = np.max(igl.all_pairs_distances(mirror_blendshape, blendshapes[j], True))
    #             if max_diff < 1e-3:
    #                 if i not in symmetric_blendshapes:
    #                     symmetric_blendshapes[i] = [j]
    #                 else:
    #                     symmetric_blendshapes[i].append(j)
    #                 if j not in symmetric_blendshapes:
    #                     symmetric_blendshapes[j] = [i]
    #                 else:
    #                     symmetric_blendshapes[j].append(i)

    # TODO: remove symmetric blendshapes from the computation to improve performance
    # compute region of influence for each blendshape
    regions = np.zeros((n_blendshapes, blendshapes.V.shape[0]), dtype=bool)
    for i in range(n_blendshapes):
        blendshape_max_norm = np.max(blendshapes_norm[i])
        # valid if the relative displacement magnitude is larger than the activate threshold
        regions[i] = (blendshapes_norm[i] / blendshape_max_norm) > activate_threshold

    # compute IoU between blendshapes
    IoU = np.zeros((regions.shape[0], regions.shape[0]))
    for i in range(regions.shape[0]):
        for j in range(i+1, regions.shape[0]):
            IoU[i, j] = np.sum(regions[i] & regions[j]) / np.sum(regions[i] | regions[j])
            IoU[j, i] = IoU[i, j]
    
    # cluster blendshapes, walk through the IoU matrix until found all clusters
    clusters = []
    # sort the sum of IoU for each blendshape
    blendshapes_IoU = np.sum(IoU, axis=0)
    blendshapes_IoU_rank = np.argsort(blendshapes_IoU)[::-1]
    visited = np.zeros(regions.shape[0], dtype=bool)
    for i in blendshapes_IoU_rank:
        if visited[i]:
            continue
        # iteratively add blendshapes to the cluster with BFS based on IoU
        cluster = [i]
        Q = [i]
        while len(Q) > 0:
            blendshape = Q.pop(0)
            visited[blendshape] = True
            # for s in symmetric_blendshapes[blendshape]:
            #     visited[s] = True
            blendshape_IoU_rank = np.argsort(IoU[blendshape])[::-1]
            for j in blendshape_IoU_rank:
                # compare the IoU between the blendshape and the cluster
                if not visited[j] and np.all(IoU[cluster, j] > cluster_threshold):
                    cluster.append(j)
                    Q.append(j)
        clusters.append(cluster)

    # ! doesn't work
    # from sklearn.cluster import AgglomerativeClustering
    # model = AgglomerativeClustering(n_clusters=None, distance_threshold=1, linkage="average")
    # model = model.fit(IoU)
    # clusters = []
    # for i in range(model.n_clusters_):
    #     clusters.append(np.where(model.labels_ == i)[0])

    # return clusters, symmetric_blendshapes
    return clusters

if __name__ == "__main__":
    # load the blendshape model
    import os
    PROJ_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
    BLENDSHAPES_PATH = os.path.join(PROJ_ROOT, "data", "AppleAR", "OBJs")
    blendshapes = load_blendshape(BLENDSHAPES_PATH)
    # compute clusters
    import time
    start_time = time.time()
    clusters = cluster_blendshapes(blendshapes, cluster_threshold=0.05, activate_threshold=0.2)
    end_time = time.time()
    print(f"Clustering took {end_time - start_time} seconds.")
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
    bounding_box = (np.max(blendshapes.V, axis=0) - np.min(blendshapes.V, axis=0))*1.2

    for cluster_id, cluster in enumerate(clusters):
        N = len(cluster)
        NV = blendshapes.V.shape[0]
        NF = blendshapes.F.shape[0]
        V = np.zeros((N*NV, 3))
        F = np.zeros((N*NF, 3), dtype=np.int32)
        D = np.zeros(N*NV)
        for i, blendshape in enumerate(cluster):
            D[i*NV:(i+1)*NV] = np.linalg.norm(blendshapes[blendshape], axis=1)
            V[i*NV:(i+1)*NV] = blendshapes.V + blendshapes[blendshape]
            V[i*NV:(i+1)*NV,0] += i * bounding_box[0]
            F[i*NF:(i+1)*NF] = blendshapes.F + i*NV
        V[:,1] += cluster_id * bounding_box[1]
        vis_cluster = ps.register_surface_mesh(f"cluster {cluster_id}", V, F, smooth_shade=True)
        vis_cluster.add_scalar_quantity("displacement", D, enabled=True, cmap="turbo")
        vis_clusters.append(vis_cluster)
    ps.show()
