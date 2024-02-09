import numpy as np
from blendshapes import BasicBlendshapes, load_blendshape
import polyscope as ps

def compute_ruzicka_similarity(blendshapes:BasicBlendshapes):
    """compute Ruzicka similarity between blendshapes
    Args:
        blendshapes (BasicBlendshapes): blendshape model
    Returns:
        similarity (np.ndarray): similarity matrix
    """
    # number of blendshapes
    n_blendshapes = len(blendshapes)
    # compute displacement magnitude for each blendshape
    blendshapes_norm = np.linalg.norm(blendshapes.blendshapes, axis=2)
    # Ruzicka similarity
    similarity = np.zeros((n_blendshapes, n_blendshapes))
    for i in range(n_blendshapes):
        similarity[i, i] = 1.0
        for j in range(i+1, n_blendshapes):
            similarity[i, j] = np.sum(np.minimum(blendshapes_norm[i], blendshapes_norm[j])) / np.sum(np.maximum(blendshapes_norm[i], blendshapes_norm[j]))
            similarity[j, i] = similarity[i, j]
    return similarity

def compute_maximum_deformation_position_similarity(blendshapes:BasicBlendshapes):
    n_blendshapes = len(blendshapes)
    blend_shape_maximum_deformation_position = np.mean(blendshapes.blendshapes + blendshapes.neu, axis=1)
    # reflected
    reflected_blend_shape_maximum_deformation_position = blend_shape_maximum_deformation_position.copy()
    reflected_blend_shape_maximum_deformation_position[:, 1] = -reflected_blend_shape_maximum_deformation_position[:, 1]
    # compluete similarity matrix
    similarity = np.zeros((n_blendshapes, n_blendshapes))
    for i in range(n_blendshapes):
        similarity[i, i] = 1.0
        for j in range(i+1, n_blendshapes):
            similarity_ij = np.linalg.norm(blend_shape_maximum_deformation_position[i] - blend_shape_maximum_deformation_position[j])
            similarity_ij_reflected = np.linalg.norm(blend_shape_maximum_deformation_position[i] - reflected_blend_shape_maximum_deformation_position[j])
            similarity[i, j] = np.max([similarity_ij, similarity_ij_reflected])
            similarity[j, i] = similarity[i, j]
    similarity = similarity/np.max(similarity)
    return similarity
    

def compute_jaccard_similarity(blendshapes:BasicBlendshapes, threshold=0.1):
    """compute Jaccard similarity between blendshapes
    Args:
        blendshapes (BasicBlendshapes): blendshape model
        threshold (float, optional): threshold for activating a vertex. Defaults to 0.1.
    Returns:
        similarity (np.ndarray): similarity matrix
    """
    # number of blendshapes
    n_blendshapes = len(blendshapes)
    # compute displacement magnitude for each blendshape
    blendshapes_norm = np.linalg.norm(blendshapes.blendshapes, axis=2)
    blendshapes_activation = blendshapes_norm > threshold
    # Jaccard similarity
    similarity = np.zeros((n_blendshapes, n_blendshapes))
    for i in range(n_blendshapes):
        similarity[i, i] = 1.0
        for j in range(i+1, n_blendshapes):
            similarity[i, j] = np.sum(blendshapes_activation[i] & blendshapes_activation[j]) / np.sum(blendshapes_activation[i] | blendshapes_activation[j])
            similarity[j, i] = similarity[i, j]
    return similarity

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
    # for i in range(n_blendshapes):
    #     for j in range(i+1, n_blendshapes):
    #         Vi = blendshapes.V + blendshapes[i]
    #         Vj = blendshapes.V + blendshapes[j]
    #         # mirror Vj
    #         Vj[:,0] = -Vj[:,0]
    #         Aj = blendshapes[j] > 0
    #         # check if the two blendshapes are symmetric
    #         Vj = Vj[Aj]
    #         sqrD, I, J = igl.point_mesh_squared_distance(Vj, Vi, blendshapes.F)
    #         dist = np.mean(np.sqrt(sqrD))
    #         if dist != np.nan:
    #             print(dist)
    #         if dist < 0.1:
    #             symmetric_blendshapes[i].append(j)
    #             symmetric_blendshapes[j].append(i)
    # print(f"Found {len(symmetric_blendshapes)} symmetric blendshapes.")
    
    # # fit k-means to cluster blendshapes
    # from sklearn.cluster import KMeans
    # n_clusters = 8
    # kmeans = KMeans(n_init=10, n_clusters=n_clusters, random_state=0).fit(blendshapes_norm)
    # clusters = []
    # for i in range(n_clusters):
    #     clusters.append(np.where(kmeans.labels_ == i)[0])
    # return clusters
    
    # # fit spectral clustering to cluster blendshapes
    # from sklearn.cluster import SpectralClustering
    # n_clusters = 8
    # spectral = SpectralClustering(n_init=10, n_clusters=n_clusters, affinity="nearest_neighbors", n_neighbors=10).fit(blendshapes_norm)
    # clusters = []
    # for i in range(n_clusters):
    #     clusters.append(np.where(spectral.labels_ == i)[0])
    # return clusters

    # # TODO: remove symmetric blendshapes from the computation to improve performance
    # # compute region of influence for each blendshape
    # regions = np.zeros((n_blendshapes, blendshapes.V.shape[0]), dtype=bool)
    # for i in range(n_blendshapes):
    #     blendshape_max_norm = np.max(blendshapes_norm[i])
    #     # valid if the relative displacement magnitude is larger than the activate threshold
    #     regions[i] = (blendshapes_norm[i] / blendshape_max_norm) > activate_threshold

    # # compute IoU between blendshapes
    # IoU = np.zeros((regions.shape[0], regions.shape[0]))
    # for i in range(regions.shape[0]):
    #     for j in range(i+1, regions.shape[0]):
    #         IoU[i, j] = np.sum(regions[i] & regions[j]) / np.sum(regions[i] | regions[j])
    #         IoU[j, i] = IoU[i, j]
    
    # # cluster blendshapes, walk through the IoU matrix until found all clusters
    # clusters = []
    # # sort the sum of IoU for each blendshape
    # blendshapes_IoU = np.sum(IoU, axis=0)
    # blendshapes_IoU_rank = np.argsort(blendshapes_IoU)[::-1]
    # visited = np.zeros(regions.shape[0], dtype=bool)
    # for i in blendshapes_IoU_rank:
    #     if visited[i]:
    #         continue
    #     # iteratively add blendshapes to the cluster with BFS based on IoU
    #     cluster = [i]
    #     Q = [i]
    #     while len(Q) > 0:
    #         blendshape = Q.pop(0)
    #         visited[blendshape] = True
    #         # for s in symmetric_blendshapes[blendshape]:
    #         #     visited[s] = True
    #         blendshape_IoU_rank = np.argsort(IoU[blendshape])[::-1]
    #         for j in blendshape_IoU_rank:
    #             # compare the IoU between the blendshape and the cluster
    #             if not visited[j] and np.all(IoU[cluster, j] > cluster_threshold):
    #                 cluster.append(j)
    #                 Q.append(j)
    #     clusters.append(cluster)
    
    # Ruzicka similarity
    IoU = compute_ruzicka_similarity(blendshapes)
    # IoU = compute_jaccard_similarity(blendshapes, threshold=activate_threshold)
    
    # zero out the bottom activate_threshold%
    cutoff = np.percentile(IoU, activate_threshold*100)
    IoU[IoU < cutoff] = 0.0
    cluster_cutoff = np.max(IoU) * cluster_threshold
    
    # cluster blendshapes, walk through the IoU matrix until found all clusters
    clusters = []
    # sort the sum of IoU for each blendshape
    blendshapes_IoU = np.sum(IoU, axis=0)
    blendshapes_IoU_rank = np.argsort(blendshapes_IoU)[::-1]
    visited = np.zeros(n_blendshapes, dtype=bool)
    for i in blendshapes_IoU_rank:
        if visited[i]:
            continue
        # iteratively add blendshapes to the cluster with BFS based on IoU
        cluster = [i]
        Q = [i]
        while len(Q) > 0:
            blendshape = Q.pop(0)
            visited[blendshape] = True
            blendshape_IoU_rank = np.argsort(IoU[blendshape])[::-1]
            # select unvisited in the rank
            # if blendshape in symmetric_blendshapes:
            #     visited[symmetric_blendshapes[blendshape]] = True
            for j in blendshape_IoU_rank:
                # compare the IoU between the blendshape and the cluster
                if not visited[j] and np.all(IoU[cluster, j] > cluster_cutoff):
                    cluster.append(j)
                    # if j in symmetric_blendshapes:
                    #     cluster.extend(symmetric_blendshapes[j])
                    Q.append(j)
        clusters.append(cluster)
    
    return clusters
def cluster_blendshapes_ec8ec1a(blendshapes:BasicBlendshapes, cluster_threshold=0.25, activate_threshold=0.1):
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
    blendshapes = load_blendshape(model="SP")
    # compute clusters
    import time
    start_time = time.time()
    clusters = cluster_blendshapes(blendshapes, cluster_threshold=0.05, activate_threshold=0.2)
    end_time = time.time()
    print(f"Clustering took {end_time - start_time} seconds.")
    print(f"Found {len(clusters)} clusters.")

    # plot the clusters as heatmap
    import matplotlib.pyplot as plt
    # plot the similarity matrix
    similarity = compute_maximum_deformation_position_similarity(blendshapes)
    plt.imshow(similarity, cmap="turbo")
    # label each column
    blendshape_names = blendshapes.names
    plt.xticks(range(len(blendshape_names)), blendshape_names, rotation=90)

    plt.colorbar()
    # add the values
    plt.show()




    # visualize clusters
    import polyscope as ps
    ps.init()
    ps.set_program_name("Clustering Visualization")
    ps.set_verbosity(0)
    ps.set_SSAA_factor(3)
    ps.set_max_fps(60)
    ps.set_ground_plane_mode("none")
    ps.set_view_projection_mode("orthographic")
    ps.set_autocenter_structures(False)
    ps.set_autoscale_structures(False)
    ps.set_front_dir("z_front")
    # compute the bounding box of the model
    bounding_box = (np.max(blendshapes.V, axis=0) - np.min(blendshapes.V, axis=0))*1.2

    for cluster_id, cluster in enumerate(clusters):
        group = ps.create_group(f"cluster {cluster_id}")
        for i, blendshape in enumerate(cluster):
            D = np.linalg.norm(blendshapes[blendshape], axis=1) / 100
            V = blendshapes.V + blendshapes[blendshape]
            cluster_i = ps.register_surface_mesh(f"{blendshape}", V, blendshapes.F, smooth_shade=True)
            cluster_i.add_scalar_quantity("displacement", D, enabled=True, cmap="turbo")
            cluster_i.add_to_group(group)
            transform = np.zeros(3)
            transform[0] = i * bounding_box[0]
            transform[1] = cluster_id * bounding_box[1]
            cluster_i.set_position(transform)
        group.set_hide_descendants_from_structure_lists(True)
        group.set_show_child_details(False)
    ps.show()
