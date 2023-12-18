from blendshapes import *
from train import *
from inference import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import igl
import scipy.sparse as sp
import os

RESULTS_PATH = os.path.join(os.pardir, "results")

def distance_energy(V0, V1):
    # compute the distance between two meshes at each vertex
    return np.sum(np.linalg.norm(V0 - V1, axis=-1))

def sample_distance_energy(model:CollisionBlendshapeModel, id_pair, N=100):
    # sample grid coordiantes in [0, 1]^2
    coords = sample_cspace(N)
    value = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            weights = np.zeros_like(model.weights)
            weights[id_pair[0]] = coords[i, j, 0]
            weights[id_pair[1]] = coords[i, j, 1]
            value[i, j] = distance_energy(model.V, model.eval(weights))
    return coords, value

def arap_precompute(V0, F):
    N = V0.shape[0]
    L = igl.cotmatrix(V0, F)
    K_triplets = []
    for f in F:
        for j in range(3):
            u = f[j]
            v = f[(j+1)%3]
            e = L[u, v] * (V0[u] - V0[v])
            for k in range(3):
                w = f[k]
                for dim in range(3):
                    K_triplets.append((u, 3*w+dim, e[dim]))
                    K_triplets.append((v, 3*w+dim, -e[dim]))
    K_triplets = np.array(K_triplets)
    K = sp.coo_matrix((K_triplets[:, 2], (K_triplets[:, 0], K_triplets[:, 1])), shape=(N, 3*N))
    return L, K

def arap_energy(V0, V1, F, L=None, K=None):
    # compute the ARAP energy between two meshes
    # V0: reference mesh
    # V1: deformed mesh
    # F: face indices
    # L: cotangent (discrete Laplacian) matrix
    # K: bilinear form matrix
    # return: ARAP energy
    N = V0.shape[0]
    if L is None or K is None:
        L, K = arap_precompute(V0, F)
    C = (V1.T @ K).T
    R = np.zeros((N*3, 3))
    for i in range(N):
        U, S, VT = np.linalg.svd(C[3*i:3*i+3,:])
        R[3*i:3*i+3,:] = U @ VT
    E = np.trace(V1.T @ L @ V1) / 6 + np.trace(C.T @ R) / 6
    return E

def arap_per_vertex_energy(V0, V1, F, L=None, K=None):
    # compute the ARAP energy between two meshes
    # V0: reference mesh
    # V1: deformed mesh
    # F: face indices
    # L: cotangent (discrete Laplacian) matrix
    # K: bilinear form matrix
    # return: ARAP energy
    N = V0.shape[0]
    if L is None or K is None:
        L, K = arap_precompute(V0, F)
    C = (V1.T @ K).T
    R = np.zeros((N*3, 3))
    for i in range(N):
        U, S, VT = np.linalg.svd(C[3*i:3*i+3,:])
        R[3*i:3*i+3,:] = U @ VT
    R = R.reshape((N, 3, 3))
    E = np.zeros(N)
    adj = igl.adjacency_matrix(F)
    for i in range(N):
        for j in range(N):
            if adj[i, j] != 0:
                e1 = V1[i] - V1[j]
                e0 = V0[i] - V0[j]
                E[i] += L[i, j] * np.linalg.norm(e1 - R[i] @ e0)**2
    return E

def arap_gradient(V0, V1, F, L=None, K=None):
    # compute the ARAP energy between two meshes
    # V0: reference mesh
    # V1: deformed mesh
    # F: face indices
    # L: cotangent (discrete Laplacian) matrix
    # K: bilinear form matrix
    # return: ARAP energy
    N = V0.shape[0]
    if L is None or K is None:
        L, K = arap_precompute(V0, F)
    C = (V1.T @ K).T
    R = np.zeros((N*3, 3))
    for i in range(N):
        U, S, VT = np.linalg.svd(C[3*i:3*i+3,:])
        R[3*i:3*i+3,:] = U @ VT
    E = L @ V1 + L.T @ V1 + K @ R
    return E

def sample_arap_energy(model:CollisionBlendshapeModel, id_pair, N=100):
    # sample grid coordiantes in [0, 1]^2
    L, K = arap_precompute(model.V, model.F)
    coords = sample_cspace(N)
    value = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            weights = np.zeros_like(model.weights)
            weights[id_pair[0]] = coords[i, j, 0]
            weights[id_pair[1]] = coords[i, j, 1]
            value[i, j] = arap_energy(model.V, model.eval(weights), model.F, L, K)
    return coords, value

def sample_cspace(N=100):
    # sample grid coordiantes in [0, 1]^2
    coords = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
    coords = np.stack(coords, axis=-1)
    return coords

def sample_data(model:CollisionBlendshapeModel, id_pair, N=100, energy="arap"):
    if energy == "arap":
        return sample_arap_energy(model, id_pair, N)
    elif energy == "distance":
        return sample_distance_energy(model, id_pair, N)

def submanifold_metric(model:CollisionBlendshapeModel, clusters, N=101, energy="arap"):
    N_CLUSTERS = len(clusters)
    cmap = cm.get_cmap('viridis')
    for c in range(N_CLUSTERS):
        NB = len(clusters[c])
        for i in range(NB):
            id0 = clusters[c][i]
            for j in range(i+1, NB):
                id1 = clusters[c][j]
                print(f"Computing C-space metric for cluster {c}, blendshapes {id0} and {id1}.")
                coords, value = sample_data(model, (id0, id1), N=N, energy=energy)
                # if np.sum(value) <= 0.01 * N**2:
                #     print(f"skip computing for cluster {c}, blendshapes {id0} and {id1}.")
                #     continue
                # compute submanifold metric
                net = train(coords, value)
                pred = infer(net, coords)
                # save submanifold metric
                SAVE_FOLDER = os.path.join(RESULTS_PATH, f"{id0}_{id1}")
                os.makedirs(SAVE_FOLDER, exist_ok=True)
                SAVE_PATH = os.path.join(SAVE_FOLDER, f"network.pt")
                torch.save(net.state_dict(), SAVE_PATH)
                # save plot of submanifold metric
                plt.figure()
                fig, ax = plt.subplots(1,2)
                vmin, vmax = np.min(value), np.max(value)
                ax[0].imshow(value, vmin=vmin, vmax=vmax)
                ax[0].set_title("Ground Truth")
                # ax[1].imshow(pred, vmin=vmin, vmax=vmax)
                ax[1].pcolormesh(pred, cmap=cmap, vmin=vmin, vmax=vmax)
                ax[1].set_title("Prediction")
                fig.savefig(f"c{c}_{id0}_{id1}.png", bbox_inches='tight')


def submanifold_metric_samples(model:CollisionBlendshapeModel, clusters, N=101, energy="arap"):
    N_CLUSTERS = len(clusters)
    cmap = cm.get_cmap('viridis')
    for c in range(N_CLUSTERS):
        NB = len(clusters[c])
        for i in range(NB):
            id0 = clusters[c][i]
            for j in range(i+1, NB):
                id1 = clusters[c][j]
                print(f"Computing C-space metric for cluster {c}, blendshapes {id0} and {id1}.")
                coords, value = sample_data(model, (id0, id1), N=N, energy=energy)
                vmin, vmax = np.min(value), np.max(value)
                SAVE_FOLDER = os.path.join(RESULTS_PATH, f"{id0}_{id1}")
                os.makedirs(SAVE_FOLDER, exist_ok=True)
                # save submanifold metric
                np.save(os.path.join(SAVE_FOLDER, "data.npy"), value)
                # save plot of submanifold metric
                plt.figure(figsize=(6,6))
                plt.pcolormesh(value, cmap=cmap, vmin=vmin, vmax=vmax)
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(os.path.join(SAVE_FOLDER, "plot.png"), bbox_inches='tight')
