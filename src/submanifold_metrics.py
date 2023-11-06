from blendshapes import *
from train import *
from inference import *
import matplotlib.pyplot as plt

def sample_cspace(N=100):
    # sample grid coordiantes in [-1, 1]^2
    coords = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    coords = np.stack(coords, axis=-1)
    return coords

def sample_data(model:CollisionBlendshapeModel, id_pair, N=100):
    # sample grid coordiantes in [-1, 1]^2
    coords = sample_cspace(N)
    labels = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            weights = np.zeros_like(model.weights)
            weights[id_pair[0]] = coords[i, j, 0]
            weights[id_pair[1]] = coords[i, j, 1]
            if model.has_intersections(weights):
                labels[i, j] = 1
    return coords, labels

def submanifold_metric(model:CollisionBlendshapeModel, clusters, N=101):
    N_CLUSTERS = len(clusters)
    for c in range(N_CLUSTERS):
        NB = len(clusters[c])
        for i in range(NB):
            id0 = clusters[c][i]
            for j in range(i+1, NB):
                id1 = clusters[c][j]
                print(f"Computing C-space metric for cluster {c}, blendshapes {id0} and {id1}.")
                coords, labels = sample_data(model, (id0, id1), N)
                if np.sum(labels) <= 0.01 * N**2:
                    print(f"skip computing for cluster {c}, blendshapes {id0} and {id1}.")
                    continue
                # compute submanifold metric
                net = train(coords, labels)
                pred = infer(net, coords)
                # save submanifold metric
                torch.save(net.state_dict(), f"c{c}_{id0}_{id1}.pt")
                # save plot of submanifold metric
                plt.figure()
                fig, ax = plt.subplots(1,2)
                ax[0].imshow(labels, vmin=0, vmax=1)
                ax[0].set_title("Ground Truth")
                ax[1].imshow(pred, vmin=0, vmax=1)
                ax[1].set_title("Prediction")
                fig.savefig(f"c{c}_{id0}_{id1}.png")

