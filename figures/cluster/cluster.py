import os
import sys
import numpy as np
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))
src_path = os.path.join(PROJ_ROOT, "src")
sys.path.append(src_path)
FIGURES_FOLDER = os.path.join(PROJ_ROOT, "figures")
FIGURE_NAME = os.path.dirname(os.path.abspath(__file__))
from blendshapes import load_blendshape
from utils import *
from submanifold import *

def compute_error(weights, weights_gt):
    return np.linalg.norm(weights - weights_gt, axis=1).mean()

if __name__ == "__main__":
    # load the blendshape model
    import os
    BLENDSHAPES_PATH = os.path.join(PROJ_ROOT, "data", "AppleAR", "OBJs")
    blendshapes = load_blendshape(BLENDSHAPES_PATH)
    
    # compute clusters
    from clustering import *
    clusters = cluster_blendshapes(blendshapes, cluster_threshold=0.05, activate_threshold=0.2)
    
    # save per-vertex offsets
    cluster_path = os.path.join(FIGURES_FOLDER, FIGURE_NAME, "cluster")
    os.makedirs(cluster_path, exist_ok=True)
    for i, cluster in enumerate(clusters):
        weights = np.zeros(len(blendshapes))
        weights[cluster] = 1
        displacement = blendshapes.displacement(weights)
        displacement = np.linalg.norm(displacement, axis=1)
        np.save(os.path.join(cluster_path, f"{i}"), displacement)
    