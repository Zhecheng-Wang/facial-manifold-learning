from blendshapes import *
from train import *
from inference import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.sparse as sp
import os
from model import *
from utils import *

def sample_configurations(blendshapes, weights):
    # generate # n_sample random weights
    n_samples = weights.shape[0]
    V = np.zeros((n_samples, *blendshapes.V.shape))
    for i in range(n_samples):
        V[i] = blendshapes.eval(weights[i])
    return V

def manifold_construction(save_path, cluster=[], dataset=None, network_type="dae"):
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    train(save_path, cluster, dataset, network_type)

def manifold_projection(blendshapes, weights, model):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    
    n_samples = weights.shape[0]
    
    # project to the manifold
    proj_weights = infer(model, weights)
    
    # geometry of the blendshapes
    V_proj = np.zeros((n_samples, *blendshapes.V.shape))
    for i in range(n_samples):
        V_proj[i] = blendshapes.eval(proj_weights[i])
    
    return proj_weights, V_proj

def submanifolds_construction(save_path, clusters):
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    dataset = load_dataset()
    for i, cluster in enumerate(clusters):
        cluster_save_path = os.path.join(save_path, f"cluster_{i}")
        if not os.path.exists(cluster_save_path):
            os.makedirs(cluster_save_path, exist_ok=True)
        train(cluster_save_path, cluster, dataset)

def submanifolds_projection(blendshapes, weights, ensemble):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    n_samples = weights.shape[0]
    
    # project to the submanifold
    proj_weights = np.zeros((n_samples, len(blendshapes)))
    for i, (model, cluster) in enumerate(ensemble):
        model = model.to(device)
        proj_weights[:,cluster] = infer(model, weights[:,cluster])
    
    # geometry of the blendshapes
    V_proj = np.zeros((n_samples, *blendshapes.V.shape))
    for i in range(n_samples):
        V_proj[i] = blendshapes.eval(proj_weights[i])
    
    return proj_weights, V_proj

if __name__ == "__main__":
    # load the blendshape model
    import os
    PROJ_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
    BLENDSHAPES_PATH = os.path.join(PROJ_ROOT, "data", "AppleAR", "OBJs")
    blendshapes = load_blendshape(BLENDSHAPES_PATH)
    
    # compute clusters
    from clustering import *
    clusters = cluster_blendshapes(blendshapes, cluster_threshold=0.05, activate_threshold=0.2)
    print(clusters)
    
    # load dataset
    dataset = load_dataset()
    print(f"dataset # of samples: {len(dataset.dataset)}")
    
    manifold_path = os.path.join(PROJ_ROOT, "experiments", "manifold")
    if not model_exists(manifold_path):
        print(f"Manifold model does not exist. Constructing {manifold_path}")
        manifold_construction(manifold_path, dataset=dataset, network_type="ae")
        
    manifold_path = os.path.join(PROJ_ROOT, "experiments", "dae_manifold")
    if not model_exists(manifold_path):
        print(f"Manifold model does not exist. Constructing {manifold_path}")
        manifold_construction(manifold_path, dataset=dataset, network_type="dae")
        
    manifold_path = os.path.join(PROJ_ROOT, "experiments", "vae_manifold")
    if not model_exists(manifold_path):
        print(f"Manifold model does not exist. Constructing {manifold_path}")
        manifold_construction(manifold_path, dataset=dataset, network_type="vae")

    submanifold_path = os.path.join(PROJ_ROOT, "experiments", "submanifold")
    for i, cluster in enumerate(clusters):
        cluster_path = os.path.join(submanifold_path, f"cluster_{i}")
        if not model_exists(cluster_path):
            print(f"Manifold model does not exist. Constructing {cluster_path}")
            manifold_construction(cluster_path, cluster, dataset=dataset, network_type="ae")
            
    submanifold_path = os.path.join(PROJ_ROOT, "experiments", "dae_submanifold")
    for i, cluster in enumerate(clusters):
        cluster_path = os.path.join(submanifold_path, f"cluster_{i}")
        if not model_exists(cluster_path):
            print(f"Manifold model does not exist. Constructing {cluster_path}")
            manifold_construction(cluster_path, cluster, dataset=dataset, network_type="dae")
            
    vae_submanifold_path = os.path.join(PROJ_ROOT, "experiments", "vae_submanifold")
    for i, cluster in enumerate(clusters):
        cluster_path = os.path.join(vae_submanifold_path, f"cluster_{i}")
        if not model_exists(cluster_path):
            print(f"Manifold model does not exist. Constructing {cluster_path}")
            manifold_construction(cluster_path, cluster, dataset=dataset, network_type="vae")
