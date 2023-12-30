import os
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import polyscope as ps
import PIL
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))
src_path = os.path.join(PROJ_ROOT, "src")
sys.path.append(src_path)
FIGURES_FOLDER = os.path.join(PROJ_ROOT, "figures")
FIGURE_NAME = os.path.dirname(os.path.abspath(__file__))
from blendshapes import load_blendshape
from utils import *
from submanifold import *

if __name__ == "__main__":
    # load the blendshape model
    import os
    BLENDSHAPES_PATH = os.path.join(PROJ_ROOT, "data", "AppleAR", "OBJs")
    blendshapes = load_blendshape(BLENDSHAPES_PATH)
    
    # compute clusters
    from clustering import *
    clusters = cluster_blendshapes(blendshapes, cluster_threshold=0.05, activate_threshold=0.2)
    
    vae_submanifold_path = os.path.join(PROJ_ROOT, "experiments", "vae_submanifold")
    cluster_path = os.path.join(vae_submanifold_path, f"cluster_0")
    if not model_exists(cluster_path):
        print(f"Manifold model does not exist. Constructing {cluster_path}")
        vae_submanifold_construction(cluster_path, cluster)
    config = load_config(cluster_path)
    model = load_model(config)
    cluster = config["clusters"]
    cluster_dim = len(cluster)
    slice_dims = [0,2,3]
    n_slice_dims = len(slice_dims)
    n_samples = 5
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    weight_samples = np.zeros((n_slice_dims, n_samples, len(blendshapes)))
    latent_mean_samples = torch.linspace(-10, 10, n_samples).to(device)
    # x = torch.zeros(cluster_dim).to(device)
    # w = torch.zeros(len(blendshapes)).to(device)
    # w[cluster] = model(x)[0]
    # w = w.cpu().detach().numpy().copy()
    for i, slice_dim in enumerate(slice_dims):
        for j in range(n_samples):
            mu, logvar = model.encode(x)
            mu[slice_dim] += latent_mean_samples[j]
            z = model.reparameterize(mu, logvar)
            weight_samples[i,j,cluster] = model.decode(z).cpu().detach().numpy().copy()
    
    ps.init()
    ps.set_program_name("Latent Slicing")
    ps.set_verbosity(0)
    ps.set_SSAA_factor(3)
    ps.set_max_fps(60)
    ps.set_ground_plane_mode("none")
    ps.set_view_projection_mode("orthographic")
    import matplotlib.pyplot as plt
    import PIL
    # ps_V = ps.register_surface_mesh("V", blendshapes.eval(w), blendshapes.F, smooth_shade=True, enabled=True)
    # ps.show()
    # ps_V.set_enabled(False)
    fig, axes = plt.subplots(3, n_samples, figsize=(n_slice_dims*2, n_samples*2))
    for i in range(n_slice_dims):
        for j in range(n_samples):
            ps_V = ps.register_surface_mesh(f"V{i,j}", blendshapes.eval(weight_samples[i,j]), blendshapes.F, smooth_shade=True, enabled=True)
            ps.screenshot(f"temp.png")
            img = PIL.Image.open("temp.png")
            w, h = img.size
            diff = 0.35 * w
            img = img.crop((diff, 100, w-diff, h-100))
            axes[i,j].imshow(img)
            axes[i,j].axis("off")
            ps_V.set_enabled(False)
    os.remove("temp.png")
    fig.tight_layout()
    fig.savefig("latent.png", dpi=300, bbox_inches="tight", transparent=True)
    fig.savefig("latent.pdf", dpi=300, bbox_inches="tight", transparent=True)