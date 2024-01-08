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

if __name__ == "__main__":
    # load the blendshape model
    import os
    path = f'{os.path.expanduser("~")}/.local/share/fonts/LinBiolinum_R.ttf'
    biolinum_font = fm.FontProperties(fname=path)
    sns.set(font=biolinum_font.get_name())
    PROJ_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
    BLENDSHAPES_PATH = os.path.join(PROJ_ROOT, "data", "AppleAR", "OBJs")
    blendshapes = load_blendshape(BLENDSHAPES_PATH)

    ps.init()
    ps.set_verbosity(0)
    ps.set_SSAA_factor(3)
    ps.set_ground_plane_mode("none")
    ps.set_view_projection_mode("orthographic")
    random_weights = np.random.rand(4, len(blendshapes))
    fig, axes = plt.subplots(2, 2, figsize=(5, 5))
    for i in range(2):
        for j in range(2):
            idx = i*2+j
            V = blendshapes.eval(random_weights[idx])
            ps_mesh = ps.register_surface_mesh(f"random", V, blendshapes.F, smooth_shade=True, material="wax", color=(0.7, 0.7, 0.7), edge_width=0.5, enabled=True)
            ps.screenshot(f"temp.png")
            img = PIL.Image.open("temp.png")
            w, h = img.size
            diff = 0.35 * w
            img = img.crop((diff, 175, w-diff, h-90))
            axes[i, j].imshow(img)
            axes[i, j].axis("off")
    os.remove("temp.png")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_FOLDER, FIGURE_NAME, "random_config.pdf"), dpi=300, bbox_inches="tight", transparent=True)
    plt.savefig(os.path.join(FIGURES_FOLDER, FIGURE_NAME, "random_config.png"), dpi=300, bbox_inches="tight", transparent=True)
