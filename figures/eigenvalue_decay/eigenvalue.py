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

def singular_value_decay(A):
    """
    Compute the eigenvalue decay of a list of eigenvalues.
    """
    U, S, V = np.linalg.svd(A)
    index = np.argsort(S)[::-1]
    return S[index], index

if __name__ == "__main__":
    # load the blendshape model
    import os
    path = '/home/zhecheng/.local/share/fonts/LinBiolinum_R.ttf'
    biolinum_font = fm.FontProperties(fname=path)
    sns.set(font=biolinum_font.get_name())
    PROJ_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
    BLENDSHAPES_PATH = os.path.join(PROJ_ROOT, "data", "AppleAR", "OBJs")
    blendshapes = load_blendshape(BLENDSHAPES_PATH)
    
    A = blendshapes.blendshapes
    A = A.reshape((A.shape[0], -1))
    S, index = singular_value_decay(A)
    # lowest_3 = index[-3:].tolist()
    # highest_3 = index[:3].tolist()
    # ps.init()
    # ps.set_verbosity(0)
    # ps.set_SSAA_factor(3)
    # ps.set_ground_plane_mode("none")
    # ps.set_view_projection_mode("orthographic")
    # Create the figure and subplots
    # fig = plt.figure(figsize=(8, 5))
    # subfigs = fig.subfigures(1, 2, wspace=0.1)
    # axs0 = subfigs[0].subplots(1, 1)
    # sns.lineplot(x=range(len(S)), y=S, ax=axs0)
    # axs0.set_xlabel("Index")
    # axs0.set_ylabel("Singular Value")
    # axs0.set_title("Singular Value Decay")
    # inner_grid = subfigs[1].subplots(2, 3)
    # # plot highest & lowest 3 singular values blendshapes
    # vmin, vmax = blendshapes.delta.min(), blendshapes.delta.max()
    # for i in range(2):
    #     if i == 0:
    #         index = highest_3
    #     else:
    #         index = lowest_3
    #     for j, idx in enumerate(index):
    #         ax = inner_grid[i, j]
    #         ax.axis("off")
    #         ps_V = ps.register_surface_mesh(f"V", blendshapes.V + blendshapes[idx], blendshapes.F, smooth_shade=True, enabled=True)
    #         ps_V.add_scalar_quantity("delta", blendshapes.delta[idx], cmap="turbo", vminmax=(vmin, vmax), enabled=True)
    #         # ps.show()
    #         ps.screenshot(f"temp.png")
    #         img = PIL.Image.open("temp.png")
    #         w, h = img.size
    #         diff = 0.35 * w
    #         img = img.crop((diff, 250, w-diff, h))
    #         ax.imshow(img)
    #         ax.axis("off")
    #         ps_V.set_enabled(False)
    # os.remove("temp.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    sns.lineplot(x=range(len(S)), y=S, ax=axes[0])
    axes[0].set_xlabel("Index")
    axes[0].set_ylabel("Singular Value")
    cond_number = np.max(S) / np.min(S)
    axes[0].set_title(f"Singular Value Decay")
    # remove that max_idx
    ls_A = A @ A.T
    # rows, cols = np.nonzero(ls_A)
    # sorted_rows = np.argsort(rows)
    # sorted_cols = np.argsort(cols)
    # sorted_ls_A = ls_A[rows[sorted_rows]][:, cols[sorted_cols]]
    percentages = np.percentile(ls_A, [5, 95])
    vmax = np.max([np.abs(percentages[0]), np.abs(percentages[1])])
    # print(np.min(ls_A), np.max(ls_A))
    # ls_A[ls_A < 0] = 0
    # print(np.abs(A).min(), np.abs(A).max())
    # sns.heatmap(np.abs(A), ax=axes[1], cmap="Blues")
    # cov = np.cov(A)
    # percentages = np.percentile(cov, [5, 95])
    sns.heatmap(ls_A, ax=axes[1], cmap="vlag", vmax=vmax, vmin=-vmax)
    axes[1].set_title(f"Least-Square Matrix")
    axes[1].axis("off")
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(FIGURES_FOLDER, FIGURE_NAME, "singular_value_decay.pdf"), dpi=300)
    plt.savefig(os.path.join(FIGURES_FOLDER, FIGURE_NAME, "singular_value_decay.png"), dpi=300)
