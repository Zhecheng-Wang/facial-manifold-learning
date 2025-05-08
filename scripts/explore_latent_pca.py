import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.append("/Users/evanpan/Documents/GitHub/ManifoldExploration/src")
sys.path.append("/Users/evanpan/Documents/GitHub/ManifoldExploration")


import json
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import load_dataset, load_blendshape, SPDataset
from scripts.SMOTE import BalancedSMOTEDataset
from model import build_model, save_model, KL_divergence
from clustering import compute_jaccard_similarity, cluster_blendshapes_kmeans
import matplotlib.pyplot as plt
from scripts.plot_heads import *

from utils import load_blendshape, SPDataset, load_config
from clustering import compute_jaccard_similarity
from model import load_model

def visualize_explained_variance(pca, n_components):
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.bar(range(1, n_components + 1), explained_variance)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Component')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, n_components + 1), cumulative_variance, marker='o')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance')
    plt.legend()
    plt.tight_layout()
    plt.show()

def visualize_components(pca, n_components):
    explained_variance = pca.explained_variance_ratio_
    num_to_show = min(5, n_components)  # Show up to 5 components
    plt.figure(figsize=(15, 3 * num_to_show))
    for i in range(num_to_show):  # Show top components
        plt.subplot(num_to_show, 1, i+1)
        component = pca.components_[i]
        plt.bar(range(len(component)), component)
        plt.title(f'Principal Component {i+1} (Variance: {explained_variance[i]:.2%})')
        plt.ylabel('Weight')
        plt.xlim(-1, len(component))
    plt.tight_layout()
    plt.show()

    # 3. Visualize the mean face
    # plt.figure(figsize=(12, 4))
    # plt.bar(range(len(blendshapes)), pca.mean_)
    # plt.title('Mean Blendshape Weights')
    # plt.xlabel('Blendshape Index')
    # plt.ylabel('Weight')
    # plt.tight_layout()
    # plt.show()

def visualize_pca_projection(pca, n_components, in_data):
    projected_data = pca.transform(in_data)
    explained_variance = pca.explained_variance_ratio_
    # Visualize the first two principal components (if we have at least 2)
    if n_components >= 2:
        plt.figure(figsize=(10, 8))
        plt.scatter(projected_data[:, 0], projected_data[:, 1], alpha=0.5)
        plt.xlabel(f'PC1 ({explained_variance[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({explained_variance[1]:.2%} variance)')
        plt.title('Data Projected onto First Two Principal Components')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def visualize(pca, n_components):
    # 1. Visualize explained variance
    visualize_explained_variance(pca, n_components)
    # 2. Visualize the first few principal components

    # 5. Reconstruct a random example with different numbers of components
    sample_idx = np.random.randint(0, n_frames)  # Choose a random sample
    original_sample = frame_weights[sample_idx]

    reconstructions = []
    for i in range(1, n_components + 1):
        # Transform and inverse transform to get reconstruction
        transformed = pca.transform([original_sample])[:, :i]
        # Create a zero-filled array with the right shape for reconstruction
        full_transformed = np.zeros((1, n_components))
        full_transformed[0, :i] = transformed[0, :i]
        reconstructed = pca.inverse_transform(full_transformed)
        reconstructions.append(reconstructed[0])

    # Calculate reconstruction error for each number of components
    mse_errors = [np.mean((original_sample - rec)**2) for rec in reconstructions]

    plt.figure(figsize=(12, 4))
    plt.plot(range(1, n_components + 1), mse_errors, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Mean Squared Error')
    plt.title('Reconstruction Error vs. Number of Components')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('reconstruction_error.png')

    # 6. Visualize original vs reconstructed (using different numbers of components)
    mid_point = max(1, n_components // 3)
    
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.bar(range(len(blendshapes)), original_sample)
    plt.title('Original Blendshape Weights')
    plt.ylabel('Weight')

    plt.subplot(3, 1, 2)
    plt.bar(range(len(blendshapes)), reconstructions[mid_point - 1])  # Using mid_point components
    plt.title(f'Reconstructed Weights (Using {mid_point} Components)')
    plt.ylabel('Weight')

    plt.subplot(3, 1, 3)
    plt.bar(range(len(blendshapes)), reconstructions[-1])  # Using all components
    plt.title(f'Reconstructed Weights (Using All {n_components} Components)')
    plt.xlabel('Blendshape Index')
    plt.ylabel('Weight')
    plt.tight_layout()
    plt.savefig('reconstruction_comparison.png')

    # Print summary information
    print("PCA Summary:")
    print(f"Number of original blendshapes: {len(blendshapes)}")
    print(f"Number of principal components: {n_components}")
    print(f"Shape of mean vector: {pca.mean_.shape}")
    print(f"Shape of principal components: {pca.components_.shape}")
    
    # Print variance explained at different thresholds
    thresholds = [3, 5, 10, n_components]
    for t in thresholds:
        if t <= n_components:
            print(f"First {t} components explain {cumulative_variance[t-1]:.2%} of variance")


PROJ_ROOT = "/Users/evanpan/Documents/GitHub/ManifoldExploration/data"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

blendshapes = load_blendshape(model="SP")

dataset        = SPDataset()
frame_weights  = dataset.data.numpy()                 # [N, m]
n_frames       = len(dataset)
n_blendshapes  = len(blendshapes)

#  ========================================  perform PCA on the dataset ======================================== 
from sklearn.decomposition import PCA
n_components = 5
pca = PCA(n_components=n_components)
pca.fit(frame_weights)

# ======================================== visualize PCA resutls as faces ========================================

if False:
    to_show = []
    for i in range(n_components):
        to_show.append(pca.components_[i] + pca.mean_)

    ps.init()
    ps.remove_all_structures()
    rows = max(n_components//5, 1)
    cols = min(n_components, 5)
    plot_multiple_faces(blendshapes, to_show, grid_size=[rows, cols], spacing=1)
    ps.show()

# ========================================= conduct PCA on clustered local manifolds ========================================
# How do we get PCA from latent space?
PROJ_ROOT = "/Users/evanpan/Documents/GitHub/ManifoldExploration/"

config = load_config(os.path.join(PROJ_ROOT, "experiments", "10-clusters"))
model  = load_model(config)

latent_frames = {}
for cluster_name in model.cluster_names:
    latent_frames[cluster_name] = []

for i in range(frame_weights.shape[0]):
    latents = model.encode(torch.from_numpy(frame_weights[i]).unsqueeze(0).to(device))
    for m, cluster_name in enumerate(model.cluster_names):
        latent_frames[cluster_name].append(latents[m].detach().numpy())

latent_frames = {k: np.array(v)[:, 0, :] for k, v in latent_frames.items()}


minimum_dim = 5
for i in range(0, len(model.cluster_names)):
    cluster_name_i = model.cluster_names[i]
    # latent_frames[cluster_name].shape
    local_manifold_components = min(minimum_dim, latent_frames[cluster_name_i].shape[1])
    pca = PCA(n_components=local_manifold_components)
    __ = pca.fit(latent_frames[cluster_name_i])
    mean = pca.mean_
    components = pca.components_
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    print(f"Cluster {cluster_name_i}: has {latent_frames[cluster_name_i].shape[0]} frames, {len(model.clustering[cluster_name_i])} dimensions")
    print(f"Cluster {cluster_name_i}: cumulative variance: {cumulative_variance}")

    # visualize the PCA results for the local_manifold_i
    to_show = []
    for i in range(local_manifold_components):
        component_i_latent = mean + components[i] 
        component_i_latent = torch.from_numpy(component_i_latent).to(device).unsqueeze(0)
        component_i_weight = model.decode_cluster_i(component_i_latent, cluster_name_i)
        component_i_weight = component_i_weight / component_i_weight.max()
        to_show.append(component_i_weight.detach().cpu().numpy()[0])


    


