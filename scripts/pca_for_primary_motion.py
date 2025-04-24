import os
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
import sys
sys.path.append("/Users/evanpan/Documents/GitHub/ManifoldExploration")
from scripts.plot_heads import *

def visualize(pca, n_components):
    # 1. Visualize explained variance
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
    plt.savefig('pca_variance.png')

    # 2. Visualize the first few principal components
    num_to_show = min(5, n_components)  # Show up to 5 components
    plt.figure(figsize=(15, 3 * num_to_show))
    for i in range(num_to_show):  # Show top components
        plt.subplot(num_to_show, 1, i+1)
        component = pca.components_[i]
        plt.bar(range(len(blendshapes)), component)
        plt.title(f'Principal Component {i+1} (Variance: {explained_variance[i]:.2%})')
        plt.ylabel('Weight')
        plt.xlim(-1, len(blendshapes))
    plt.tight_layout()
    plt.savefig('pca_components.png')

    # 3. Visualize the mean face
    plt.figure(figsize=(12, 4))
    plt.bar(range(len(blendshapes)), pca.mean_)
    plt.title('Mean Blendshape Weights')
    plt.xlabel('Blendshape Index')
    plt.ylabel('Weight')
    plt.tight_layout()
    plt.savefig('mean_blendshapes.png')

    # 4. Project data into principal component space
    projected_data = pca.transform(frame_weights)

    # Visualize the first two principal components (if we have at least 2)
    if n_components >= 2:
        plt.figure(figsize=(10, 8))
        plt.scatter(projected_data[:, 0], projected_data[:, 1], alpha=0.5)
        plt.xlabel(f'PC1 ({explained_variance[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({explained_variance[1]:.2%} variance)')
        plt.title('Data Projected onto First Two Principal Components')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('pca_projection_2d.png')

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

# perform PCA on the dataset
from sklearn.decomposition import PCA
n_components = 20
pca = PCA(n_components=n_components)
pca.fit(frame_weights)

to_show = []
for i in range(n_components):
    to_show.append(pca.components_[i] + pca.mean_)


ps.init()
ps.remove_all_structures()
rows = max(n_components//5, 1)
cols = min(n_components, 5)
plot_multiple_faces(blendshapes, cluster_heads, grid_size=[rows, cols], spacing=1)
ps.show()
