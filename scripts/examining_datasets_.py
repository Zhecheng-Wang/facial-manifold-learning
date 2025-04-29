# we need to answer the question: "in faces, is it common for one part to be highly similar, and other parts to be highly dissimilar?"

import os
import json
import torch
import sys
import numpy as np
import time
sys.path.append("/Users/evanpan/Documents/GitHub/ManifoldExploration/src")
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import load_dataset, load_blendshape
from model import build_model, save_model, KL_divergence
from clustering import compute_jaccard_similarity

loader = load_dataset(
        batch_size=1,
        dataset="SP",
    )
# dataset
dataset = loader.dataset


S = 30
D = 5
epsilon = 0.02
thre = 0.2
samples = 200

all_similarity = {}
all_dissimilarity = {}
for D in [1, 3, 5, 7, 10]:
    all_similarity_for_D = []
    all_dissimilarity_for_D = []
    # for n in range(0, len(dataset)):
    for n_idx in range(0, samples):
        n = np.random.randint(0, len(dataset))
        w_n, __, __ = dataset[n]
        w_n = w_n.numpy()
        similar_with_n = 0
        similar_but_different_with_n = 0 
        for m in range(0, len(dataset)):
            w_m, __, __ = dataset[m]
            w_m = w_m.numpy()
            # see if two faces are similar
            diff = np.abs(w_n - w_m)
            num_of_similar_weights = np.where(diff < epsilon, 1, 0)
            num_of_similar_weights = np.sum(num_of_similar_weights)
            if num_of_similar_weights >= S:
                similar_with_n+=1 
                # see if two faces have enough sufficiently different weights
                num_of_dissimilar_weights = np.where(diff > thre, 1, 0)
                num_of_dissimilar_weights = np.sum(num_of_dissimilar_weights)
                if num_of_dissimilar_weights >= D:
                    similar_but_different_with_n+=1
        print("similar_with_n: ", similar_with_n, "for D: ", D, "with n: ", n)
        print("similar_but_different_with_n: ", similar_but_different_with_n, "for D: ", D, "with n: ", n)
        all_similarity_for_D.append(similar_with_n/len(dataset))
        all_dissimilarity_for_D.append(similar_but_different_with_n/len(dataset))
    all_similarity[D] = all_similarity_for_D
    all_dissimilarity[D] = all_dissimilarity_for_D


    print(n, "similar_with_n: ", similar_with_n)
    print(n, "similar_but_different_with_n: ", similar_but_different_with_n)

##############################################################
################### Plotting the results #####################
##############################################################

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Sample data - a dictionary of lists
data = all_dissimilarity

# Set the style
sns.set_style("whitegrid")
plt.figure(figsize=(12, 8))

# Create the boxplot
ax = sns.boxplot(data=data, palette="viridis", width=0.6, linewidth=2)


print(data.keys())

# Customize the plot
plt.title('Percentage of configurations that are very similar but differs significantly at D blendshapes', fontsize=18, pad=20)
plt.xlabel('D=', fontsize=14, labelpad=10)
plt.ylabel('%', fontsize=14, labelpad=10)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Adding a subtle background color
ax.set_facecolor('#f8f9fa')

# Adding a grid for better readability
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add median values as text on each box

# Tight layout and show
plt.tight_layout()
plt.show()

