import os
import sys
import numpy as np
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))
FIGURES_FOLDER = os.path.join(PROJ_ROOT, "figures")
FIGURE_NAME = os.path.dirname(os.path.abspath(__file__))

import PIL
import matplotlib.pyplot as plt

IMAGES_FOLDER = os.path.join(FIGURES_FOLDER, FIGURE_NAME, "cluster_render")
# load all images in the folder
images = []
for root, dirs, files in os.walk(IMAGES_FOLDER):
    for file in files:
        if file.endswith(".png"):
            images.append(PIL.Image.open(os.path.join(root, file)))
            
# plot
n_images = len(images)
fig, axes = plt.subplots(1, n_images, figsize=(n_images*4, n_images))
for i, image in enumerate(images):
    w, h = image.size
    diff = 0.35 * w
    image = image.crop((diff, 100, w-diff, h-100))
    axes[i].imshow(image)
    axes[i].axis("off")
    
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_FOLDER, FIGURE_NAME, "cluster.png"), dpi=300, transparent=True)
plt.savefig(os.path.join(FIGURES_FOLDER, FIGURE_NAME, "cluster.pdf"), dpi=300, transparent=True)
