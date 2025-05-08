import pickle
import numpy as np
import torch 
import sys
sys.path.append("/Users/evanpan/Documents/GitHub/ManifoldExploration/src")
from blendshapes import FLAMEBlendshapes
import polyscope as ps
import polyscope.imgui as psim

flame = FLAMEBlendshapes()

# load the pickle_dataset:
dataset_path = "/Users/evanpan/Documents/GitHub/ManifoldExploration/data/MeadRavdess/val_mead_ravdess_0.1.pickle"
with open(dataset_path, "rb") as f:
    data = pickle.load(f)

# get the keys of the data
data_keys = list(data.keys())

# get one sample
sample = data[data_keys[0]]
exp = sample["exp"][0]
jaw = sample["jaw"][0]
weight = np.concatenate([exp, jaw], axis=1)

# animate it in polyscope 

# convert to vertex space
v_sample_i = flame.V
v_sample_i = np.expand_dims(v_sample_i, axis=0)
v_sample_i = [v_sample_i]
v_sample_i[0].shape
for i in range(0, len(weight)):
    v_sample_i.append(np.expand_dims(flame.eval(weight[i]), axis=0))
v_sample_i = np.concatenate(v_sample_i, axis=0)
v_sample_i.shape