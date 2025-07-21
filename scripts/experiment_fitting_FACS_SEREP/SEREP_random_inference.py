import os
from omegaconf import OmegaConf
import sys
sys.path.append("/code/speech2face_baselines/")
sys.path.append("/code/expressive-speech2face/")
sys.path.append("/code/facial-manifold-learning/")
from speech2face.mesh.utils import loadObj
from speech2face.models.spiral.get_model import get_model
from speech2face.scripts.id_exp_convert_tests import replace_on_cfg
import torch 
import numpy as np
from tqdm import trange
import polyscope as ps
from matplotlib import pyplot as plt
from src.web_visualizer_server import *
import pickle 

class DummyArgs:
    def __init__(self, input, output):
        self.config = "/code/models/id_exp_apply_model/config.yaml"
        self.checkpoint = "/code/models/id_exp_apply_model/checkpoint_epoch41.pth"
        self.neutral = "/code/models/S077_HSP_M_20/Head/S077_HSP_M_20_Head.obj"
        self.output = output
        self.input = input        
        self.scale = 0.01
        self.shift = (0, 169.44, 5.2)

faces_path = "/mnt/e/Projects/Ubi_Speech2face/models/flame_retopo/faces.pickle" 
FLAME_TINGS_ROOT = "/code/models/flame2ubi"
FLAME_IN_UBI_fname = "generic_model_ubito_flame_v2.pkl"
flame_lmk_path =  "landmark_embedding.npy"
FLAME_IN_UBI_fname = os.path.join(FLAME_TINGS_ROOT, FLAME_IN_UBI_fname)
flame_lmk_path = os.path.join(FLAME_TINGS_ROOT, flame_lmk_path)
flame_in_ubi_path = FLAME_IN_UBI_fname
args = DummyArgs(input=None, output=None)

config = OmegaConf.load(args.config)
replace_on_cfg(config)
config.model.data_root = r"/expnet_root"
device = torch.device('cuda')

# load model
model = get_model(config.model, device)
checkpoint = torch.load(args.checkpoint)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.to(device)

# load landmarks
with open(flame_in_ubi_path, 'rb') as f:
    flame_in_ubi = pickle.load(f, encoding="latin1")

flame_in_ubi["f"] # faces
flame_in_ubi["weights"].shape # vertices


flame_in_ubi.keys()
# load neutral mesh
batch_size = 3
neutral = loadObj(args.neutral)['verts']
neutral = (neutral - np.array(args.shift)) * args.scale
neutral = neutral.astype(np.float32)
neutral = torch.from_numpy(neutral).to(device)
mean, std = checkpoint['meanstd']
neutral = (neutral - mean) / std
neutral = neutral.tile(batch_size, 1, 1)
F = loadObj(args.neutral)["tris"]
# input code
code = torch.zeros([batch_size, 64], device=device)
code = torch.randn([batch_size, 64], device=device) * 0.1
mesh =  model.id_encoder(neutral, code) # [1, 13473, 3]
mesh = mesh * std + mean

# visualize the mesh in polyscope:
V = mesh.detach().cpu().numpy()[0]

# in jupyter
visualizer, task = run_visualizer()
visualizer.add_mesh("my_mesh flame in ubi", V+np.array([[0, 0.0, 0]]), F)
visualizer.add_mesh("flame", flame_in_ubi["v_template"]+np.array([[0, 0.3, 0]]), F)
