import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
from blendshapes import BasicBlendshapes
import igl
import pandas as pd
import sys
<<<<<<< HEAD
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

=======
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
>>>>>>> cvae
from scripts.SMOTE import *


PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
path = f'{os.path.expanduser("~")}/.local/share/fonts/LinBiolinum_R.ttf'
# biolinum_font = fm.FontProperties(fname=path)
# sns.set(font=biolinum_font.get_name())
sns.set_theme()


def load_blendshape(model: str = "ARKit"):
    if model == "ARKit":
        return load_ARKit_blendshape()
    elif model == "SP":
        return load_SP_blendshape()
    else:
        raise NotImplementedError


def load_ARKit_blendshape():
    path = os.path.join(PROJ_ROOT, "data", "AppleAR", "OBJs")
    blendshape_names = [
        "browDownLeft",
        "browDownRight",
        "browInnerUp",
        "browOuterUpLeft",
        "browOuterUpRight",
        "cheekPuff",
        "cheekSquintLeft",
        "cheekSquintRight",
        "eyeBlinkLeft",
        "eyeBlinkRight",
        "eyeLookDownLeft",
        "eyeLookDownRight",
        "eyeLookInLeft",
        "eyeLookInRight",
        "eyeLookOutLeft",
        "eyeLookOutRight",
        "eyeLookUpLeft",
        "eyeLookUpRight",
        "eyeSquintLeft",
        "eyeSquintRight",
        "eyeWideLeft",
        "eyeWideRight",
        "jawForward",
        "jawLeft",
        "jawOpen",
        "jawRight",
        "mouthClose",
        "mouthDimpleLeft",
        "mouthDimpleRight",
        "mouthFrownLeft",
        "mouthFrownRight",
        "mouthFunnel",
        "mouthLeft",
        "mouthLowerDownLeft",
        "mouthLowerDownRight",
        "mouthPressLeft",
        "mouthPressRight",
        "mouthPucker",
        "mouthRight",
        "mouthRollLower",
        "mouthRollUpper",
        "mouthShrugLower",
        "mouthShrugUpper",
        "mouthSmileLeft",
        "mouthSmileRight",
        "mouthStretchLeft",
        "mouthStretchRight",
        "mouthUpperUpLeft",
        "mouthUpperUpRight",
        "noseSneerLeft",
        "noseSneerRight",
    ]
    neutral_path = os.path.join(path, "Neutral.obj")
    V, F = igl.read_triangle_mesh(neutral_path)
    N_BLENDSHAPES = len(blendshape_names)
    blendshapes = np.zeros((N_BLENDSHAPES, *V.shape))
    for i, blendshape_name in enumerate(blendshape_names):
        blendshape_path = os.path.join(path, f"{blendshape_name}.obj")
        assert os.path.exists(blendshape_path)
        VB, _ = igl.read_triangle_mesh(blendshape_path)
        blendshapes[i] = VB - V
    return BasicBlendshapes(V, F, blendshapes, blendshape_names)


def load_SP_blendshape():
    path = os.path.join(PROJ_ROOT, "data", "SP", "blendshapes")
    valid_names_path = os.path.join(path, "valid_names.txt")
    V, F = igl.read_triangle_mesh(os.path.join(path, "neutral.obj"))
    valid_names = []
    sequences = []
    if not os.path.exists(valid_names_path):
        dataset_path = os.path.join(PROJ_ROOT, "data", "SP", "dataset")
        df = pd.DataFrame()
        for root, _, files in os.walk(dataset_path):
            for file in sorted(files):
                if file.endswith(".txt"):
                    sequences.append(file)
                    # append every text file to the data frame
                    df = pd.concat([df, parse_SP_txt(os.path.join(root, file))])
        # cull invalid columns where all values are zero
        df = df.loc[:, (df != 0).any(axis=0)]
        # save valid names
        valid_names = df.columns
        with open(valid_names_path, "w") as f:
            for name in valid_names:
                f.write(f"{name}\n")
    else:
        with open(valid_names_path, "r") as f:
            valid_names = f.read().splitlines()
    blendshapes = []
    for _, (blendshape_name, file_name) in enumerate(SP_BLENDSHAPE_MAPPING):
        if blendshape_name not in valid_names:
            continue
        file_path = os.path.join(path, file_name)
        VB, _ = igl.read_triangle_mesh(file_path)
        blendshapes.append(VB - V)
    blendshapes = np.array(blendshapes)
    return BasicBlendshapes(V, F, blendshapes, valid_names)


def parse_BEAT_json(json_path):
    j = json.load(open(json_path))
    n_frames = len(j["frames"])
    n_controller = len(j["names"])
    data = np.zeros((n_frames, n_controller), dtype=np.float32)
    for i in range(n_frames):
        data[i, :] = np.array(j["frames"][i]["weights"])
    return data


def parse_BEAT_dataset(dataset_path):
    bin_dataset_path = os.path.join(dataset_path, "data.npy")
    if not os.path.exists(bin_dataset_path):
        sequences = []
        n_frames = 0
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.endswith(".json"):
                    sequences.append(file)
                    j = json.load(open(os.path.join(root, file)))
                    n_frames += len(j["frames"])
        n_controller = len(j["names"])
        data = np.zeros((n_frames, n_controller), dtype=np.float32)
        c_n_frames = 0
        for seq in sequences:
            json_path = os.path.join(dataset_path, seq)
            frames = parse_BEAT_json(json_path)
            data[c_n_frames : c_n_frames + frames.shape[0], :] = frames
            c_n_frames += frames.shape[0]
        np.save(os.path.join(dataset_path, "data"), data)
    else:
        data = np.load(bin_dataset_path)
    return data


class BEATDataset(Dataset):
    def __init__(self):
        dataset_path = os.path.join(PROJ_ROOT, "data", "BEAT", "1")
        self.data = parse_BEAT_dataset(dataset_path)
        self.n_frames = self.data.shape[0]
        self.data = torch.from_numpy(self.data).to(torch.float32)

    def __len__(self):
        return self.n_frames

    def __getitem__(self, idx):
        return self.data[idx]


SP_BLENDSHAPE_MAPPING = [
    ("faceMuscles.frontalis", "frontalis.obj"),
    ("faceMuscles.currogator", "currogator.obj"),
    ("faceMuscles.medFrontalis", "medFrontalis.obj"),
    ("faceMuscles.orbOculi", "orbOculi.obj"),
    ("faceMuscles.orbOculi_L", "orbOculi_L.obj"),
    ("faceMuscles.orbOculi_R", "orbOculi_R.obj"),
    ("faceMuscles.orbOculi_out", "orbOculi_out.obj"),
    ("faceMuscles.latFrontalis", "latFrontalis.obj"),
    ("faceMuscles.latFrontalis_L", "latFrontalis_L.obj"),
    ("faceMuscles.latFrontalis_R", "latFrontalis_R.obj"),
    ("faceMuscles.LabInf", "LabInf.obj"),
    ("faceMuscles.zygomatic", "zygomatic.obj"),
    ("faceMuscles.labSup", "labSup.obj"),
    ("faceMuscles.labSup_AN", "labSup_AN.obj"),
    ("faceMuscles.triangularis", "triangularis.obj"),
    ("faceMuscles.incisivus", "incisivus.obj"),
    ("faceMuscles.mentalis", "mentalis.obj"),
    ("faceMuscles.risoriusPlatysma", "risoriusPlatysma.obj"),
    ("faceMuscles.orbOris_loose_lo", "orbOris_loose_lo.obj"),
    ("faceMuscles.orbOris_loose_hi", "orbOris_loose_hi.obj"),
    ("faceMuscles.orbOris_tight_lo", "orbOris_tight_lo.obj"),
    ("faceMuscles.orbOris_tight_hi", "orbOris_tight_hi.obj"),
    ("faceMuscles.orbOri0s_tight_hi2", "orbOri0s_tight_hi2.obj"),
    ("faceMuscles.orbOris_tight_lo2", "orbOris_tight_lo2.obj"),
    ("faceMuscles.mouthClose", "mouthClose.obj"),
    ("faceMuscles.orbOculi_lo", "orbOculi_lo.obj"),
    ("faceMuscles.buccinator", "buccinator.obj"),
    ("faceMuscles.orbOculi_lo_L", "orbOculi_lo_L.obj"),
    ("faceMuscles.orbOculi_lo_R", "orbOculi_lo_R.obj"),
    ("faceMuscles.labSup_L", "labSup_L.obj"),
    ("faceMuscles.labSup_R", "labSup_R.obj"),
    ("faceMuscles.zygomatic_L", "zygomatic_L.obj"),
    ("faceMuscles.zygomatic_R", "zygomatic_R.obj"),
    ("faceMuscles.risoriusPlatysma_L", "risoriusPlatysma_L.obj"),
    ("faceMuscles.risoriusPlatysma_R", "risoriusPlatysma_R.obj"),
    ("faceMuscles.levAnguliOris", "levAnguliOris.obj"),
    ("faceMuscles.dilatorNaris", "dilatorNaris.obj"),
    ("faceMuscles.Zyg_Minor", "Zyg_Minor.obj"),
    ("faceMuscles.mentalis_lowerLip", "mentalis_lowerLip.obj"),
    ("faceMuscles.triangularis_L", "triangularis_L.obj"),
    ("faceMuscles.triangularis_R", "triangularis_R.obj"),
    ("faceMuscles.orbOris_up_hi", "orbOris_up_hi.obj"),
    ("faceMuscles.jawOpenComp", "jawOpenComp.obj"),
    ("faceMuscles.blow", "blow.obj"),
    ("jaw.rotateZ", "jawUpDn.obj"),
    ("jaw.translateX", "jawInOut.obj"),
    ("blink_ctl.translateY_pos", "topeyelids_up.obj"),
    ("blink_ctl.translateY_neg", "topeyelids_down.obj"),
    ("loBlink_ctl.translateY_pos", "btmeyelids_up.obj"),
    ("loBlink_ctl.translateY_neg", "btmeyelids_down.obj"),
]


def parse_SP_txt(txt_path):
    df = pd.read_csv(txt_path, dtype=np.float32, index_col=0)
    del df["jaw.rotateY"]
    # split columns for blink_ctl.translateY and loBlink_ctl.translateY
    df["blink_ctl.translateY_pos"] = df["blink_ctl.translateY"].clip(lower=0)
    df["blink_ctl.translateY_neg"] = -df["blink_ctl.translateY"].clip(upper=0)
    df["loBlink_ctl.translateY_pos"] = df["loBlink_ctl.translateY"].clip(lower=0)
    df["loBlink_ctl.translateY_neg"] = -df["loBlink_ctl.translateY"].clip(upper=0)
    del df["blink_ctl.translateY"]
    del df["loBlink_ctl.translateY"]
    return df


def parse_SP_dataset(dataset_path):
    bin_dataset_path = os.path.join(dataset_path, "data.npy")
    if not os.path.exists(bin_dataset_path):
        sequences = []
        # make a empty data frame
        df = pd.DataFrame()
        for root, _, files in os.walk(dataset_path):
            for file in sorted(files):
                if file.endswith(".txt"):
                    sequences.append(file)
                    # append every text file to the data frame
                    df = pd.concat([df, parse_SP_txt(os.path.join(root, file))])
        # cull invalid columns where all values are zero
        df = df.loc[:, (df != 0).any(axis=0)]
        data = df.to_numpy()
        np.save(os.path.join(dataset_path, "data"), data)
    else:
        data = np.load(bin_dataset_path)
    return data


def load_smote_dataset(cluster_n=15, alpha_and_mask=False):
    # load dataset:
    dataset = load_dataset(
        batch_size=32,
        dataset="SP",
    )

    # load smote related data
    cluster_n = 15
    smote_X = dataset.dataset.base.data.numpy()
    try:
        smote_Y_path = os.path.join(
            "data", "SP", f"k={cluster_n}_cluster_assignment.npy"
        )
    except:
        print(f"No cluster assignment file found at {smote_Y_path}")
        raise FileNotFoundError

    smote_Y = np.load(smote_Y_path)
    smote_dataset = BalancedSMOTEDataset(
        smote_X,
        smote_Y,
        balance_strategy="both",
        k=cluster_n,
        alpha_and_mask=alpha_and_mask,
    )
    return smote_dataset


class SPDataset(Dataset):
    def __init__(self):
        dataset_path = os.path.join(PROJ_ROOT, "data", "SP", "dataset")
        self.data = parse_SP_dataset(dataset_path)
        self.n_frames = self.data.shape[0]
        self.data = torch.from_numpy(self.data).to(torch.float32)

    def __len__(self):
        return self.n_frames

    def __getitem__(self, idx):
        return self.data[idx]


class DynamicMaskDataset(Dataset):
    def __init__(self, base_dataset):
        """
        base_dataset: yields clean w_gt (torch.Tensor shape [m])
        """
        self.base = base_dataset
        self.m = self.base.data.shape[1]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        w_gt = self.base[idx]  # torch.Tensor [m]
        # 1) pick which blendshape the user is editing
        selected_id = torch.randint(0, self.m, (1,), dtype=torch.long)
        # 2) pick cutoff alpha in [0,1]
        alpha = torch.rand(1)
        return w_gt, selected_id, alpha


def load_dataset(batch_size: int = 32, dataset: str = "SP"):
    """
    Returns a DataLoader yielding:
        (w_gt: Tensor [B, m],
         selected_id: Tensor [B, 1],
         alpha: Tensor [B, 1])
    """
    if dataset == "SP":
        from utils import SPDataset

        base = SPDataset()
        ds = DynamicMaskDataset(base)
    elif dataset == "SP_SMOTE":
        ds = load_smote_dataset(alpha_and_mask=True)
    else:
        raise NotImplementedError(f"Dataset '{dataset}' not supported.")
    print(f"Loaded {dataset} dataset with {len(ds)} samples")
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)


def load_config(path):
    json_path = os.path.join(path, "config.json")
    return json.load(open(json_path, "r"))


def model_exists(path):
    return os.path.exists(os.path.join(path, "model.pt"))


def compute_error(weights, weights_gt):
    return np.linalg.norm(weights - weights_gt, axis=1).mean()
