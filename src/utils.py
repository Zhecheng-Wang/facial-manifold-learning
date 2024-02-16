import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
import matplotlib.font_manager as fm
from blendshapes import BasicBlendshapes
import igl

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
path = f'{os.path.expanduser("~")}/.local/share/fonts/LinBiolinum_R.ttf'
# biolinum_font = fm.FontProperties(fname=path)
# sns.set(font=biolinum_font.get_name())
sns.set_theme()

def load_blendshape(model="ARKit"):
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
        "noseSneerRight"
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
    n_blendshapes = len(SP_BLENDSHAPE_MAPPING)
    V, F = igl.read_triangle_mesh(os.path.join(path, "neutral.obj"))
    blendshapes = np.zeros((n_blendshapes, *V.shape))
    names = []
    for i, (_, file_name) in enumerate(SP_BLENDSHAPE_MAPPING):
        file_path = os.path.join(path, file_name)
        VB, _ = igl.read_triangle_mesh(file_path)
        blendshapes[i] = VB - V
        names.append(file_name.split(".")[0])
    return BasicBlendshapes(V, F, blendshapes, names)

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
        for root, dirs, files in os.walk(dataset_path):
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
            data[c_n_frames:c_n_frames+frames.shape[0], :] = frames
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
                ('faceMuscles.frontalis',           "frontalis.obj"),
                ('faceMuscles.currogator',          "currogator.obj" ),
                ('faceMuscles.medFrontalis',        "medFrontalis.obj"),
                ('faceMuscles.orbOculi',            "orbOculi.obj"),
                ('faceMuscles.orbOculi_L',          "orbOculi_L.obj"),
                ('faceMuscles.orbOculi_R',          "orbOculi_R.obj"),
                ('faceMuscles.orbOculi_out',        "orbOculi_out.obj"),
                ('faceMuscles.latFrontalis',        "latFrontalis.obj"),
                ('faceMuscles.latFrontalis_L',      "latFrontalis_L.obj"),
                ('faceMuscles.latFrontalis_R',      "latFrontalis_R.obj"),
                ('faceMuscles.LabInf',              "LabInf.obj"),
                ('faceMuscles.zygomatic',           "zygomatic.obj" ),
                ('faceMuscles.labSup',              "labSup.obj"),
                ('faceMuscles.labSup_AN',           "labSup_AN.obj"),
                ('faceMuscles.triangularis',        "triangularis.obj"),
                ('faceMuscles.incisivus',           "incisivus.obj"),
                ('faceMuscles.mentalis',            "mentalis.obj"),
                ('faceMuscles.risoriusPlatysma',    "risoriusPlatysma.obj"),
                ('faceMuscles.orbOris_loose_lo',    "orbOris_loose_lo.obj"),
                ('faceMuscles.orbOris_loose_hi',    "orbOris_loose_hi.obj"),
                ('faceMuscles.orbOris_tight_lo',    "orbOris_tight_lo.obj"),
                ('faceMuscles.orbOris_tight_hi',    "orbOris_tight_hi.obj"),
                ('faceMuscles.orbOri0s_tight_hi2',  "orbOri0s_tight_hi2.obj"),
                ('faceMuscles.orbOris_tight_lo2',   "orbOris_tight_lo2.obj"),
                ('faceMuscles.mouthClose',          "mouthClose.obj"),
                ('faceMuscles.orbOculi_lo',         "orbOculi_lo.obj" ),
                ('faceMuscles.buccinator',          "buccinator.obj" ),
                ('faceMuscles.orbOculi_lo_L',       "orbOculi_lo_L.obj" ),
                ('faceMuscles.orbOculi_lo_R',       "orbOculi_lo_R.obj" ),
                ('faceMuscles.labSup_L',            "labSup_L.obj" ),
                ('faceMuscles.labSup_R',            "labSup_R.obj" ),
                ('faceMuscles.zygomatic_L',         "zygomatic_L.obj" ),
                ('faceMuscles.zygomatic_R',         "zygomatic_R.obj" ),
                ('faceMuscles.risoriusPlatysma_L',  "risoriusPlatysma_L.obj" ),
                ('faceMuscles.risoriusPlatysma_R',  "risoriusPlatysma_R.obj" ),
                ('faceMuscles.levAnguliOris',       "levAnguliOris.obj" ),
                ('faceMuscles.dilatorNaris',        "dilatorNaris.obj" ),
                ('faceMuscles.Zyg_Minor',           "Zyg_Minor.obj" ),
                ('faceMuscles.mentalis_lowerLip',   "mentalis_lowerLip.obj" ),
                ('faceMuscles.triangularis_L',      "triangularis_L.obj" ),
                ('faceMuscles.triangularis_R',      "triangularis_R.obj" ),
                ('faceMuscles.orbOris_up_hi',       "orbOris_up_hi.obj" ),
                ('faceMuscles.jawOpenComp',         "jawOpenComp.obj" ),
                ('faceMuscles.blow',                "blow.obj"),
                ('jaw.rotateZ',                     "jawUpDn.obj"),
                ('jaw.translateX',                  "jawInOut.obj"),
                ('blink_ctl.translateY_pos',        "topeyelids_up.obj"),
                ('blink_ctl.translateY_neg',        "topeyelids_down.obj"),
                ('loBlink_ctl.translateY_pos',      "btmeyelids_up.obj"),
                ('loBlink_ctl.translateY_neg',      "btmeyelids_down.obj")
            ]

def parse_SP_txt(txt_path):
    import pandas as pd
    df = pd.read_csv(txt_path, dtype=np.float32, index_col=0)
    del df["jaw.rotateY"]
    # split columns for blink_ctl.translateY and loBlink_ctl.translateY
    df["blink_ctl.translateY_pos"] = df["blink_ctl.translateY"].clip(lower=0)
    df["blink_ctl.translateY_neg"] = -df["blink_ctl.translateY"].clip(upper=0)
    df["loBlink_ctl.translateY_pos"] = df["loBlink_ctl.translateY"].clip(lower=0)
    df["loBlink_ctl.translateY_neg"] = -df["loBlink_ctl.translateY"].clip(upper=0)
    del df["blink_ctl.translateY"]
    del df["loBlink_ctl.translateY"]
    data = df.to_numpy()
    return data

def detect_keyframes(data, threshold=0.5):
    # detect keyframes
    keyframes = []
    for i in range(data.shape[0]-1):
        if np.linalg.norm(data[i+1] - data[i]) > threshold:
            keyframes.append(i)
    return keyframes

def parse_SP_dataset(dataset_path):
    bin_dataset_path = os.path.join(dataset_path, "data.npy")
    if not os.path.exists(bin_dataset_path):
        sequences = []
        n_frames = 0
        for root, dirs, files in os.walk(dataset_path):
            for file in sorted(files):
                if file.endswith("Charles.txt"):
                    sequences.append(file)
                    df = parse_SP_txt(os.path.join(root, file))
                    n_frames += df.shape[0]
        n_controller = len(SP_BLENDSHAPE_MAPPING)
        data = np.zeros((n_frames, n_controller), dtype=np.float32)
        c_n_frames = 0
        for seq in sequences:
            df = parse_SP_txt(os.path.join(dataset_path, seq))
            data[c_n_frames:c_n_frames+df.shape[0], :] = df
            c_n_frames += df.shape[0]
        np.save(os.path.join(dataset_path, "data"), data)
    else:
        data = np.load(bin_dataset_path)
    return data

class SPDeltaWeightDataset(Dataset):
    def __init__(self):
        dataset_path = os.path.join(PROJ_ROOT, "data", "SP", "dataset")
        self.data = parse_SP_dataset(dataset_path)
        # reverse the data in the first dimension
        data2 = self.data[::-1]
        # concate data and data2 in dimension 0
        self.data = np.concatenate([self.data, data2], axis=0)
        self.n_frames = self.data.shape[0]
        self.data = torch.from_numpy(self.data).to(torch.float32)
    def __len__(self):
        return self.n_frames
    def __getitem__(self, idx):
        return (self.data[idx] - self.data[max(0, idx-1)])

class SPKeyframeDataset(Dataset):
    def __init__(self):
        dataset_path = os.path.join(PROJ_ROOT, "data", "SP", "dataset")
        self.data = parse_SP_dataset(dataset_path)
        self.keyframes = detect_keyframes(self.data)
        self.n_frames = len(self.keyframes)
        self.data = torch.from_numpy(self.data).to(torch.float32)
    
    def __len__(self):
        return self.n_frames
    
    def __getitem__(self, idx):
        return self.data[self.keyframes[idx]]

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
    
class SingleActivationDataset(Dataset):
    def __init__(self, n_blendshapes):
        self.data = torch.zeros((n_blendshapes+1, n_blendshapes))
        for i in range(n_blendshapes):
            self.data[i, i] = 1.0
        self.n_frames = self.data.shape[0]
    
    def __len__(self):
        return self.n_frames
    
    def __getitem__(self, idx):
        return self.data[idx], 

def load_dataset(batch_size=32, dataset="BEAT", augment=False):
    dataset_name = dataset
    from torch.utils.data import ConcatDataset
    if dataset_name == "BEAT":
        dataset = BEATDataset()
    elif dataset_name == "SP":
        dataset = SPDataset()
    elif dataset_name == "SPKeyframe":
        dataset = SPKeyframeDataset()
    elif dataset_name == "SPDeltaWeight":
        dataset = SPDeltaWeightDataset()
    else:
        raise NotImplementedError
    print(f"{dataset_name} dataset size: {len(dataset)}")
    if augment:
        n_blendshapes = dataset.data.shape[1]
        augment_dataset = SingleActivationDataset(n_blendshapes)
        print(f"Augment dataset size: {len(augment_dataset)}")
        dataset = ConcatDataset([dataset, augment_dataset])
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

def random_sample(dataset, n_samples):
    if isinstance(dataset, DataLoader):
        dataset = dataset.dataset
    return dataset[np.random.choice(len(dataset), n_samples, replace=False)].numpy()

def load_config(path):
    json_path = os.path.join(path, "config.json")
    return json.load(open(json_path, "r"))

def model_exists(path):
    return os.path.exists(os.path.join(path, "model.pt"))

def compute_error(weights, weights_gt):
    return np.linalg.norm(weights - weights_gt, axis=1).mean()