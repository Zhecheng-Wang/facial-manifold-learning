import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
path = '/home/zhecheng/.local/share/fonts/LinBiolinum_R.ttf'
biolinum_font = fm.FontProperties(fname=path)
sns.set(font=biolinum_font.get_name())
sns.set_theme()

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

def load_dataset(batch_size=32):
    dataset = BEATDataset()
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