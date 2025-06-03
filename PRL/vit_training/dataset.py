import json
import numpy as np
import torch
from torch.utils.data import Dataset

class SegmentationDataset3D(Dataset):
    def __init__(self, json_path):
        with open(json_path) as f:
            self.data_info = json.load(f)

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        item = self.data_info[idx]
        image = np.load(item["image"]).astype(np.float32)
        mask = np.load(item["mask"]).astype(np.int64)

        # Normalize and add channel dimension
        image = (image - image.mean()) / (image.std() + 1e-5)
        image = np.expand_dims(image, axis=0)

        return torch.from_numpy(image), torch.from_numpy(mask)
