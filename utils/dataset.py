import cv2
import os
import torch
from torch.utils.data import Dataset
import numpy as np

class UCSDFrameDataset(Dataset):
    def __init__(self, root_folder):
        # root_folder = "data/ucsd_ped2/Train"
        self.paths = []

        folders = sorted(os.listdir(root_folder))
        for fold in folders:
            folder_path = os.path.join(root_folder, fold)
            if not os.path.isdir(folder_path):
                continue

            for fname in sorted(os.listdir(folder_path)):
                if fname.endswith(".tif"):
                    self.paths.append(os.path.join(folder_path, fname))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))
        img = img.astype("float32") / 255.0

        img = torch.tensor(img).unsqueeze(0)  # shape: (1, H, W)
        return img
