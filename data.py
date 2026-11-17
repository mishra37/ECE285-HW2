import numpy as np
import torch
from torch.utils.data import Dataset


class NpyImages(Dataset):
    """Images cached by preprocess.py, returned as float tensors in [-1, 1]."""

    def __init__(self, path, hflip=True):
        self.x = torch.from_numpy(np.load(path)).permute(0, 3, 1, 2).contiguous()  # (N,3,H,W) uint8
        self.hflip = hflip

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        img = self.x[i].float().div(127.5).sub(1.0)
        if self.hflip and torch.rand(()) < 0.5:
            img = img.flip(-1)
        return img
