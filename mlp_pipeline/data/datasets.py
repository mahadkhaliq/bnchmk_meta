"""Torch Dataset wrapper shared by both variants.

The original notebook had separate ADMDataset (flat) and GridDataset classes,
but they are identical apart from the shape of `x`. A single wrapper works for
both: the baseline passes x of shape (n, INPUT_DIM) and the neighbourhood model
passes x of shape (n, GRID_N, GRID_N, CHANNELS).
"""
import torch
from torch.utils.data import Dataset


class ArrayDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
