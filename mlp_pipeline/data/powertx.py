"""Data loading for the power-transmission .npz datasets.

Separate from data/loaders.py (which handles the ADM CSVs). Reuses the shared
normalize() and ArrayDataset; the model and engine code are dataset agnostic.

Each .npz holds:
    params   (n, P)  float64   geometry, grouped BY CHANNEL (d,g,l,w) then cell
    T_clean  (n, F)  float32   power transmission |S21|^2 in [0,1]  -> the target

Because there is a single file (no separate test set) we carve out a held-out
test split first, then a train/val split from the remainder.
"""
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import config_powertx as C
from data.normalize import normalize
from data.datasets import ArrayDataset


def _load_npz():
    d = np.load(C.NPZ_PATH, allow_pickle=True)
    X = d["params"].astype("float32")     # (n, P)
    Y = d["T_clean"].astype("float32")    # (n, F)
    return X, Y


def build_grid_powertx(params, x_max=None, x_min=None, grid_n=2, channels=4):
    """flat params (n, P) -> normalised grid (n, grid_n, grid_n, channels).

    Columns are grouped BY CHANNEL (d, g, l, w) then row-major cell, i.e.
        column = channel * n_cells + (r * grid_n + c)
    Normalisation (min-max to [-1, 1] per column) is applied BEFORE reshaping.
    """
    params, x_max, x_min = normalize(params, x_max, x_min)
    n = params.shape[0]
    n_cells = grid_n * grid_n
    grid = np.zeros((n, grid_n, grid_n, channels), dtype="float32")
    for ch in range(channels):
        for k in range(n_cells):
            r, c = k // grid_n, k % grid_n
            grid[:, r, c, ch] = params[:, ch * n_cells + k]
    return grid, x_max, x_min


def load_flat(batch_size=None):
    """Baseline loaders. Returns (train_loader, val_loader, test_x, test_y)."""
    batch_size = batch_size or C.BATCH_SIZE
    X, Y = _load_npz()

    x_tr, test_x, y_tr, test_y = train_test_split(X, Y, test_size=C.TEST_SPLIT, random_state=0)

    x_tr, x_max, x_min = normalize(x_tr)             # fit on train
    test_x, _, _ = normalize(test_x, x_max, x_min)   # apply to test

    x_train, x_val, y_train, y_val = train_test_split(x_tr, y_tr, test_size=0.2, random_state=0)

    train_loader = DataLoader(ArrayDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(ArrayDataset(x_val,   y_val),   batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_x, test_y


def load_grid(batch_size=None):
    """Neighbourhood loaders. Returns (train_loader, val_loader, test_grid, test_y)."""
    batch_size = batch_size or C.BATCH_SIZE
    X, Y = _load_npz()

    x_tr, test_x, y_tr, test_y = train_test_split(X, Y, test_size=C.TEST_SPLIT, random_state=0)

    train_grid, x_max, x_min = build_grid_powertx(x_tr, grid_n=C.GRID_N, channels=C.CHANNELS)
    test_grid, _, _ = build_grid_powertx(test_x, x_max, x_min, grid_n=C.GRID_N, channels=C.CHANNELS)

    xg_train, xg_val, y_train, y_val = train_test_split(train_grid, y_tr, test_size=0.2, random_state=0)

    train_loader = DataLoader(ArrayDataset(xg_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(ArrayDataset(xg_val,   y_val),   batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_grid, test_y
