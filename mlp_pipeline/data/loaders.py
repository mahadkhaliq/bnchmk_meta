"""DataLoader builders for both variants.

    load_flat()  -> baseline (no neighbourhood). x stays a flat (n, INPUT_DIM) vector.
    load_grid()  -> neighbourhood model. x is reshaped to a (n, N, N, C) grid.

Both read the same 4 CSVs (paths in config.py), fit normalisation on the train
split, apply it to the test split, and carve out a 20% validation split with a
fixed seed (random_state=0) so runs are comparable.
"""
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import config
from data.normalize import normalize
from data.datasets import ArrayDataset
from data.grid import build_grid


def _read_csvs():
    train_x = pd.read_csv(config.TRAIN_X_PATH, header=None).astype("float32").values
    train_y = pd.read_csv(config.TRAIN_Y_PATH, header=None).astype("float32").values
    test_x  = pd.read_csv(config.TEST_X_PATH,  header=None).astype("float32").values
    test_y  = pd.read_csv(config.TEST_Y_PATH,  header=None).astype("float32").values
    return train_x, train_y, test_x, test_y


def load_flat(batch_size=config.BATCH_SIZE):
    """Baseline loaders. Returns (train_loader, val_loader, test_x, test_y)."""
    train_x, train_y, test_x, test_y = _read_csvs()

    train_x, x_max, x_min = normalize(train_x)        # fit on train
    test_x, _, _ = normalize(test_x, x_max, x_min)     # apply to test

    x_train, x_val, y_train, y_val = train_test_split(
        train_x, train_y, test_size=0.2, random_state=0)

    train_loader = DataLoader(ArrayDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(ArrayDataset(x_val,   y_val),   batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_x, test_y


def load_grid(batch_size=config.BATCH_SIZE):
    """Neighbourhood loaders. Returns (train_loader, val_loader, test_grid, test_y)."""
    train_x, train_y, test_x, test_y = _read_csvs()

    train_grid, x_max, x_min = build_grid(
        train_x, grid_n=config.GRID_N, channels=config.CHANNELS)          # fit on train
    test_grid, _, _ = build_grid(
        test_x, x_max, x_min, grid_n=config.GRID_N, channels=config.CHANNELS)  # apply to test

    xg_train, xg_val, y_train, y_val = train_test_split(
        train_grid, train_y, test_size=0.2, random_state=0)

    train_loader = DataLoader(ArrayDataset(xg_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(ArrayDataset(xg_val,   y_val),   batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_grid, test_y
