"""Data loading for the power-transmission .npz datasets.

Separate from data/loaders.py (which handles the ADM CSVs). Reuses the shared
normalize() and ArrayDataset; the model and engine code are dataset agnostic.

Two on-disk schemas are supported (auto-detected from the npz keys):
    legacy (v1/v2): params  (n, P)  geometry grouped BY CHANNEL (d,g,l,w) then cell
                    T_clean (n, F)  power transmission |S21|^2 in [0,1]  -> target
    v3            : geom    (n, P)  geometry grouped BY ATOM, [d,l,w,g] per cell
                    T       (n, F)  preprocessed power transmission in [0,1] -> target

The flat baseline is order-agnostic (it learns whatever consistent column order
it is given), so only the grid path cares about the layout difference.

Because there is a single file (no separate test set) we carve out a held-out
test split first, then a train/val split from the remainder.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import config_powertx as C
from data.normalize import normalize
from data.datasets import ArrayDataset


def _load_npz():
    """Return (X, Y, schema) with schema in {'legacy', 'v3'} (auto-detected)."""
    with np.load(C.NPZ_PATH, allow_pickle=True) as d:
        if "geom" in d and "T" in d:                      # v3 (by-atom, [d,l,w,g])
            X = d["geom"].astype("float32")
            Y = d["T"].astype("float32")
            schema = "v3"
        elif "params" in d and "T_clean" in d:
            X = d["params"].astype("float32")
            Y = d["T_clean"].astype("float32")
            schema = "legacy"
        else:
            raise ValueError(
                f"{C.NPZ_PATH} has unsupported keys: {sorted(d.files)}"
            )

    if X.ndim != 2 or Y.ndim != 2 or len(X) != len(Y):
        raise ValueError(f"Invalid data shapes: X={X.shape}, Y={Y.shape}")
    if X.shape[1] != C.INPUT_DIM or Y.shape[1] != C.OUTPUT_DIM:
        raise ValueError(
            f"Data/config mismatch for {C.GRID}: X={X.shape}, Y={Y.shape}, "
            f"expected (*,{C.INPUT_DIM}) and (*,{C.OUTPUT_DIM})"
        )
    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError(f"{C.NPZ_PATH} contains NaN or infinite values")
    if Y.min() < -1e-6 or Y.max() > 1.0 + 1e-6:
        raise ValueError(
            f"Transmission target is outside [0,1]: min={Y.min()}, max={Y.max()}"
        )
    return X, Y, schema


def _split(X, Y):
    """Deterministic 68/17/15 train/validation/test split."""
    x_fit, test_x, y_fit, test_y = train_test_split(
        X, Y, test_size=C.TEST_SPLIT, random_state=C.SEED
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_fit, y_fit, test_size=0.2, random_state=C.SEED
    )
    return x_train, x_val, y_train, y_val, test_x, test_y


def _loader(x, y, batch_size, shuffle):
    generator = torch.Generator().manual_seed(C.SEED) if shuffle else None
    return DataLoader(
        ArrayDataset(x, y),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def _grid_from_geom(geom, x_max=None, x_min=None):
    """v3 grid: geom is already by-atom row-major with channel order [d,l,w,g],
    so (normalised) geom reshapes directly to (n, grid_n, grid_n, channels)."""
    geom, x_max, x_min = normalize(geom, x_max, x_min)
    n = geom.shape[0]
    grid = geom.reshape(n, C.GRID_N, C.GRID_N, C.CHANNELS).astype("float32")
    return grid, x_max, x_min


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
    X, Y, _ = _load_npz()
    x_train, x_val, y_train, y_val, test_x, test_y = _split(X, Y)

    x_train, x_max, x_min = normalize(x_train)       # fit on train only
    x_val, _, _ = normalize(x_val, x_max, x_min)
    test_x, _, _ = normalize(test_x, x_max, x_min)   # apply to test

    train_loader = _loader(x_train, y_train, batch_size, shuffle=True)
    val_loader = _loader(x_val, y_val, batch_size, shuffle=False)
    return train_loader, val_loader, test_x, test_y


def load_grid(batch_size=None):
    """Neighbourhood loaders. Returns (train_loader, val_loader, test_grid, test_y)."""
    batch_size = batch_size or C.BATCH_SIZE
    X, Y, schema = _load_npz()
    x_train, x_val, y_train, y_val, test_x, test_y = _split(X, Y)

    if schema == "v3":
        # v3 geom is already by-atom row-major with [d,l,w,g] -> reshape directly
        train_grid, x_max, x_min = _grid_from_geom(x_train)
        val_grid, _, _ = _grid_from_geom(x_val, x_max, x_min)
        test_grid, _, _ = _grid_from_geom(test_x, x_max, x_min)
    else:
        # legacy: columns grouped BY CHANNEL (d,g,l,w) then cell -> mapped loop
        train_grid, x_max, x_min = build_grid_powertx(
            x_train, grid_n=C.GRID_N, channels=C.CHANNELS
        )
        val_grid, _, _ = build_grid_powertx(
            x_val, x_max, x_min, grid_n=C.GRID_N, channels=C.CHANNELS
        )
        test_grid, _, _ = build_grid_powertx(test_x, x_max, x_min, grid_n=C.GRID_N, channels=C.CHANNELS)

    train_loader = _loader(train_grid, y_train, batch_size, shuffle=True)
    val_loader = _loader(val_grid, y_val, batch_size, shuffle=False)
    return train_loader, val_loader, test_grid, test_y


def get_freq_axis():
    """Return the npz freq_GHz axis (float array) if present, else None.

    Used by the vector-fitting / Lorentz models to evaluate their rational at the
    real frequencies. Legacy v1 files without freq_GHz return None (fallback axis).
    """
    with np.load(C.NPZ_PATH, allow_pickle=True) as d:
        if "freq_GHz" in d:
            return np.asarray(d["freq_GHz"], dtype="float32")
    return None
