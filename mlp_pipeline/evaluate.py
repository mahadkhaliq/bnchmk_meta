"""Evaluate trained checkpoints on the test set.

    python evaluate.py            # evaluates whichever checkpoints exist
"""
import os

import numpy as np
import torch

import config
from data.loaders import load_flat, load_grid
from models.mlp import MLP
from models.scale_invariant import ScaleInvariantMetasurface


def _predict(model, test_x, device, batch=512):
    """Batched inference (kept small so the grid model stays memory-friendly)."""
    model = model.to(device).eval()
    preds = []
    with torch.no_grad():
        tx = torch.tensor(test_x, dtype=torch.float32)
        for i in range(0, len(tx), batch):
            preds.append(model(tx[i:i + batch].to(device)).cpu().numpy())
    return np.concatenate(preds, axis=0)


def evaluate_baseline():
    if not os.path.exists(config.BASELINE_CKPT):
        print(f"[skip] baseline checkpoint not found: {config.BASELINE_CKPT}")
        return None, None
    _, _, test_x, test_y = load_flat(config.BATCH_SIZE)
    model = MLP(config.LAYERS)
    model.load_state_dict(torch.load(config.BASELINE_CKPT, map_location=config.device))
    pred = _predict(model, test_x, config.device)
    mse = ((pred - test_y) ** 2).mean()
    print(f"Baseline MLP        test MSE: {mse:.6f}  ({mse:.3e})")
    return pred, test_y


def evaluate_neighborhood():
    if not os.path.exists(config.NEIGHBORHOOD_CKPT):
        print(f"[skip] neighbourhood checkpoint not found: {config.NEIGHBORHOOD_CKPT}")
        return None, None
    _, _, test_grid, test_y = load_grid(config.BATCH_SIZE)
    model = ScaleInvariantMetasurface(
        N=config.GRID_N, K=config.KERNEL, C=config.CHANNELS,
        n_freq=config.OUTPUT_DIM, hidden=config.HIDDEN, n_hidden=config.N_HIDDEN)
    model.load_state_dict(torch.load(config.NEIGHBORHOOD_CKPT, map_location=config.device))
    pred = _predict(model, test_grid, config.device)
    mse = ((pred - test_y) ** 2).mean()
    print(f"Neighbourhood MLP   test MSE: {mse:.6f}  ({mse:.3e})")
    return pred, test_y


if __name__ == "__main__":
    evaluate_baseline()
    evaluate_neighborhood()
