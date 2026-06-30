"""Evaluate trained power-transmission checkpoints on the held-out test split.

    python evaluate_powertx.py     # evaluates whichever checkpoints exist for the
                                   # GRID size currently selected in config_powertx.py
"""
import os

import numpy as np
import torch

import config_powertx as C
from data.powertx import load_flat, load_grid
from models.mlp import MLP
from models.scale_invariant import ScaleInvariantMetasurface
from models.concept_one import ConceptOneMetasurface


def _predict(model, test_x, device, batch=256):
    model = model.to(device).eval()
    preds = []
    with torch.no_grad():
        tx = torch.tensor(test_x, dtype=torch.float32)
        for i in range(0, len(tx), batch):
            preds.append(model(tx[i:i + batch].to(device)).cpu().numpy())
    return np.concatenate(preds, axis=0)


def evaluate_baseline():
    if not os.path.exists(C.BASELINE_CKPT):
        print(f"[skip] baseline checkpoint not found: {C.BASELINE_CKPT}")
        return
    _, _, test_x, test_y = load_flat()
    model = MLP(C.LAYERS)
    model.load_state_dict(torch.load(C.BASELINE_CKPT, map_location=C.device))
    pred = _predict(model, test_x, C.device)
    mse = ((pred - test_y) ** 2).mean()
    print(f"[{C.GRID}] Baseline MLP        test MSE: {mse:.6f}  ({mse:.3e})")


def evaluate_neighborhood():
    if not os.path.exists(C.NEIGHBORHOOD_CKPT):
        print(f"[skip] neighbourhood checkpoint not found: {C.NEIGHBORHOOD_CKPT}")
        return
    _, _, test_grid, test_y = load_grid()
    model = ScaleInvariantMetasurface(
        N=C.GRID_N, K=C.KERNEL, C=C.CHANNELS,
        n_freq=C.OUTPUT_DIM, hidden=C.HIDDEN, n_hidden=C.N_HIDDEN)
    model.load_state_dict(torch.load(C.NEIGHBORHOOD_CKPT, map_location=C.device))
    pred = _predict(model, test_grid, C.device)
    mse = ((pred - test_y) ** 2).mean()
    print(f"[{C.GRID}] Neighbourhood MLP   test MSE: {mse:.6f}  ({mse:.3e})")


def evaluate_concept_one():
    if not os.path.exists(C.CONCEPT_ONE_CKPT):
        print(f"[skip] Concept #1 checkpoint not found: {C.CONCEPT_ONE_CKPT}")
        return
    _, _, test_grid, test_y = load_grid()
    model = ConceptOneMetasurface(
        K=C.KERNEL, C=C.CHANNELS,
        n_freq=C.OUTPUT_DIM, latent_dim=C.CONCEPT_LATENT_DIM,
        hidden=C.HIDDEN, n_hidden=C.N_HIDDEN)
    model.load_state_dict(torch.load(C.CONCEPT_ONE_CKPT, map_location=C.device))
    pred = _predict(model, test_grid, C.device)
    mse = ((pred - test_y) ** 2).mean()
    print(f"[{C.GRID}] Concept #1 MLP      test MSE: {mse:.6f}  ({mse:.3e})")


if __name__ == "__main__":
    evaluate_baseline()
    evaluate_neighborhood()
    evaluate_concept_one()
