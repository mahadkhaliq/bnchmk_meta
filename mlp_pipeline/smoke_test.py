"""Wiring check with synthetic data (no CSVs needed).

Verifies that:
  - the baseline MLP maps (B, INPUT_DIM) -> (B, OUTPUT_DIM)
  - build_grid -> build_neighborhoods produce the expected shapes
  - the periodic-padded centre of each neighbourhood equals the original cell
  - the neighbourhood model maps (B, N, N, C) -> (B, OUTPUT_DIM)
  - the NumPy neighbourhood matches the model's internal torch windowing

Run:  python smoke_test.py
"""
import numpy as np
import torch

import config
from data.grid import build_grid, build_neighborhoods
from models.mlp import MLP
from models.scale_invariant import ScaleInvariantMetasurface
from models.concept_one import ConceptOneMetasurface


def test_baseline_mlp():
    model = MLP(config.LAYERS).eval()
    x = torch.randn(8, config.INPUT_DIM)
    y = model(x)
    assert y.shape == (8, config.OUTPUT_DIM), y.shape
    print(f"baseline MLP: {tuple(x.shape)} -> {tuple(y.shape)}  OK")


def test_grid_and_neighborhoods():
    n = 16
    raw = np.random.randn(n, config.INPUT_DIM).astype("float32")
    grid, _, _ = build_grid(raw, grid_n=config.GRID_N, channels=config.CHANNELS)
    assert grid.shape == (n, config.GRID_N, config.GRID_N, config.CHANNELS), grid.shape

    nb = build_neighborhoods(grid, K=config.KERNEL)
    N, K, C = config.GRID_N, config.KERNEL, config.CHANNELS
    assert nb.shape == (n, N * N, K * K * C), nb.shape

    # the centre of every neighbourhood must equal the original cell
    nb5 = nb.reshape(n, N * N, K, K, C)
    pad = K // 2
    for idx in range(N * N):
        i, j = idx // N, idx % N
        center = nb5[0, idx, pad, pad, :]
        original = grid[0, i, j, :]
        assert np.allclose(center, original), f"cell {idx} centre mismatch"
    print(f"build_grid -> {grid.shape},  build_neighborhoods -> {nb.shape}  OK")


def test_neighborhood_model():
    model = ScaleInvariantMetasurface(
        N=config.GRID_N, K=config.KERNEL, C=config.CHANNELS,
        n_freq=config.OUTPUT_DIM, hidden=64, n_hidden=2).eval()  # tiny net for speed
    g = torch.randn(8, config.GRID_N, config.GRID_N, config.CHANNELS)
    y = model(g)
    assert y.shape == (8, config.OUTPUT_DIM), y.shape
    print(f"neighbourhood model: {tuple(g.shape)} -> {tuple(y.shape)}  OK")


def test_concept_one_model():
    model = ConceptOneMetasurface(
        K=config.KERNEL, C=config.CHANNELS,
        n_freq=config.OUTPUT_DIM, latent_dim=16,
        hidden=32, n_hidden=1).eval()
    g = torch.randn(8, config.GRID_N, config.GRID_N, config.CHANNELS)
    y = model(g)
    assert y.shape == (8, config.OUTPUT_DIM), y.shape
    print(f"Concept #1 model: {tuple(g.shape)} -> {tuple(y.shape)}  OK")


def test_numpy_matches_torch():
    """The NumPy windowing must equal the model's internal torch windowing."""
    n = 4
    raw = np.random.randn(n, config.INPUT_DIM).astype("float32")
    grid, _, _ = build_grid(raw, grid_n=config.GRID_N, channels=config.CHANNELS)
    nb = build_neighborhoods(grid, K=config.KERNEL)  # (n, N*N, K*K*C)

    # reproduce the model's step-1/2 windowing in torch
    N, K, C = config.GRID_N, config.KERNEL, config.CHANNELS
    pad = K // 2
    g = torch.tensor(grid)
    src = (torch.arange(N + 2 * pad) - pad) % N
    padded = g[:, src[:, None], src[None, :], :]
    x = torch.zeros(n, N * N, K * K * C)
    for idx in range(N * N):
        i, j = idx // N, idx % N
        x[:, idx, :] = padded[:, i:i + K, j:j + K, :].reshape(n, K * K * C)

    assert np.allclose(nb, x.numpy(), atol=1e-6), "numpy vs torch windowing mismatch"
    print("numpy neighbourhoods == torch neighbourhoods  OK")


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    test_baseline_mlp()
    test_grid_and_neighborhoods()
    test_neighborhood_model()
    test_concept_one_model()
    test_numpy_matches_torch()
    print("\nAll smoke tests passed.")
