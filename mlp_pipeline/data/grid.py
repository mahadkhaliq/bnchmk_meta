"""Grid construction + neighbourhood extraction (the "neighbouring" mechanism).

This module turns a flat geometry vector into a spatial grid of cells, and
extracts the wrap-around K x K neighbourhood of every cell. It is the heart of
the WITH-neighbouring model.

Two pieces live here:

    build_grid()         flat (n, INPUT_DIM)  ->  grid (n, N, N, C)   [DATASET SPECIFIC]
    build_neighborhoods() grid (n, N, N, C)   ->  (n, N*N, K*K*C)     [dataset agnostic]

`build_neighborhoods` is a NumPy reference implementation of exactly what the
ScaleInvariantMetasurface model does internally in torch — handy for inspecting
or visualising the neighbourhoods. Training uses the torch version inside the
model; this NumPy version is for analysis.
"""
import numpy as np

from data.normalize import normalize


def build_grid(g, x_max=None, x_min=None, grid_n=2, channels=5):
    """Raw geometry array (n, INPUT_DIM) -> normalised grid (n, grid_n, grid_n, channels).

    Columns are normalised to [-1, 1] BEFORE being reshaped into the grid, using
    the same min-max scheme as the baseline.

    >>> DATASET-SPECIFIC MAPPING (ADM) <<<
    The 14 ADM columns are laid out as:
        col 0          -> h      (height,  broadcast to every cell)
        col 1          -> p      (period,  broadcast to every cell)
        col 2 + 2*i    -> rx     of resonator i
        col 3 + 2*i    -> ry     of resonator i
        col 10 + i     -> theta  of resonator i      (i = 0 .. grid_n*grid_n - 1)
    Cell (r, c) for resonator i is r = i // grid_n, c = i % grid_n.

    For a DIFFERENT dataset, rewrite ONLY the assignment loop below so that each
    (cell, channel) slot is filled from the right raw column. Everything
    downstream (padding, windowing, the model) is dataset agnostic.
    """
    g, x_max, x_min = normalize(g, x_max, x_min)

    n = g.shape[0]
    grid = np.zeros((n, grid_n, grid_n, channels), dtype="float32")
    n_cells = grid_n * grid_n
    for i in range(n_cells):
        r = i // grid_n
        c = i % grid_n
        grid[:, r, c, 0] = g[:, 2 + 2 * i]   # rx
        grid[:, r, c, 1] = g[:, 3 + 2 * i]   # ry
        grid[:, r, c, 2] = g[:, 10 + i]      # theta
        grid[:, r, c, 3] = g[:, 0]           # h  (same for every cell)
        grid[:, r, c, 4] = g[:, 1]           # p  (same for every cell)
    return grid, x_max, x_min


def periodic_pad(grid, pad):
    """Wrap-around (periodic) padding of the two spatial dims of (n, N, N, C).

    A metasurface supercell is periodic, so cell (0,0)'s left neighbour is the
    last column, etc. We build wrap-around source indices and gather:

        src = (arange(N + 2*pad) - pad) % N
    """
    N = grid.shape[1]
    src = (np.arange(N + 2 * pad) - pad) % N
    return grid[:, src[:, None], src[None, :], :]


def build_neighborhoods(grid, K=3):
    """grid (n, N, N, C) -> flattened neighbourhoods (n, N*N, K*K*C).

    For each of the N*N cells, take the K x K window centred on it (using
    periodic padding) and flatten it. NumPy reference for the torch logic in
    ScaleInvariantMetasurface.forward.
    """
    n, N, _, C = grid.shape
    pad = K // 2
    padded = periodic_pad(grid, pad)

    neighborhoods = np.zeros((n, N * N, K, K, C), dtype="float32")
    for idx in range(N * N):
        i, j = idx // N, idx % N
        neighborhoods[:, idx] = padded[:, i:i + K, j:j + K, :]

    return neighborhoods.reshape(n, N * N, K * K * C)
