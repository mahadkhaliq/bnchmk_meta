"""Shared helpers for the power-tx result plots (grid-agnostic).

Every plot script reads the current GRID from config_powertx.py and the best
model size from that grid's sweep results, so the same scripts work for
1x1 / 2x2 / 3x3 without edits.
"""
import csv
import numpy as np
import torch

import config_powertx as C
from models.mlp import MLP
from models.scale_invariant import ScaleInvariantMetasurface

# consistent palette across all plots
INK = "#22303f"
BASE_CLR = "#457b9d"      # baseline (blue)
NEIGH_CLR = "#2a9d8f"     # neighbourhood (teal)
CH = {"d": "#2a9d8f", "g": "#457b9d", "l": "#e9c46a", "w": "#e76f51"}


def best_size(grid=None, by="neigh_test"):
    """Pick (HIDDEN, N_HIDDEN) = the sweep row with the lowest `by` metric."""
    grid = grid or C.GRID
    rows = list(csv.DictReader(open(f"logs/sweep_{grid}_results.csv")))
    r = min(rows, key=lambda d: float(d[by]))
    return int(r["hidden"]), int(r["n_hidden"])


def predict(model, x, batch=256):
    model = model.to(C.device).eval()
    out = []
    with torch.no_grad():
        tx = torch.tensor(x, dtype=torch.float32)
        for i in range(0, len(tx), batch):
            out.append(model(tx[i:i + batch].to(C.device)).cpu().numpy())
    return np.concatenate(out, axis=0)


def load_models(hidden, n_hidden):
    """Load the best baseline + neighbourhood checkpoints for the current GRID."""
    base = MLP([C.INPUT_DIM] + [hidden] * n_hidden + [C.OUTPUT_DIM])
    base.load_state_dict(torch.load(
        f"ckpts/sweep_{C.GRID}_baseline_{hidden}x{n_hidden}.pt", map_location=C.device))
    neigh = ScaleInvariantMetasurface(N=C.GRID_N, K=C.KERNEL, C=C.CHANNELS,
                                      n_freq=C.OUTPUT_DIM, hidden=hidden, n_hidden=n_hidden)
    neigh.load_state_dict(torch.load(
        f"ckpts/sweep_{C.GRID}_neigh_{hidden}x{n_hidden}.pt", map_location=C.device))
    return base, neigh


def smooth(a, w=21):
    """light boxcar smoothing along frequency (for stable dip detection)."""
    k = np.ones(w) / w
    return np.array([np.convolve(row, k, mode="same") for row in a])


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color="#e5ebf0", lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9, colors=INK)
