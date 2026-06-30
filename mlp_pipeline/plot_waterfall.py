"""Waterfall (heatmap) of transmission over ALL test samples, for current GRID.

Each row is one test sample's spectrum, colour = transmission T; rows sorted by
ground-truth dip frequency. Ground truth and both best models share the SAME
row order.

    python plot_waterfall.py  ->  plots/waterfall_<GRID>.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config_powertx as C
from data.powertx import load_flat, load_grid
from plot_common import best_size, load_models, predict, smooth, INK

os.makedirs("plots", exist_ok=True)
HIDDEN, N_HIDDEN = best_size()
OUT = f"plots/waterfall_{C.GRID}.png"
CMAP = "turbo"


def main():
    freq = np.asarray(np.load(C.NPZ_PATH, allow_pickle=True)["freq_GHz"], dtype=float)
    _, _, test_x, test_y = load_flat()
    _, _, test_grid, _ = load_grid()
    base, neigh = load_models(HIDDEN, N_HIDDEN)
    p_base, p_neigh = predict(base, test_x), predict(neigh, test_grid)

    n = len(test_y)
    order = np.argsort(freq[smooth(test_y).argmin(1)])
    panels = [("Ground truth", test_y[order]),
              (f"Baseline {HIDDEN}×{N_HIDDEN}", p_base[order]),
              (f"Neighbourhood {HIDDEN}×{N_HIDDEN}", p_neigh[order])]

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
    extent = [freq.min(), freq.max(), 0, n]
    im = None
    for ax, (title, img) in zip(axes, panels):
        im = ax.imshow(np.clip(img, 0, 1), aspect="auto", origin="lower", cmap=CMAP,
                       extent=extent, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=12, weight="bold", color=INK)
        ax.set_xlabel("Frequency (GHz)", fontsize=11)
        ax.tick_params(labelsize=9, colors=INK)
    axes[0].set_ylabel("test sample  (sorted by dip frequency)", fontsize=11)

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Transmission  T = |S21|²", fontsize=11, color=INK)
    cbar.ax.tick_params(labelsize=9, colors=INK)
    fig.suptitle(
        f"Power-tx {C.GRID} — transmission waterfall over all {n} test samples (blue = dip / low T)",
        fontsize=15, weight="bold", color=INK, y=1.0)
    fig.savefig(OUT, dpi=130, bbox_inches="tight", facecolor="white")
    print("saved", OUT, f"| size {HIDDEN}x{N_HIDDEN}")


if __name__ == "__main__":
    main()
