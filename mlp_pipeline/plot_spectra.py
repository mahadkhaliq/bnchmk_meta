"""Spectra: ground truth vs both models, for the current GRID (config_powertx.py).

Uses the best sweep size for this grid and the shared test split. Picks 6 test
samples spanning the error range (best -> worst by neighbourhood MSE).

    python plot_spectra.py   ->  plots/spectra_compare_<GRID>.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config_powertx as C
from data.powertx import load_flat, load_grid
from plot_common import best_size, load_models, predict, style, INK, BASE_CLR, NEIGH_CLR

os.makedirs("plots", exist_ok=True)
HIDDEN, N_HIDDEN = best_size()
OUT = f"plots/spectra_compare_{C.GRID}.png"
GT_CLR = INK


def main():
    freq = np.load(C.NPZ_PATH, allow_pickle=True)["freq_GHz"]
    _, _, test_x, test_y = load_flat()
    _, _, test_grid, _ = load_grid()

    base, neigh = load_models(HIDDEN, N_HIDDEN)
    p_base = predict(base, test_x)
    p_neigh = predict(neigh, test_grid)
    mse_base = ((p_base - test_y) ** 2).mean()
    mse_neigh = ((p_neigh - test_y) ** 2).mean()

    err = ((p_neigh - test_y) ** 2).mean(axis=1)
    order = np.argsort(err)
    pcts = [0, 20, 40, 60, 80, 99]
    idxs = [order[min(len(order) - 1, int(round(p / 100 * (len(order) - 1))))] for p in pcts]
    tags = ["best", "p20", "p40", "p60", "p80", "worst"]

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4), sharex=True)
    axes = axes.flatten()
    for ax, idx, tag in zip(axes, idxs, tags):
        ax.plot(freq, test_y[idx], color=GT_CLR, lw=2.6, alpha=0.9, label="ground truth", zorder=3)
        ax.plot(freq, p_base[idx], color=BASE_CLR, lw=1.7, ls="--", alpha=0.95, label="baseline", zorder=4)
        ax.plot(freq, p_neigh[idx], color=NEIGH_CLR, lw=1.9, alpha=0.95, label="neighbourhood", zorder=5)
        eb = ((p_base[idx] - test_y[idx]) ** 2).mean()
        en = ((p_neigh[idx] - test_y[idx]) ** 2).mean()
        ax.set_title(f"{tag}  (sample {idx})", fontsize=11, weight="bold", color=INK)
        ax.text(0.025, 0.04, f"base MSE {eb:.4f}\nneigh MSE {en:.4f}", transform=ax.transAxes,
                fontsize=8.6, va="bottom", ha="left", color=INK,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d7dee5", lw=1))
        ax.set_ylim(-0.03, 1.05)
        style(ax)
    for ax in axes[3:]:
        ax.set_xlabel("Frequency (GHz)", fontsize=10)
    for ax in (axes[0], axes[3]):
        ax.set_ylabel("Transmission  T = |S21|²", fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               fontsize=11, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(
        f"Power-tx {C.GRID}: predicted vs ground-truth spectra  (best {HIDDEN}×{N_HIDDEN} models)\n"
        f"overall test MSE — baseline {mse_base:.4f}   |   neighbourhood {mse_neigh:.4f}",
        fontsize=14, weight="bold", color=INK, y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT, dpi=130, bbox_inches="tight", facecolor="white")
    print("saved", OUT, f"| size {HIDDEN}x{N_HIDDEN} | base {mse_base:.5f} neigh {mse_neigh:.5f}")


if __name__ == "__main__":
    main()
