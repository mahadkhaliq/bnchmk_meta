"""Aggregate frequency view over ALL test samples, for the current GRID.

  A  all ground-truth spectra overlaid + mean + 10-90% band
  B  mean spectrum: ground truth vs baseline vs neighbourhood
  C  dip-frequency histogram (where each sample's main resonance falls)
  D  dip-depth histogram      (how deep the resonance goes)

    python plot_spectra_aggregate.py  ->  plots/spectra_aggregate_<GRID>.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config_powertx as C
from data.powertx import load_flat, load_grid
from plot_common import best_size, load_models, predict, smooth, style, INK, BASE_CLR, NEIGH_CLR

os.makedirs("plots", exist_ok=True)
HIDDEN, N_HIDDEN = best_size()
OUT = f"plots/spectra_aggregate_{C.GRID}.png"
GT_CLR = INK


def main():
    freq = np.asarray(np.load(C.NPZ_PATH, allow_pickle=True)["freq_GHz"], dtype=float)
    _, _, test_x, test_y = load_flat()
    _, _, test_grid, _ = load_grid()
    base, neigh = load_models(HIDDEN, N_HIDDEN)
    p_base, p_neigh = predict(base, test_x), predict(neigh, test_grid)
    n = len(test_y)

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(2, 2, figsize=(14.5, 9.2))
    A, B, Cp, D = ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]

    for row in test_y:
        A.plot(freq, row, color=NEIGH_CLR, lw=0.5, alpha=0.10, zorder=1)
    lo, hi = np.percentile(test_y, [10, 90], axis=0)
    A.fill_between(freq, lo, hi, color=GT_CLR, alpha=0.12, zorder=2, label="10–90% band")
    A.plot(freq, test_y.mean(0), color=GT_CLR, lw=2.6, zorder=4, label="mean")
    A.set_title(f"A · all {n} ground-truth spectra (combined)", fontsize=11, weight="bold", color=INK)
    A.set_ylabel("Transmission  T = |S21|²"); A.set_ylim(-0.02, 1.05)
    A.legend(frameon=False, fontsize=9, loc="lower left"); style(A)

    mu_gt, sd_gt = test_y.mean(0), test_y.std(0)
    B.fill_between(freq, mu_gt - sd_gt, mu_gt + sd_gt, color=GT_CLR, alpha=0.10, label="ground truth ±1σ")
    B.plot(freq, mu_gt, color=GT_CLR, lw=2.6, label="ground truth (mean)", zorder=5)
    B.plot(freq, p_base.mean(0), color=BASE_CLR, lw=1.9, ls="--", label="baseline (mean)", zorder=4)
    B.plot(freq, p_neigh.mean(0), color=NEIGH_CLR, lw=2.0, label="neighbourhood (mean)", zorder=4)
    B.set_title("B · mean spectrum — truth vs models", fontsize=11, weight="bold", color=INK)
    B.set_ylim(-0.02, 1.05); B.legend(frameon=False, fontsize=8.6, loc="lower left"); style(B)

    f_gt = freq[smooth(test_y).argmin(1)]
    f_b = freq[smooth(p_base).argmin(1)]
    f_n = freq[smooth(p_neigh).argmin(1)]
    bins = np.linspace(freq.min(), freq.max(), 28)
    Cp.hist(f_gt, bins=bins, color=GT_CLR, alpha=0.55, label="ground truth")
    Cp.hist(f_b, bins=bins, histtype="step", lw=2, color=BASE_CLR, label="baseline")
    Cp.hist(f_n, bins=bins, histtype="step", lw=2, color=NEIGH_CLR, label="neighbourhood")
    Cp.set_title("C · where the resonance dip falls", fontsize=11, weight="bold", color=INK)
    Cp.set_xlabel("dip frequency (GHz)"); Cp.set_ylabel("count (samples)")
    Cp.legend(frameon=False, fontsize=8.6); style(Cp)

    d_gt, d_b, d_n = test_y.min(1), p_base.min(1), p_neigh.min(1)
    dbins = np.linspace(0, max(d_gt.max(), d_b.max(), d_n.max(), 1.0), 28)
    D.hist(d_gt, bins=dbins, color=GT_CLR, alpha=0.55, label=f"ground truth (μ={d_gt.mean():.2f})")
    D.hist(d_b, bins=dbins, histtype="step", lw=2, color=BASE_CLR, label=f"baseline (μ={d_b.mean():.2f})")
    D.hist(d_n, bins=dbins, histtype="step", lw=2, color=NEIGH_CLR, label=f"neighbourhood (μ={d_n.mean():.2f})")
    D.set_title("D · how deep the dip goes (min T)", fontsize=11, weight="bold", color=INK)
    D.set_xlabel("minimum transmission"); D.set_ylabel("count (samples)")
    D.legend(frameon=False, fontsize=8.6); style(D)

    for a in (A, B):
        a.set_xlabel("Frequency (GHz)")
    fig.suptitle(
        f"Power-tx {C.GRID} — aggregate transmission over all {n} test samples  (best {HIDDEN}×{N_HIDDEN})",
        fontsize=15, weight="bold", color=INK, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT, dpi=130, bbox_inches="tight", facecolor="white")
    print("saved", OUT, f"| size {HIDDEN}x{N_HIDDEN}")


if __name__ == "__main__":
    main()
