"""Per-sample test-MSE histograms for both best models, for the current GRID.

  left   distribution of per-sample MSE (baseline vs neighbourhood)
  right  paired per-sample change (neighbourhood - baseline)

    python plot_mse_hist.py  ->  plots/mse_hist_compare_<GRID>.png
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
OUT = f"plots/mse_hist_compare_{C.GRID}.png"


def main():
    _, _, test_x, test_y = load_flat()
    _, _, test_grid, _ = load_grid()
    base, neigh = load_models(HIDDEN, N_HIDDEN)
    eb = ((predict(base, test_x) - test_y) ** 2).mean(axis=1)
    en = ((predict(neigh, test_grid) - test_y) ** 2).mean(axis=1)
    n = len(test_y)

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1.5, 1]})

    bins = np.linspace(0, max(eb.max(), en.max()) * 1.02, 32)
    axL.hist(eb, bins=bins, color=BASE_CLR, alpha=0.55,
             label=f"baseline   (mean {eb.mean():.4f}, median {np.median(eb):.4f})")
    axL.hist(en, bins=bins, color=NEIGH_CLR, alpha=0.55,
             label=f"neighbourhood   (mean {en.mean():.4f}, median {np.median(en):.4f})")
    for v, c in [(eb.mean(), BASE_CLR), (en.mean(), NEIGH_CLR)]:
        axL.axvline(v, color=c, lw=2, ls="--")
    axL.set_xlabel("per-sample test MSE"); axL.set_ylabel("count (samples)")
    axL.set_title(f"Per-sample test-MSE distribution  (all {n} samples)", fontsize=12, weight="bold", color=INK)
    axL.legend(frameon=False, fontsize=9.2); style(axL)

    diff = en - eb
    better = diff < 0
    axR.hist(diff[better], bins=24, color=NEIGH_CLR, alpha=0.7, label=f"neigh better ({better.sum()}/{n})")
    axR.hist(diff[~better], bins=24, color=BASE_CLR, alpha=0.7, label=f"baseline better ({(~better).sum()}/{n})")
    axR.axvline(0, color=INK, lw=1.4)
    axR.set_xlabel("ΔMSE  (neighbourhood − baseline)"); axR.set_ylabel("count (samples)")
    axR.set_title("Paired per-sample change\n(left of 0 = neighbourhood wins)", fontsize=11.5, weight="bold", color=INK)
    axR.legend(frameon=False, fontsize=9.2); style(axR)

    fig.suptitle(f"Power-tx {C.GRID} — per-sample MSE, baseline vs neighbourhood (best {HIDDEN}×{N_HIDDEN})",
                 fontsize=14, weight="bold", color=INK, y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT, dpi=130, bbox_inches="tight", facecolor="white")
    print("saved", OUT, f"| size {HIDDEN}x{N_HIDDEN} | base {eb.mean():.5f} neigh {en.mean():.5f} | neigh better {better.sum()}/{n}")


if __name__ == "__main__":
    main()
