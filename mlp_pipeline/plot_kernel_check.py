"""Aesthetic plot of the K=3 vs K=5 'number of neighbours' check on 2x2.

Parses logs/kernel_check_<GRID>.log and produces plots/kernel_check_<GRID>.png:
    left  - training/validation loss curves for both kernels
    right - final val/test MSE bars (the headline: it's a wash on 2x2)

    python plot_kernel_check.py
"""
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

import config_powertx as C

LOG = f"logs/kernel_check_{C.GRID}.log"
OUT = f"plots/kernel_check_{C.GRID}.png"
os.makedirs("plots", exist_ok=True)

# ---- palette ----
INK = "#22303f"
GRID_CLR = "#d7dee5"
COLORS = {3: "#2a9d8f", 5: "#e76f51"}      # teal, coral

BLOCK_RE = re.compile(r"=+ K=(\d+)")
EPOCH_RE = re.compile(r"Epoch\s+(\d+) \| train (\S+) \| val (\S+)")
SUMM_RE = re.compile(r"\[K=(\d+)\] params [\d,]+ \| best val (\S+) \| test (\S+)")


def parse(path):
    """-> {K: dict(epoch, train, val, best_val, test)}"""
    runs = {}
    cur = None
    with open(path) as f:
        for line in f:
            mb = BLOCK_RE.search(line)
            if mb:
                cur = int(mb.group(1))
                runs[cur] = dict(epoch=[], train=[], val=[])
                continue
            me = EPOCH_RE.search(line)
            if me and cur is not None:
                runs[cur]["epoch"].append(int(me.group(1)))
                runs[cur]["train"].append(float(me.group(2)))
                runs[cur]["val"].append(float(me.group(3)))
            ms = SUMM_RE.search(line)
            if ms:
                k = int(ms.group(1))
                runs[k]["best_val"] = float(ms.group(2))
                runs[k]["test"] = float(ms.group(3))
    return runs


def style_ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK)
    ax.tick_params(colors=INK, labelsize=10)
    ax.grid(True, color=GRID_CLR, lw=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def main():
    runs = parse(LOG)
    ks = sorted(runs)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 13, "axes.titleweight": "bold",
        "axes.labelsize": 11, "figure.dpi": 130,
    })

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2),
                                   gridspec_kw={"width_ratios": [1.55, 1]})

    # ---------- LEFT: loss curves ----------
    for k in ks:
        r = runs[k]
        c = COLORS[k]
        axL.plot(r["epoch"], r["val"], color=c, lw=2.4,
                 label=f"K={k}  (val)", zorder=3)
        axL.plot(r["epoch"], r["train"], color=c, lw=1.3, ls="--",
                 alpha=0.55, label=f"K={k}  (train)", zorder=2)
        # mark best-val point
        bi = int(np.argmin(r["val"]))
        axL.scatter(r["epoch"][bi], r["val"][bi], color=c, s=46,
                    edgecolor="white", lw=1.4, zorder=5)

    axL.set_yscale("log")
    axL.yaxis.set_major_formatter(ScalarFormatter())
    axL.yaxis.set_minor_formatter(plt.NullFormatter())
    axL.set_xlabel("epoch")
    axL.set_ylabel("MSE (log scale)")
    axL.set_title("Training dynamics — 128×3 neighbourhood")
    style_ax(axL)
    leg = axL.legend(frameon=False, fontsize=9.5, ncol=2,
                     loc="upper right", labelcolor=INK)

    # ---------- RIGHT: final val/test bars ----------
    metrics = ["best_val", "test"]
    labels = ["best val", "test"]
    x = np.arange(len(metrics))
    w = 0.36
    for i, k in enumerate(ks):
        vals = [runs[k][m] for m in metrics]
        bars = axR.bar(x + (i - 0.5) * w, vals, w, color=COLORS[k],
                       label=f"K={k}", zorder=3, edgecolor="white", lw=1.2)
        for b, v in zip(bars, vals):
            axR.annotate(f"{v:.4f}", (b.get_x() + b.get_width() / 2, v),
                         ha="center", va="bottom", fontsize=9.5,
                         color=INK, fontweight="bold", xytext=(0, 3),
                         textcoords="offset points")

    axR.set_xticks(x)
    axR.set_xticklabels(labels)
    axR.set_ylabel("MSE")
    axR.set_ylim(0, max(runs[k]["best_val"] for k in ks) * 1.38)
    axR.set_title("Final error — a wash on 2×2")
    style_ax(axR)
    axR.grid(axis="x", visible=False)
    axR.legend(frameon=False, fontsize=10, labelcolor=INK)

    # headline annotation: the delta on test
    dt = (runs[5]["test"] - runs[3]["test"]) / runs[3]["test"] * 100
    axR.annotate(f"Δtest = {dt:+.1f}%\n(within init noise)",
                 (1, runs[3]["test"]), xytext=(0.50, 0.93),
                 textcoords="axes fraction", fontsize=9.5, color=INK,
                 ha="center", va="top",
                 bbox=dict(boxstyle="round,pad=0.4", fc="#fdf3ef",
                           ec=COLORS[5], lw=1))

    fig.suptitle("Adding neighbours (K=3 → K=5) on the 2×2 grid",
                 fontsize=15, fontweight="bold", color=INK, y=1.005)
    fig.text(0.5, 0.945,
             "on 2×2 the 3×3 window already wraps the whole torus — K=5 only "
             "re-tiles the same 4 cells (input 36 → 100 dims), so error is unchanged",
             ha="center", fontsize=9.5, color="#5a6b7a", style="italic")

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print("saved", OUT)


if __name__ == "__main__":
    main()
