"""Visual comparison of the DATA SHAPE through baseline vs neighbourhood model,
for whatever GRID is set in config_powertx.py (1x1 / 2x2 / 3x3).

Shapes (grid, padded, windows, ...) are derived from GRID_N, KERNEL, CHANNELS,
OUTPUT_DIM, so the diagram is correct for any size. At 1x1 it also flags the
degeneracy (a single cell has no real neighbours).

    python plot_shape_compare.py  ->  plots/shape_compare_<GRID>.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

import config_powertx as C
from plot_common import best_size, INK, CH

os.makedirs("plots", exist_ok=True)
OUT = f"plots/shape_compare_{C.GRID}.png"
HIDDEN, N_HIDDEN = best_size()

MUTED = "#5a6b7a"
BOXFC = "#f2f6f8"
BASE_EC = "#457b9d"
NEIGH_EC = "#2a9d8f"
CH_ORDER = ["d", "g", "l", "w"]

# ---- derived shapes ----
N, K, Cch, F, P = C.GRID_N, C.KERNEL, C.CHANNELS, C.OUTPUT_DIM, C.INPUT_DIM
pad = K // 2
Np = N + 2 * pad
ncell = N * N
win = K * K * Cch
DEGEN = (N == 1)


def rbox(ax, cx, cy, w, h, ec, fc=BOXFC, lw=1.8, ls="-"):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.0,rounding_size=0.10",
                 fc=fc, ec=ec, lw=lw, ls=ls, mutation_aspect=1, zorder=2))


def arrow(ax, cx, y0, y1, label=None):
    ax.annotate("", xy=(cx, y1), xytext=(cx, y0),
                arrowprops=dict(arrowstyle="-|>", lw=2, color=INK, shrinkA=2, shrinkB=2), zorder=1)
    if label:
        ax.text(cx, (y0 + y1) / 2, label, ha="center", va="center", fontsize=8.6,
                color=MUTED, zorder=4, bbox=dict(fc="white", ec="none", pad=1.2))


def channel_row(ax, cx, cy, n_per_ch, sq=0.34):
    gap_in, gap_grp = 0.04, 0.34
    grp_w = n_per_ch * sq + (n_per_ch - 1) * gap_in
    total = 4 * grp_w + 3 * gap_grp
    x = cx - total / 2
    for ch in CH_ORDER:
        for _ in range(n_per_ch):
            ax.add_patch(Rectangle((x, cy - sq / 2), sq, sq, fc=CH[ch], ec="white", lw=1, zorder=3))
            x += sq + gap_in
        ax.text(x - grp_w / 2 - gap_in / 2, cy - sq / 2 - 0.26, ch, ha="center", va="center",
                fontsize=10, color=INK, weight="bold")
        x += gap_grp - gap_in


def cell_quad(ax, cx, cy, s):
    h = s / 2
    for ch, (dx, dy) in {"d": (-h, 0), "g": (0, 0), "l": (-h, -h), "w": (0, -h)}.items():
        ax.add_patch(Rectangle((cx + dx, cy + dy), h, h, fc=CH[ch], ec="white", lw=1.2, zorder=3))
    ax.add_patch(Rectangle((cx - h, cy - h), s, s, fc="none", ec=INK, lw=1.2, zorder=4))


def cell_grid(ax, cx, cy, n, s=0.62, gap=0.16):
    step = s + gap
    off = (n - 1) / 2
    for r in range(n):
        for c in range(n):
            cell_quad(ax, cx + (c - off) * step, cy - (r - off) * step, s)


def footer(ax, cx, cy, w, text, ec):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - 0.7), w, 1.4,
                 boxstyle="round,pad=0.0,rounding_size=0.10", fc="#fbfdfe", ec=ec, lw=1.4, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=9.2, color=INK, zorder=4)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, (axB, axN) = plt.subplots(1, 2, figsize=(14.5, 9.6))
    for ax in (axB, axN):
        ax.set_xlim(0, 10); ax.set_ylim(0, 15.2); ax.axis("off")

    # ---------- LEFT : BASELINE ----------
    axB.text(5, 14.7, "BASELINE  (no neighbouring)", ha="center", fontsize=14, weight="bold", color=BASE_EC)
    rbox(axB, 5, 12.5, 8.6, 2.3, BASE_EC)
    axB.text(5, 13.35, f"params   (B, {P})", ha="center", fontsize=11.5, weight="bold", color=INK)
    channel_row(axB, 5, 12.05, n_per_ch=ncell)
    axB.text(5, 11.15, f"flat vector — order d, g, l, w  ({ncell} cell{'s' if ncell>1 else ''} each)",
             ha="center", fontsize=8.6, color=MUTED, style="italic")
    arrow(axB, 5, 11.35, 3.1)
    axB.add_patch(FancyBboxPatch((2.1, 6.3), 5.8, 3.0, boxstyle="round,pad=0.0,rounding_size=0.10",
                  fc="#eef4f8", ec=BASE_EC, lw=1.4, ls="--", zorder=3))
    axB.text(5, 8.5, "one big MLP", ha="center", fontsize=11, weight="bold", color=INK)
    axB.text(5, 7.85, f"[ {P} → {HIDDEN} × {N_HIDDEN} → {F} ]", ha="center", fontsize=10, color=INK)
    axB.text(5, 7.0, "single forward pass\nover the whole supercell", ha="center", fontsize=8.8,
             color=MUTED, style="italic")
    rbox(axB, 5, 2.6, 5.0, 1.2, BASE_EC)
    axB.text(5, 2.6, f"y_hat   (B, {F})", ha="center", fontsize=11.5, weight="bold", color=INK)
    footer(axB, 5, 0.7, 9.0,
           f"input width  {P} = (#cells × #channels)  →  LOCKED to grid size\n"
           f"cannot run on a different grid size (width changes)", BASE_EC)

    # ---------- RIGHT : NEIGHBOURHOOD ----------
    axN.text(5, 14.7, "NEIGHBOURHOOD  (with neighbouring)", ha="center", fontsize=14, weight="bold", color=NEIGH_EC)
    rbox(axN, 5, 12.55, 5.0, 2.5, NEIGH_EC)
    axN.text(5, 13.55, f"grid   (B, {N}, {N}, {Cch})", ha="center", fontsize=11.5, weight="bold", color=INK)
    cell_grid(axN, 5, 11.8, n=N)

    arrow(axN, 5, 11.3, 10.15, f"wrap-around pad  (K={K}, pad={pad})")
    rbox(axN, 5, 9.6, 4.8, 1.0, NEIGH_EC)
    axN.text(5, 9.6, f"padded   (B, {Np}, {Np}, {Cch})", ha="center", fontsize=10.8, weight="bold", color=INK)

    arrow(axN, 5, 9.1, 7.85, "K×K window per cell")
    rbox(axN, 5, 7.3, 6.6, 1.0, NEIGH_EC)
    axN.text(5, 7.3, f"windows   (B, {ncell}, {win})", ha="center", fontsize=10.8, weight="bold", color=INK)
    note_w = f"the {Cch} params\ntiled {K*K}×" if DEGEN else f"N²={ncell} cells\n× K²C={win}"
    axN.text(8.7, 7.3, note_w, ha="center", va="center", fontsize=8.0,
             color="#b5651d" if DEGEN else MUTED, style="italic")

    arrow(axN, 5, 6.8, 5.55, f"SHARED  f_theta   [ {win} → {HIDDEN} × {N_HIDDEN} → {F} ]")
    rbox(axN, 5, 5.0, 6.2, 1.0, NEIGH_EC)
    axN.text(5, 5.0, f"per-cell spectra   (B, {ncell}, {F})", ha="center", fontsize=10.6, weight="bold", color=INK)
    axN.text(8.7, 5.0, "same weights\non every cell", ha="center", va="center", fontsize=8.0,
             color=MUTED, style="italic")

    arrow(axN, 5, 4.5, 3.2, f"mean over the {ncell} cell{'s' if ncell>1 else ''}"
                            + ("  (= identity)" if DEGEN else ""))
    rbox(axN, 5, 2.6, 5.0, 1.2, NEIGH_EC)
    axN.text(5, 2.6, f"y_hat   (B, {F})", ha="center", fontsize=11.5, weight="bold", color=INK)
    if DEGEN:
        footer(axN, 5, 0.7, 9.0,
               "single cell → NO real neighbours: the window is the 4 params tiled 9×\n"
               "⇒ degenerates to a baseline with a redundant (replicated) input", "#b5651d")
    else:
        footer(axN, 5, 0.7, 9.0,
               f"input width  {win} = K·K·C  →  INDEPENDENT of grid size\n"
               f"the same model runs on 1×1 / 2×2 / 3×3  →  cross-size transfer", NEIGH_EC)

    sub = ("at 1×1 there is a single cell — its 'neighbourhood' is just itself tiled K², "
           "so the neighbourhood model collapses to the baseline") if DEGEN else \
          ("the SAME geometry numbers (d, g, l, w per cell) enter both models — "
           "the baseline flattens them, the neighbourhood keeps them spatial")
    fig.suptitle(f"How the data is shaped — baseline vs neighbourhood model  (power-tx {C.GRID})",
                 fontsize=16, weight="bold", color=INK, y=0.99)
    fig.text(0.5, 0.945, sub, ha="center", fontsize=10.5, color=MUTED, style="italic")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT, dpi=130, bbox_inches="tight", facecolor="white")
    print("saved", OUT)


if __name__ == "__main__":
    main()
