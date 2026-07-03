"""Plot test spectra for the 500-epoch 2x2 experiments.

Creates two no-sweep figures:
    plots/test_spectra_2x2_706_500ep.png
    plots/test_spectra_2x2_integrated_500ep.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from data.normalize import normalize
from models.mlp import MLP
from models.mlp_silu import MLPSiLU
from models.scale_invariant import ScaleInvariantMetasurface
from models.scale_invariant_silu import ScaleInvariantSiLU


PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

OLD_NPZ = "/Users/mkfqm/malof_lab/power_tx_data/dataset_2x2.npz"
INT_NPZ = "/Users/mkfqm/malof_lab/power_tx_data/version_2/dataset_2x2_integrated.npz"

INK = "#233142"
GRID = "#dce3ea"
GT = "#111827"
COLORS = {
    "SiLU beta2": "#e76f51",
    "SiLU beta2 K=3": "#b56576",
    "Full ReLU": "#6c7fb5",
    "Full ReLU K=3": "#2d6a4f",
}


def load_npz_test(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    x = d["params"].astype("float32")
    y = d["T_clean"].astype("float32")
    freq = d["freq_GHz"].astype("float32")

    x_tr, test_x_raw, _, test_y = train_test_split(
        x, y, test_size=0.15, random_state=0
    )
    _, x_max, x_min = normalize(x_tr)
    test_x, _, _ = normalize(test_x_raw, x_max, x_min)
    test_grid = build_grid(test_x)
    return freq, test_x, test_grid, test_y


def build_grid(params, grid_n=2, channels=4):
    n = params.shape[0]
    n_cells = grid_n * grid_n
    grid = np.zeros((n, grid_n, grid_n, channels), dtype="float32")
    for ch in range(channels):
        for k in range(n_cells):
            r, c = k // grid_n, k % grid_n
            grid[:, r, c, ch] = params[:, ch * n_cells + k]
    return grid


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
    ax.tick_params(colors=INK, labelsize=8.8)
    ax.grid(True, color=GRID, lw=0.8, alpha=0.85)
    ax.set_axisbelow(True)


@torch.no_grad()
def predict(model, x, batch=256):
    model = model.to(DEVICE).eval()
    out = []
    tx = torch.tensor(x, dtype=torch.float32)
    for i in range(0, len(tx), batch):
        out.append(model(tx[i:i + batch].to(DEVICE)).cpu().numpy())
    return np.concatenate(out, axis=0)


def load_state(model, path):
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model


def select_samples(test_y, ref_pred, n=6):
    err = ((ref_pred - test_y) ** 2).mean(axis=1)
    order = np.argsort(err)
    pcts = [0, 20, 40, 60, 80, 99][:n]
    idxs = [order[min(len(order) - 1, int(round(p / 100 * (len(order) - 1))))] for p in pcts]
    tags = ["best", "p20", "p40", "p60", "p80", "worst"][:n]
    return idxs, tags


def plot_grid(freq, test_y, preds, idxs, tags, title, subtitle, out):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "figure.dpi": 140,
    })
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.6), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, idx, tag in zip(axes, idxs, tags):
        ax.plot(freq, test_y[idx], color=GT, lw=2.5, label="ground truth", zorder=5)
        lines = []
        for name, pred in preds.items():
            mse = ((pred[idx] - test_y[idx]) ** 2).mean()
            ls = "--" if "Full" in name and "K=3" not in name else "-"
            line, = ax.plot(
                freq, pred[idx], color=COLORS[name], lw=1.35,
                alpha=0.9, ls=ls, label=name, zorder=3)
            lines.append((name, mse, line))
        best_name, best_mse, _ = min(lines, key=lambda row: row[1])
        ax.set_title(f"{tag} sample {idx}", color=INK)
        ax.text(
            0.025, 0.045, f"best: {best_name}\nMSE {best_mse:.4f}",
            transform=ax.transAxes, fontsize=8.2, color=INK, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID, lw=1.0, alpha=0.92),
        )
        ax.set_ylim(-0.035, 1.05)
        style(ax)

    for ax in axes[3:]:
        ax.set_xlabel("Frequency (GHz)")
    for ax in (axes[0], axes[3]):
        ax.set_ylabel("Transmission T = |S21|^2")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=3,
        frameon=False, fontsize=9, labelcolor=INK, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle(title, fontsize=16, fontweight="bold", color=INK, y=1.015)
    fig.text(0.5, 0.955, subtitle, ha="center", fontsize=9.6, color="#687789")
    fig.tight_layout(rect=(0, 0.075, 1, 0.925))
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print("saved", out)


def plot_old_706():
    freq, test_x, test_grid, test_y = load_npz_test(OLD_NPZ)

    preds = {
        "SiLU beta2": predict(load_state(
            MLPSiLU(d_in=16, d_out=2001, hidden=512, n_layers=4),
            "ckpts/silu_2x2_512x4_500ep.pt"), test_x),
        "SiLU beta2 K=3": predict(load_state(
            ScaleInvariantSiLU(N=2, K=3, C=4, n_freq=2001, hidden=512, n_layers=4),
            "ckpts/silu_neigh_2x2_K3_512x4_500ep.pt"), test_grid),
    }
    idxs, tags = select_samples(test_y, preds["SiLU beta2 K=3"])
    plot_grid(
        freq, test_y, preds, idxs, tags,
        "2x2 Test Spectra, 706-Sample Dataset",
        "500-epoch SiLU variants; panels are ranked by SiLU beta2 K=3 per-sample error.",
        f"{PLOT_DIR}/test_spectra_2x2_706_500ep.png",
    )


def plot_integrated():
    freq, test_x, test_grid, test_y = load_npz_test(INT_NPZ)

    preds = {
        "SiLU beta2": predict(load_state(
            MLPSiLU(d_in=16, d_out=2001, hidden=512, n_layers=4),
            "ckpts/silu_2x2_integrated_512x4_500ep.pt"), test_x),
        "SiLU beta2 K=3": predict(load_state(
            ScaleInvariantSiLU(N=2, K=3, C=4, n_freq=2001, hidden=512, n_layers=4),
            "ckpts/silu_neigh_2x2_integrated_K3_512x4_500ep.pt"), test_grid),
        "Full ReLU": predict(load_state(
            MLP([16] + [2000] * 10 + [2001]),
            "ckpts/2x2_integrated_full_relu_cuda_baseline_2000x10.pt"), test_x, batch=128),
        "Full ReLU K=3": predict(load_state(
            ScaleInvariantMetasurface(N=2, K=3, C=4, n_freq=2001, hidden=2000, n_hidden=10),
            "ckpts/2x2_integrated_full_relu_cuda_neigh_K3_2000x10.pt"), test_grid, batch=128),
    }
    idxs, tags = select_samples(test_y, preds["Full ReLU K=3"])
    plot_grid(
        freq, test_y, preds, idxs, tags,
        "2x2 Test Spectra, 21,625-Sample Integrated Dataset",
        "500-epoch models; panels are ranked by full ReLU K=3 per-sample error.",
        f"{PLOT_DIR}/test_spectra_2x2_integrated_500ep.png",
    )


def main():
    plot_old_706()
    plot_integrated()


if __name__ == "__main__":
    main()
