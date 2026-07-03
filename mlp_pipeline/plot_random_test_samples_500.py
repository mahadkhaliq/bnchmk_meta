"""Plot fixed random test samples for the 500-epoch 2x2 experiments.

Creates no-sweep figures where every model is shown on the same random held-out
samples within each dataset:
    plots/random_test_spectra_2x2_706_500ep.png
    plots/random_test_spectra_2x2_integrated_500ep.png
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

OLD_NPZ = "/Users/mkfqm/malof_lab/power_tx_data/dataset_2x2.npz"
INT_NPZ = "/Users/mkfqm/malof_lab/power_tx_data/version_2/dataset_2x2_integrated.npz"
RANDOM_SEED = 17
N_SAMPLES = 3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

INK = "#233142"
GRID = "#dce3ea"
GT = "#111827"
COLORS = {
    "Full ReLU": "#3b6fb6",
    "Full ReLU K=3": "#118a5b",
    "SiLU beta2": "#e66100",
    "SiLU beta2 K=3": "#7b3294",
}
LINESTYLES = {
    "Full ReLU": (0, (5, 2)),
    "Full ReLU K=3": (0, (3, 1, 1, 1)),
    "SiLU beta2": "-",
    "SiLU beta2 K=3": (0, (6, 2, 1, 2)),
}
ALPHAS = {
    "Full ReLU": 0.62,
    "Full ReLU K=3": 0.66,
    "SiLU beta2": 0.9,
    "SiLU beta2 K=3": 0.9,
}


def build_grid(params, grid_n=2, channels=4):
    n = params.shape[0]
    n_cells = grid_n * grid_n
    grid = np.zeros((n, grid_n, grid_n, channels), dtype="float32")
    for ch in range(channels):
        for k in range(n_cells):
            r, c = k // grid_n, k % grid_n
            grid[:, r, c, ch] = params[:, ch * n_cells + k]
    return grid


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
    return freq, test_x, build_grid(test_x), test_y


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(True, color=GRID, lw=0.8, alpha=0.85)
    ax.set_axisbelow(True)


@torch.no_grad()
def predict(model, x, batch=128):
    model = model.to(DEVICE).eval()
    tx = torch.tensor(x, dtype=torch.float32)
    out = []
    for i in range(0, len(tx), batch):
        out.append(model(tx[i:i + batch].to(DEVICE)).cpu().numpy())
    return np.concatenate(out, axis=0)


def load_state(model, path):
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model


def random_indices(n_test):
    rng = np.random.default_rng(RANDOM_SEED)
    return sorted(rng.choice(n_test, size=N_SAMPLES, replace=False).tolist())


def plot_random(freq, test_y, preds, idxs, title, subtitle, out):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.7,
        "figure.dpi": 150,
    })
    fig, axes = plt.subplots(
        2, len(idxs), figsize=(16, 6.8), sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.15], "hspace": 0.12},
    )
    if len(idxs) == 1:
        axes = np.array(axes).reshape(2, 1)

    for col, idx in enumerate(idxs):
        ax = axes[0, col]
        err_ax = axes[1, col]
        ax.plot(freq, test_y[idx], color=GT, lw=3.0, label="ground truth", zorder=8)
        per_model = []
        for name, pred in preds.items():
            mse = float(((pred[idx] - test_y[idx]) ** 2).mean())
            ax.plot(
                freq, pred[idx], color=COLORS[name], lw=1.35,
                alpha=ALPHAS[name], ls=LINESTYLES[name], label=name, zorder=3
            )
            err_ax.plot(
                freq, np.abs(pred[idx] - test_y[idx]), color=COLORS[name],
                lw=1.15, alpha=min(0.9, ALPHAS[name] + 0.08),
                ls=LINESTYLES[name], zorder=3
            )
            per_model.append((name, mse))

        best_name, best_mse = min(per_model, key=lambda row: row[1])
        ax.set_title(f"sample {idx}  |  best: {best_name} ({best_mse:.4f})", color=INK)
        ax.set_ylim(-0.035, 1.05)
        style(ax)
        style(err_ax)
        err_ax.set_ylim(0.0, max(0.12, min(1.0, max(
            float(np.abs(pred[idx] - test_y[idx]).max()) for pred in preds.values()
        ) * 1.12)))
        err_ax.set_xlabel("Frequency (GHz)")

    axes[0, 0].set_ylabel("Transmission T = |S21|^2")
    axes[1, 0].set_ylabel("|error|")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=min(5, len(labels)),
        frameon=False, fontsize=9.2, labelcolor=INK, bbox_to_anchor=(0.5, -0.02)
    )
    fig.suptitle(title, fontsize=15.5, fontweight="bold", color=INK, y=1.025)
    fig.text(0.5, 0.955, subtitle, ha="center", fontsize=9.5, color="#687789")
    fig.tight_layout(rect=(0, 0.065, 1, 0.925))
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print("random test indices:", idxs)
    print("saved", out)


def old_706_preds(test_x, test_grid):
    return {
        "Full ReLU": predict(load_state(
            MLP([16] + [2000] * 10 + [2001]),
            "best_powertx_baseline_2x2.pt"), test_x),
        "Full ReLU K=3": predict(load_state(
            ScaleInvariantMetasurface(N=2, K=3, C=4, n_freq=2001, hidden=2000, n_hidden=10),
            "best_powertx_neighborhood_2x2.pt"), test_grid),
        "SiLU beta2": predict(load_state(
            MLPSiLU(d_in=16, d_out=2001, hidden=512, n_layers=4),
            "ckpts/silu_2x2_512x4_500ep.pt"), test_x),
        "SiLU beta2 K=3": predict(load_state(
            ScaleInvariantSiLU(N=2, K=3, C=4, n_freq=2001, hidden=512, n_layers=4),
            "ckpts/silu_neigh_2x2_K3_512x4_500ep.pt"), test_grid),
    }


def integrated_preds(test_x, test_grid):
    return {
        "Full ReLU": predict(load_state(
            MLP([16] + [2000] * 10 + [2001]),
            "ckpts/2x2_integrated_full_relu_cuda_baseline_2000x10.pt"), test_x),
        "Full ReLU K=3": predict(load_state(
            ScaleInvariantMetasurface(N=2, K=3, C=4, n_freq=2001, hidden=2000, n_hidden=10),
            "ckpts/2x2_integrated_full_relu_cuda_neigh_K3_2000x10.pt"), test_grid),
        "SiLU beta2": predict(load_state(
            MLPSiLU(d_in=16, d_out=2001, hidden=512, n_layers=4),
            "ckpts/silu_2x2_integrated_512x4_500ep.pt"), test_x),
        "SiLU beta2 K=3": predict(load_state(
            ScaleInvariantSiLU(N=2, K=3, C=4, n_freq=2001, hidden=512, n_layers=4),
            "ckpts/silu_neigh_2x2_integrated_K3_512x4_500ep.pt"), test_grid),
    }


def main():
    freq, test_x, test_grid, test_y = load_npz_test(OLD_NPZ)
    idxs = random_indices(len(test_y))
    plot_random(
        freq, test_y, old_706_preds(test_x, test_grid), idxs,
        "Random Held-Out Test Samples, 706-Sample 2x2 Dataset",
        f"Same fixed random samples for every model; seed={RANDOM_SEED}; spectra above, absolute error below.",
        f"{PLOT_DIR}/random_test_spectra_2x2_706_500ep_clean.png",
    )

    freq, test_x, test_grid, test_y = load_npz_test(INT_NPZ)
    idxs = random_indices(len(test_y))
    plot_random(
        freq, test_y, integrated_preds(test_x, test_grid), idxs,
        "Random Held-Out Test Samples, 21,625-Sample Integrated 2x2 Dataset",
        f"Same fixed random samples for every model; seed={RANDOM_SEED}; spectra above, absolute error below.",
        f"{PLOT_DIR}/random_test_spectra_2x2_integrated_500ep_clean.png",
    )


if __name__ == "__main__":
    main()
