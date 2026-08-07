"""Plot three separate random test samples for each v3 dataset.

Each figure compares the stored target against the verified 500-epoch flat
SiLU and K=3 SiLU checkpoints, with absolute errors in a second panel.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from data.normalize import normalize
from models.mlp_silu import MLPSiLU
from models.scale_invariant_silu import ScaleInvariantSiLU


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "power_tx_data" / "version_3"
PLOT_DIR = ROOT / "plots" / "v3_random_test_samples"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_SEED = 0
SAMPLE_SEED = 17
N_SAMPLES = 3
DEVICE = torch.device("cpu")

INK = "#27313b"
GRID_COLOR = "#dce3e8"
TARGET_COLOR = "#17191c"
FLAT_COLOR = "#d95f02"
NEIGH_COLOR = "#007c83"

SPECS = {
    "1x1": {"n": 1, "d_in": 4, "samples": 2000},
    "2x2": {"n": 2, "d_in": 16, "samples": 6497},
    "3x3": {"n": 3, "d_in": 36, "samples": 397},
}


def load_test(tag):
    with np.load(DATA_DIR / f"preprocessed_{tag}.npz", allow_pickle=True) as data:
        x = data["geom"].astype("float32")
        y = data["T"].astype("float32")
        freq = data["freq_GHz"].astype("float32")

    x_fit, test_x, y_fit, test_y = train_test_split(
        x, y, test_size=0.15, random_state=SPLIT_SEED
    )
    x_train, _, _, _ = train_test_split(
        x_fit, y_fit, test_size=0.2, random_state=SPLIT_SEED
    )
    _, x_max, x_min = normalize(x_train)
    test_x, _, _ = normalize(test_x, x_max, x_min)

    n = SPECS[tag]["n"]
    test_grid = test_x.reshape(len(test_x), n, n, 4).astype("float32")
    return freq, test_x, test_grid, test_y


def load_models(tag):
    spec = SPECS[tag]
    flat = MLPSiLU(
        d_in=spec["d_in"], d_out=2001, hidden=512, n_layers=4
    )
    neigh = ScaleInvariantSiLU(
        N=spec["n"], K=3, C=4, n_freq=2001, hidden=512, n_layers=4
    )
    flat_path = ROOT / "ckpts" / f"silu_{tag}v3_512x4_500ep_verified_seed0.pt"
    neigh_path = (
        ROOT / "ckpts"
        / f"silu_neigh_{tag}v3_K3_512x4_500ep_verified_seed0.pt"
    )
    flat.load_state_dict(torch.load(flat_path, map_location=DEVICE, weights_only=True))
    neigh.load_state_dict(torch.load(neigh_path, map_location=DEVICE, weights_only=True))
    return flat.eval(), neigh.eval()


@torch.inference_mode()
def predict(model, x):
    return model(torch.as_tensor(x, dtype=torch.float32)).cpu().numpy()


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(colors=INK, labelsize=10)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)


def plot_sample(tag, idx, freq, target, flat, neigh):
    flat_mse = float(np.mean((flat - target) ** 2))
    neigh_mse = float(np.mean((neigh - target) ** 2))

    fig, (ax, err_ax) = plt.subplots(
        2,
        1,
        figsize=(10.5, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3.1, 1.25], "hspace": 0.10},
    )
    ax.plot(
        freq, target, color=TARGET_COLOR, linewidth=2.8,
        label="Stored target T", zorder=5,
    )
    ax.plot(
        freq, flat, color=FLAT_COLOR, linewidth=1.8,
        label=f"Flat SiLU (MSE {flat_mse:.5f})", zorder=3,
    )
    ax.plot(
        freq, neigh, color=NEIGH_COLOR, linewidth=1.8,
        linestyle=(0, (6, 2)), label=f"SiLU K=3 (MSE {neigh_mse:.5f})", zorder=4,
    )

    err_ax.plot(
        freq, np.abs(flat - target), color=FLAT_COLOR, linewidth=1.5,
        label="Flat absolute error",
    )
    err_ax.plot(
        freq, np.abs(neigh - target), color=NEIGH_COLOR, linewidth=1.5,
        linestyle=(0, (6, 2)), label="K=3 absolute error",
    )

    ax.set_ylim(-0.03, 1.04)
    error_max = max(
        float(np.max(np.abs(flat - target))),
        float(np.max(np.abs(neigh - target))),
    )
    err_ax.set_ylim(0, max(0.05, min(1.0, error_max * 1.08)))
    ax.set_ylabel("Transmission T", color=INK)
    err_ax.set_ylabel("Absolute error", color=INK)
    err_ax.set_xlabel("Frequency (GHz)", color=INK)
    style_axis(ax)
    style_axis(err_ax)

    winner = "K=3" if neigh_mse < flat_mse else "Flat"
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.90),
        ncol=3, frameon=False, fontsize=10,
    )
    fig.suptitle(
        f"V3 {tag} Random Test Sample {idx}",
        fontsize=17, fontweight="bold", color=INK, y=0.985,
    )
    fig.text(
        0.5, 0.94,
        f"500 epochs, seed {SPLIT_SEED}; lower sample MSE: {winner}",
        ha="center", fontsize=10.5, color="#667581",
    )
    fig.subplots_adjust(
        left=0.10, right=0.985, bottom=0.10, top=0.82, hspace=0.10
    )

    out = PLOT_DIR / f"v3_{tag}_test_sample_{idx:04d}.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(
        f"{tag} sample {idx}: flat={flat_mse:.6f}, "
        f"K3={neigh_mse:.6f} -> {out}"
    )


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.labelsize": 11,
    })
    rng = np.random.default_rng(SAMPLE_SEED)

    for tag in SPECS:
        freq, test_x, test_grid, test_y = load_test(tag)
        indices = sorted(
            rng.choice(len(test_y), size=N_SAMPLES, replace=False).tolist()
        )
        flat_model, neigh_model = load_models(tag)
        flat_pred = predict(flat_model, test_x[indices])
        neigh_pred = predict(neigh_model, test_grid[indices])

        print(f"{tag} random test indices: {indices}")
        for row, idx in enumerate(indices):
            plot_sample(
                tag, idx, freq, test_y[idx], flat_pred[row], neigh_pred[row]
            )


if __name__ == "__main__":
    main()
