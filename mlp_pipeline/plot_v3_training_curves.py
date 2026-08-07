"""Plot training and validation curves for all verified v3 experiments."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
HISTORY_DIR = ROOT / "logs" / "history"
PLOT_DIR = ROOT / "plots" / "v3_training_curves"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

INK = "#27313b"
GRID_COLOR = "#dce3e8"
FLAT_COLOR = "#d95f02"
K3_COLOR = "#007c83"

SPECS = {
    "1x1": {"samples": 2000, "flat_test": 0.000323, "k3_test": 0.000265},
    "2x2": {"samples": 6497, "flat_test": 0.008661, "k3_test": 0.008124},
    "3x3": {"samples": 397, "flat_test": 0.061176, "k3_test": 0.049949},
}


def load_history(tag, neighborhood=False):
    if neighborhood:
        name = f"silu_neigh_{tag}v3_K3_512x4_500ep_verified_seed0.csv"
    else:
        name = f"silu_{tag}v3_512x4_500ep_verified_seed0.csv"
    return np.genfromtxt(HISTORY_DIR / name, delimiter=",", names=True)


def ema(values, alpha=0.08):
    trend = np.empty_like(values)
    trend[0] = values[0]
    for idx in range(1, len(values)):
        trend[idx] = alpha * values[idx] + (1.0 - alpha) * trend[idx - 1]
    return trend


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(colors=INK, labelsize=10)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)
    ax.set_yscale("log")


def draw_curves(train_ax, val_ax, flat, k3, compact=False):
    configs = (
        ("Flat SiLU", flat, FLAT_COLOR, "-"),
        ("SiLU K=3", k3, K3_COLOR, (0, (6, 2))),
    )
    raw_width = 0.75 if compact else 0.9
    trend_width = 1.8 if compact else 2.2

    for label, history, color, linestyle in configs:
        epoch = history["epoch"]
        train = history["train_beta2"]
        val = history["val_mse"]
        train_ax.plot(epoch, train, color=color, linewidth=raw_width, alpha=0.20)
        train_ax.plot(
            epoch, ema(train), color=color, linewidth=trend_width,
            linestyle=linestyle, label=label,
        )
        val_ax.plot(epoch, val, color=color, linewidth=raw_width, alpha=0.20)
        val_ax.plot(
            epoch, ema(val), color=color, linewidth=trend_width,
            linestyle=linestyle, label=label,
        )

        best_idx = int(np.argmin(val))
        val_ax.scatter(
            epoch[best_idx], val[best_idx], color=color, s=55 if compact else 75,
            marker="o", edgecolor="white", linewidth=1.2, zorder=6,
        )
        if not compact:
            val_ax.annotate(
                f"best {val[best_idx]:.6f}\nepoch {int(epoch[best_idx])}",
                xy=(epoch[best_idx], val[best_idx]),
                xytext=(8, 8), textcoords="offset points",
                fontsize=8.5, color=color, fontweight="bold",
            )

    style_axis(train_ax)
    style_axis(val_ax)
    train_ax.set_ylabel("Train beta2 loss", color=INK)
    val_ax.set_ylabel("Validation MSE", color=INK)
    val_ax.set_xlabel("Epoch", color=INK)


def plot_dataset(tag):
    spec = SPECS[tag]
    flat = load_history(tag)
    k3 = load_history(tag, neighborhood=True)
    fig, (train_ax, val_ax) = plt.subplots(
        2, 1, figsize=(10.5, 7.2), sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12},
    )
    draw_curves(train_ax, val_ax, flat, k3)

    handles, labels = train_ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.90),
        ncol=2, frameon=False, fontsize=10.5,
    )
    fig.suptitle(
        f"V3 {tag} Training Curves",
        fontsize=17, fontweight="bold", color=INK, y=0.985,
    )
    fig.text(
        0.5, 0.94,
        f"{spec['samples']:,} samples, 500 epochs; test MSE: "
        f"Flat {spec['flat_test']:.6f}, K=3 {spec['k3_test']:.6f}",
        ha="center", fontsize=10.5, color="#667581",
    )
    fig.subplots_adjust(
        left=0.11, right=0.98, bottom=0.09, top=0.82, hspace=0.12
    )
    out = PLOT_DIR / f"v3_{tag}_training_curves.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)


def plot_overview():
    fig, axes = plt.subplots(
        2, 3, figsize=(16, 7.4), sharex=True,
        gridspec_kw={"hspace": 0.15, "wspace": 0.25},
    )
    for col, tag in enumerate(SPECS):
        flat = load_history(tag)
        k3 = load_history(tag, neighborhood=True)
        draw_curves(axes[0, col], axes[1, col], flat, k3, compact=True)
        axes[0, col].set_title(
            f"{tag}  |  {SPECS[tag]['samples']:,} samples",
            fontsize=11.5, fontweight="bold", color=INK,
        )
        if col:
            axes[0, col].set_ylabel("")
            axes[1, col].set_ylabel("")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.925),
        ncol=2, frameon=False, fontsize=10.5,
    )
    fig.suptitle(
        "V3 Training and Validation Curves",
        fontsize=18, fontweight="bold", color=INK, y=0.995,
    )
    fig.text(
        0.5, 0.955,
        "Raw curves shown faintly; bold lines are EMA trends; circles mark best validation epochs.",
        ha="center", fontsize=10.2, color="#667581",
    )
    fig.subplots_adjust(
        left=0.07, right=0.985, bottom=0.09, top=0.84,
        hspace=0.15, wspace=0.25,
    )
    out = PLOT_DIR / "v3_training_curves_overview.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.labelsize": 11,
    })
    for tag in SPECS:
        plot_dataset(tag)
    plot_overview()


if __name__ == "__main__":
    main()
