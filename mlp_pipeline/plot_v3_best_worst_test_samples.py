"""Plot the best and worst test samples for every verified v3 model.

Best/worst is determined independently for Flat SiLU and SiLU K=3 using
per-sample plain MSE over all 2001 output frequencies.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from plot_v3_random_test_samples import (
    DATA_DIR,
    FLAT_COLOR,
    GRID_COLOR,
    INK,
    NEIGH_COLOR,
    PLOT_DIR as RANDOM_PLOT_DIR,
    SPECS,
    TARGET_COLOR,
    load_models,
    load_test,
)


ROOT = Path(__file__).resolve().parent
PLOT_DIR = RANDOM_PLOT_DIR.parent / "v3_best_worst_test_samples"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY = PLOT_DIR / "v3_best_worst_summary.csv"
DEVICE = torch.device("cpu")


@torch.inference_mode()
def predict_batched(model, x, batch_size=64):
    outputs = []
    tensor = torch.as_tensor(x, dtype=torch.float32)
    for start in range(0, len(tensor), batch_size):
        outputs.append(model(tensor[start:start + batch_size]).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(colors=INK, labelsize=10)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)


def plot_selected(
    tag, selected_model, rank, idx, original_row, freq, target, flat, neigh
):
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
    )
    err_ax.plot(
        freq, np.abs(neigh - target), color=NEIGH_COLOR, linewidth=1.5,
        linestyle=(0, (6, 2)),
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

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.90),
        ncol=3, frameon=False, fontsize=10,
    )
    model_label = "Flat SiLU" if selected_model == "flat" else "SiLU K=3"
    fig.suptitle(
        f"V3 {tag} {model_label} {rank.title()} Test Sample",
        fontsize=17, fontweight="bold", color=INK, y=0.985,
    )
    selected_mse = flat_mse if selected_model == "flat" else neigh_mse
    fig.text(
        0.5, 0.94,
        f"Test position {idx}, original NPZ row {original_row}; selected by "
        f"{model_label} MSE ({selected_mse:.6f})",
        ha="center", fontsize=10.5, color="#667581",
    )
    fig.subplots_adjust(
        left=0.10, right=0.985, bottom=0.10, top=0.82, hspace=0.10
    )

    out = PLOT_DIR / f"v3_{tag}_{selected_model}_{rank}_sample_{idx:04d}.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(
        f"{tag} {selected_model} {rank}: index={idx}, "
        f"flat={flat_mse:.6f}, K3={neigh_mse:.6f} -> {out}"
    )
    return flat_mse, neigh_mse


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.labelsize": 11,
    })
    rows = []

    for tag in SPECS:
        freq, test_x, test_grid, test_y = load_test(tag)
        with np.load(DATA_DIR / f"preprocessed_{tag}.npz", allow_pickle=True) as data:
            n_total = len(data["T"])
        _, original_test_rows = train_test_split(
            np.arange(n_total), test_size=0.15, random_state=0
        )
        flat_model, neigh_model = load_models(tag)
        flat_pred = predict_batched(flat_model, test_x)
        neigh_pred = predict_batched(neigh_model, test_grid)
        errors = {
            "flat": np.mean((flat_pred - test_y) ** 2, axis=1),
            "k3": np.mean((neigh_pred - test_y) ** 2, axis=1),
        }

        for model_name, sample_errors in errors.items():
            selections = {
                "best": int(np.argmin(sample_errors)),
                "worst": int(np.argmax(sample_errors)),
            }
            for rank, idx in selections.items():
                flat_mse, neigh_mse = plot_selected(
                    tag, model_name, rank, idx, int(original_test_rows[idx]),
                    freq, test_y[idx],
                    flat_pred[idx], neigh_pred[idx],
                )
                rows.append({
                    "dataset": tag,
                    "selected_model": model_name,
                    "rank": rank,
                    "test_position": idx,
                    "original_npz_row": int(original_test_rows[idx]),
                    "selected_mse": float(sample_errors[idx]),
                    "flat_mse": flat_mse,
                    "k3_mse": neigh_mse,
                })

    with SUMMARY.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print("saved", SUMMARY)


if __name__ == "__main__":
    main()
