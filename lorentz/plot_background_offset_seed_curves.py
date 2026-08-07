"""Plot paired seed histories for the background-offset interaction study."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_ROOT = (
    Path(__file__).resolve().parent
    / "experiments"
    / "mapping_background_offsets_1x1_20260805"
)
PROFILES = {
    "baseline": {"label": "Reference", "color": "#27647b"},
    "no_background_offsets": {
        "label": "Remove both +1 offsets",
        "color": "#b05a8c",
    },
}
SEED_COLORS = {0: "#0072b2", 1: "#d55e00", 2: "#009e73"}
PROFILE_STYLES = {"baseline": "-", "no_background_offsets": "--"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--smoothing-window", type=int, default=11)
    return parser.parse_args()


def read_history(path):
    with path.open(newline="") as handle:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def read_json(path):
    return json.loads(path.read_text())


def moving_average(values, window):
    if window <= 1:
        return np.arange(len(values)), values
    kernel = np.ones(window, dtype=float) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    return np.arange(window - 1, len(values)), smoothed


def test_effect(reference, ablation):
    return 100.0 * (ablation / reference - 1.0)


def main():
    args = parse_args()
    root = args.experiment_root.resolve()
    output = (args.output or root / "plots" / "06_seed_training_curves.png").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    seeds = (0, 1, 2)
    histories = {}
    metrics = {}
    for seed in seeds:
        for profile in PROFILES:
            run_dir = root / "runs" / profile / f"seed_{seed}"
            histories[(profile, seed)] = read_history(run_dir / "history.csv")
            metrics[(profile, seed)] = read_json(run_dir / "metrics.json")

    fig, axes = plt.subplots(3, 2, figsize=(15, 13), sharex=True)
    summary_rows = []
    for row, seed in enumerate(seeds):
        baseline_test = metrics[("baseline", seed)]["test"]["mse"]
        ablation_test = metrics[("no_background_offsets", seed)]["test"]["mse"]
        effect = test_effect(baseline_test, ablation_test)
        for profile, style in PROFILES.items():
            history = histories[(profile, seed)]
            epoch = np.asarray([entry["epoch"] for entry in history])
            train = np.asarray([entry["train_beta2"] for entry in history])
            validation = np.asarray([entry["val_mse"] for entry in history])
            axes[row, 0].plot(
                epoch, train, color=style["color"], linewidth=0.7, alpha=0.22
            )
            axes[row, 1].plot(
                epoch, validation, color=style["color"], linewidth=0.7, alpha=0.22
            )
            train_index, train_smooth = moving_average(
                train, args.smoothing_window
            )
            val_index, val_smooth = moving_average(
                validation, args.smoothing_window
            )
            axes[row, 0].plot(
                epoch[train_index],
                train_smooth,
                color=style["color"],
                linewidth=1.8,
                label=style["label"],
            )
            axes[row, 1].plot(
                epoch[val_index],
                val_smooth,
                color=style["color"],
                linewidth=1.8,
                label=style["label"],
            )
            best_epoch = int(metrics[(profile, seed)]["best_epoch"])
            best_value = validation[best_epoch - 1]
            axes[row, 1].scatter(
                best_epoch,
                best_value,
                color=style["color"],
                edgecolor="white",
                marker="*",
                s=85,
                linewidth=0.7,
                zorder=4,
            )

        direction = "lower" if effect < 0 else "higher"
        axes[row, 0].set_title(f"Seed {seed}: training beta-2 loss")
        axes[row, 1].set_title(f"Seed {seed}: validation MSE")
        axes[row, 1].text(
            0.98,
            0.96,
            f"Test MSE\nReference: {baseline_test:.3e}\n"
            f"Remove both: {ablation_test:.3e}\n"
            f"{abs(effect):.1f}% {direction}",
            transform=axes[row, 1].transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
        )
        summary_rows.append(
            {
                "seed": seed,
                "reference_test_mse": baseline_test,
                "remove_both_test_mse": ablation_test,
                "percent_change": effect,
            }
        )

    for row in range(3):
        for column in range(2):
            axes[row, column].set_yscale("log")
            axes[row, column].set_ylabel(
                "Training beta-2" if column == 0 else "Validation MSE"
            )
            axes[row, column].grid(alpha=0.25)
            axes[row, column].legend(loc="lower left", fontsize=8)
    axes[2, 0].set_xlabel("Epoch")
    axes[2, 1].set_xlabel("Epoch")
    fig.suptitle(
        "Reference vs simultaneous epsilon/mu offset removal: paired seed curves"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)

    individual_paths = []
    for seed in seeds:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
        baseline_test = metrics[("baseline", seed)]["test"]["mse"]
        ablation_test = metrics[("no_background_offsets", seed)]["test"]["mse"]
        effect = test_effect(baseline_test, ablation_test)
        for profile, style in PROFILES.items():
            history = histories[(profile, seed)]
            epoch = np.asarray([entry["epoch"] for entry in history])
            train = np.asarray([entry["train_beta2"] for entry in history])
            validation = np.asarray([entry["val_mse"] for entry in history])
            for axis, values in zip(axes, (train, validation)):
                axis.plot(
                    epoch,
                    values,
                    color=style["color"],
                    linewidth=0.7,
                    alpha=0.22,
                )
                smooth_index, smooth_values = moving_average(
                    values, args.smoothing_window
                )
                axis.plot(
                    epoch[smooth_index],
                    smooth_values,
                    color=style["color"],
                    linewidth=1.8,
                    label=style["label"],
                )
            best_epoch = int(metrics[(profile, seed)]["best_epoch"])
            axes[1].scatter(
                best_epoch,
                validation[best_epoch - 1],
                color=style["color"],
                edgecolor="white",
                marker="*",
                s=90,
                linewidth=0.7,
                zorder=4,
            )
        direction = "lower" if effect < 0 else "higher"
        axes[0].set_title("Training beta-2 loss")
        axes[1].set_title("Validation MSE")
        axes[1].text(
            0.98,
            0.96,
            f"Test MSE\nReference: {baseline_test:.3e}\n"
            f"Remove both: {ablation_test:.3e}\n"
            f"{abs(effect):.1f}% {direction}",
            transform=axes[1].transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
        )
        for index, axis in enumerate(axes):
            axis.set_yscale("log")
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Training beta-2" if index == 0 else "Validation MSE")
            axis.grid(alpha=0.25)
            axis.legend(loc="lower left", fontsize=8)
        fig.suptitle(
            f"Seed {seed}: reference vs simultaneous epsilon/mu offset removal"
        )
        fig.tight_layout()
        individual_path = output.with_name(f"07_seed_{seed}_training_curves.png")
        fig.savefig(individual_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        individual_paths.append(individual_path)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.8))
    for seed in seeds:
        for profile, style in PROFILES.items():
            history = histories[(profile, seed)]
            epoch = np.asarray([entry["epoch"] for entry in history])
            train = np.asarray([entry["train_beta2"] for entry in history])
            validation = np.asarray([entry["val_mse"] for entry in history])
            label = f"Seed {seed}, {style['label']}"
            for axis, values in zip(axes, (train, validation)):
                axis.plot(
                    epoch,
                    values,
                    color=SEED_COLORS[seed],
                    linestyle=PROFILE_STYLES[profile],
                    linewidth=0.65,
                    alpha=0.12,
                )
                smooth_index, smooth_values = moving_average(
                    values, args.smoothing_window
                )
                axis.plot(
                    epoch[smooth_index],
                    smooth_values,
                    color=SEED_COLORS[seed],
                    linestyle=PROFILE_STYLES[profile],
                    linewidth=1.8,
                    label=label,
                )
            axes[0].scatter(
                epoch[-1],
                train[-1],
                color=SEED_COLORS[seed],
                marker="o" if profile == "baseline" else "s",
                s=28,
                edgecolor="white",
                linewidth=0.6,
                zorder=4,
            )
            best_epoch = int(metrics[(profile, seed)]["best_epoch"])
            axes[1].scatter(
                best_epoch,
                validation[best_epoch - 1],
                color=SEED_COLORS[seed],
                marker="*",
                s=70,
                edgecolor="white",
                linewidth=0.6,
                zorder=4,
            )
    axes[0].set_title("All seeds: training beta-2 loss")
    axes[1].set_title("All seeds: validation MSE")
    for index, axis in enumerate(axes):
        axis.set_yscale("log")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Training beta-2" if index == 0 else "Validation MSE")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right", fontsize=8, ncol=2)
    fig.suptitle(
        "Reference and simultaneous epsilon/mu offset removal on common axes"
    )
    fig.tight_layout()
    common_path = output.with_name("08_all_seeds_common_axes.png")
    fig.savefig(common_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    csv_path = output.with_name("06_seed_test_mse_summary.csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(output)
    for path in individual_paths:
        print(path)
    print(common_path)
    print(csv_path)


if __name__ == "__main__":
    main()
