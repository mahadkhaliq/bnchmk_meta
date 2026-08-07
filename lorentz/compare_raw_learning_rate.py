"""Compare mapped and raw Lorentz outputs at standard and reduced learning rates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .analyze_constraint_ablation import (
    checkpoint_predictions,
    collect_runs,
    read_history,
)
from .train_1x1 import DEFAULT_DATASET


CONDITIONS = (
    ("mapped_standard", "All mappings, lr=3e-4", "standard", "all", "#27647b"),
    ("raw_standard", "All raw, lr=3e-4", "standard", "raw", "#c94f46"),
    ("mapped_low", "All mappings, lr=3e-5", "low", "all", "#319e8f"),
    ("raw_low", "All raw, lr=3e-5", "low", "raw", "#d28b26"),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--standard-root", type=Path, required=True)
    parser.add_argument("--low-lr-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--spectrum-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def select_rows(standard_rows, low_rows):
    selected = []
    for key, label, source, profile, color in CONDITIONS:
        source_rows = standard_rows if source == "standard" else low_rows
        for row in source_rows:
            if row["profile"] != profile:
                continue
            selected.append(
                {
                    **row,
                    "condition": key,
                    "condition_label": label,
                    "learning_rate_group": source,
                    "color": color,
                }
            )
    return selected


def finite(values):
    values = np.asarray(list(values), dtype=float)
    return values[np.isfinite(values)]


def summarize(rows):
    summary = {}
    for key, label, _, _, _ in CONDITIONS:
        condition_rows = [row for row in rows if row["condition"] == key]
        completed = [row for row in condition_rows if row["state"] == "completed"]
        entry = {
            "label": label,
            "runs": len(condition_rows),
            "completed": len(completed),
            "failed": len(condition_rows) - len(completed),
            "failure_epochs": [
                int(row["failure_epoch"])
                for row in condition_rows
                if str(row["failure_epoch"]).strip()
            ],
        }
        for metric in (
            "test_mse",
            "test_beta2",
            "max_R_plus_T",
            "negative_gamma_fraction",
            "max_gradient_norm",
        ):
            all_values = finite(row[metric] for row in condition_rows)
            complete_values = finite(row[metric] for row in completed)
            entry[metric] = {
                "all_best_checkpoints_mean": (
                    float(all_values.mean()) if all_values.size else None
                ),
                "all_best_checkpoints_std": (
                    float(all_values.std(ddof=1)) if all_values.size > 1 else 0.0
                ),
                "completed_only_mean": (
                    float(complete_values.mean()) if complete_values.size else None
                ),
                "completed_only_std": (
                    float(complete_values.std(ddof=1))
                    if complete_values.size > 1
                    else 0.0
                ),
            }
        summary[key] = entry
    return summary


def write_outputs(rows, summary, output):
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "raw_learning_rate_runs.csv"
    serializable_rows = [
        {key: value for key, value in row.items() if key != "color"} for row in rows
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(serializable_rows[0]))
        writer.writeheader()
        writer.writerows(serializable_rows)
    json_path = output / "raw_learning_rate_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    return csv_path, json_path


def condition_metric(rows, condition, metric):
    return [row[metric] for row in rows if row["condition"] == condition]


def scatter_metric(ax, rows, metric, *, log=False, reference=None):
    offsets = np.asarray((-0.12, 0.0, 0.12))
    for x, (key, _, _, _, color) in enumerate(CONDITIONS):
        condition_rows = sorted(
            (row for row in rows if row["condition"] == key),
            key=lambda row: row["seed"],
        )
        values = []
        for index, row in enumerate(condition_rows):
            value = float(row[metric])
            if not np.isfinite(value):
                continue
            values.append(value)
            marker = "o" if row["state"] == "completed" else "X"
            ax.scatter(
                x + offsets[index % len(offsets)],
                value,
                marker=marker,
                s=75,
                color=color,
                edgecolor="#222222" if marker == "o" else color,
                linewidth=0.7,
                zorder=3,
            )
        if values:
            ax.hlines(
                np.median(values), x - 0.24, x + 0.24, color="#222222", linewidth=1.5
            )
    if log:
        ax.set_yscale("log")
    if reference is not None:
        ax.axhline(reference, color="#333333", linestyle="--", linewidth=1.0)
    ax.grid(axis="y", alpha=0.25)


def plot_summary(rows, output):
    labels = [label for _, label, _, _, _ in CONDITIONS]
    colors = [color for _, _, _, _, color in CONDITIONS]
    x = np.arange(len(CONDITIONS))
    completion = [
        np.mean(
            [row["state"] == "completed" for row in rows if row["condition"] == key]
        )
        for key, _, _, _, _ in CONDITIONS
    ]

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    scatter_metric(axes[0, 0], rows, "test_mse", log=True)
    axes[0, 0].set_ylabel("Best-checkpoint test MSE")
    axes[0, 0].set_title("Prediction error")

    axes[0, 1].bar(x, completion, color=colors)
    axes[0, 1].set_ylim(0, 1.08)
    axes[0, 1].set_ylabel("Completed fraction")
    axes[0, 1].set_title("500-epoch completion")
    axes[0, 1].grid(axis="y", alpha=0.25)

    scatter_metric(axes[0, 2], rows, "completed_epochs")
    axes[0, 2].axhline(500, color="#333333", linestyle="--", linewidth=1)
    axes[0, 2].set_ylabel("Epochs completed")
    axes[0, 2].set_title("Failure timing")

    scatter_metric(axes[1, 0], rows, "max_R_plus_T", reference=1.0)
    axes[1, 0].set_ylabel("Maximum R+T")
    axes[1, 0].set_title("Passivity diagnostic")

    scatter_metric(axes[1, 1], rows, "negative_gamma_fraction")
    axes[1, 1].set_ylim(-0.03, 1.03)
    axes[1, 1].set_ylabel("Fraction gamma < 0")
    axes[1, 1].set_title("Learned negative damping")

    scatter_metric(axes[1, 2], rows, "max_gradient_norm", log=True)
    axes[1, 2].set_ylabel("Maximum pre-clipping gradient norm")
    axes[1, 2].set_title("Optimization stiffness")

    for ax in axes.flat:
        ax.set_xticks(x, labels, rotation=25, ha="right")
    axes[0, 0].scatter([], [], marker="o", color="#777777", label="Completed")
    axes[0, 0].scatter([], [], marker="X", color="#777777", label="Failed")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Raw-output learning-rate control, three paired seeds")
    fig.tight_layout()
    path = output / "01_raw_learning_rate_control.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def find_row(rows, condition, seed):
    return next(
        row
        for row in rows
        if row["condition"] == condition and int(row["seed"]) == int(seed)
    )


def plot_seed_comparison(rows, standard_root, low_root, dataset, seed, batch_size, output):
    roots = {"standard": standard_root, "low": low_root}
    predictions = {}
    target_reference = None
    frequency_reference = None
    for key, label, source, profile, color in CONDITIONS:
        checkpoint = roots[source] / "runs" / profile / f"seed_{seed}" / "model.pt"
        frequency, target, prediction, reflection = checkpoint_predictions(
            checkpoint, dataset, batch_size
        )
        if frequency_reference is None:
            frequency_reference = frequency
            target_reference = target
        elif not np.array_equal(frequency_reference, frequency) or not np.array_equal(
            target_reference, target
        ):
            raise ValueError("Learning-rate controls do not share a paired test split.")
        row = find_row(rows, key, seed)
        state_note = "completed" if row["state"] == "completed" else f"failed after {row['completed_epochs']} epochs"
        predictions[key] = {
            "label": f"{label}, {state_note}",
            "short_label": label,
            "color": color,
            "prediction": prediction,
            "reflection": reflection,
            "sample_mse": np.mean((prediction - target) ** 2, axis=1),
            "row": row,
        }

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes[0, 0].plot(
        frequency_reference,
        target_reference.mean(axis=0),
        color="#222222",
        linewidth=2.2,
        label="CST target",
    )
    for run in predictions.values():
        axes[0, 0].plot(
            frequency_reference,
            run["prediction"].mean(axis=0),
            color=run["color"],
            linewidth=1.35,
            label=run["label"],
        )
    axes[0, 0].set_xlabel("Frequency (GHz)")
    axes[0, 0].set_ylabel("Mean power transmittance")
    axes[0, 0].set_title("Held-out mean spectrum")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7)

    all_errors = np.concatenate([run["sample_mse"] for run in predictions.values()])
    positive = all_errors[all_errors > 0]
    bins = np.geomspace(positive.min(), positive.max(), 38)
    for run in predictions.values():
        axes[0, 1].hist(
            run["sample_mse"],
            bins=bins,
            histtype="step",
            linewidth=1.7,
            color=run["color"],
            label=run["short_label"],
        )
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_xlabel("Per-sample MSE")
    axes[0, 1].set_ylabel("Test samples")
    axes[0, 1].set_title("Error distribution")
    axes[0, 1].legend(fontsize=8)

    x = np.arange(len(CONDITIONS))
    width = 0.36
    max_t = [predictions[key]["prediction"].max() for key, *_ in CONDITIONS]
    max_energy = [
        (predictions[key]["prediction"] + predictions[key]["reflection"]).max()
        for key, *_ in CONDITIONS
    ]
    axes[1, 0].bar(x - width / 2, max_t, width, color="#d28b26", label="Maximum T")
    axes[1, 0].bar(x + width / 2, max_energy, width, color="#27647b", label="Maximum R+T")
    axes[1, 0].axhline(1.0, color="#333333", linestyle="--", linewidth=1)
    axes[1, 0].set_xticks(x, [label for _, label, _, _, _ in CONDITIONS], rotation=25, ha="right")
    axes[1, 0].set_ylabel("Maximum on test set")
    axes[1, 0].set_title("Passivity symptoms")
    axes[1, 0].grid(axis="y", alpha=0.25)
    axes[1, 0].legend(fontsize=8)

    for key, _, source, profile, color in CONDITIONS:
        history = read_history(
            roots[source] / "runs" / profile / f"seed_{seed}" / "history.csv"
        )
        axes[1, 1].plot(
            [row["epoch"] for row in history],
            [row["val_mse"] for row in history],
            color=color,
            linewidth=1.35,
            label=predictions[key]["short_label"],
        )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Validation MSE")
    axes[1, 1].set_title("Training trajectory")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(fontsize=8)
    fig.suptitle(f"Mapped versus raw outputs across learning rates, paired seed {seed}")
    fig.tight_layout()
    path = output / "02_raw_learning_rate_seed_comparison.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    args = parse_args()
    standard_root = args.standard_root.resolve()
    low_root = args.low_lr_root.resolve()
    dataset = args.dataset.resolve()
    output = (
        args.output or standard_root / "plots" / "raw_learning_rate_control"
    ).resolve()
    rows = select_rows(collect_runs(standard_root), collect_runs(low_root))
    if len(rows) != 12:
        raise ValueError(f"Expected 12 endpoint runs, found {len(rows)}.")
    summary = summarize(rows)
    csv_path, json_path = write_outputs(rows, summary, output)
    figures = [
        plot_summary(rows, output),
        plot_seed_comparison(
            rows,
            standard_root,
            low_root,
            dataset,
            args.spectrum_seed,
            args.batch_size,
            output,
        ),
    ]
    print(
        json.dumps(
            {
                "runs": len(rows),
                "summary_csv": str(csv_path),
                "summary_json": str(json_path),
                "figures": [str(path) for path in figures],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
