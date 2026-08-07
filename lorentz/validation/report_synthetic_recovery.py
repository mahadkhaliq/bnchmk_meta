"""Aggregate multi-seed synthetic unary recovery results and create figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import PARAMETER_NAMES, ROOT, write_json


DEFAULT_ROOT = (
    ROOT / "lorentz" / "experiments" / "unary_validation_1x1_20260806"
    / "synthetic_recovery"
)
COLORS = ("#2b6f84", "#c44e52", "#3f8f6b", "#d28e2b", "#7b5ea7")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def read_history(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        key: np.asarray([float(row[key]) for row in rows])
        for key in rows[0]
    }


def load_runs(root):
    runs = []
    for run_dir in sorted(root.glob("seed_*")):
        metrics_path = run_dir / "metrics.json"
        prediction_path = run_dir / "test_predictions.npz"
        history_path = run_dir / "history.csv"
        if not all(path.exists() for path in (metrics_path, prediction_path, history_path)):
            continue
        metrics = json.loads(metrics_path.read_text())
        if metrics.get("status") != "completed":
            continue
        with np.load(prediction_path) as data:
            predictions = {name: np.asarray(data[name]) for name in data.files}
        runs.append(
            {
                "dir": run_dir,
                "seed": int(metrics["seed"]),
                "metrics": metrics,
                "history": read_history(history_path),
                "predictions": predictions,
            }
        )
    if not runs:
        raise FileNotFoundError(f"No completed seed runs found under {root}.")
    return runs


def plot_training(output, runs):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    for index, run in enumerate(runs):
        history = run["history"]
        color = COLORS[index % len(COLORS)]
        label = f"Seed {run['seed']}"
        epoch = history["epoch"]
        axes[0].plot(epoch, history["train_beta2"], color=color, label=label)
        axes[1].plot(epoch, history["val_mse"], color=color, label=label)
        axes[2].plot(
            epoch,
            history["val_physical_parameter_standardized_mse"],
            color=color,
            label=label,
        )
    titles = (
        "Training spectrum loss",
        "Validation spectrum error",
        "Validation hidden-parameter error",
    )
    ylabels = (
        "Beta-2 weighted MSE",
        "Plain T MSE",
        "Standardized physical-parameter MSE",
    )
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.2)
    axes[0].legend(loc="upper right", frameon=False)
    path = output / "01_training_curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_metric_summary(output, runs):
    spectral = np.asarray([run["metrics"]["test"]["mse"] for run in runs])
    mean_predictor = np.asarray(
        [run["metrics"]["test"]["training_target_mean_mse"] for run in runs]
    )
    complex_s = np.asarray(
        [run["metrics"]["test"]["complex_s_mse"] for run in runs]
    )
    physical = np.asarray(
        [
            run["metrics"]["test"]["physical_parameter_standardized_mse"]
            for run in runs
        ]
    )
    labels = ("Training-target mean", "Unary T prediction")
    values = (mean_predictor, spectral)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)
    axes[0].bar(
        labels,
        [value.mean() for value in values],
        yerr=[value.std() for value in values],
        color=("#d28e2b", "#2b6f84"),
        capsize=4,
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Mean test MSE")
    axes[0].set_title("Synthetic spectrum recovery")
    axes[0].tick_params(axis="x", rotation=12)
    axes[0].grid(axis="y", alpha=0.2)

    diagnostic_labels = ("Complex S MSE", "Physical P std-MSE")
    diagnostic_values = (complex_s, physical)
    axes[1].bar(
        diagnostic_labels,
        [value.mean() for value in diagnostic_values],
        yerr=[value.std() for value in diagnostic_values],
        color=("#c44e52", "#3f8f6b"),
        capsize=4,
    )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Test diagnostic error")
    axes[1].set_title("Unsupervised intermediate recovery")
    axes[1].tick_params(axis="x", rotation=12)
    axes[1].grid(axis="y", alpha=0.2)
    path = output / "02_test_metrics.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_parameter_nrmse(output, runs):
    values = np.asarray(
        [
            [
                run["metrics"]["test"]["physical_parameter_metrics"][name][
                    "nrmse_by_target_std"
                ]
                for name in PARAMETER_NAMES
            ]
            for run in runs
        ],
        dtype=float,
    )
    x = np.arange(len(PARAMETER_NAMES))
    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    width = 0.8 / len(runs)
    for index, run in enumerate(runs):
        ax.bar(
            x - 0.4 + width / 2 + index * width,
            values[index],
            width=width,
            color=COLORS[index % len(COLORS)],
            label=f"Seed {run['seed']}",
        )
    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=0.9)
    ax.set_yscale("log")
    ax.set_xticks(x, PARAMETER_NAMES, rotation=22)
    ax.set_ylabel("Physical-parameter NRMSE / target standard deviation")
    parameter_weight = runs[0]["metrics"]["parameter_loss_weight"]
    title = (
        "Does spectrum-only training recover the teacher parameters?"
        if parameter_weight == 0
        else "Parameter-supervised control recovers the teacher parameters"
    )
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="upper right", frameon=False)
    path = output / "03_parameter_nrmse.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_parameter_scatter(output, run):
    prediction = run["predictions"]["prediction_physical"][:, 0]
    target = run["predictions"]["target_physical"][:, 0]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    for index, (ax, name) in enumerate(zip(axes.flat, PARAMETER_NAMES)):
        lower = float(min(target[:, index].min(), prediction[:, index].min()))
        upper = float(max(target[:, index].max(), prediction[:, index].max()))
        ax.scatter(
            target[:, index],
            prediction[:, index],
            s=8,
            alpha=0.28,
            color=COLORS[index % len(COLORS)],
            edgecolors="none",
        )
        ax.plot((lower, upper), (lower, upper), color="#222222", linewidth=1.0)
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_title(name)
        ax.set_xlabel("Teacher")
        ax.set_ylabel("Predicted")
        ax.grid(alpha=0.18)
    fig.suptitle(f"Physical parameter recovery, seed {run['seed']}")
    path = output / "04_parameter_scatter.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_spectra(output, run):
    data = run["predictions"]
    target = data["target_T"]
    prediction = data["prediction_T"]
    freq = data["freq_GHz"]
    error = np.mean((prediction - target) ** 2, axis=1)
    order = np.argsort(error)
    selected = (order[0], order[len(order) // 2], order[-1])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(freq, target.mean(0), color="#222222", linewidth=2.0, label="Teacher")
    axes[0, 0].plot(freq, prediction.mean(0), color="#2b6f84", linewidth=1.4, label="Unary")
    axes[0, 0].set_title("Mean synthetic spectrum")
    axes[0, 0].legend(frameon=False)
    for ax, index, label in zip(axes.flat[1:], selected, ("Best", "Median", "Worst")):
        ax.plot(freq, target[index], color="#222222", linewidth=1.8, label="Teacher")
        ax.plot(freq, prediction[index], color="#2b6f84", linewidth=1.2, label="Unary")
        ax.set_title(f"{label} sample | MSE={error[index]:.3e}")
    for ax in axes.flat:
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Power transmittance T")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.2)
    path = output / "05_spectra.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main():
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = args.experiment_root / "report"
    runs = load_runs(args.experiment_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plots = [
        plot_training(args.output_dir, runs),
        plot_metric_summary(args.output_dir, runs),
        plot_parameter_nrmse(args.output_dir, runs),
        plot_parameter_scatter(args.output_dir, runs[0]),
        plot_spectra(args.output_dir, runs[0]),
    ]
    rows = []
    for run in runs:
        test = run["metrics"]["test"]
        rows.append(
            {
                "seed": run["seed"],
                "best_epoch": run["metrics"]["best_epoch"],
                "test_T_mse": test["mse"],
                "test_beta2": test["beta2"],
                "test_complex_S_mse": test["complex_s_mse"],
                "test_physical_parameter_standardized_mse": test[
                    "physical_parameter_standardized_mse"
                ],
                "training_target_mean_mse": test["training_target_mean_mse"],
            }
        )
    csv_path = args.output_dir / "seed_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    scalar_names = tuple(name for name in rows[0] if name not in {"seed", "best_epoch"})
    aggregate = {
        name: {
            "mean": float(np.mean([float(row[name]) for row in rows])),
            "std": float(np.std([float(row[name]) for row in rows])),
        }
        for name in scalar_names
    }
    summary = {
        "experiment": "multi-seed self-consistent synthetic unary recovery",
        "completed_runs": len(runs),
        "seeds": [run["seed"] for run in runs],
        "parameter_loss_weight": runs[0]["metrics"]["parameter_loss_weight"],
        "aggregate": aggregate,
        "runs": rows,
        "artifacts": {
            "seed_summary": str(csv_path.resolve()),
            "plots": [str(path.resolve()) for path in plots],
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
