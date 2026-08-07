"""Create test-set and training reports for the 1x1 Lorentz checkpoint."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .lorentz import Model
from .train_1x1 import DEFAULT_CHECKPOINT, DEFAULT_DATASET, load_data


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "artifacts" / "report_1x1_500ep"
TARGET_COLOR = "#d1495b"
PRED_COLOR = "#2166ac"
ERROR_COLOR = "#3a7d44"
INK = "#222222"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--history", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--random-count", type=int, default=6)
    parser.add_argument("--random-seed", type=int, default=20260730)
    return parser.parse_args()


def predict(model, geometry, batch_size):
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(geometry), batch_size):
            x = torch.from_numpy(geometry[start : start + batch_size])
            outputs.append(model.power_transmittance(x).numpy())
    return np.concatenate(outputs)


def geometry_label(feature_names, geometry):
    return ", ".join(
        f"{name}={value:.4f}" for name, value in zip(feature_names, geometry)
    )


def add_spectrum(ax, freq, target, prediction, title):
    ax.plot(freq, target, color=TARGET_COLOR, linewidth=1.8, label="CST target")
    ax.plot(freq, prediction, color=PRED_COLOR, linewidth=1.4, label="Lorentz")
    ax.set_title(title)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel(r"Power transmittance $T$")
    ax.grid(alpha=0.22)


def plot_aggregate(output, freq, target, prediction, per_sample_mse):
    per_frequency_mse = np.mean((prediction - target) ** 2, axis=0)
    mean_mse = float(per_sample_mse.mean())
    median_mse = float(np.median(per_sample_mse))

    fig = plt.figure(figsize=(11, 7.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.45, 1.0))
    mean_ax = fig.add_subplot(grid[0, :])
    freq_ax = fig.add_subplot(grid[1, 0])
    hist_ax = fig.add_subplot(grid[1, 1])

    mean_ax.plot(
        freq,
        target.mean(axis=0),
        color=TARGET_COLOR,
        linewidth=2.0,
        label="Mean CST target",
    )
    mean_ax.plot(
        freq,
        prediction.mean(axis=0),
        color=PRED_COLOR,
        linewidth=1.6,
        label="Mean Lorentz prediction",
    )
    mean_ax.set_title(
        f"1x1 held-out test aggregate | mean MSE={mean_mse:.8f}, "
        f"median sample MSE={median_mse:.8f}"
    )
    mean_ax.set_ylabel(r"Mean power transmittance $T$")
    mean_ax.set_ylim(-0.03, 1.03)
    mean_ax.grid(alpha=0.22)
    mean_ax.legend(frameon=False)

    freq_ax.plot(freq, per_frequency_mse, color=ERROR_COLOR, linewidth=1.3)
    freq_ax.axhline(mean_mse, color=INK, linestyle="--", linewidth=1.0, label="Test MSE")
    freq_ax.set_yscale("log")
    freq_ax.set_xlabel("Frequency (GHz)")
    freq_ax.set_ylabel("MSE across test samples")
    freq_ax.set_title("Error by frequency")
    freq_ax.grid(alpha=0.22)
    freq_ax.legend(frameon=False)

    positive = np.maximum(per_sample_mse, np.finfo(np.float32).tiny)
    bins = np.geomspace(positive.min(), positive.max(), 30)
    hist_ax.hist(positive, bins=bins, color="#6c8ebf", edgecolor="white")
    hist_ax.axvline(mean_mse, color=TARGET_COLOR, linewidth=1.5, label="Mean")
    hist_ax.axvline(median_mse, color=ERROR_COLOR, linewidth=1.5, label="Median")
    hist_ax.set_xscale("log")
    hist_ax.set_xlabel("Per-sample MSE")
    hist_ax.set_ylabel("Test samples")
    hist_ax.set_title("Per-sample error distribution")
    hist_ax.grid(alpha=0.18)
    hist_ax.legend(frameon=False)

    path = output / "01_mean_and_mse.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_best_worst(
    output, freq, target, prediction, geometry, feature_names, per_sample_mse
):
    selected = (int(np.argmin(per_sample_mse)), int(np.argmax(per_sample_mse)))
    labels = ("Best", "Worst")
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 7.5),
        sharex="col",
        gridspec_kw={"height_ratios": (3, 1)},
        constrained_layout=True,
    )
    for column, (sample, label) in enumerate(zip(selected, labels)):
        title = (
            f"{label} test sample | index={sample}, MSE={per_sample_mse[sample]:.8f}\n"
            f"{geometry_label(feature_names, geometry[sample, 0])}"
        )
        spectrum_ax = axes[0, column]
        add_spectrum(
            spectrum_ax, freq, target[sample], prediction[sample], title
        )
        if column == 0:
            spectrum_ax.legend(frameon=False)

        residual_ax = axes[1, column]
        residual_ax.axhline(0.0, color=INK, linewidth=0.8)
        residual_ax.plot(
            freq, prediction[sample] - target[sample], color=ERROR_COLOR, linewidth=1.1
        )
        residual_ax.set_xlabel("Frequency (GHz)")
        residual_ax.set_ylabel("Pred. - target")
        residual_ax.grid(alpha=0.22)

    path = output / "02_best_and_worst.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path, selected


def plot_random(
    output,
    freq,
    target,
    prediction,
    geometry,
    feature_names,
    per_sample_mse,
    count,
    seed,
):
    if count < 1:
        raise ValueError("--random-count must be positive.")
    count = min(count, len(target))
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(target), size=count, replace=False)
    columns = min(3, count)
    rows = math.ceil(count / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.0 * columns, 3.6 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, sample in zip(axes.flat, selected):
        add_spectrum(
            ax,
            freq,
            target[sample],
            prediction[sample],
            f"Test index {sample} | MSE={per_sample_mse[sample]:.6f}\n"
            f"{geometry_label(feature_names, geometry[sample, 0])}",
        )
    for ax in axes.flat[count:]:
        ax.set_visible(False)
    axes.flat[0].legend(frameon=False)
    fig.suptitle(f"Fixed-seed random 1x1 test samples | seed={seed}", fontsize=15)

    path = output / "03_random_samples.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path, selected.tolist()


def plot_training(output, history_path, best_epoch):
    with history_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    epoch = np.asarray([int(row["epoch"]) for row in rows])
    train_beta2 = np.asarray([float(row["train_beta2"]) for row in rows])
    val_mse = np.asarray([float(row["val_mse"]) for row in rows])
    val_beta2 = np.asarray([float(row["val_beta2"]) for row in rows])
    lr = np.asarray([float(row["lr"]) for row in rows])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    axes[0].plot(epoch, train_beta2, color=PRED_COLOR, linewidth=1.3)
    axes[0].set_title("Training beta-2 loss")
    axes[0].set_ylabel("Beta-2 weighted MSE")

    axes[1].plot(epoch, val_mse, color=TARGET_COLOR, linewidth=1.3, label="Plain MSE")
    axes[1].plot(epoch, val_beta2, color=ERROR_COLOR, linewidth=1.1, label="Beta-2")
    axes[1].scatter(
        [best_epoch],
        [val_mse[best_epoch - 1]],
        color=INK,
        s=32,
        zorder=4,
        label=f"Selected epoch {best_epoch}",
    )
    axes[1].set_title("Validation metrics")
    axes[1].set_ylabel("Loss")
    axes[1].legend(frameon=False)

    axes[2].plot(epoch, lr, color="#7a5195", linewidth=1.3)
    axes[2].set_title("Learning-rate schedule")
    axes[2].set_ylabel("Learning rate")

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_yscale("log")
        ax.grid(alpha=0.22)

    path = output / "04_training_curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_sample_metrics(
    output, geometry, feature_names, per_sample_mse, per_sample_beta2, per_sample_mae
):
    order = np.argsort(per_sample_mse)
    rank = np.empty_like(order)
    rank[order] = np.arange(1, len(order) + 1)
    path = output / "test_sample_metrics.csv"
    with path.open("w", newline="") as handle:
        fields = [
            "test_index",
            "mse_rank",
            "mse",
            "beta2",
            "mae",
            *feature_names,
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(len(geometry)):
            row = {
                "test_index": index,
                "mse_rank": int(rank[index]),
                "mse": float(per_sample_mse[index]),
                "beta2": float(per_sample_beta2[index]),
                "mae": float(per_sample_mae[index]),
            }
            row.update(
                {
                    name: float(value)
                    for name, value in zip(feature_names, geometry[index, 0])
                }
            )
            writer.writerow(row)
    return path


def main():
    args = parse_args()
    if args.history is None:
        args.history = args.checkpoint.with_suffix(".history.csv")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    freq, feature_names, splits, normalization = load_data(
        args.dataset, checkpoint["seed"]
    )
    test_x, test_y = splits["test"]

    model = Model(freq_GHz=freq, **checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    prediction = predict(model, test_x, args.batch_size)

    error = prediction - test_y
    weight = 1.0 + 2.0 * np.maximum(1.0 - test_y, 0.0) ** 2
    per_sample_mse = np.mean(error**2, axis=1)
    per_sample_beta2 = np.mean(weight * error**2, axis=1)
    per_sample_mae = np.mean(np.abs(error), axis=1)

    x_min = np.asarray(normalization["min"], dtype=np.float32).reshape(1, 1, 4)
    x_max = np.asarray(normalization["max"], dtype=np.float32).reshape(1, 1, 4)
    geometry = 0.5 * (test_x + 1.0) * (x_max - x_min) + x_min

    args.output.mkdir(parents=True, exist_ok=True)
    aggregate_path = plot_aggregate(
        args.output, freq, test_y, prediction, per_sample_mse
    )
    best_worst_path, best_worst = plot_best_worst(
        args.output,
        freq,
        test_y,
        prediction,
        geometry,
        feature_names,
        per_sample_mse,
    )
    random_path, random_samples = plot_random(
        args.output,
        freq,
        test_y,
        prediction,
        geometry,
        feature_names,
        per_sample_mse,
        args.random_count,
        args.random_seed,
    )
    training_path = plot_training(
        args.output, args.history, checkpoint["best_epoch"]
    )
    sample_metrics_path = write_sample_metrics(
        args.output,
        geometry,
        feature_names,
        per_sample_mse,
        per_sample_beta2,
        per_sample_mae,
    )

    summary = {
        "checkpoint": str(args.checkpoint),
        "history": str(args.history),
        "test_samples": len(test_y),
        "test_mse": float(per_sample_mse.mean()),
        "test_beta2": float(per_sample_beta2.mean()),
        "test_mae": float(per_sample_mae.mean()),
        "median_sample_mse": float(np.median(per_sample_mse)),
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_test_index": best_worst[0],
        "best_sample_mse": float(per_sample_mse[best_worst[0]]),
        "worst_test_index": best_worst[1],
        "worst_sample_mse": float(per_sample_mse[best_worst[1]]),
        "random_seed": args.random_seed,
        "random_test_indices": random_samples,
        "plots": [
            str(aggregate_path),
            str(best_worst_path),
            str(random_path),
            str(training_path),
        ],
        "sample_metrics": str(sample_metrics_path),
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
