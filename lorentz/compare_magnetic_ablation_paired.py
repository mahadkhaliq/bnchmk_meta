"""Run a paired test-set comparison for the 1x1 magnetic ablation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .lorentz import Model
from .train_1x1 import DEFAULT_CHECKPOINT, DEFAULT_DATASET, load_data


ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
DEFAULT_ZERO_MAGNETIC = (
    ARTIFACTS / "best_1x1_silu_beta2_512_500ep_ne1_nm0.pt"
)
DEFAULT_OUTPUT = ARTIFACTS / "compare_magnetic_ablation_500ep"

TARGET_COLOR = "#d1495b"
FULL_COLOR = "#2166ac"
ZERO_COLOR = "#3a7d44"
INK = "#222222"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--full", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--zero-magnetic", type=Path, default=DEFAULT_ZERO_MAGNETIC
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def validate_experiment(full, zero):
    if full["seed"] != zero["seed"]:
        raise ValueError("The checkpoints use different dataset-split seeds.")

    full_config = full["model_config"]
    zero_config = zero["model_config"]
    controlled = (
        "n_geom",
        "n_e",
        "hidden",
        "depth",
        "activation",
        "thickness_mm",
    )
    mismatched = [key for key in controlled if full_config[key] != zero_config[key]]
    if mismatched:
        raise ValueError(f"Non-ablation settings differ: {mismatched}")
    if full_config["n_m"] < 1:
        raise ValueError("The full checkpoint has no magnetic oscillator.")
    if zero_config["n_m"] != 0:
        raise ValueError("The zero-magnetic checkpoint must have n_m=0.")
    if not np.allclose(full["freq_GHz"], zero["freq_GHz"]):
        raise ValueError("The checkpoints use different frequency grids.")


def build_model(checkpoint, freq):
    model = Model(freq_GHz=freq, **checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def predict(model, geometry, batch_size):
    transmissions = []
    energy_sums = []
    with torch.no_grad():
        for start in range(0, len(geometry), batch_size):
            x = torch.from_numpy(geometry[start : start + batch_size])
            reflection, transmission = model(x)
            reflectance = reflection.abs().square()
            transmittance = transmission.abs().square()
            transmissions.append(transmittance.numpy())
            energy_sums.append((reflectance + transmittance).numpy())
    return np.concatenate(transmissions), np.concatenate(energy_sums)


def calculate_metrics(prediction, target, energy_sum):
    error = prediction.astype(np.float64) - target.astype(np.float64)
    target64 = target.astype(np.float64)
    weight = 1.0 + 2.0 * np.maximum(1.0 - target64, 0.0) ** 2
    squared_error = error**2
    per_sample_mse = squared_error.mean(axis=1)
    per_frequency_mse = squared_error.mean(axis=0)
    aggregate = {
        "mse": float(squared_error.mean()),
        "beta2": float((weight * squared_error).mean()),
        "mae": float(np.abs(error).mean()),
        "median_sample_mse": float(np.median(per_sample_mse)),
        "prediction_min": float(prediction.min()),
        "prediction_max": float(prediction.max()),
        "max_reflectance_plus_transmittance": float(energy_sum.max()),
    }
    return aggregate, per_sample_mse, per_frequency_mse


def load_history(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "epoch": np.asarray([int(row["epoch"]) for row in rows]),
        "val_mse": np.asarray([float(row["val_mse"]) for row in rows]),
    }


def plot_comparison(
    output,
    freq,
    target,
    full_prediction,
    zero_prediction,
    full_sample_mse,
    zero_sample_mse,
    full_frequency_mse,
    zero_frequency_mse,
    full_history,
    zero_history,
    full_best_epoch,
    zero_best_epoch,
):
    fig, axes = plt.subplots(
        2, 2, figsize=(12.5, 8.2), constrained_layout=True
    )

    mean_ax = axes[0, 0]
    mean_ax.plot(
        freq,
        target.mean(axis=0),
        color=TARGET_COLOR,
        linewidth=2.0,
        label="CST target",
    )
    mean_ax.plot(
        freq,
        full_prediction.mean(axis=0),
        color=FULL_COLOR,
        linewidth=1.5,
        label="Electric + magnetic",
    )
    mean_ax.plot(
        freq,
        zero_prediction.mean(axis=0),
        color=ZERO_COLOR,
        linewidth=1.4,
        label="Electric only",
    )
    mean_ax.set_title("Mean held-out spectrum")
    mean_ax.set_xlabel("Frequency (GHz)")
    mean_ax.set_ylabel(r"Mean power transmittance $T$")
    mean_ax.set_ylim(-0.03, 1.03)
    mean_ax.legend(frameon=False)

    frequency_ax = axes[0, 1]
    frequency_ax.plot(
        freq,
        full_frequency_mse,
        color=FULL_COLOR,
        linewidth=1.4,
        label="Electric + magnetic",
    )
    frequency_ax.plot(
        freq,
        zero_frequency_mse,
        color=ZERO_COLOR,
        linewidth=1.4,
        label="Electric only",
    )
    frequency_ax.set_title("Test MSE by frequency")
    frequency_ax.set_xlabel("Frequency (GHz)")
    frequency_ax.set_ylabel("MSE across test samples")
    frequency_ax.set_yscale("log")
    frequency_ax.legend(frameon=False)

    paired_ax = axes[1, 0]
    full_better = full_sample_mse < zero_sample_mse
    zero_better = zero_sample_mse < full_sample_mse
    paired_ax.scatter(
        full_sample_mse[full_better],
        zero_sample_mse[full_better],
        color=FULL_COLOR,
        s=18,
        alpha=0.7,
        label=f"Magnetic model lower: {full_better.sum()}",
    )
    paired_ax.scatter(
        full_sample_mse[zero_better],
        zero_sample_mse[zero_better],
        color=ZERO_COLOR,
        s=18,
        alpha=0.7,
        label=f"Electric-only lower: {zero_better.sum()}",
    )
    positive = np.concatenate((full_sample_mse, zero_sample_mse))
    lower = positive[positive > 0].min()
    upper = positive.max()
    paired_ax.plot(
        [lower, upper], [lower, upper], color=INK, linestyle="--", linewidth=1.0
    )
    paired_ax.set_xscale("log")
    paired_ax.set_yscale("log")
    paired_ax.set_xlim(lower * 0.8, upper * 1.25)
    paired_ax.set_ylim(lower * 0.8, upper * 1.25)
    paired_ax.set_title("Paired error on the same 300 samples")
    paired_ax.set_xlabel("Electric + magnetic sample MSE")
    paired_ax.set_ylabel("Electric-only sample MSE")
    paired_ax.legend(frameon=False)

    history_ax = axes[1, 1]
    history_ax.plot(
        full_history["epoch"],
        full_history["val_mse"],
        color=FULL_COLOR,
        linewidth=1.2,
        label=f"Electric + magnetic (best {full_best_epoch})",
    )
    history_ax.plot(
        zero_history["epoch"],
        zero_history["val_mse"],
        color=ZERO_COLOR,
        linewidth=1.2,
        label=f"Electric only (best {zero_best_epoch})",
    )
    history_ax.scatter(
        [full_best_epoch],
        [full_history["val_mse"][full_best_epoch - 1]],
        color=FULL_COLOR,
        s=28,
        zorder=4,
    )
    history_ax.scatter(
        [zero_best_epoch],
        [zero_history["val_mse"][zero_best_epoch - 1]],
        color=ZERO_COLOR,
        s=28,
        zorder=4,
    )
    history_ax.set_title("Matched validation histories")
    history_ax.set_xlabel("Epoch")
    history_ax.set_ylabel("Validation MSE")
    history_ax.set_yscale("log")
    history_ax.legend(frameon=False)

    for ax in axes.flat:
        ax.grid(alpha=0.22)

    fig.suptitle(
        "1x1 magnetic-oscillator ablation | one electric oscillator",
        fontsize=16,
    )
    path = output / "magnetic_ablation_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_paired_metrics(output, full_sample_mse, zero_sample_mse):
    path = output / "paired_test_metrics.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "test_index",
                "electric_plus_magnetic_mse",
                "electric_only_mse",
                "electric_only_over_full_ratio",
                "lower_mse_model",
            ),
        )
        writer.writeheader()
        for index, (full_mse, zero_mse) in enumerate(
            zip(full_sample_mse, zero_sample_mse)
        ):
            if full_mse < zero_mse:
                lower_model = "electric_plus_magnetic"
            elif zero_mse < full_mse:
                lower_model = "electric_only"
            else:
                lower_model = "tie"
            writer.writerow(
                {
                    "test_index": index,
                    "electric_plus_magnetic_mse": float(full_mse),
                    "electric_only_mse": float(zero_mse),
                    "electric_only_over_full_ratio": float(
                        zero_mse / max(full_mse, np.finfo(float).tiny)
                    ),
                    "lower_mse_model": lower_model,
                }
            )
    return path


def main():
    args = parse_args()
    full_checkpoint = torch.load(
        args.full, map_location="cpu", weights_only=False
    )
    zero_checkpoint = torch.load(
        args.zero_magnetic, map_location="cpu", weights_only=False
    )
    validate_experiment(full_checkpoint, zero_checkpoint)

    freq, _, splits, _ = load_data(args.dataset, full_checkpoint["seed"])
    if not np.allclose(freq, full_checkpoint["freq_GHz"]):
        raise ValueError("Dataset and checkpoint frequency grids differ.")
    test_x, test_y = splits["test"]

    full_model = build_model(full_checkpoint, freq)
    zero_model = build_model(zero_checkpoint, freq)
    full_prediction, full_energy_sum = predict(
        full_model, test_x, args.batch_size
    )
    zero_prediction, zero_energy_sum = predict(
        zero_model, test_x, args.batch_size
    )
    full_metrics, full_sample_mse, full_frequency_mse = calculate_metrics(
        full_prediction, test_y, full_energy_sum
    )
    zero_metrics, zero_sample_mse, zero_frequency_mse = calculate_metrics(
        zero_prediction, test_y, zero_energy_sum
    )

    full_history = load_history(args.full.with_suffix(".history.csv"))
    zero_history = load_history(
        args.zero_magnetic.with_suffix(".history.csv")
    )
    full_better = full_sample_mse < zero_sample_mse
    zero_better = zero_sample_mse < full_sample_mse
    tied_samples = ~(full_better | zero_better)
    full_frequency_better = full_frequency_mse < zero_frequency_mse
    zero_frequency_better = zero_frequency_mse < full_frequency_mse
    tied_frequencies = ~(full_frequency_better | zero_frequency_better)

    args.output.mkdir(parents=True, exist_ok=True)
    plot_path = plot_comparison(
        args.output,
        freq,
        test_y,
        full_prediction,
        zero_prediction,
        full_sample_mse,
        zero_sample_mse,
        full_frequency_mse,
        zero_frequency_mse,
        full_history,
        zero_history,
        full_checkpoint["best_epoch"],
        zero_checkpoint["best_epoch"],
    )
    csv_path = write_paired_metrics(
        args.output, full_sample_mse, zero_sample_mse
    )

    with torch.no_grad():
        raw_zero = zero_model.f1(torch.from_numpy(test_x))
        mu_inf = 1.0 + F.softplus(raw_zero[..., -1]).mean(dim=1)

    full_parameters = sum(parameter.numel() for parameter in full_model.parameters())
    zero_parameters = sum(parameter.numel() for parameter in zero_model.parameters())
    mse_ratio = zero_metrics["mse"] / full_metrics["mse"]
    summary = {
        "dataset": str(args.dataset),
        "test_samples": len(test_y),
        "frequency_points": len(freq),
        "controlled_settings": {
            "seed": full_checkpoint["seed"],
            "n_e": full_checkpoint["model_config"]["n_e"],
            "hidden": full_checkpoint["model_config"]["hidden"],
            "depth": full_checkpoint["model_config"]["depth"],
            "activation": full_checkpoint["model_config"]["activation"],
            "thickness_mm": full_checkpoint["model_config"]["thickness_mm"],
            "epochs": len(full_history["epoch"]),
        },
        "electric_plus_magnetic": {
            "checkpoint": str(args.full),
            "n_m": full_checkpoint["model_config"]["n_m"],
            "f1_output_size": 3
            * (
                full_checkpoint["model_config"]["n_e"]
                + full_checkpoint["model_config"]["n_m"]
            )
            + 2,
            "trainable_parameters": full_parameters,
            "best_epoch": int(full_checkpoint["best_epoch"]),
            "best_val_mse": float(full_checkpoint["best_val_mse"]),
            "test": full_metrics,
        },
        "electric_only": {
            "checkpoint": str(args.zero_magnetic),
            "n_m": zero_checkpoint["model_config"]["n_m"],
            "f1_output_size": 3
            * (
                zero_checkpoint["model_config"]["n_e"]
                + zero_checkpoint["model_config"]["n_m"]
            )
            + 2,
            "trainable_parameters": zero_parameters,
            "best_epoch": int(zero_checkpoint["best_epoch"]),
            "best_val_mse": float(zero_checkpoint["best_val_mse"]),
            "test": zero_metrics,
            "learned_mu_inf_test_range": [
                float(mu_inf.min()),
                float(mu_inf.max()),
            ],
        },
        "comparison": {
            "electric_only_over_full_test_mse_ratio": float(mse_ratio),
            "test_mse_increase_when_removing_magnetic_pct": float(
                100.0 * (mse_ratio - 1.0)
            ),
            "test_mse_reduction_with_magnetic_pct": float(
                100.0 * (1.0 - 1.0 / mse_ratio)
            ),
            "electric_plus_magnetic_lower_sample_mse_count": int(
                full_better.sum()
            ),
            "electric_only_lower_sample_mse_count": int(zero_better.sum()),
            "tied_sample_mse_count": int(tied_samples.sum()),
            "electric_plus_magnetic_lower_frequency_mse_points": int(
                full_frequency_better.sum()
            ),
            "electric_only_lower_frequency_mse_points": int(
                zero_frequency_better.sum()
            ),
            "tied_frequency_mse_points": int(tied_frequencies.sum()),
        },
        "plot": str(plot_path),
        "paired_test_metrics": str(csv_path),
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
