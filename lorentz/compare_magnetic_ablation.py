"""Compare the 1x1 full Lorentz model with the zero-magnetic ablation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
DEFAULT_FULL = ARTIFACTS / "best_1x1_silu_beta2_512_500ep.pt"
DEFAULT_ZERO_MAGNETIC = (
    ARTIFACTS / "best_1x1_silu_beta2_512_500ep_ne1_nm0.pt"
)
DEFAULT_OUTPUT = ARTIFACTS / "report_1x1_500ep_ne1_nm0"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=Path, default=DEFAULT_FULL)
    parser.add_argument(
        "--zero-magnetic", type=Path, default=DEFAULT_ZERO_MAGNETIC
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_run(checkpoint_path):
    metrics_path = checkpoint_path.with_suffix(".metrics.json")
    history_path = checkpoint_path.with_suffix(".history.csv")
    metrics = json.loads(metrics_path.read_text())
    with history_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return {
        "checkpoint": checkpoint,
        "metrics": metrics,
        "epoch": np.asarray([int(row["epoch"]) for row in rows]),
        "val_mse": np.asarray([float(row["val_mse"]) for row in rows]),
        "parameter_count": sum(
            value.numel()
            for name, value in checkpoint["model_state"].items()
            if name.startswith("f1.")
        ),
    }


def main():
    args = parse_args()
    full = load_run(args.full)
    zero = load_run(args.zero_magnetic)
    args.output.mkdir(parents=True, exist_ok=True)

    full_test = full["metrics"]["test"]
    zero_test = zero["metrics"]["test"]
    metric_names = ("mse", "beta2", "mae")
    ratios = {
        name: zero_test[name] / full_test[name] for name in metric_names
    }
    comparison = {
        "full_model": {
            "oscillators": {"electric": 1, "magnetic": 1},
            "theta_dim": 8,
            "parameter_count": full["parameter_count"],
            "best_epoch": full["metrics"]["best_epoch"],
            "best_val_mse": full["metrics"]["best_val_mse"],
            "test": full_test,
        },
        "zero_magnetic": {
            "oscillators": {"electric": 1, "magnetic": 0},
            "theta_dim": 5,
            "parameter_count": zero["parameter_count"],
            "best_epoch": zero["metrics"]["best_epoch"],
            "best_val_mse": zero["metrics"]["best_val_mse"],
            "test": zero_test,
        },
        "zero_magnetic_over_full": ratios,
        "interpretation": (
            "Removing the magnetic oscillator also removes one dispersive pole. "
            "Transmission-only supervision cannot uniquely identify whether an "
            "effective pole is electric or magnetic."
        ),
    }
    json_path = args.output / "magnetic_oscillator_ablation.json"
    json_path.write_text(json.dumps(comparison, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    full_label = "1 electric + 1 magnetic"
    zero_label = "1 electric + 0 magnetic"

    axes[0].plot(full["epoch"], full["val_mse"], label=full_label, linewidth=1.5)
    axes[0].plot(zero["epoch"], zero["val_mse"], label=zero_label, linewidth=1.5)
    axes[0].scatter(
        full["metrics"]["best_epoch"],
        full["metrics"]["best_val_mse"],
        s=35,
        zorder=3,
    )
    axes[0].scatter(
        zero["metrics"]["best_epoch"],
        zero["metrics"]["best_val_mse"],
        s=35,
        zorder=3,
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation MSE")
    axes[0].set_title("Validation learning curves")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    x = np.arange(len(metric_names))
    width = 0.36
    axes[1].bar(
        x - width / 2,
        [full_test[name] for name in metric_names],
        width,
        label=full_label,
    )
    axes[1].bar(
        x + width / 2,
        [zero_test[name] for name in metric_names],
        width,
        label=zero_label,
    )
    axes[1].set_yscale("log")
    axes[1].set_xticks(x, ("MSE", "Beta-2", "MAE"))
    axes[1].set_ylabel("Held-out test error")
    axes[1].set_title("Test metrics")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    fig.suptitle("1x1 magnetic-oscillator ablation | same split, seed, and training")
    fig.tight_layout()
    figure_path = args.output / "05_magnetic_oscillator_ablation.png"
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps({"figure": str(figure_path), **comparison}, indent=2))


if __name__ == "__main__":
    main()
