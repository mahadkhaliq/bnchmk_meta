"""Compare temporary 1x1 Lorentz wp/gamma scale experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / "artifacts" / "scale_ablation_100ep"
RUNS = (
    ("baseline", 0.5, 0.1, "wp0p5_gamma0p1"),
    ("unit wp scale", 1.0, 0.1, "wp1_gamma0p1"),
    ("unit gamma scale", 0.5, 1.0, "wp0p5_gamma1"),
    ("unit wp + gamma scales", 1.0, 1.0, "wp1_gamma1"),
)


def load_run(label, wp_scale, gamma_scale, stem):
    metrics_path = RUN_DIR / f"{stem}.metrics.json"
    history_path = RUN_DIR / f"{stem}.history.csv"
    metrics = json.loads(metrics_path.read_text())
    with history_path.open(newline="") as handle:
        history = list(csv.DictReader(handle))
    return {
        "label": label,
        "wp_scale": wp_scale,
        "gamma_scale": gamma_scale,
        "metrics": metrics,
        "epoch": np.asarray([int(row["epoch"]) for row in history]),
        "val_mse": np.asarray([float(row["val_mse"]) for row in history]),
    }


def main():
    runs = [load_run(*spec) for spec in RUNS]
    baseline = runs[0]["metrics"]["test"]
    summary = {
        "experiment": "100-epoch wp/gamma scale ablation",
        "controlled_variables": (
            "Same v3 1x1 split, seed 0, 512x3 SiLU model, beta-2 loss, "
            "optimizer, scheduler, and one electric plus one magnetic oscillator."
        ),
        "scope": (
            "The multipliers are set to one; positivity, frequency bounds, "
            "damping floor, background constraints, and passive branches remain."
        ),
        "runs": [],
        "caution": (
            "This is a temporary single-seed optimization screen, not a "
            "multi-seed or converged 500-epoch model-selection result."
        ),
    }
    for run in runs:
        test = run["metrics"]["test"]
        summary["runs"].append(
            {
                "label": run["label"],
                "wp_scale": run["wp_scale"],
                "gamma_scale": run["gamma_scale"],
                "best_epoch": run["metrics"]["best_epoch"],
                "best_val_mse": run["metrics"]["best_val_mse"],
                "test": test,
                "test_mse_change_percent": 100.0
                * (test["mse"] / baseline["mse"] - 1.0),
                "test_beta2_change_percent": 100.0
                * (test["beta2"] / baseline["beta2"] - 1.0),
                "test_mae_change_percent": 100.0
                * (test["mae"] / baseline["mae"] - 1.0),
            }
        )

    summary_path = RUN_DIR / "scale_ablation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for run in runs:
        axes[0].plot(
            run["epoch"],
            run["val_mse"],
            linewidth=1.35,
            label=(
                f"{run['label']} "
                f"({run['wp_scale']:g}, {run['gamma_scale']:g})"
            ),
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation MSE")
    axes[0].set_title("Validation learning curves")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    labels = [run["label"] for run in runs]
    values = [run["metrics"]["test"]["mse"] for run in runs]
    colors = ("#4477aa", "#228833", "#cc6677", "#aa3377")
    bars = axes[1].bar(np.arange(len(runs)), values, color=colors)
    axes[1].axhline(
        baseline["mse"],
        color="#333333",
        linestyle="--",
        linewidth=1.0,
        label="100-epoch baseline",
    )
    axes[1].set_xticks(np.arange(len(runs)), labels, rotation=18, ha="right")
    axes[1].set_ylabel("Held-out test MSE")
    axes[1].set_title("Selected-checkpoint test error")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)
    for bar, run in zip(bars, summary["runs"]):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{run['test_mse_change_percent']:+.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.suptitle(
        "1x1 Lorentz scale ablation | values are (wp scale, gamma scale)"
    )
    fig.tight_layout()
    figure_path = RUN_DIR / "scale_ablation_100ep.png"
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(
        json.dumps(
            {
                "figure": str(figure_path),
                "summary": str(summary_path),
                "runs": summary["runs"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
