"""Compare spectrum-only unary recovery with a parameter-supervised control."""

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


DEFAULT_EXPERIMENT = (
    ROOT / "lorentz" / "experiments" / "unary_validation_1x1_20260806"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spectrum-only-root",
        type=Path,
        default=DEFAULT_EXPERIMENT / "synthetic_recovery",
    )
    parser.add_argument(
        "--supervised-root",
        type=Path,
        default=DEFAULT_EXPERIMENT / "synthetic_recovery_supervised",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT / "synthetic_comparison",
    )
    return parser.parse_args()


def load_runs(root):
    runs = {}
    for run_dir in root.glob("seed_*"):
        path = run_dir / "metrics.json"
        if not path.exists():
            continue
        metrics = json.loads(path.read_text())
        if metrics.get("status") == "completed":
            runs[int(metrics["seed"])] = metrics
    if not runs:
        raise FileNotFoundError(f"No completed runs under {root}.")
    return runs


def paired_values(runs, seeds, key):
    return np.asarray([runs[seed]["test"][key] for seed in seeds], dtype=float)


def main():
    args = parse_args()
    spectrum = load_runs(args.spectrum_only_root)
    supervised = load_runs(args.supervised_root)
    seeds = sorted(set(spectrum).intersection(supervised))
    if not seeds:
        raise ValueError("The two campaigns have no paired seeds.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    conditions = {
        "Spectrum only": spectrum,
        "Parameter-supervised control": supervised,
    }
    colors = ("#2b6f84", "#c44e52")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    metric_specs = (
        ("mse", "Test T MSE", "Spectrum accuracy"),
        (
            "physical_parameter_standardized_mse",
            "Physical P standardized MSE",
            "Intermediate-parameter recovery",
        ),
    )
    for ax, (metric, ylabel, title) in zip(axes, metric_specs):
        for index, (label, runs) in enumerate(conditions.items()):
            values = paired_values(runs, seeds, metric)
            x = np.full(len(seeds), index, dtype=float)
            ax.scatter(x, values, color=colors[index], s=42, zorder=3)
            ax.bar(
                index,
                values.mean(),
                color=colors[index],
                alpha=0.45,
                width=0.65,
            )
        for seed_index in range(len(seeds)):
            ax.plot(
                (0, 1),
                (
                    paired_values(spectrum, seeds, metric)[seed_index],
                    paired_values(supervised, seeds, metric)[seed_index],
                ),
                color="#666666",
                alpha=0.45,
                linewidth=0.9,
            )
        ax.set_yscale("log")
        ax.set_xticks((0, 1), tuple(conditions), rotation=12)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    metrics_path = args.output_dir / "01_supervision_effect.png"
    fig.savefig(metrics_path, dpi=180)
    plt.close(fig)

    nrmse = {}
    for label, runs in conditions.items():
        nrmse[label] = np.asarray(
            [
                [
                    runs[seed]["test"]["physical_parameter_metrics"][name][
                        "nrmse_by_target_std"
                    ]
                    for name in PARAMETER_NAMES
                ]
                for seed in seeds
            ],
            dtype=float,
        )
    x = np.arange(len(PARAMETER_NAMES))
    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    width = 0.34
    for index, (label, values) in enumerate(nrmse.items()):
        offset = (index - 0.5) * width
        ax.bar(
            x + offset,
            values.mean(axis=0),
            yerr=values.std(axis=0),
            width=width,
            color=colors[index],
            capsize=3,
            label=label,
        )
    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=0.9)
    ax.set_yscale("log")
    ax.set_xticks(x, PARAMETER_NAMES, rotation=22)
    ax.set_ylabel("Physical-parameter NRMSE / target standard deviation")
    ax.set_title("Parameter supervision tests F1 representational capacity")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="upper right", frameon=False)
    parameter_path = args.output_dir / "02_parameter_recovery.png"
    fig.savefig(parameter_path, dpi=180)
    plt.close(fig)

    rows = []
    for label, runs in conditions.items():
        for seed in seeds:
            test = runs[seed]["test"]
            rows.append(
                {
                    "condition": label,
                    "seed": seed,
                    "test_T_mse": test["mse"],
                    "test_complex_S_mse": test["complex_s_mse"],
                    "physical_parameter_standardized_mse": test[
                        "physical_parameter_standardized_mse"
                    ],
                }
            )
    csv_path = args.output_dir / "comparison.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "experiment": "synthetic unary spectrum-only versus supervised control",
        "paired_seeds": seeds,
        "conditions": {
            label: {
                "parameter_loss_weight": runs[seeds[0]]["parameter_loss_weight"],
                "mean_test_T_mse": float(
                    paired_values(runs, seeds, "mse").mean()
                ),
                "mean_test_complex_S_mse": float(
                    paired_values(runs, seeds, "complex_s_mse").mean()
                ),
                "mean_physical_parameter_standardized_mse": float(
                    paired_values(
                        runs, seeds, "physical_parameter_standardized_mse"
                    ).mean()
                ),
            }
            for label, runs in conditions.items()
        },
        "interpretation": (
            "The supervised condition is a capacity control, not the primary "
            "physics-informed training result."
        ),
        "artifacts": {
            "comparison_csv": str(csv_path.resolve()),
            "plots": [str(metrics_path.resolve()), str(parameter_path.resolve())],
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
