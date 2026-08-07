"""Create direct-fit oracle figures from saved numerical artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .common import ROOT, write_json
from .fit_oracle_1x1 import (
    configure_plotting,
    plot_error_ladder,
    plot_optimization,
    plot_spectra,
)


DEFAULT_ROOT = (
    ROOT / "lorentz" / "experiments" / "unary_validation_1x1_20260806"
    / "oracle_v3_1x1"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    fits_path = args.experiment_root / "fits.npz"
    history_path = args.experiment_root / "optimization_history.csv"
    samples_path = args.experiment_root / "sample_metrics.csv"
    if not all(path.exists() for path in (fits_path, history_path, samples_path)):
        raise FileNotFoundError("The oracle numerical artifacts are incomplete.")

    with np.load(fits_path) as data:
        fits = {name: np.asarray(data[name]) for name in data.files}
    with history_path.open(newline="") as handle:
        history = []
        for row in csv.DictReader(handle):
            history.append(
                {
                    "sample_batch": int(row["sample_batch"]),
                    "stage": row["stage"],
                    "step": int(row["step"]),
                    "mean_best_mse": float(row["mean_best_mse"]),
                    "median_best_mse": float(row["median_best_mse"]),
                    "minimum_candidate_mse": float(row["minimum_candidate_mse"]),
                    "finite_candidate_fraction": float(row["finite_candidate_fraction"]),
                    "lr": float(row["lr"]),
                }
            )
    with samples_path.open(newline="") as handle:
        sample_rows = list(csv.DictReader(handle))
    per_sample = {
        name: np.asarray([float(row[name]) for row in sample_rows])
        for name in ("mean_mse", "reference_mse", "oracle_mse")
    }

    configure_plotting()
    ladder = plot_error_ladder(args.experiment_root, per_sample)
    spectra, selected = plot_spectra(
        args.experiment_root,
        fits["freq_GHz"],
        fits["target_T"],
        fits["reference_T"],
        fits["oracle_T"],
    )
    optimization = plot_optimization(args.experiment_root, history)
    report = {
        "experiment": "direct-fit oracle report",
        "selected_plot_examples": selected,
        "plots": [str(path.resolve()) for path in (ladder, spectra, optimization)],
    }
    write_json(args.experiment_root / "report.json", report)

    summary_path = args.experiment_root / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        summary.setdefault("artifacts", {})["plots"] = report["plots"]
        summary["selected_plot_examples"] = selected
        write_json(summary_path, summary)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
