"""Aggregate and plot the Lorentz mapping-constant ablation campaign."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .analyze_constraint_ablation import checkpoint_predictions, nested, read_history, read_json
from .run_mapping_constant_ablation import PROFILES
from .train_1x1 import DEFAULT_DATASET


PROFILE_ORDER = tuple(PROFILES)
PROFILE_LABELS = {
    "baseline": "Reference",
    "no_wp_scale": "No 0.5 wp scale",
    "no_wp_floor": "No wp floor",
    "wp_softplus_only": "wp = softplus(x)",
    "no_gamma_scale": "No 0.1 gamma scale",
    "no_gamma_floor": "No gamma floor",
    "gamma_softplus_only": "gamma = softplus(x)",
    "no_epsilon_offset": "No +1 eps_inf",
    "no_mu_offset": "No +1 mu_inf",
    "no_background_offsets": "No eps/mu +1",
}
COLORS = {
    "baseline": "#27647b",
    "no_wp_scale": "#6c8e3c",
    "no_wp_floor": "#8c6d31",
    "wp_softplus_only": "#4c78a8",
    "no_gamma_scale": "#d28b26",
    "no_gamma_floor": "#73549d",
    "gamma_softplus_only": "#4c78a8",
    "no_epsilon_offset": "#319e8f",
    "no_mu_offset": "#c94f46",
    "no_background_offsets": "#b05a8c",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--spectrum-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=tuple(PROFILES),
        default=None,
        help="Profiles to analyze; defaults to those recorded in manifest.json.",
    )
    return parser.parse_args()


def parameter_stat(metrics, names, statistic, reducer="mean"):
    values = np.asarray(
        [
            nested(metrics, "parameter_diagnostics", "physical", name, statistic)
            for name in names
        ],
        dtype=float,
    )
    values = values[np.isfinite(values)]
    if not values.size:
        return np.nan
    return float(values.min() if reducer == "min" else values.mean())


def collect_runs(experiment_root):
    rows = []
    for profile in PROFILE_ORDER:
        for run_dir in sorted((experiment_root / "runs" / profile).glob("seed_*")):
            config = read_json(run_dir / "config.json")
            status = read_json(run_dir / "status.json")
            metrics = read_json(run_dir / "metrics.json")
            history = read_history(run_dir / "history.csv")
            test = metrics.get("test") or {}
            failure = metrics.get("failure") or status.get("failure") or {}
            constants = config.get("constants", PROFILES[profile])
            rows.append(
                {
                    "profile": profile,
                    "seed": int(config.get("seed", run_dir.name.removeprefix("seed_"))),
                    "state": status.get("state", metrics.get("status", "missing")),
                    "completed_epochs": metrics.get("completed_epochs", len(history)),
                    "best_epoch": metrics.get("best_epoch", np.nan),
                    "best_val_mse": metrics.get("best_val_mse", np.nan),
                    "test_mse": test.get("mse", np.nan),
                    "test_beta2": test.get("beta2", np.nan),
                    "test_mae": test.get("mae", np.nan),
                    "prediction_max": test.get("prediction_max", np.nan),
                    "max_R_plus_T": test.get("max_R_plus_T", np.nan),
                    "fraction_R_plus_T_above_one": test.get(
                        "fraction_R_plus_T_above_one", np.nan
                    ),
                    "min_gamma": parameter_stat(
                        metrics, ("gamma_e", "gamma_m"), "min", reducer="min"
                    ),
                    "mean_gamma": parameter_stat(
                        metrics, ("gamma_e", "gamma_m"), "mean"
                    ),
                    "mean_wp": parameter_stat(metrics, ("wp_e", "wp_m"), "mean"),
                    "mean_epsilon_inf": parameter_stat(
                        metrics, ("epsilon_inf",), "mean"
                    ),
                    "mean_mu_inf": parameter_stat(metrics, ("mu_inf",), "mean"),
                    "epsilon_below_one_fraction": parameter_stat(
                        metrics, ("epsilon_inf",), "below_one_fraction"
                    ),
                    "mu_below_one_fraction": parameter_stat(
                        metrics, ("mu_inf",), "below_one_fraction"
                    ),
                    "max_gradient_norm": max(
                        (entry.get("max_gradient_norm", np.nan) for entry in history),
                        default=np.nan,
                    ),
                    "failure_type": failure.get("type", ""),
                    "failure_epoch": failure.get("failed_during_epoch", ""),
                    **constants,
                    "run_dir": str(run_dir),
                }
            )
    return rows


def finite_metric(rows, profile, metric, completed_only=False):
    values = np.asarray(
        [
            row[metric]
            for row in rows
            if row["profile"] == profile
            and (not completed_only or row["state"] == "completed")
        ],
        dtype=float,
    )
    return values[np.isfinite(values)]


def mean_std(rows, profile, metric):
    values = finite_metric(rows, profile, metric)
    if not values.size:
        return np.nan, 0.0
    return values.mean(), values.std(ddof=1) if values.size > 1 else 0.0


def write_summary(rows, output):
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "run_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    profiles = {}
    for profile in PROFILE_ORDER:
        selected = [row for row in rows if row["profile"] == profile]
        entry = {
            "constants": PROFILES[profile],
            "runs": len(selected),
            "completed": sum(row["state"] == "completed" for row in selected),
            "failed": sum(row["state"] == "failed" for row in selected),
        }
        for metric in (
            "test_mse",
            "test_beta2",
            "test_mae",
            "max_R_plus_T",
            "min_gamma",
            "mean_gamma",
            "mean_wp",
            "mean_epsilon_inf",
            "mean_mu_inf",
            "epsilon_below_one_fraction",
            "mu_below_one_fraction",
            "max_gradient_norm",
        ):
            mean, std = mean_std(rows, profile, metric)
            entry[metric] = {
                "mean": float(mean) if np.isfinite(mean) else None,
                "std": float(std),
            }
        profiles[profile] = entry
    summary = {"experiment": "mapping constant ablation", "profiles": profiles}
    json_path = output / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    return csv_path, json_path


def metric_arrays(rows, metric):
    values = [mean_std(rows, profile, metric) for profile in PROFILE_ORDER]
    return np.asarray([item[0] for item in values]), np.asarray(
        [item[1] for item in values]
    )


def plot_performance(rows, output):
    labels = [PROFILE_LABELS[profile] for profile in PROFILE_ORDER]
    colors = [COLORS[profile] for profile in PROFILE_ORDER]
    x = np.arange(len(PROFILE_ORDER))
    mse, mse_std = metric_arrays(rows, "test_mse")
    beta2, beta2_std = metric_arrays(rows, "test_beta2")
    completion = [
        np.mean(
            [row["state"] == "completed" for row in rows if row["profile"] == profile]
        )
        for profile in PROFILE_ORDER
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8))
    axes[0].bar(x, mse, yerr=mse_std, color=colors, capsize=3)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Test MSE")
    axes[0].set_title("Prediction error")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, beta2, yerr=beta2_std, color=colors, capsize=3)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Test beta-2")
    axes[1].set_title("Dip-weighted error")
    axes[1].grid(axis="y", alpha=0.25)
    axes[2].bar(x, completion, color=colors)
    axes[2].set_ylim(0, 1.08)
    axes[2].set_ylabel("Completed fraction")
    axes[2].set_title("500-epoch stability")
    axes[2].grid(axis="y", alpha=0.25)
    for ax in axes:
        ax.set_xticks(x, labels, rotation=30, ha="right")
    fig.suptitle("1x1 Lorentz mapping-constant ablation")
    fig.tight_layout()
    path = output / "01_performance_and_stability.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def aggregate_history(experiment_root, profile, key):
    series = []
    for path in sorted((experiment_root / "runs" / profile).glob("seed_*/history.csv")):
        history = read_history(path)
        if history:
            series.append(np.asarray([entry[key] for entry in history], dtype=float))
    if not series:
        return np.asarray([]), np.asarray([]), np.asarray([])
    length = max(len(values) for values in series)
    stacked = np.full((len(series), length), np.nan)
    for index, values in enumerate(series):
        stacked[index, : len(values)] = values
    return np.arange(1, length + 1), np.nanmean(stacked, axis=0), np.nanstd(
        stacked, axis=0
    )


def plot_training(experiment_root, output):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    for profile in PROFILE_ORDER:
        epoch, mean, std = aggregate_history(experiment_root, profile, "val_mse")
        color = COLORS[profile]
        axes[0].plot(epoch, mean, color=color, linewidth=1.45, label=PROFILE_LABELS[profile])
        axes[0].fill_between(
            epoch,
            np.maximum(mean - std, np.maximum(0.1 * mean, 1e-12)),
            mean + std,
            color=color,
            alpha=0.12,
        )
        grad_epoch, grad_mean, _ = aggregate_history(
            experiment_root, profile, "max_gradient_norm"
        )
        axes[1].plot(
            grad_epoch,
            grad_mean,
            color=color,
            linewidth=1.3,
            label=PROFILE_LABELS[profile],
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation MSE")
    axes[0].set_title("Learning curves")
    axes[0].grid(alpha=0.25)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Maximum pre-clipping gradient norm")
    axes[1].set_title("Optimization conditioning")
    axes[1].grid(alpha=0.25)
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = output / "02_training_and_gradients.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_physics(rows, output):
    labels = [PROFILE_LABELS[profile] for profile in PROFILE_ORDER]
    colors = [COLORS[profile] for profile in PROFILE_ORDER]
    x = np.arange(len(PROFILE_ORDER))
    max_energy, max_energy_std = metric_arrays(rows, "max_R_plus_T")
    min_gamma, min_gamma_std = metric_arrays(rows, "min_gamma")
    eps_below, _ = metric_arrays(rows, "epsilon_below_one_fraction")
    mu_below, _ = metric_arrays(rows, "mu_below_one_fraction")
    eps_mean, eps_mean_std = metric_arrays(rows, "mean_epsilon_inf")
    mu_mean, mu_mean_std = metric_arrays(rows, "mean_mu_inf")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0, 0].bar(x, max_energy, yerr=max_energy_std, color=colors, capsize=3)
    axes[0, 0].axhline(1.0, color="#333333", linestyle="--", linewidth=1)
    axes[0, 0].set_ylabel("Maximum R+T")
    axes[0, 0].set_title("Passivity")
    axes[0, 0].grid(axis="y", alpha=0.25)

    axes[0, 1].bar(x, min_gamma, yerr=min_gamma_std, color=colors, capsize=3)
    axes[0, 1].axhline(1e-4, color="#333333", linestyle="--", linewidth=1, label="Reference floor")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylabel("Smallest learned gamma")
    axes[0, 1].set_title("Distance from a zero-width pole")
    axes[0, 1].grid(axis="y", alpha=0.25)
    axes[0, 1].legend(fontsize=8)

    width = 0.36
    axes[1, 0].bar(x - width / 2, eps_below, width, color="#319e8f", label="eps_inf < 1")
    axes[1, 0].bar(x + width / 2, mu_below, width, color="#c94f46", label="mu_inf < 1")
    axes[1, 0].set_ylim(0, 1.08)
    axes[1, 0].set_ylabel("Fraction of test samples")
    axes[1, 0].set_title("Use of newly available background range")
    axes[1, 0].grid(axis="y", alpha=0.25)
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].bar(
        x - width / 2,
        eps_mean,
        width,
        yerr=eps_mean_std,
        color="#319e8f",
        capsize=3,
        label="Mean eps_inf",
    )
    axes[1, 1].bar(
        x + width / 2,
        mu_mean,
        width,
        yerr=mu_mean_std,
        color="#c94f46",
        capsize=3,
        label="Mean mu_inf",
    )
    axes[1, 1].axhline(1.0, color="#333333", linestyle="--", linewidth=1)
    axes[1, 1].set_ylabel("Learned background value")
    axes[1, 1].set_title("Effective backgrounds")
    axes[1, 1].grid(axis="y", alpha=0.25)
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xticks(x, labels, rotation=30, ha="right")
    fig.tight_layout()
    path = output / "03_physical_diagnostics.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_effect_sizes(rows, output):
    indexed = {(row["profile"], row["seed"]): row for row in rows}
    profiles = PROFILE_ORDER[1:]
    effects = []
    for profile in profiles:
        values = []
        for seed in sorted({row["seed"] for row in rows}):
            baseline = indexed.get(("baseline", seed), {}).get("test_mse", np.nan)
            ablation = indexed.get((profile, seed), {}).get("test_mse", np.nan)
            if np.isfinite(baseline) and np.isfinite(ablation) and ablation > 0:
                values.append(np.log10(baseline / ablation))
        effects.append(np.asarray(values))
    means = [value.mean() if value.size else np.nan for value in effects]
    stds = [value.std(ddof=1) if value.size > 1 else 0.0 for value in effects]
    colors = [COLORS[profile] for profile in profiles]
    labels = [PROFILE_LABELS[profile] for profile in profiles]
    x = np.arange(len(profiles))
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.bar(x, means, yerr=stds, color=colors, capsize=4)
    ax.axhline(0.0, color="#333333", linewidth=1)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("Removal benefit, log10(reference MSE / ablation MSE)")
    ax.set_title(
        "Positive means lower checkpoint MSE; failed runs use their best finite checkpoint"
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = output / "04_paired_effect_sizes.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_spectra(experiment_root, dataset, seed, batch_size, output):
    runs = {}
    frequency_reference = None
    target_reference = None
    for profile in PROFILE_ORDER:
        checkpoint = experiment_root / "runs" / profile / f"seed_{seed}" / "model.pt"
        frequency, target, prediction, reflection = checkpoint_predictions(
            checkpoint, dataset, batch_size
        )
        if frequency_reference is None:
            frequency_reference = frequency
            target_reference = target
        elif not np.array_equal(frequency_reference, frequency) or not np.array_equal(
            target_reference, target
        ):
            raise ValueError("Constant-ablation runs do not share a paired test split.")
        runs[profile] = {
            "prediction": prediction,
            "reflection": reflection,
            "sample_mse": np.mean((prediction - target) ** 2, axis=1),
        }
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.8))
    axes[0].plot(
        frequency_reference,
        target_reference.mean(axis=0),
        color="#222222",
        linewidth=2.2,
        label="CST target",
    )
    for profile in PROFILE_ORDER:
        axes[0].plot(
            frequency_reference,
            runs[profile]["prediction"].mean(axis=0),
            color=COLORS[profile],
            linewidth=1.35,
            label=PROFILE_LABELS[profile],
        )
    axes[0].set_xlabel("Frequency (GHz)")
    axes[0].set_ylabel("Mean power transmittance")
    axes[0].set_title("Held-out mean spectrum")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    all_errors = np.concatenate([run["sample_mse"] for run in runs.values()])
    positive = all_errors[all_errors > 0]
    bins = np.geomspace(positive.min(), positive.max(), 38)
    for profile in PROFILE_ORDER:
        axes[1].hist(
            runs[profile]["sample_mse"],
            bins=bins,
            histtype="step",
            linewidth=1.5,
            color=COLORS[profile],
            label=PROFILE_LABELS[profile],
        )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Per-sample MSE")
    axes[1].set_ylabel("Test samples")
    axes[1].set_title("Paired error distribution")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Mapping-constant ablation, paired seed {seed}")
    fig.tight_layout()
    path = output / "05_spectra_and_sample_errors.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    global PROFILE_ORDER
    args = parse_args()
    experiment_root = args.experiment_root.resolve()
    manifest = read_json(experiment_root / "manifest.json")
    PROFILE_ORDER = tuple(args.profiles or manifest.get("profiles", PROFILES))
    if not PROFILE_ORDER or PROFILE_ORDER[0] != "baseline":
        raise ValueError("The first analyzed profile must be baseline.")
    dataset = args.dataset.resolve()
    output = (args.output or experiment_root / "plots").resolve()
    rows = collect_runs(experiment_root)
    expected = len(PROFILE_ORDER) * len(manifest.get("seeds", (0, 1, 2)))
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} runs, found {len(rows)}.")
    csv_path, json_path = write_summary(rows, output)
    figures = [
        plot_performance(rows, output),
        plot_training(experiment_root, output),
        plot_physics(rows, output),
        plot_effect_sizes(rows, output),
        plot_spectra(
            experiment_root,
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
