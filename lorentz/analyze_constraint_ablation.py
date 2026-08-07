"""Aggregate and plot a Lorentz output-mapping ablation campaign."""

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
from torch.utils.data import DataLoader, TensorDataset

from .lorentz import Model
from .run_constraint_ablation import PROFILES
from .train_1x1 import CONSTRAINT_NAMES, DEFAULT_DATASET, load_data


PROFILE_ORDER = tuple(PROFILES)
PROFILE_LABELS = {
    "all": "All mappings",
    "raw": "All raw",
    "wp_only": "wp only",
    "w0_only": "w0 only",
    "gamma_only": "gamma only",
    "epsilon_inf_only": "eps_inf only",
    "mu_inf_only": "mu_inf only",
    "drop_wp": "Without wp",
    "drop_w0": "Without w0",
    "drop_gamma": "Without gamma",
    "drop_epsilon_inf": "Without eps_inf",
    "drop_mu_inf": "Without mu_inf",
}
COLORS = {
    "all": "#27647b",
    "raw": "#c94f46",
    "wp_only": "#6c8e3c",
    "w0_only": "#d28b26",
    "gamma_only": "#73549d",
    "epsilon_inf_only": "#319e8f",
    "mu_inf_only": "#a95c87",
    "drop_wp": "#7c8da6",
    "drop_w0": "#b27341",
    "drop_gamma": "#7b5f4b",
    "drop_epsilon_inf": "#4d8993",
    "drop_mu_inf": "#8f6c79",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--spectrum-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def read_history(path):
    try:
        with path.open(newline="") as handle:
            return [
                {key: float(value) for key, value in row.items()}
                for row in csv.DictReader(handle)
            ]
    except FileNotFoundError:
        return []


def nested(mapping, *keys, default=np.nan):
    value = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value or value[key] is None:
            return default
        value = value[key]
    return value


def mean_parameter_fraction(metrics, names, statistic):
    values = [
        nested(metrics, "parameter_diagnostics", "physical", name, statistic)
        for name in names
    ]
    values = np.asarray(values, dtype=float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else np.nan


def collect_runs(experiment_root):
    rows = []
    for profile in PROFILE_ORDER:
        profile_dir = experiment_root / "runs" / profile
        if not profile_dir.exists():
            continue
        for run_dir in sorted(profile_dir.glob("seed_*")):
            config = read_json(run_dir / "config.json")
            status = read_json(run_dir / "status.json")
            metrics = read_json(run_dir / "metrics.json")
            history = read_history(run_dir / "history.csv")
            try:
                seed = int(config.get("seed", run_dir.name.removeprefix("seed_")))
            except ValueError:
                continue
            test = metrics.get("test") or {}
            failure = metrics.get("failure") or status.get("failure") or {}
            row = {
                "profile": profile,
                "seed": seed,
                "state": status.get("state", metrics.get("status", "missing")),
                "completed_epochs": metrics.get("completed_epochs", len(history)),
                "best_epoch": metrics.get("best_epoch", np.nan),
                "best_val_mse": metrics.get("best_val_mse", np.nan),
                "test_mse": test.get("mse", np.nan),
                "test_beta2": test.get("beta2", np.nan),
                "test_mae": test.get("mae", np.nan),
                "prediction_max": test.get("prediction_max", np.nan),
                "max_R_plus_T": test.get("max_R_plus_T", np.nan),
                "fraction_T_above_one": test.get("fraction_T_above_one", np.nan),
                "fraction_R_plus_T_above_one": test.get(
                    "fraction_R_plus_T_above_one", np.nan
                ),
                "negative_wp_fraction": mean_parameter_fraction(
                    metrics, ("wp_e", "wp_m"), "negative_fraction"
                ),
                "negative_w0_fraction": mean_parameter_fraction(
                    metrics, ("w0_e", "w0_m"), "negative_fraction"
                ),
                "w0_outside_bounds_fraction": mean_parameter_fraction(
                    metrics,
                    ("w0_e", "w0_m"),
                    "outside_mapping_bounds_fraction",
                ),
                "negative_gamma_fraction": mean_parameter_fraction(
                    metrics, ("gamma_e", "gamma_m"), "negative_fraction"
                ),
                "epsilon_inf_below_one_fraction": mean_parameter_fraction(
                    metrics, ("epsilon_inf",), "below_one_fraction"
                ),
                "mu_inf_below_one_fraction": mean_parameter_fraction(
                    metrics, ("mu_inf",), "below_one_fraction"
                ),
                "max_gradient_norm": max(
                    (entry.get("max_gradient_norm", np.nan) for entry in history),
                    default=np.nan,
                ),
                "failure_type": failure.get("type", ""),
                "failure_message": failure.get("message", ""),
                "failure_epoch": failure.get("failed_during_epoch", ""),
                "run_dir": str(run_dir),
            }
            constraints = config.get("constraints", {})
            row.update(
                {f"constraint_{name}": int(bool(constraints.get(name))) for name in CONSTRAINT_NAMES}
            )
            rows.append(row)
    return rows


def write_summary(rows, output):
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "run_summary.csv"
    if rows:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    profiles = {}
    for profile in PROFILE_ORDER:
        selected = [row for row in rows if row["profile"] == profile]
        if not selected:
            continue
        profile_summary = {
            "runs": len(selected),
            "completed": sum(row["state"] == "completed" for row in selected),
            "failed": sum(row["state"] == "failed" for row in selected),
        }
        for metric in (
            "best_val_mse",
            "test_mse",
            "test_beta2",
            "test_mae",
            "prediction_max",
            "max_R_plus_T",
            "negative_gamma_fraction",
            "max_gradient_norm",
        ):
            values = np.asarray([row[metric] for row in selected], dtype=float)
            finite = values[np.isfinite(values)]
            profile_summary[metric] = {
                "mean": float(finite.mean()) if finite.size else None,
                "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
                "count": int(finite.size),
            }
        profiles[profile] = profile_summary
    summary = {"profiles": profiles, "runs": rows}
    json_path = output / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    return csv_path, json_path


def profile_values(rows, metric):
    means = []
    errors = []
    for profile in PROFILE_ORDER:
        values = np.asarray(
            [row[metric] for row in rows if row["profile"] == profile], dtype=float
        )
        values = values[np.isfinite(values)]
        means.append(values.mean() if values.size else np.nan)
        errors.append(values.std(ddof=1) if values.size > 1 else 0.0)
    return np.asarray(means), np.asarray(errors)


def plot_performance(rows, output):
    labels = [PROFILE_LABELS[profile] for profile in PROFILE_ORDER]
    colors = [COLORS[profile] for profile in PROFILE_ORDER]
    mse, mse_std = profile_values(rows, "test_mse")
    beta2, beta2_std = profile_values(rows, "test_beta2")
    completion = np.asarray(
        [
            np.mean(
                [row["state"] == "completed" for row in rows if row["profile"] == profile]
            )
            if any(row["profile"] == profile for row in rows)
            else np.nan
            for profile in PROFILE_ORDER
        ]
    )

    x = np.arange(len(PROFILE_ORDER))
    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)
    axes[0].bar(x, mse, yerr=mse_std, color=colors, capsize=3)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Best-checkpoint test MSE")
    axes[0].set_title("Prediction accuracy")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, beta2, yerr=beta2_std, color=colors, capsize=3)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Best-checkpoint test beta-2")
    axes[1].set_title("Dip-weighted prediction error")
    axes[1].grid(axis="y", alpha=0.25)

    axes[2].bar(x, completion, color=colors)
    axes[2].set_ylim(0, 1.08)
    axes[2].set_ylabel("Completed fraction")
    axes[2].set_title("Training stability")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].set_xticks(x, labels, rotation=35, ha="right")
    fig.suptitle("1x1 Lorentz output-mapping ablation")
    fig.tight_layout()
    path = output / "01_performance_and_stability.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def aggregate_history(experiment_root, profile, key):
    series = []
    profile_dir = experiment_root / "runs" / profile
    for path in sorted(profile_dir.glob("seed_*/history.csv")):
        history = read_history(path)
        if history:
            series.append(np.asarray([row[key] for row in history], dtype=float))
    if not series:
        return np.asarray([]), np.asarray([]), np.asarray([])
    length = max(len(values) for values in series)
    stacked = np.full((len(series), length), np.nan)
    for index, values in enumerate(series):
        stacked[index, : len(values)] = values
    return (
        np.arange(1, length + 1),
        np.nanmean(stacked, axis=0),
        np.nanstd(stacked, axis=0),
    )


def plot_training(experiment_root, output):
    groups = (
        ("Add one mapping to raw", PROFILE_ORDER[:7]),
        ("Remove one mapping from all", ("all", *PROFILE_ORDER[7:])),
    )
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for row_index, (title, profiles) in enumerate(groups):
        for profile in profiles:
            epoch, mean, std = aggregate_history(
                experiment_root, profile, "val_mse"
            )
            if not epoch.size:
                continue
            color = COLORS[profile]
            axes[row_index, 0].plot(
                epoch, mean, color=color, label=PROFILE_LABELS[profile], linewidth=1.5
            )
            axes[row_index, 0].fill_between(
                epoch,
                np.maximum(mean - std, np.maximum(0.1 * mean, 1e-12)),
                mean + std,
                color=color,
                alpha=0.12,
            )

            grad_epoch, grad_mean, _ = aggregate_history(
                experiment_root, profile, "max_gradient_norm"
            )
            axes[row_index, 1].plot(
                grad_epoch,
                grad_mean,
                color=color,
                label=PROFILE_LABELS[profile],
                linewidth=1.25,
            )
        axes[row_index, 0].set_yscale("log")
        axes[row_index, 0].set_ylabel("Validation MSE")
        axes[row_index, 0].set_title(title)
        axes[row_index, 0].grid(alpha=0.25)
        axes[row_index, 0].legend(fontsize=8, ncol=2)
        axes[row_index, 1].set_yscale("log")
        axes[row_index, 1].set_ylabel("Maximum pre-clipping gradient norm")
        axes[row_index, 1].set_title(title)
        axes[row_index, 1].grid(alpha=0.25)
        axes[row_index, 1].legend(fontsize=8, ncol=2)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 1].set_xlabel("Epoch")
    fig.suptitle("Optimization behavior of output mappings")
    fig.tight_layout()
    path = output / "02_training_and_gradients.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_physics(rows, output):
    labels = [PROFILE_LABELS[profile] for profile in PROFILE_ORDER]
    prediction, _ = profile_values(rows, "prediction_max")
    energy, _ = profile_values(rows, "max_R_plus_T")
    diagnostic_names = (
        "negative_wp_fraction",
        "w0_outside_bounds_fraction",
        "negative_gamma_fraction",
        "epsilon_inf_below_one_fraction",
        "mu_inf_below_one_fraction",
    )
    heatmap = np.asarray(
        [profile_values(rows, metric)[0] for metric in diagnostic_names]
    )
    x = np.arange(len(PROFILE_ORDER))
    width = 0.38
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    axes[0].bar(x - width / 2, prediction, width, label="Maximum T", color="#d28b26")
    axes[0].bar(x + width / 2, energy, width, label="Maximum R+T", color="#27647b")
    axes[0].axhline(1.0, color="#333333", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("Mean maximum over seeds")
    axes[0].set_title("Power and passivity diagnostics")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()
    axes[0].set_xticks(x, labels, rotation=35, ha="right")

    image = axes[1].imshow(
        np.ma.masked_invalid(heatmap), aspect="auto", vmin=0.0, vmax=1.0, cmap="magma"
    )
    axes[1].set_yticks(
        np.arange(len(diagnostic_names)),
        (
            "wp < 0",
            "w0 outside mapped bounds",
            "gamma < 0",
            "eps_inf < 1",
            "mu_inf < 1",
        ),
    )
    axes[1].set_title("Fraction of learned effective parameters outside mapped ranges")
    fig.colorbar(image, ax=axes[1], label="Fraction")
    axes[1].set_xticks(x, labels, rotation=35, ha="right")
    fig.tight_layout()
    path = output / "03_physical_diagnostics.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_effect_sizes(rows, output):
    factors = tuple(CONSTRAINT_NAMES)
    only_profiles = {
        "wp": "wp_only",
        "w0": "w0_only",
        "gamma": "gamma_only",
        "epsilon_inf": "epsilon_inf_only",
        "mu_inf": "mu_inf_only",
    }
    drop_profiles = {name: f"drop_{name}" for name in factors}
    indexed = {(row["profile"], row["seed"]): row for row in rows}

    def paired_benefit(numerator_profile, denominator_profile):
        values = []
        seeds = sorted({row["seed"] for row in rows})
        for seed in seeds:
            numerator = indexed.get((numerator_profile, seed), {}).get("test_mse", np.nan)
            denominator = indexed.get((denominator_profile, seed), {}).get("test_mse", np.nan)
            if np.isfinite(numerator) and np.isfinite(denominator) and denominator > 0:
                values.append(np.log10(numerator / denominator))
        return np.asarray(values)

    add = [paired_benefit("raw", only_profiles[name]) for name in factors]
    remove = [paired_benefit(drop_profiles[name], "all") for name in factors]
    add_mean = np.asarray([value.mean() if value.size else np.nan for value in add])
    add_std = np.asarray([value.std(ddof=1) if value.size > 1 else 0.0 for value in add])
    remove_mean = np.asarray([value.mean() if value.size else np.nan for value in remove])
    remove_std = np.asarray([value.std(ddof=1) if value.size > 1 else 0.0 for value in remove])

    x = np.arange(len(factors))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.bar(
        x - width / 2,
        add_mean,
        width,
        yerr=add_std,
        capsize=3,
        color="#319e8f",
        label="Add mapping to raw model",
    )
    ax.bar(
        x + width / 2,
        remove_mean,
        width,
        yerr=remove_std,
        capsize=3,
        color="#73549d",
        label="Remove mapping from full model",
    )
    ax.axhline(0.0, color="#333333", linewidth=1.0)
    ax.set_xticks(x, ("wp", "w0", "gamma", "eps_inf", "mu_inf"))
    ax.set_ylabel("Mapping benefit, log10 best-checkpoint MSE ratio")
    ax.set_title("Positive values favor the mapping; completion is reported separately")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = output / "04_mapping_effect_sizes.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


@torch.no_grad()
def checkpoint_predictions(checkpoint_path, dataset, batch_size):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    frequency, _, splits, _ = load_data(dataset, checkpoint["seed"])
    model = Model(freq_GHz=frequency, **checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    x, target = splits["test"]
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size)
    predictions = []
    reflections = []
    for (geometry,) in loader:
        reflection, transmission = model(geometry)
        reflections.append(reflection.abs().square().numpy())
        predictions.append(transmission.abs().square().numpy())
    return frequency, target, np.concatenate(predictions), np.concatenate(reflections)


def plot_raw_endpoint(experiment_root, dataset, seed, batch_size, output):
    checkpoints = {
        profile: experiment_root / "runs" / profile / f"seed_{seed}" / "model.pt"
        for profile in ("all", "raw")
    }
    if not all(path.exists() for path in checkpoints.values()):
        return None
    constrained = checkpoint_predictions(checkpoints["all"], dataset, batch_size)
    raw = checkpoint_predictions(checkpoints["raw"], dataset, batch_size)
    frequency, target, constrained_prediction, constrained_reflection = constrained
    raw_frequency, raw_target, raw_prediction, raw_reflection = raw
    if not np.array_equal(frequency, raw_frequency) or not np.array_equal(target, raw_target):
        raise ValueError("Raw and constrained endpoint runs do not share a paired split.")

    constrained_sample_mse = np.mean((constrained_prediction - target) ** 2, axis=1)
    raw_sample_mse = np.mean((raw_prediction - target) ** 2, axis=1)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0, 0].plot(frequency, target.mean(axis=0), color="#222222", label="CST target", linewidth=2)
    axes[0, 0].plot(
        frequency,
        constrained_prediction.mean(axis=0),
        color=COLORS["all"],
        label="All mappings",
        linewidth=1.6,
    )
    axes[0, 0].plot(
        frequency,
        raw_prediction.mean(axis=0),
        color=COLORS["raw"],
        label="All raw, best checkpoint",
        linewidth=1.4,
    )
    axes[0, 0].set_xlabel("Frequency (GHz)")
    axes[0, 0].set_ylabel("Mean power transmittance")
    axes[0, 0].set_title("Held-out mean spectrum")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend()

    positive = np.concatenate((constrained_sample_mse, raw_sample_mse))
    positive = positive[positive > 0]
    bins = np.geomspace(positive.min(), positive.max(), 35)
    axes[0, 1].hist(constrained_sample_mse, bins=bins, alpha=0.65, color=COLORS["all"], label="All mappings")
    axes[0, 1].hist(raw_sample_mse, bins=bins, alpha=0.6, color=COLORS["raw"], label="All raw")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_xlabel("Per-sample MSE")
    axes[0, 1].set_ylabel("Test samples")
    axes[0, 1].set_title("Error distribution")
    axes[0, 1].legend()

    axes[1, 0].scatter(
        constrained_sample_mse,
        raw_sample_mse,
        s=18,
        alpha=0.6,
        color="#73549d",
        edgecolors="none",
    )
    bounds = positive.min(), positive.max()
    axes[1, 0].plot(bounds, bounds, color="#333333", linestyle="--", linewidth=1)
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("All mappings: sample MSE")
    axes[1, 0].set_ylabel("All raw: sample MSE")
    axes[1, 0].set_title("Paired sample errors")
    axes[1, 0].grid(alpha=0.2)

    endpoint_labels = ("All mappings", "All raw")
    max_t = (constrained_prediction.max(), raw_prediction.max())
    max_energy = (
        (constrained_prediction + constrained_reflection).max(),
        (raw_prediction + raw_reflection).max(),
    )
    x = np.arange(2)
    width = 0.36
    axes[1, 1].bar(x - width / 2, max_t, width, label="Maximum T", color="#d28b26")
    axes[1, 1].bar(x + width / 2, max_energy, width, label="Maximum R+T", color="#27647b")
    axes[1, 1].axhline(1.0, color="#333333", linestyle="--", linewidth=1)
    axes[1, 1].set_xticks(x, endpoint_labels)
    axes[1, 1].set_ylabel("Maximum on test set")
    axes[1, 1].set_title("Passivity symptoms")
    axes[1, 1].grid(axis="y", alpha=0.25)
    axes[1, 1].legend()
    fig.suptitle(f"Effect of bypassing every output mapping, paired seed {seed}")
    fig.tight_layout()
    path = output / "05_raw_vs_constrained.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    args = parse_args()
    args.experiment_root = args.experiment_root.resolve()
    args.dataset = args.dataset.resolve()
    output = (args.output or args.experiment_root / "plots").resolve()
    rows = collect_runs(args.experiment_root)
    if not rows:
        raise FileNotFoundError(f"No campaign runs found under {args.experiment_root}.")
    csv_path, json_path = write_summary(rows, output)
    figures = [
        plot_performance(rows, output),
        plot_training(args.experiment_root, output),
        plot_physics(rows, output),
        plot_effect_sizes(rows, output),
    ]
    raw_figure = plot_raw_endpoint(
        args.experiment_root,
        args.dataset,
        args.spectrum_seed,
        args.batch_size,
        output,
    )
    if raw_figure is not None:
        figures.append(raw_figure)
    print(
        json.dumps(
            {
                "runs_found": len(rows),
                "summary_csv": str(csv_path),
                "summary_json": str(json_path),
                "figures": [str(path) for path in figures],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
