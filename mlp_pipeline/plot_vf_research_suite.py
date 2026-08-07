"""Create publication-oriented figures for the stable VF ablation study.

The top configurations are ranked across the four native benchmarks using the
geometric mean of MSE / MSE(n_real=0, rel=none). This avoids ranking unlike
datasets directly by raw MSE.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

ROOT_HINT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT_HINT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from plot_vf_relative_ablation import (
    GRID,
    INK,
    MODE_LABELS,
    MUTED,
    ROOT,
    STUDY_ID,
    dataset_label,
    history_path,
    load_results,
    one_result,
    prediction_file,
    selected_spectrum,
)

CASES = (("v4", "2x2"), ("v4", "3x3"), ("v3", "2x2"), ("v3", "3x3"))
CONFIG_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")
CONFIG_STYLES = ("-", (0, (6, 2)), (0, (2, 1)), (0, (8, 2, 2, 2)), (0, (4, 2, 1, 2)))
SILU_BASELINES = {
    "2x2": {"SiLU flat": 0.008661, "SiLU K=3": 0.008124},
    "3x3": {"SiLU flat": 0.061176, "SiLU K=3": 0.049949},
}
DATASET_SPLITS = {
    ("v4", "2x2"): (13600, 3400, 3000),
    ("v4", "3x3"): (13600, 3400, 3000),
    ("v3", "2x2"): (4417, 1105, 975),
    ("v3", "3x3"): (269, 68, 60),
}
REL_DIMS = {"none": 0, "offset": 2, "offset_dist": 3, "embed": 8, "polar": 3}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "logs" / f"vf_rel_ablation_{STUDY_ID}" / "evaluations.csv",
    )
    parser.add_argument("--history-dir", type=Path, default=ROOT / "logs" / "history")
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=ROOT / "logs" / f"vf_rel_ablation_{STUDY_ID}" / "preds",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "plots" / f"vf_research_suite_{STUDY_ID}",
    )
    parser.add_argument("--top", type=int, default=5)
    return parser.parse_args()


def set_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_axis(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.75, alpha=0.9)
    ax.set_axisbelow(True)


def config_label(config):
    n_real, mode = config
    return f"{MODE_LABELS[mode]}, $n_{{real}}={n_real}$"


def native_rows(rows):
    return [row for row in rows if row["train_set"] == row["test_set"]]


def rank_configs(rows):
    native = native_rows(rows)
    reference = {}
    for dataset, grid in CASES:
        reference[(dataset, grid)] = one_result(
            native, dataset, dataset, grid, 0, "none"
        )["test_mse"]

    configs = sorted({(row["n_real"], row["mode"]) for row in native})
    ranked = []
    for config in configs:
        n_real, mode = config
        ratios = []
        raw = []
        for dataset, grid in CASES:
            mse = one_result(
                native, dataset, dataset, grid, n_real, mode
            )["test_mse"]
            ratios.append(mse / reference[(dataset, grid)])
            raw.append(mse)
        score = math.exp(np.log(ratios).mean())
        ranked.append(
            {"config": config, "score": score, "ratios": ratios, "mse": raw}
        )
    return sorted(ranked, key=lambda item: item["score"])


def save_figure(fig, output_dir, stem):
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, dpi in (("png", 220), ("pdf", 300)):
        fig.savefig(
            output_dir / f"{stem}.{suffix}",
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def write_ranking_csv(ranked, output_dir):
    path = output_dir / "top_configuration_ranking.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "relative_encoding",
                "n_real",
                "geometric_mean_normalized_mse",
                "v4_2x2_mse",
                "v4_3x3_mse",
                "v3_2x2_mse",
                "v3_3x3_mse",
            ]
        )
        for rank, item in enumerate(ranked, 1):
            n_real, mode = item["config"]
            writer.writerow([rank, mode, n_real, item["score"], *item["mse"]])


def write_cross_domain_csv(rows, output_dir):
    path = output_dir / "cross_domain_test_metrics.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    transfer = sorted(
        (row for row in rows if row["train_set"] != row["test_set"]),
        key=lambda row: (
            row["train_set"],
            row["test_set"],
            row["grid_size"],
            row["n_real"],
            row["mode"],
        ),
    )
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "train_dataset",
                "test_dataset",
                "grid",
                "relative_encoding",
                "n_real",
                "test_samples",
                "test_mse",
                "test_beta2",
                "checkpoint",
            ]
        )
        for row in transfer:
            writer.writerow(
                [
                    dataset_label(row["train_set"]),
                    dataset_label(row["test_set"]),
                    row["grid_size"],
                    row["mode"],
                    row["n_real"],
                    row["n"],
                    row["test_mse"],
                    row["test_beta2"],
                    row["checkpoint"],
                ]
            )
    if len(transfer) != 40:
        raise ValueError(f"Expected 40 cross-domain rows, got {len(transfer)}")


def write_all_experiment_configurations(rows, output_dir):
    path = output_dir / "all_80_experiment_configurations.csv"
    ordered = sorted(
        rows,
        key=lambda row: (
            row["train_set"],
            row["grid_size"],
            row["n_real"],
            row["mode"],
            row["test_set"],
        ),
    )
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "experiment_id",
                "evaluation_protocol",
                "train_dataset",
                "test_dataset",
                "grid",
                "train_samples",
                "validation_samples",
                "test_samples",
                "ablation_relative_encoding",
                "relative_feature_dimension",
                "ablation_n_real",
                "K",
                "n_pole",
                "hidden_width",
                "hidden_layers",
                "latent_dimension",
                "training_loss",
                "validation_metric",
                "reported_test_metrics",
                "epochs",
                "batch_size",
                "learning_rate",
                "weight_decay",
                "gradient_clip",
                "seed",
                "test_mse",
                "test_beta2",
                "checkpoint",
            ]
        )
        for experiment_id, row in enumerate(ordered, 1):
            train_count, val_count, _ = DATASET_SPLITS[
                (row["train_set"], row["grid_size"])
            ]
            writer.writerow(
                [
                    experiment_id,
                    (
                        "native"
                        if row["train_set"] == row["test_set"]
                        else "cross-domain transfer"
                    ),
                    dataset_label(row["train_set"]),
                    dataset_label(row["test_set"]),
                    row["grid_size"],
                    train_count,
                    val_count,
                    row["n"],
                    row["mode"],
                    REL_DIMS[row["mode"]],
                    row["n_real"],
                    3,
                    8,
                    512,
                    4,
                    64,
                    "beta2 weighted MSE",
                    "plain MSE",
                    "plain MSE and beta2 weighted MSE",
                    500,
                    128,
                    0.0003,
                    0.00001,
                    1.0,
                    0,
                    row["test_mse"],
                    row["test_beta2"],
                    row["checkpoint"],
                ]
            )
    if len(ordered) != 80:
        raise ValueError(f"Expected 80 experiment rows, got {len(ordered)}")


def plot_ablation_ranking(ranked, output_dir):
    labels = [config_label(item["config"]) for item in ranked]
    scores = np.array([item["score"] for item in ranked])
    y = np.arange(len(ranked))
    fig, ax = plt.subplots(figsize=(9.6, 6.0))
    colors = [
        CONFIG_COLORS[idx] if idx < len(CONFIG_COLORS) else "#A7B0B7"
        for idx in range(len(ranked))
    ]
    ax.barh(y, scores, color=colors, alpha=0.88, height=0.67)
    markers = ("o", "s", "^", "D")
    for case_idx, (dataset, grid) in enumerate(CASES):
        ax.scatter(
            [item["ratios"][case_idx] for item in ranked],
            y,
            marker=markers[case_idx],
            s=31,
            facecolor="white",
            edgecolor=INK,
            linewidth=0.8,
            zorder=4,
            label=f"{dataset_label(dataset)}, {grid}",
        )
    ax.axvline(1.0, color=INK, linewidth=1.1, linestyle=(0, (3, 2)))
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Normalized test MSE (lower is better)")
    fig.suptitle(
        "Relative-Position Ablation: Cross-Benchmark Ranking",
        fontsize=15,
        fontweight="bold",
        color=INK,
        y=0.995,
    )
    fig.text(
        0.5,
        0.945,
        "Bars: geometric mean across four native tasks; symbols: individual tasks. "
        "Reference = none, $n_{real}=0$.",
        ha="center",
        color=MUTED,
        fontsize=9,
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        fontsize=8,
    )
    clean_axis(ax, "x")
    fig.subplots_adjust(top=0.82)
    save_figure(fig, output_dir, "01_ablation_ranking")


def plot_v3_baselines(rows, top, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    for ax, grid in zip(axes, ("2x2", "3x3")):
        labels = list(SILU_BASELINES[grid])
        values = list(SILU_BASELINES[grid].values())
        colors = ["#7A8790", "#343A40"]
        for idx, item in enumerate(top):
            n_real, mode = item["config"]
            labels.append(config_label(item["config"]))
            values.append(
                one_result(rows, "v3", "v3", grid, n_real, mode)["test_mse"]
            )
            colors.append(CONFIG_COLORS[idx])
        x = np.arange(len(labels))
        bars = ax.bar(x, values, color=colors, width=0.72)
        best = min(values)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + max(values) * 0.018,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=INK,
            )
            if value == best:
                bar.set_edgecolor("#F2B134")
                bar.set_linewidth(2.3)
        ax.set_xticks(x, labels, rotation=28, ha="right")
        ax.set_ylabel("Test MSE")
        ax.set_title(f"v3 CST, {grid}")
        ax.set_ylim(0, max(values) * 1.19)
        clean_axis(ax)
    fig.suptitle(
        "VF Ablations Compared with Stored SiLU Baselines",
        fontsize=15,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.5,
        0.925,
        "Same v3 files, deterministic seed 0, and 68/17/15 sample split; "
        "all values are plain test MSE.",
        ha="center",
        color=MUTED,
        fontsize=9,
    )
    fig.subplots_adjust(top=0.83, bottom=0.30, wspace=0.22)
    save_figure(fig, output_dir, "02_v3_baseline_comparison")


def plot_two_metric_protocols(rows, top, output_dir, cross_domain=False):
    if cross_domain:
        cases = (
            ("v4", "v3", "2x2"),
            ("v4", "v3", "3x3"),
            ("v3", "v4", "2x2"),
            ("v3", "v4", "3x3"),
        )
        stem = "06_top5_cross_domain_mse_vs_beta2"
        title = "Top-Five Configurations: Cross-Domain Test Metrics"
    else:
        cases = tuple((dataset, dataset, grid) for dataset, grid in CASES)
        stem = "05_top5_native_mse_vs_beta2"
        title = "Top-Five Configurations: Native-Domain Test Metrics"

    fig, axes = plt.subplots(1, 4, figsize=(16.0, 5.2))
    x = np.arange(1, len(top) + 1)
    for ax, (train_set, test_set, grid) in zip(axes, cases):
        mse = []
        beta2 = []
        for item in top:
            n_real, mode = item["config"]
            row = one_result(
                rows, train_set, test_set, grid, n_real, mode
            )
            mse.append(row["test_mse"])
            beta2.append(row["test_beta2"])
        for xpos, low, high in zip(x, mse, beta2):
            ax.plot(
                [xpos, xpos],
                [low, high],
                color="#B8C2C9",
                linewidth=1.2,
                zorder=1,
            )
        ax.plot(
            x,
            mse,
            color="#0072B2",
            marker="o",
            linewidth=1.8,
            markersize=6,
            markerfacecolor="white",
            markeredgewidth=1.4,
            label="Plain MSE",
            zorder=3,
        )
        ax.plot(
            x,
            beta2,
            color="#D55E00",
            marker="s",
            linestyle=(0, (5, 2)),
            linewidth=1.8,
            markersize=5.5,
            markerfacecolor="white",
            markeredgewidth=1.4,
            label=r"Weighted MSE ($\beta=2$)",
            zorder=3,
        )
        heading = (
            f"{dataset_label(train_set)}, {grid}"
            if train_set == test_set
            else f"{train_set} $\\rightarrow$ {test_set}, {grid}"
        )
        ax.set_title(heading)
        ax.set_xticks(x, [f"#{idx}" for idx in x])
        ax.set_xlabel("Aggregate configuration rank")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        clean_axis(ax)
    axes[0].set_ylabel("Held-out test error")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.865),
        ncol=2,
    )
    fig.suptitle(title, fontsize=15, fontweight="bold", color=INK, y=0.995)
    fig.text(
        0.5,
        0.935,
        "Ranks: #1 embed/$n_{real}=4$, #2 embed/0, #3 polar/4, "
        "#4 offset+distance/4, #5 offset/4.",
        ha="center",
        color=MUTED,
        fontsize=9,
    )
    fig.subplots_adjust(top=0.74, bottom=0.15, left=0.065, right=0.99, wspace=0.28)
    save_figure(fig, output_dir, stem)


def load_native_prediction(rows, pred_dir, dataset, grid, config):
    n_real, mode = config
    row = one_result(rows, dataset, dataset, grid, n_real, mode)
    return row, np.load(prediction_file(pred_dir, row["tag"]))


def positive_log_bins(arrays, count=42):
    values = np.concatenate([np.asarray(array) for array in arrays])
    positive = values[values > 0]
    low = max(float(np.min(positive)), 1e-9)
    high = float(np.max(positive))
    return np.geomspace(low, high * 1.001, count)


def plot_mse_histograms(rows, pred_dir, top, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.6))
    for ax, (dataset, grid) in zip(axes.flat, CASES):
        loaded = [
            load_native_prediction(rows, pred_dir, dataset, grid, item["config"])
            for item in top
        ]
        losses = [data["per_sample_mse"] for _, data in loaded]
        bins = positive_log_bins(losses)
        for idx, (item, (_, data)) in enumerate(zip(top, loaded)):
            values = data["per_sample_mse"]
            ax.hist(
                values,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.8,
                color=CONFIG_COLORS[idx],
                linestyle=CONFIG_STYLES[idx],
                label=config_label(item["config"]),
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Per-sample spectral MSE")
        ax.set_ylabel("Probability density")
        ax.set_title(
            f"{dataset_label(dataset)}, {grid} (test n={len(losses[0]):,})"
        )
        clean_axis(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.905),
        ncol=5,
        fontsize=8,
    )
    fig.suptitle(
        "Test-Set Error Distributions for the Top Five Ablation Configurations",
        fontsize=15,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.5,
        0.94,
        "Histograms use identical log-spaced bins within each benchmark. "
        "v4 is synthetic simulation data; v3 is CST-generated data.",
        ha="center",
        color=MUTED,
        fontsize=9,
    )
    fig.subplots_adjust(top=0.82, hspace=0.32, wspace=0.22)
    save_figure(fig, output_dir, "03_test_mse_histograms")


def plot_individual_mse_histograms(rows, pred_dir, top, output_dir):
    histogram_dir = output_dir / "individual_test_mse_histograms"
    for dataset, grid in CASES:
        loaded = [
            load_native_prediction(rows, pred_dir, dataset, grid, item["config"])
            for item in top
        ]
        losses = [data["per_sample_mse"] for _, data in loaded]
        bins = positive_log_bins(losses)
        fig, ax = plt.subplots(figsize=(9.2, 5.8))
        for idx, (item, (_, data)) in enumerate(zip(top, loaded)):
            values = data["per_sample_mse"]
            ax.hist(
                values,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=2.0,
                color=CONFIG_COLORS[idx],
                linestyle=CONFIG_STYLES[idx],
                label=(
                    f"{config_label(item['config'])} "
                    f"(median {np.median(values):.2e})"
                ),
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Per-sample spectral MSE")
        ax.set_ylabel("Probability density")
        ax.set_title(
            f"{dataset_label(dataset)}, {grid}: Held-Out Test Dataset",
            pad=12,
        )
        ax.text(
            0.0,
            1.015,
            f"Test samples: n={len(losses[0]):,}; identical log-spaced bins "
            "for all five configurations.",
            transform=ax.transAxes,
            color=MUTED,
            fontsize=9,
        )
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=2,
            fontsize=8.2,
        )
        clean_axis(ax)
        fig.subplots_adjust(top=0.86, bottom=0.30)
        save_figure(
            fig,
            histogram_dir,
            f"{dataset}_{grid}_test_mse_histogram",
        )


def smooth(values, alpha=0.06):
    output = np.empty_like(values, dtype=float)
    output[0] = values[0]
    for idx in range(1, len(values)):
        output[idx] = alpha * values[idx] + (1.0 - alpha) * output[idx - 1]
    return output


def plot_top_training(top, history_dir, output_dir):
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(15.8, 7.5),
        sharex="col",
        gridspec_kw={"hspace": 0.14, "wspace": 0.23},
    )
    for col, (dataset, grid) in enumerate(CASES):
        axes[0, col].set_title(f"{dataset_label(dataset)}, {grid}")
        for idx, item in enumerate(top):
            n_real, mode = item["config"]
            history = np.genfromtxt(
                history_path(history_dir, dataset, grid, n_real, mode),
                delimiter=",",
                names=True,
            )
            for row_idx, field in enumerate(("train_loss", "val_mse")):
                ax = axes[row_idx, col]
                values = history[field]
                ax.plot(
                    history["epoch"],
                    values,
                    color=CONFIG_COLORS[idx],
                    linewidth=0.45,
                    alpha=0.13,
                )
                ax.plot(
                    history["epoch"],
                    smooth(values),
                    color=CONFIG_COLORS[idx],
                    linestyle=CONFIG_STYLES[idx],
                    linewidth=1.7,
                    label=config_label(item["config"]),
                )
        for ax in axes[:, col]:
            ax.set_yscale("log")
            clean_axis(ax)
        axes[1, col].set_xlabel("Epoch")
    axes[0, 0].set_ylabel(r"Training weighted MSE ($\beta=2$)")
    axes[1, 0].set_ylabel("Validation MSE")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=5,
    )
    fig.suptitle(
        "Training Dynamics of the Top Five Ablation Configurations",
        fontsize=15,
        fontweight="bold",
        color=INK,
        y=1.02,
    )
    fig.text(
        0.5,
        0.975,
        "Faint traces are raw epochs; emphasized traces are exponential moving averages.",
        ha="center",
        color=MUTED,
        fontsize=9,
    )
    fig.subplots_adjust(top=0.82, left=0.07, right=0.99, bottom=0.08)
    save_figure(fig, output_dir, "04_top5_training_curves")


def plot_best_worst(rows, pred_dir, top, output_dir):
    sample_dir = output_dir / "top5_best_worst"
    for dataset, grid in CASES:
        fig, axes = plt.subplots(
            len(top),
            2,
            figsize=(13.5, 2.35 * len(top) + 1.2),
            sharex=True,
            sharey=True,
        )
        for row_idx, item in enumerate(top):
            config = item["config"]
            _, data = load_native_prediction(rows, pred_dir, dataset, grid, config)
            if int(data["best_idx"]) != int(np.argmin(data["per_sample_mse"])):
                raise ValueError(f"Stored best index is invalid for {dataset} {grid} {config}")
            if int(data["worst_idx"]) != int(np.argmax(data["per_sample_mse"])):
                raise ValueError(f"Stored worst index is invalid for {dataset} {grid} {config}")
            for col, kind in enumerate(("best", "worst")):
                ax = axes[row_idx, col]
                sample_idx = int(data[f"{kind}_idx"])
                truth, prediction = selected_spectrum(data, sample_idx)
                recomputed = float(np.mean((truth - prediction) ** 2))
                recorded = float(data["per_sample_mse"][sample_idx])
                if not np.isclose(recomputed, recorded, rtol=1e-5, atol=1e-8):
                    raise ValueError(
                        f"MSE mismatch for {dataset} {grid} {config} {kind}"
                    )
                ax.plot(data["freq"], truth, color="#111827", linewidth=1.8)
                ax.plot(
                    data["freq"],
                    prediction,
                    color=CONFIG_COLORS[row_idx],
                    linestyle=CONFIG_STYLES[row_idx],
                    linewidth=1.45,
                )
                clean_axis(ax)
                ax.set_ylim(-0.03, 1.03)
                ax.text(
                    0.985,
                    0.94,
                    f"index {sample_idx}  |  MSE {recorded:.3e}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                    color=INK,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
                )
            axes[row_idx, 0].set_ylabel(
                f"#{row_idx + 1} {config_label(config)}\nTransmission $T$",
                fontsize=8.5,
            )
        axes[0, 0].set_title("Best-performing test sample")
        axes[0, 1].set_title("Worst-performing test sample")
        for ax in axes[-1, :]:
            ax.set_xlabel("Frequency (GHz)")
        legend = [
            Line2D([], [], color="#111827", linewidth=1.8, label="Ground truth"),
            Line2D(
                [],
                [],
                color=CONFIG_COLORS[0],
                linewidth=1.5,
                label="Model prediction",
            ),
        ]
        fig.legend(legend, ["Ground truth", "Model prediction"], loc="upper right")
        fig.suptitle(
            f"Top-Five Models: Per-Model Best and Worst Spectra\n"
            f"{dataset_label(dataset)}, {grid}",
            fontsize=15,
            fontweight="bold",
            color=INK,
            y=0.995,
        )
        fig.subplots_adjust(top=0.91, hspace=0.27, wspace=0.12, left=0.12)
        save_figure(fig, sample_dir, f"{dataset}_{grid}_top5_best_worst")


def main():
    args = parse_args()
    if args.top < 1 or args.top > 10:
        raise ValueError("--top must be between 1 and 10")
    set_style()
    rows = load_results(args.summary)
    ranked = rank_configs(rows)
    top = ranked[: args.top]
    write_ranking_csv(ranked, args.output_dir)
    write_cross_domain_csv(rows, args.output_dir)
    write_all_experiment_configurations(rows, args.output_dir)
    plot_ablation_ranking(ranked, args.output_dir)
    plot_v3_baselines(rows, top, args.output_dir)
    plot_two_metric_protocols(rows, top, args.output_dir, cross_domain=False)
    plot_two_metric_protocols(rows, top, args.output_dir, cross_domain=True)
    plot_mse_histograms(rows, args.pred_dir, top, args.output_dir)
    plot_individual_mse_histograms(rows, args.pred_dir, top, args.output_dir)
    plot_top_training(top, args.history_dir, args.output_dir)
    plot_best_worst(rows, args.pred_dir, top, args.output_dir)
    print("Top configurations:")
    for rank, item in enumerate(top, 1):
        print(f"  {rank}. {config_label(item['config'])}: {item['score']:.6f}")
    print("Saved research suite to", args.output_dir)


if __name__ == "__main__":
    main()
