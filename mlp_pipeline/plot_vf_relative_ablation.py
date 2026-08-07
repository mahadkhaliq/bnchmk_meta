"""Plot the verified ConceptOneVF relative-position ablation.

Produces:
  - native and transfer metric summaries;
  - train-beta2 and validation-MSE curves for n_real=0 and 4;
  - three identical seeded native test samples across all relative modes;
  - best/worst spectra for the best native mode in each dataset/grid/n_real case.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

STUDY_ID = "relabl_stable_v3"
TAG_RE = re.compile(
    r"rel_(v[34])to(v[34])_(2x2|3x3)_K3_nr([04])_"
    r"(none|offset|offset_dist|embed|polar)$"
)
MODES = ("none", "offset", "offset_dist", "embed", "polar")
MODE_LABELS = {
    "none": "None",
    "offset": "Offset",
    "offset_dist": "Offset + distance",
    "embed": "Embedding",
    "polar": "Polar",
}
MODE_COLORS = {
    "none": "#6B7280",
    "offset": "#0072B2",
    "offset_dist": "#D55E00",
    "embed": "#009E73",
    "polar": "#CC79A7",
}
MODE_STYLES = {
    "none": "-",
    "offset": (0, (6, 2)),
    "offset_dist": (0, (2, 1)),
    "embed": (0, (8, 2, 2, 2)),
    "polar": (0, (4, 2, 1, 2)),
}
NR_STYLES = {
    0: {"color": "#185FA5", "marker": "o", "linestyle": "-", "label": "n_real = 0"},
    4: {"color": "#D1495B", "marker": "s", "linestyle": (0, (6, 2)), "label": "n_real = 4"},
}
INK = "#24313B"
MUTED = "#64727D"
GRID = "#DDE5EA"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "logs" / f"vf_rel_ablation_{STUDY_ID}" / "evaluations.csv",
    )
    ap.add_argument("--history-dir", type=Path, default=ROOT / "logs" / "history")
    ap.add_argument(
        "--pred-dir",
        type=Path,
        default=ROOT / "logs" / f"vf_rel_ablation_{STUDY_ID}" / "preds",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "plots" / f"vf_relative_ablation_{STUDY_ID}",
    )
    ap.add_argument("--n-random", type=int, default=3)
    return ap.parse_args()


def load_results(path):
    rows = []
    with path.open(newline="") as f:
        for raw in csv.DictReader(f):
            match = TAG_RE.fullmatch(raw["tag"])
            if not match:
                raise ValueError(f"Unexpected result tag: {raw['tag']}")
            train_set, test_set, grid, n_real, mode = match.groups()
            rows.append(
                {
                    **raw,
                    "train_set": train_set,
                    "test_set": test_set,
                    "grid_size": grid,
                    "n_real": int(n_real),
                    "mode": mode,
                    "test_mse": float(raw["test_mse"]),
                    "test_beta2": float(raw["test_beta2"]),
                }
            )
    if len(rows) != 80 or len({row["tag"] for row in rows}) != 80:
        raise ValueError(f"Expected 80 unique rows, got {len(rows)}")
    return rows


def one_result(rows, train_set, test_set, grid, n_real, mode):
    matches = [
        row
        for row in rows
        if (
            row["train_set"],
            row["test_set"],
            row["grid_size"],
            row["n_real"],
            row["mode"],
        )
        == (train_set, test_set, grid, n_real, mode)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one row for {(train_set, test_set, grid, n_real, mode)}, "
            f"got {len(matches)}"
        )
    return matches[0]


def style_axis(ax, log=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    if log:
        ax.set_yscale("log")


def dataset_label(version):
    return "v4 synthetic" if version == "v4" else "v3 CST"


def plot_metric_overview(rows, native, output_dir):
    if native:
        cases = [
            ("v4", "v4", "2x2"),
            ("v4", "v4", "3x3"),
            ("v3", "v3", "2x2"),
            ("v3", "v3", "3x3"),
        ]
        filename = "native_test_metrics.png"
        title = "Native Test Performance by Relative Encoding"
    else:
        cases = [
            ("v4", "v3", "2x2"),
            ("v4", "v3", "3x3"),
            ("v3", "v4", "2x2"),
            ("v3", "v4", "3x3"),
        ]
        filename = "transfer_test_metrics.png"
        title = "Cross-Domain Test Performance by Relative Encoding"

    fig, axes = plt.subplots(
        2, 4, figsize=(18, 7.5), sharex="col",
        gridspec_kw={"hspace": 0.20, "wspace": 0.23},
    )
    x = np.arange(len(MODES))
    for col, (train_set, test_set, grid) in enumerate(cases):
        heading = (
            f"{dataset_label(train_set)} · {grid}"
            if native
            else f"{train_set} → {test_set} · {grid}"
        )
        axes[0, col].set_title(heading, color=INK, fontsize=11.5, fontweight="bold")
        for row_idx, metric in enumerate(("test_mse", "test_beta2")):
            ax = axes[row_idx, col]
            all_points = []
            for n_real, line_style in NR_STYLES.items():
                values = [
                    one_result(
                        rows, train_set, test_set, grid, n_real, mode
                    )[metric]
                    for mode in MODES
                ]
                all_points.extend((value, idx, n_real) for idx, value in enumerate(values))
                ax.plot(
                    x,
                    values,
                    color=line_style["color"],
                    marker=line_style["marker"],
                    linestyle=line_style["linestyle"],
                    linewidth=2.0,
                    markersize=6,
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    label=line_style["label"],
                )
            best_value, best_x, _ = min(all_points)
            ax.scatter(
                [best_x], [best_value], marker="*", s=145,
                color="#F2B134", edgecolor=INK, linewidth=0.6, zorder=6,
            )
            style_axis(ax)
            ax.set_xticks(x, [MODE_LABELS[mode] for mode in MODES], rotation=18)
            ax.margins(x=0.06, y=0.16)
            if col == 0:
                ax.set_ylabel(
                    "Test MSE" if metric == "test_mse" else "Test beta2",
                    color=INK,
                )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(
        Line2D(
            [], [], marker="*", linestyle="", markersize=11,
            markerfacecolor="#F2B134", markeredgecolor=INK,
        )
    )
    labels.append("Best MSE / beta2 in panel")
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.925),
        ncol=3, frameon=False, fontsize=10,
    )
    fig.suptitle(title, fontsize=18, fontweight="bold", color=INK, y=0.995)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.12, top=0.84)
    out = output_dir / filename
    fig.savefig(out, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)


def history_path(history_dir, dataset, grid, n_real, mode):
    return history_dir / (
        f"vf_{grid}{dataset}_K3_np8_nr{n_real}_{mode}_beta2_"
        f"512x4_500ep_seed0_{STUDY_ID}.csv"
    )


def ema(values, alpha=0.06):
    result = np.empty_like(values)
    result[0] = values[0]
    for idx in range(1, len(values)):
        result[idx] = alpha * values[idx] + (1.0 - alpha) * result[idx - 1]
    return result


def plot_training_curves(history_dir, n_real, output_dir):
    cases = [
        ("v4", "2x2"),
        ("v4", "3x3"),
        ("v3", "2x2"),
        ("v3", "3x3"),
    ]
    fig, axes = plt.subplots(
        2, 4, figsize=(18, 7.7), sharex="col",
        gridspec_kw={"hspace": 0.16, "wspace": 0.23},
    )
    for col, (dataset, grid) in enumerate(cases):
        axes[0, col].set_title(
            f"{dataset_label(dataset)} · {grid}",
            color=INK, fontsize=11.5, fontweight="bold",
        )
        for mode in MODES:
            history = np.genfromtxt(
                history_path(history_dir, dataset, grid, n_real, mode),
                delimiter=",",
                names=True,
            )
            epoch = history["epoch"]
            for row_idx, field in enumerate(("train_loss", "val_mse")):
                values = history[field]
                ax = axes[row_idx, col]
                ax.plot(
                    epoch, values, color=MODE_COLORS[mode],
                    linewidth=0.7, alpha=0.13,
                )
                ax.plot(
                    epoch, ema(values), color=MODE_COLORS[mode],
                    linestyle=MODE_STYLES[mode], linewidth=1.9,
                    label=MODE_LABELS[mode],
                )
        for row_idx in range(2):
            style_axis(axes[row_idx, col], log=True)
        axes[1, col].set_xlabel("Epoch", color=INK)
        if col == 0:
            axes[0, col].set_ylabel("Train beta2", color=INK)
            axes[1, col].set_ylabel("Validation MSE", color=INK)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.925),
        ncol=5, frameon=False, fontsize=10,
    )
    fig.suptitle(
        f"Relative-Encoding Training Curves · n_real = {n_real}",
        fontsize=18, fontweight="bold", color=INK, y=0.995,
    )
    fig.text(
        0.5, 0.955,
        "Faint lines are raw epochs; bold lines are exponential moving averages.",
        ha="center", color=MUTED, fontsize=10,
    )
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.08, top=0.84)
    out = output_dir / f"training_curves_nreal{n_real}.png"
    fig.savefig(out, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)


def prediction_file(pred_dir, tag):
    path = pred_dir / f"{tag}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def selected_spectrum(data, sample_idx):
    selected = data["idx"].astype(int)
    positions = np.flatnonzero(selected == int(sample_idx))
    if len(positions) != 1:
        raise ValueError(f"Sample {sample_idx} not found exactly once in prediction file")
    pos = int(positions[0])
    return data["truth"][pos], data["pred"][pos]


def plot_random_samples(rows, pred_dir, output_dir, n_random):
    random_dir = output_dir / "random_samples"
    random_dir.mkdir(parents=True, exist_ok=True)
    for dataset in ("v4", "v3"):
        for grid in ("2x2", "3x3"):
            for n_real in (0, 4):
                mode_data = {}
                for mode in MODES:
                    row = one_result(rows, dataset, dataset, grid, n_real, mode)
                    mode_data[mode] = np.load(prediction_file(pred_dir, row["tag"]))
                random_idx = mode_data["none"]["random_idx"].astype(int)[:n_random]
                for ordinal, sample_idx in enumerate(random_idx, start=1):
                    fig, ax = plt.subplots(figsize=(10.8, 5.4))
                    reference_truth = None
                    freq = mode_data["none"]["freq"]
                    for mode in MODES:
                        truth, pred = selected_spectrum(mode_data[mode], sample_idx)
                        if reference_truth is None:
                            reference_truth = truth
                        elif not np.allclose(reference_truth, truth):
                            raise ValueError("Truth differs across relative modes")
                        sample_mse = mode_data[mode]["per_sample_mse"][sample_idx]
                        ax.plot(
                            freq, pred, color=MODE_COLORS[mode],
                            linestyle=MODE_STYLES[mode], linewidth=1.65,
                            alpha=0.82,
                            label=f"{MODE_LABELS[mode]} ({sample_mse:.5f})",
                        )
                    ax.plot(
                        freq, reference_truth, color="#111827",
                        linewidth=2.5, label="Ground truth", zorder=7,
                    )
                    style_axis(ax)
                    ax.set_ylim(-0.03, 1.03)
                    ax.set_xlabel("Frequency (GHz)", color=INK)
                    ax.set_ylabel("Power transmission T", color=INK)
                    ax.set_title(
                        f"{dataset_label(dataset)} · {grid} · n_real={n_real} · "
                        f"test sample {sample_idx}",
                        color=INK, fontsize=14, fontweight="bold",
                    )
                    ax.legend(
                        ncol=3, frameon=False, fontsize=9,
                        loc="upper center", bbox_to_anchor=(0.5, -0.16),
                    )
                    fig.subplots_adjust(bottom=0.25)
                    out = random_dir / (
                        f"{dataset}_{grid}_nr{n_real}_random{ordinal}_"
                        f"sample{sample_idx}.png"
                    )
                    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
                    plt.close(fig)


def plot_best_worst(rows, pred_dir, output_dir):
    sample_dir = output_dir / "best_worst"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for dataset in ("v4", "v3"):
        for grid in ("2x2", "3x3"):
            for n_real in (0, 4):
                candidates = [
                    one_result(rows, dataset, dataset, grid, n_real, mode)
                    for mode in MODES
                ]
                winner = min(candidates, key=lambda row: row["test_mse"])
                mode = winner["mode"]
                data = np.load(prediction_file(pred_dir, winner["tag"]))
                fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), sharey=True)
                for ax, kind in zip(axes, ("best", "worst")):
                    sample_idx = int(data[f"{kind}_idx"])
                    truth, pred = selected_spectrum(data, sample_idx)
                    ax.plot(
                        data["freq"], truth, color="#111827",
                        linewidth=2.5, label="Ground truth",
                    )
                    ax.plot(
                        data["freq"], pred, color=MODE_COLORS[mode],
                        linestyle=MODE_STYLES[mode], linewidth=2.0,
                        label=MODE_LABELS[mode],
                    )
                    style_axis(ax)
                    ax.set_ylim(-0.03, 1.03)
                    ax.set_xlabel("Frequency (GHz)", color=INK)
                    ax.set_title(
                        f"{kind.title()} sample {sample_idx}\n"
                        f"MSE {data['per_sample_mse'][sample_idx]:.6f} · "
                        f"beta2 {data['per_sample_beta2'][sample_idx]:.6f}",
                        color=INK, fontsize=11.5, fontweight="bold",
                    )
                axes[0].set_ylabel("Power transmission T", color=INK)
                axes[1].legend(frameon=False, loc="lower right")
                fig.suptitle(
                    f"{dataset_label(dataset)} · {grid} · n_real={n_real} · "
                    f"best native mode: {MODE_LABELS[mode]}",
                    color=INK, fontsize=16, fontweight="bold",
                )
                fig.subplots_adjust(top=0.80, wspace=0.16)
                out = sample_dir / f"{dataset}_{grid}_nr{n_real}_best_worst.png"
                fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
                plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": 10.5,
            "axes.titlepad": 10,
            "legend.handlelength": 2.8,
        }
    )
    rows = load_results(args.summary)
    plot_metric_overview(rows, native=True, output_dir=args.output_dir)
    plot_metric_overview(rows, native=False, output_dir=args.output_dir)
    for n_real in (0, 4):
        plot_training_curves(args.history_dir, n_real, args.output_dir)
    plot_random_samples(
        rows, args.pred_dir, args.output_dir, max(0, args.n_random)
    )
    plot_best_worst(rows, args.pred_dir, args.output_dir)
    print("all plots saved under", args.output_dir)


if __name__ == "__main__":
    main()
