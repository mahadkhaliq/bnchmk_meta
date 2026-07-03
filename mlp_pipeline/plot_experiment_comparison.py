"""Compare logged 2x2 power-transmission experiments.

Outputs:
    plots/experiment_comparison_2x2.png
    plots/experiment_comparison_2x2_no_sweep.png
    plots/experiment_comparison_2x2_no_sweep_clean.png
    plots/experiment_comparison_2x2_500ep_no_sweep.png

The figure combines:
    - training-loss curves
    - validation-loss curves
    - final test-MSE bars
"""
import csv
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter


PLOT_DIR = "plots"
OUT = f"{PLOT_DIR}/experiment_comparison_2x2.png"
OUT_NO_SWEEP = f"{PLOT_DIR}/experiment_comparison_2x2_no_sweep.png"
OUT_NO_SWEEP_CLEAN = f"{PLOT_DIR}/experiment_comparison_2x2_no_sweep_clean.png"
OUT_500_NO_SWEEP = f"{PLOT_DIR}/experiment_comparison_2x2_500ep_no_sweep.png"

LOG_FULL = "logs/powertx_2x2_full.log"
HIST_OLD_SILU = "logs/history/silu_2x2_512x4.csv"
HIST_NEW_SILU = "logs/history/silu_2x2_integrated_512x4.csv"
HIST_NEW_SILU_NEIGH = "logs/history/silu_neigh_2x2_integrated_K3_512x4.csv"
HIST_OLD_SILU_500 = "logs/history/silu_2x2_512x4_500ep.csv"
HIST_NEW_SILU_500 = "logs/history/silu_2x2_integrated_512x4_500ep.csv"
HIST_OLD_SILU_NEIGH_500 = "logs/history/silu_neigh_2x2_K3_512x4_500ep.csv"
HIST_NEW_SILU_NEIGH_500 = "logs/history/silu_neigh_2x2_integrated_K3_512x4_500ep.csv"
HIST_FULL_BASE_INT_500 = "logs/history/2x2_integrated_full_relu_cuda_baseline_2000x10.csv"
HIST_FULL_NEIGH_INT_500 = "logs/history/2x2_integrated_full_relu_cuda_neigh_K3_2000x10.csv"
RESULTS_FULL_INT_500 = "logs/2x2_integrated_full_relu_cuda_results.csv"
SWEEP_RESULTS = "logs/sweep_2x2_results.csv"
HIST_SWEEP_BASE = "logs/history/sweep_2x2_baseline_2000x10.csv"
HIST_SWEEP_NEIGH = "logs/history/sweep_2x2_neigh_128x3.csv"

INK = "#243447"
MUTED = "#637282"
GRID = "#d9e1e8"
COLORS = {
    "full_base": "#476a9f",
    "full_neigh": "#2a9d8f",
    "sweep_base": "#8a63a8",
    "sweep_neigh": "#f4a261",
    "old_silu": "#e76f51",
    "new_silu": "#1f7a5b",
    "new_silu_neigh": "#0f5c8c",
    "old_silu_neigh": "#b56576",
    "full_base_int": "#6c7fb5",
    "full_neigh_int": "#2d6a4f",
}


def style_ax(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(True, color=GRID, lw=0.8, alpha=0.85)
    ax.set_axisbelow(True)


def read_history_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    epoch = np.array([int(r["epoch"]) for r in rows])
    train_key = "train_beta2" if "train_beta2" in rows[0] else "train"
    val_key = "val_mse" if "val_mse" in rows[0] else "val"
    train = np.array([float(r[train_key]) for r in rows])
    val = np.array([float(r[val_key]) for r in rows])
    return epoch, train, val


def parse_full_log(path):
    text = open(path).read()
    sections = re.split(r"#+ MODEL \d+: ([^#]+?) #+", text)
    epoch_re = re.compile(r"Epoch\s+(\d+) \| train (\S+) \| val (\S+)")
    test_re = re.compile(r"\[2x2\]\s+(Baseline MLP|Neighbourhood MLP)\s+test MSE: (\S+)")

    runs = {}
    for i in range(1, len(sections), 2):
        name = sections[i].strip().lower()
        key = "full_base" if "baseline" in name else "full_neigh"
        body = sections[i + 1]
        e, tr, va = [], [], []
        for m in epoch_re.finditer(body):
            e.append(int(m.group(1)))
            tr.append(float(m.group(2)))
            va.append(float(m.group(3)))
        runs[key] = (np.array(e), np.array(tr), np.array(va))

    tests = {}
    for name, value in test_re.findall(text):
        key = "full_base" if "Baseline" in name else "full_neigh"
        tests[key] = float(value)
    return runs, tests


def read_sweep_tests(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    best_base = min(rows, key=lambda r: float(r["baseline_test"]))
    best_neigh = min(rows, key=lambda r: float(r["neigh_test"]))
    return {
        "sweep_base": float(best_base["baseline_test"]),
        "sweep_neigh": float(best_neigh["neigh_test"]),
    }


def read_full_integrated_tests(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        if r["model"] == "baseline_relu":
            out["full_base_int"] = float(r["test_mse"])
        elif r["model"] == "neighbourhood_relu_K3":
            out["full_neigh_int"] = float(r["test_mse"])
    return out


def plot_comparison(curves, tests, labels, curve_order, bar_order, out, subtitle, annotate=True):
    labels = {
        **labels,
    }

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "figure.dpi": 140,
    })

    fig, axes = plt.subplots(
        1, 3, figsize=(16, 5.4),
        gridspec_kw={"width_ratios": [1.2, 1.2, 1.0]},
    )
    ax_train, ax_val, ax_bar = axes

    for key in curve_order:
        e, tr, va = curves[key]
        lw = 2.8 if key == "new_silu" else 1.8
        alpha = 1.0 if key in {"old_silu", "new_silu"} else 0.78
        ax_train.plot(e, tr, color=COLORS[key], lw=lw, alpha=alpha, label=labels[key].replace("\n", " "))
        ax_val.plot(e, va, color=COLORS[key], lw=lw, alpha=alpha, label=labels[key].replace("\n", " "))
        bi = int(np.argmin(va))
        ax_val.scatter(e[bi], va[bi], s=34, color=COLORS[key], edgecolor="white", lw=1.0, zorder=5)

    for ax, title, ylabel in [
        (ax_train, "Training Curves", "training loss"),
        (ax_val, "Validation Curves", "validation MSE"),
    ]:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        style_ax(ax)

    x = np.arange(len(bar_order))
    vals = [tests[k] for k in bar_order]
    bars = ax_bar.bar(x, vals, color=[COLORS[k] for k in bar_order], edgecolor="white", lw=1.2, zorder=3)
    for b, v in zip(bars, vals):
        ax_bar.annotate(
            f"{v:.4f}",
            (b.get_x() + b.get_width() / 2, v),
            ha="center", va="bottom", fontsize=8.5, color=INK,
            xytext=(0, 3), textcoords="offset points", rotation=90,
        )
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([labels[k] for k in bar_order], rotation=38, ha="right")
    ax_bar.set_ylabel("test MSE")
    ax_bar.set_title("Held-Out Test Error")
    ax_bar.set_ylim(0, max(vals) * 1.22)
    style_ax(ax_bar)
    ax_bar.grid(axis="x", visible=False)

    if annotate and "old_silu" in tests and "full_neigh_int" in tests:
        old = tests["old_silu"]
        new = tests["full_neigh_int"]
        pct = (old - new) / old * 100
        ax_bar.annotate(
            f"best 21,625-sample model\n{pct:.1f}% lower than 706-sample SiLU",
            xy=(bar_order.index("full_neigh_int"), new),
            xytext=(0.67, 0.67),
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color=COLORS["full_neigh_int"], lw=1.4),
            bbox=dict(boxstyle="round,pad=0.45", fc="#eef7f1", ec=COLORS["full_neigh_int"], lw=1.0),
            fontsize=9.2,
            color=INK,
            ha="center",
        )

    handles, legend_labels = ax_val.get_legend_handles_labels()
    fig.legend(
        handles, legend_labels, loc="lower center", ncol=3,
        frameon=False, fontsize=9, labelcolor=INK, bbox_to_anchor=(0.5, -0.035),
    )

    fig.suptitle("2x2 Power-Transmission Experiments", fontsize=16, fontweight="bold", color=INK, y=1.02)
    fig.text(0.5, 0.945, subtitle, ha="center", fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out}")


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    curves, tests = parse_full_log(LOG_FULL)
    tests.update(read_sweep_tests(SWEEP_RESULTS))

    curves["sweep_base"] = read_history_csv(HIST_SWEEP_BASE)
    curves["sweep_neigh"] = read_history_csv(HIST_SWEEP_NEIGH)
    curves["old_silu"] = read_history_csv(HIST_OLD_SILU)
    curves["new_silu"] = read_history_csv(HIST_NEW_SILU)
    curves["new_silu_neigh"] = read_history_csv(HIST_NEW_SILU_NEIGH)

    tests["old_silu"] = 0.033898
    tests["new_silu"] = 0.007183
    tests["new_silu_neigh"] = 0.006344

    labels = {
        "full_base": "full baseline\n40M, 706 samples",
        "full_neigh": "full neighborhood K=3\n40M, 706 samples",
        "sweep_base": "sweep best baseline\n40M, 706 samples",
        "sweep_neigh": "sweep best neighborhood K=3\n0.30M, 706 samples",
        "old_silu": "SiLU beta2\n1.82M, 706 samples",
        "new_silu": "SiLU beta2\n1.82M, 21,625 samples",
        "new_silu_neigh": "SiLU beta2 neighborhood K=3\n1.83M, 21,625 samples",
    }

    subtitle = (
        "706-sample experiments use 480/120/106 train/val/test; "
        "21,625-sample SiLU uses 14,704/3,677/3,244."
    )

    plot_comparison(
        curves, tests, labels,
        ["full_base", "full_neigh", "sweep_base", "sweep_neigh", "old_silu", "new_silu", "new_silu_neigh"],
        ["full_base", "full_neigh", "sweep_base", "sweep_neigh", "old_silu", "new_silu", "new_silu_neigh"],
        OUT,
        subtitle,
    )

    plot_comparison(
        curves, tests, labels,
        ["full_base", "full_neigh", "old_silu", "new_silu", "new_silu_neigh"],
        ["full_base", "full_neigh", "old_silu", "new_silu", "new_silu_neigh"],
        OUT_NO_SWEEP,
        subtitle + " Full baseline/neighborhood ran for 500 epochs; SiLU runs used 300 epochs.",
    )

    plot_comparison(
        curves, tests, labels,
        ["full_base", "full_neigh", "old_silu", "new_silu", "new_silu_neigh"],
        ["full_base", "full_neigh", "old_silu", "new_silu", "new_silu_neigh"],
        OUT_NO_SWEEP_CLEAN,
        subtitle + " Full baseline/neighborhood ran for 500 epochs; SiLU runs used 300 epochs.",
        annotate=False,
    )

    curves_500 = {
        "old_silu": read_history_csv(HIST_OLD_SILU_500),
        "old_silu_neigh": read_history_csv(HIST_OLD_SILU_NEIGH_500),
        "new_silu": read_history_csv(HIST_NEW_SILU_500),
        "new_silu_neigh": read_history_csv(HIST_NEW_SILU_NEIGH_500),
        "full_base_int": read_history_csv(HIST_FULL_BASE_INT_500),
        "full_neigh_int": read_history_csv(HIST_FULL_NEIGH_INT_500),
    }
    tests_500 = read_full_integrated_tests(RESULTS_FULL_INT_500)
    tests_500.update({
        "old_silu": 0.033493,
        "old_silu_neigh": 0.019482,
        "new_silu": 0.006866,
        "new_silu_neigh": 0.006277,
    })
    labels_500 = {
        "old_silu": "SiLU beta2\n1.82M, 706 samples",
        "old_silu_neigh": "SiLU beta2 neighborhood K=3\n1.83M, 706 samples",
        "new_silu": "SiLU beta2\n1.82M, 21,625 samples",
        "new_silu_neigh": "SiLU beta2 neighborhood K=3\n1.83M, 21,625 samples",
        "full_base_int": "full baseline ReLU\n40.1M, 21,625 samples",
        "full_neigh_int": "full neighborhood ReLU K=3\n40.1M, 21,625 samples",
    }
    order_500 = ["old_silu", "old_silu_neigh", "new_silu", "new_silu_neigh", "full_base_int", "full_neigh_int"]
    plot_comparison(
        curves_500, tests_500, labels_500,
        order_500,
        order_500,
        OUT_500_NO_SWEEP,
        "All shown runs use 500 epochs. 706-sample split: 480/120/106; 21,625-sample split: 14,704/3,677/3,244.",
        annotate=False,
    )


if __name__ == "__main__":
    main()
