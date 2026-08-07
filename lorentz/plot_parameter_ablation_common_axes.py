"""Plot any parameter-mapping ablation campaign on common axes.

Generalizes ``plot_background_offset_seed_curves.py`` from a hard-coded two-
profile study to an arbitrary campaign directory. Colour encodes the mapping
condition and line style encodes the seed, so no distinction relies on hue
alone.

Three figures are written, each as PNG and PDF:

``<prefix>_training_loss``    training beta-2 against epoch
``<prefix>_validation_mse``   validation MSE against epoch, star at best epoch
``<prefix>_test_mse``         held-out test MSE per condition and seed

Runs that ended non-finite are kept rather than dropped: their partial curves
are drawn faintly and their bars are hatched and annotated, so an unstable
mapping stays visible instead of silently vanishing from the comparison.

    MPLCONFIGDIR=/tmp/matplotlib-lorentz python -m lorentz.plot_parameter_ablation_common_axes \
      --experiment-root lorentz/experiments/mapping_background_offsets_1x1_20260805 \
      --prefix epsilon_mu
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


# Okabe-Ito, in an order checked with the data-viz six-check validator on the
# light chart surface: lightness band, chroma floor, adjacent colour-vision-
# deficiency separation, and normal-vision separation all pass. Worst adjacent
# pair is #cc79a7 vs #009e73 at deutan dE 7.6, which sits in the 6-8 band and
# is therefore only legal alongside secondary encoding -- satisfied here,
# because every condition is also named by a panel title or an axis tick, so
# identity never rests on hue alone. Three steps fall below 3:1 contrast
# against the surface, which the same visible labels discharge.
#
# The earlier #27647b/#b05a8c pair taken from the older figures fails: #27647b
# has chroma 0.072 and reads gray, and #b05a8c vs #009e73 is deutan dE 5.4.
CONDITION_COLORS = (
    "#0072b2",  # Reference
    "#d55e00",
    "#009e73",
    "#cc79a7",
    "#e69f00",
    "#56b4e9",
)
SEED_STYLES = ("-", "--", "-.", ":", (0, (3, 1, 1, 1, 1, 1)))
SEED_MARKERS = ("o", "s", "^", "D", "v")

LABEL_OVERRIDES = {"baseline": "Reference", "reference": "Reference"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument(
        "--prefix",
        default="parameter_ablation",
        help="Filename prefix for the three figures.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--smoothing-window", type=int, default=11)
    parser.add_argument(
        "--profiles",
        default=None,
        help="Comma-separated subset; defaults to every profile in the matrix.",
    )
    return parser.parse_args()


def pretty(profile):
    if profile in LABEL_OVERRIDES:
        return LABEL_OVERRIDES[profile]
    return profile.replace("_", " ").capitalize()


def discover(root, requested):
    """Return (profiles, seeds) in experiment-matrix order where available."""
    runs = root / "runs"
    if not runs.is_dir():
        raise SystemExit(f"No runs/ directory under {root}")

    ordered = []
    matrix = root / "experiment_matrix.csv"
    if matrix.is_file():
        with matrix.open(newline="") as handle:
            for row in csv.DictReader(handle):
                name = row.get("profile")
                if name and name not in ordered:
                    ordered.append(name)
    for name in sorted(p.name for p in runs.iterdir() if p.is_dir()):
        if name not in ordered:
            ordered.append(name)

    profiles = [p for p in ordered if (runs / p).is_dir()]
    if requested:
        wanted = [p.strip() for p in requested.split(",") if p.strip()]
        missing = [p for p in wanted if p not in profiles]
        if missing:
            raise SystemExit(f"Unknown profile(s): {', '.join(missing)}")
        profiles = wanted

    seeds = set()
    for profile in profiles:
        for entry in (runs / profile).iterdir():
            if entry.is_dir() and entry.name.startswith("seed_"):
                seeds.add(int(entry.name.split("_", 1)[1]))
    return profiles, sorted(seeds)


def read_history(path):
    if not path.is_file():
        return None
    with path.open(newline="") as handle:
        rows = [
            {key: float(value) for key, value in row.items() if value != ""}
            for row in csv.DictReader(handle)
        ]
    return rows or None


def read_metrics(path):
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def moving_average(values, window):
    if window <= 1 or len(values) < window:
        return np.arange(len(values)), np.asarray(values)
    kernel = np.ones(window, dtype=float) / window
    return np.arange(window - 1, len(values)), np.convolve(
        values, kernel, mode="valid"
    )


def load_campaign(root, profiles, seeds):
    data = {}
    for profile in profiles:
        for seed in seeds:
            run = root / "runs" / profile / f"seed_{seed}"
            metrics = read_metrics(run / "metrics.json")
            data[(profile, seed)] = {
                "history": read_history(run / "history.csv"),
                "metrics": metrics,
                "ok": metrics.get("status") == "completed",
                "test_mse": (metrics.get("test") or {}).get("mse"),
                "best_epoch": metrics.get("best_epoch"),
                "epochs": metrics.get("completed_epochs"),
            }
    return data


def style_legends(axis, profiles, seeds):
    """Two legends: colour carries condition, line style carries seed."""
    condition_handles = [
        Line2D([], [], color=CONDITION_COLORS[i % len(CONDITION_COLORS)],
               linewidth=2.0, label=pretty(profile))
        for i, profile in enumerate(profiles)
    ]
    seed_handles = [
        Line2D([], [], color="#444444",
               linestyle=SEED_STYLES[i % len(SEED_STYLES)],
               linewidth=1.4, label=f"Seed {seed}")
        for i, seed in enumerate(seeds)
    ]
    first = axis.legend(
        handles=condition_handles, loc="upper right", fontsize=8,
        title="Condition", title_fontsize=8, framealpha=0.9,
    )
    axis.add_artist(first)
    axis.legend(
        handles=seed_handles, loc="lower left", fontsize=8,
        title="Seed", title_fontsize=8, framealpha=0.9,
    )


def curve_figure(data, profiles, seeds, column, title, ylabel, window,
                 mark_best):
    """Small multiples, one panel per condition, on shared log axes.

    Overlaying every condition x seed on a single axes becomes unreadable past
    roughly eight curves. Faceting keeps each condition legible while the
    shared x and y limits preserve the cross-condition comparison.
    """
    columns = min(len(profiles), 3)
    rows = int(np.ceil(len(profiles) / columns))
    fig, axes = plt.subplots(
        rows, columns, figsize=(5.4 * columns, 3.9 * rows),
        sharex=True, sharey=True, squeeze=False,
    )
    flat = axes.ravel()

    for p_index, profile in enumerate(profiles):
        axis = flat[p_index]
        color = CONDITION_COLORS[p_index % len(CONDITION_COLORS)]
        failures = 0
        for s_index, seed in enumerate(seeds):
            entry = data[(profile, seed)]
            history = entry["history"]
            if not history or column not in history[0]:
                continue
            style = SEED_STYLES[s_index % len(SEED_STYLES)]
            epoch = np.asarray([row["epoch"] for row in history])
            values = np.asarray([row[column] for row in history])
            failed = not entry["ok"]

            axis.plot(epoch, values, color=color, linestyle=style,
                      linewidth=0.6, alpha=0.13)
            index, smooth = moving_average(values, window)
            axis.plot(epoch[index], smooth, color=color, linestyle=style,
                      linewidth=1.6, alpha=0.5 if failed else 1.0)

            if failed:
                axis.scatter(epoch[-1], values[-1], color=color, marker="X",
                             s=72, edgecolor="white", linewidth=0.7, zorder=5)
                # Stack annotations so repeated failures cannot overlap.
                axis.annotate(
                    f"seed {seed} failed @ep{int(entry['epochs'] or epoch[-1])}",
                    (epoch[-1], values[-1]), textcoords="offset points",
                    xytext=(9, 3 + 11 * failures), fontsize=7.5, color=color,
                )
                failures += 1
            elif mark_best and entry["best_epoch"] is not None:
                best = int(entry["best_epoch"])
                if 1 <= best <= len(values):
                    axis.scatter(
                        best, values[best - 1], color=color,
                        marker=SEED_MARKERS[s_index % len(SEED_MARKERS)],
                        s=50, edgecolor="white", linewidth=0.7, zorder=5,
                    )

        axis.set_yscale("log")
        axis.set_title(pretty(profile), fontsize=10, color=color)
        axis.grid(alpha=0.22)

    for spare in flat[len(profiles):]:
        spare.axis("off")
    for index, axis in enumerate(flat[:len(profiles)]):
        if index % columns == 0:
            axis.set_ylabel(ylabel)
        if index >= len(profiles) - columns:
            axis.set_xlabel("Epoch")

    handles = [
        Line2D([], [], color="#444444",
               linestyle=SEED_STYLES[i % len(SEED_STYLES)],
               linewidth=1.5, label=f"Seed {seed}")
        for i, seed in enumerate(seeds)
    ]
    legend_axis = flat[len(profiles) - 1] if len(profiles) == len(flat) \
        else flat[len(profiles)]
    legend_axis.legend(handles=handles, loc="center" if
                       len(profiles) != len(flat) else "upper right",
                       fontsize=8, title="Seed", title_fontsize=8,
                       framealpha=0.9)
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    return fig


def test_mse_figure(data, profiles, seeds, title):
    """Per-seed dots with a mean rule.

    Deliberately not a bar chart: test MSE spans more than a decade here, so
    the axis must be logarithmic, and a log axis has no zero for bars to be
    anchored to. Dots also expose the per-seed spread, which is the point of
    running matched seeds at all.
    """
    fig, axis = plt.subplots(figsize=(11.5, 6.0))
    spread = 0.62 / max(len(seeds), 1)
    positions = np.arange(len(profiles), dtype=float)

    for p_index, profile in enumerate(profiles):
        color = CONDITION_COLORS[p_index % len(CONDITION_COLORS)]
        finished = []
        for s_index, seed in enumerate(seeds):
            entry = data[(profile, seed)]
            value = entry["test_mse"]
            if value is None:
                continue
            offset = (s_index - (len(seeds) - 1) / 2) * spread
            x = positions[p_index] + offset
            marker = SEED_MARKERS[s_index % len(SEED_MARKERS)]
            if entry["ok"]:
                finished.append(value)
                axis.scatter(x, value, color=color, marker=marker, s=64,
                             edgecolor="white", linewidth=0.8, zorder=4)
            else:
                axis.scatter(x, value, facecolor="none", edgecolor=color,
                             marker=marker, s=64, linewidth=1.3, zorder=4)
                axis.annotate("✗", (x, value), textcoords="offset points",
                              xytext=(7, -1), fontsize=9, color=color)

        if finished:
            mean = float(np.mean(finished))
            axis.hlines(mean, positions[p_index] - 0.42,
                        positions[p_index] + 0.42, color=color,
                        linewidth=2.2, zorder=3)
            # Show n: a mean over one surviving seed must not read like a
            # mean over all of them.
            axis.annotate(f"{mean:.3e} (n={len(finished)})",
                          (positions[p_index] + 0.42, mean),
                          textcoords="offset points", xytext=(4, -3),
                          fontsize=7.5,
                          color="#333333" if len(finished) == len(seeds)
                          else "#b3541e")

    axis.set_xticks(positions)
    axis.set_xticklabels([pretty(p) for p in profiles], fontsize=9)
    axis.set_xlim(-0.6, len(profiles) - 0.25)
    axis.set_yscale("log")
    axis.set_ylabel("Held-out test MSE")
    axis.set_title(title)
    axis.grid(alpha=0.22, axis="y")

    handles = [
        Line2D([], [], color="#444444", linestyle="none",
               marker=SEED_MARKERS[i % len(SEED_MARKERS)], markersize=7,
               label=f"Seed {seed}")
        for i, seed in enumerate(seeds)
    ]
    handles.append(Line2D([], [], color="#444444", linewidth=2.2,
                          label="Mean of completed"))
    handles.append(Line2D([], [], color="#444444", linestyle="none",
                          marker="o", markerfacecolor="none", markersize=7,
                          label="Did not complete (✗)"))
    axis.legend(handles=handles, loc="upper right", fontsize=8,
                framealpha=0.9, ncol=2)
    fig.tight_layout()
    return fig


def save(fig, directory, stem):
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in ("png", "pdf"):
        path = directory / f"{stem}.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


def main():
    args = parse_args()
    root = args.experiment_root.resolve()
    profiles, seeds = discover(root, args.profiles)
    data = load_campaign(root, profiles, seeds)
    output_dir = (args.output_dir or root / "plots").resolve()

    completed = sum(1 for entry in data.values() if entry["ok"])
    print(f"{root.name}: {len(profiles)} profiles x {len(seeds)} seeds "
          f"= {len(data)} runs ({completed} completed)")

    written = []
    written += save(
        curve_figure(data, profiles, seeds, "train_beta2",
                     f"{root.name}: training beta-2 loss",
                     "Training beta-2", args.smoothing_window, mark_best=False),
        output_dir, f"{args.prefix}_training_loss",
    )
    written += save(
        curve_figure(data, profiles, seeds, "val_mse",
                     f"{root.name}: validation MSE",
                     "Validation MSE", args.smoothing_window, mark_best=True),
        output_dir, f"{args.prefix}_validation_mse",
    )
    written += save(
        test_mse_figure(data, profiles, seeds,
                        f"{root.name}: held-out test MSE"),
        output_dir, f"{args.prefix}_test_mse",
    )
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
