"""Directly fit constrained Lorentz parameters to held-out v3 1x1 spectra.

This removes F1 from the optimization. Each spectrum gets independent raw
parameters and multiple starts, so the result estimates the best error the
current finite-slab decoder can achieve on those spectra.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..lorentz import LorentzPhysics
from .common import (
    DEFAULT_REFERENCE_CHECKPOINT,
    DEFAULT_V3_DATASET,
    PARAMETER_NAMES,
    ROOT,
    batched_model_outputs,
    load_model_checkpoint,
    load_v3_1x1,
    normalize_with,
    seed_everything,
    sha256_file,
    write_json,
)


DEFAULT_OUTPUT = (
    ROOT / "lorentz" / "experiments" / "unary_validation_1x1_20260806"
    / "oracle_v3_1x1"
)
TARGET_COLOR = "#222222"
REFERENCE_COLOR = "#2b6f84"
ORACLE_COLOR = "#c44e52"
MEAN_COLOR = "#d28e2b"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_V3_DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference-checkpoint", type=Path, default=DEFAULT_REFERENCE_CHECKPOINT)
    parser.add_argument("--no-warm-start", action="store_true")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--sample-seed", type=int, default=20260806)
    parser.add_argument("--restarts", type=int, default=64)
    parser.add_argument("--keep", type=int, default=8)
    parser.add_argument("--coarse-stride", type=int, default=4)
    parser.add_argument("--coarse-steps", type=int, default=250)
    parser.add_argument("--refine-steps", type=int, default=500)
    parser.add_argument("--coarse-lr", type=float, default=3e-2)
    parser.add_argument("--refine-lr", type=float, default=1e-2)
    parser.add_argument("--sample-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--thickness-mm", type=float, default=0.2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip Matplotlib figures; use report_oracle_1x1 after copying results.",
    )
    return parser.parse_args()


def configure_plotting():
    global plt
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    plt = pyplot


def initialize_raw(batch, restarts, seed, device):
    generator = torch.Generator().manual_seed(seed)
    raw = torch.empty(batch, restarts, 1, len(PARAMETER_NAMES))
    for offset in (0, 3):
        raw[..., offset].normal_(mean=-0.4, std=1.5, generator=generator)
        raw[..., offset + 1].uniform_(-4.0, 4.0, generator=generator)
        raw[..., offset + 2].uniform_(-6.0, 1.0, generator=generator)
    raw[..., 6:].uniform_(-5.0, 4.0, generator=generator)
    raw[:, 0].zero_()
    return raw.to(device)


def candidate_response(physics, raw):
    batch, restarts, cells, width = raw.shape
    reflection, transmission = physics(raw.reshape(batch * restarts, cells, width))
    frequencies = transmission.shape[-1]
    return (
        reflection.reshape(batch, restarts, frequencies),
        transmission.reshape(batch, restarts, frequencies),
    )


def optimize_candidates(physics, raw, target, steps, lr, stage, batch_id):
    raw = torch.nn.Parameter(raw.detach().clone())
    optimizer = torch.optim.Adam((raw,), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, steps), eta_min=lr * 0.05
    )
    best_loss = torch.full(raw.shape[:2], float("inf"), device=raw.device)
    best_raw = raw.detach().clone()
    history = []
    report_every = max(1, steps // 20)

    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        _, transmission = candidate_response(physics, raw)
        prediction = transmission.abs().square()
        candidate_loss = (prediction - target[:, None, :]).square().mean(dim=-1)
        finite = torch.isfinite(candidate_loss)
        if not finite.any():
            raise FloatingPointError(f"All {stage} candidates became non-finite.")

        safe_loss = torch.where(
            finite, candidate_loss, torch.full_like(candidate_loss, float("inf"))
        )
        improved = safe_loss < best_loss
        best_loss = torch.where(improved, safe_loss.detach(), best_loss)
        best_raw = torch.where(
            improved[..., None, None], raw.detach(), best_raw
        )

        candidate_loss[finite].sum().backward()
        if raw.grad is None or not torch.isfinite(raw.grad[finite]).all():
            raise FloatingPointError(f"Non-finite {stage} parameter gradient.")
        optimizer.step()
        with torch.no_grad():
            raw.clamp_(-20.0, 20.0)
        scheduler.step()

        if step == 1 or step == steps or step % report_every == 0:
            per_sample_best = best_loss.min(dim=1).values
            history.append(
                {
                    "sample_batch": batch_id,
                    "stage": stage,
                    "step": step,
                    "mean_best_mse": float(per_sample_best.mean()),
                    "median_best_mse": float(per_sample_best.median()),
                    "minimum_candidate_mse": float(best_loss.min()),
                    "finite_candidate_fraction": float(finite.float().mean()),
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

    return best_raw, best_loss, history


def retain_best(raw, loss, count):
    count = min(count, raw.shape[1])
    indices = torch.topk(loss, k=count, dim=1, largest=False).indices
    gather = indices[..., None, None].expand(-1, -1, raw.shape[2], raw.shape[3])
    return raw.gather(1, gather), loss.gather(1, indices)


def write_history(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_sample_metrics(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot_error_ladder(output, per_sample):
    labels = ("Training-target mean", "Unary Reference", "Direct-fit oracle")
    values = (
        per_sample["mean_mse"],
        per_sample["reference_mse"],
        per_sample["oracle_mse"],
    )
    colors = (MEAN_COLOR, REFERENCE_COLOR, ORACLE_COLOR)
    means = [float(np.mean(value)) for value in values]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].bar(labels, means, color=colors)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Mean test MSE")
    axes[0].set_title("Decoder error ladder")
    axes[0].tick_params(axis="x", rotation=18)
    axes[0].grid(axis="y", alpha=0.22)

    positions = np.arange(1, 4)
    box = axes[1].boxplot(values, positions=positions, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axes[1].set_yscale("log")
    axes[1].set_xticks(positions, labels, rotation=18)
    axes[1].set_ylabel("Per-sample MSE")
    axes[1].set_title("Error distribution")
    axes[1].grid(axis="y", alpha=0.22)
    path = output / "01_error_ladder.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_spectra(output, freq, target, reference, oracle):
    oracle_error = np.mean((oracle - target) ** 2, axis=1)
    order = np.argsort(oracle_error)
    selected = (order[0], order[len(order) // 2], order[-1])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    mean_ax = axes[0, 0]
    mean_ax.plot(freq, target.mean(0), color=TARGET_COLOR, linewidth=2.0, label="CST target")
    mean_ax.plot(freq, reference.mean(0), color=REFERENCE_COLOR, linewidth=1.4, label="Unary Reference")
    mean_ax.plot(freq, oracle.mean(0), color=ORACLE_COLOR, linewidth=1.4, label="Direct-fit oracle")
    mean_ax.set_title("Mean fitted spectrum")
    mean_ax.legend(frameon=False)

    for ax, index, label in zip(axes.flat[1:], selected, ("Best", "Median", "Worst")):
        ax.plot(freq, target[index], color=TARGET_COLOR, linewidth=1.8, label="CST target")
        ax.plot(freq, reference[index], color=REFERENCE_COLOR, linewidth=1.2, label="Unary Reference")
        ax.plot(freq, oracle[index], color=ORACLE_COLOR, linewidth=1.2, label="Oracle")
        ax.set_title(f"{label} oracle fit | MSE={oracle_error[index]:.3e}")

    for ax in axes.flat:
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Power transmittance T")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.2)
    path = output / "02_spectra.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path, [int(index) for index in selected]


def plot_optimization(output, history):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, stage in zip(axes, ("coarse", "refine")):
        batches = sorted({row["sample_batch"] for row in history if row["stage"] == stage})
        for batch in batches:
            rows = [
                row for row in history
                if row["stage"] == stage and row["sample_batch"] == batch
            ]
            ax.plot(
                [row["step"] for row in rows],
                [row["mean_best_mse"] for row in rows],
                color=ORACLE_COLOR,
                alpha=0.28,
                linewidth=0.9,
            )
        ax.set_yscale("log")
        ax.set_xlabel("Optimization step")
        ax.set_ylabel("Mean best MSE in sample batch")
        ax.set_title(f"{stage.capitalize()} search")
        ax.grid(alpha=0.2)
    path = output / "03_optimization.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main():
    args = parse_args()
    if args.samples < 1 or args.restarts < 1 or args.keep < 1:
        raise ValueError("samples, restarts, and keep must be positive.")
    if args.keep > args.restarts:
        raise ValueError("--keep cannot exceed --restarts.")
    if args.coarse_stride < 1 or args.sample_batch_size < 1:
        raise ValueError("stride and sample batch size must be positive.")
    if args.coarse_steps < 1 or args.refine_steps < 1:
        raise ValueError("Both optimization stage lengths must be positive.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    seed_everything(args.seed)
    device = torch.device(args.device)
    data = load_v3_1x1(args.dataset, seed=args.split_seed)
    split = data["splits"][args.split]
    rng = np.random.default_rng(args.sample_seed)
    selection = rng.permutation(len(split["T"]))[: min(args.samples, len(split["T"]))]
    target = split["T"][selection]
    target_s11 = split["S11"][selection]
    target_s21 = split["S21"][selection]
    atoms = split["atoms"][selection]
    source_indices = split["indices"][selection]

    checkpoint, reference_model = load_model_checkpoint(
        args.reference_checkpoint, device=device
    )
    if not np.allclose(checkpoint["freq_GHz"], data["freq_GHz"]):
        raise ValueError("Reference checkpoint and dataset frequency grids differ.")
    reference_x = normalize_with(atoms, checkpoint["normalization"])
    reference = batched_model_outputs(
        reference_model, reference_x, batch_size=128, device=device
    )
    reference_t = np.abs(reference["S21"]) ** 2
    training_mean = data["splits"]["train"]["T"].mean(axis=0)
    mean_prediction = np.broadcast_to(training_mean, target.shape)

    freq = data["freq_GHz"]
    coarse_indices = np.arange(0, len(freq), args.coarse_stride)
    if coarse_indices[-1] != len(freq) - 1:
        coarse_indices = np.append(coarse_indices, len(freq) - 1)
    physics_kwargs = {
        "thickness_mm": args.thickness_mm,
        "n_e": 1,
        "n_m": 1,
    }
    coarse_physics = LorentzPhysics(freq[coarse_indices], **physics_kwargs).to(device)
    full_physics = LorentzPhysics(freq, **physics_kwargs).to(device)

    output_raw = []
    output_physical = []
    output_s11 = []
    output_s21 = []
    history = []
    start_time = time.time()
    for start in range(0, len(target), args.sample_batch_size):
        stop = min(start + args.sample_batch_size, len(target))
        batch_id = start // args.sample_batch_size
        target_batch = torch.from_numpy(target[start:stop]).to(device)
        raw = initialize_raw(
            stop - start,
            args.restarts,
            args.seed + 1009 * batch_id,
            device,
        )
        if not args.no_warm_start:
            raw[:, 0] = torch.from_numpy(reference["raw"][start:stop]).to(device)

        raw, loss, rows = optimize_candidates(
            coarse_physics,
            raw,
            target_batch[:, coarse_indices],
            args.coarse_steps,
            args.coarse_lr,
            "coarse",
            batch_id,
        )
        history.extend(rows)
        raw, _ = retain_best(raw, loss, args.keep)
        raw, loss, rows = optimize_candidates(
            full_physics,
            raw,
            target_batch,
            args.refine_steps,
            args.refine_lr,
            "refine",
            batch_id,
        )
        history.extend(rows)
        raw, _ = retain_best(raw, loss, 1)
        best_raw = raw[:, 0]
        with torch.no_grad():
            reflection, transmission = full_physics(best_raw)
            physical = full_physics.physical_parameters(best_raw)
        output_raw.append(best_raw.cpu())
        output_physical.append(physical.cpu())
        output_s11.append(reflection.cpu())
        output_s21.append(transmission.cpu())
        current_mse = float(
            (transmission.abs().square() - target_batch).square().mean()
        )
        print(
            f"samples {start + 1}-{stop}/{len(target)} | oracle MSE {current_mse:.3e}",
            flush=True,
        )

    raw = torch.cat(output_raw).numpy()
    physical = torch.cat(output_physical).numpy()
    s11 = torch.cat(output_s11).numpy()
    s21 = torch.cat(output_s21).numpy()
    oracle_t = np.abs(s21) ** 2

    per_sample = {
        "mean_mse": np.mean((mean_prediction - target) ** 2, axis=1),
        "reference_mse": np.mean((reference_t - target) ** 2, axis=1),
        "oracle_mse": np.mean((oracle_t - target) ** 2, axis=1),
        "reference_complex_s_mse": 0.5
        * (
            np.mean(np.abs(reference["S11"] - target_s11) ** 2, axis=1)
            + np.mean(np.abs(reference["S21"] - target_s21) ** 2, axis=1)
        ),
        "oracle_complex_s_mse": 0.5
        * (
            np.mean(np.abs(s11 - target_s11) ** 2, axis=1)
            + np.mean(np.abs(s21 - target_s21) ** 2, axis=1)
        ),
    }
    sample_rows = []
    for index in range(len(target)):
        row = {
            "selection_index": index,
            "source_dataset_index": int(source_indices[index]),
            **{name: float(values[index]) for name, values in per_sample.items()},
        }
        for parameter_index, name in enumerate(PARAMETER_NAMES):
            row[f"oracle_{name}"] = float(physical[index, 0, parameter_index])
        sample_rows.append(row)

    args.output_root.mkdir(parents=True, exist_ok=True)
    history_path = args.output_root / "optimization_history.csv"
    samples_path = args.output_root / "sample_metrics.csv"
    write_history(history_path, history)
    write_sample_metrics(samples_path, sample_rows)
    np.savez_compressed(
        args.output_root / "fits.npz",
        freq_GHz=freq,
        source_indices=source_indices,
        atoms=atoms,
        target_T=target,
        target_S11=target_s11,
        target_S21=target_s21,
        mean_T=mean_prediction,
        reference_T=reference_t,
        reference_raw=reference["raw"],
        reference_physical=reference["physical"],
        oracle_T=oracle_t,
        oracle_S11=s11,
        oracle_S21=s21,
        oracle_raw=raw,
        oracle_physical=physical,
        parameter_names=np.asarray(PARAMETER_NAMES),
    )
    plot_paths = []
    selected_examples = []
    if not args.no_plots:
        configure_plotting()
        ladder_path = plot_error_ladder(args.output_root, per_sample)
        spectra_path, selected_examples = plot_spectra(
            args.output_root, freq, target, reference_t, oracle_t
        )
        optimization_path = plot_optimization(args.output_root, history)
        plot_paths = [ladder_path, spectra_path, optimization_path]

    mean_mse = {name: float(values.mean()) for name, values in per_sample.items()}
    summary = {
        "experiment": "direct constrained Lorentz fit on v3 1x1 spectra",
        "interpretation": (
            "Per-spectrum optimization result; this estimates decoder capacity "
            "and is not a geometry-to-spectrum generalization metric."
        ),
        "status": "completed",
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256_file(args.dataset),
        "reference_checkpoint": str(args.reference_checkpoint.resolve()),
        "reference_checkpoint_sha256": sha256_file(args.reference_checkpoint),
        "split": args.split,
        "split_seed": args.split_seed,
        "sample_seed": args.sample_seed,
        "sample_count": len(target),
        "source_indices": source_indices.tolist(),
        "configuration": {
            "restarts": args.restarts,
            "warm_start_included": not args.no_warm_start,
            "retained_after_coarse": args.keep,
            "coarse_stride": args.coarse_stride,
            "coarse_frequency_points": len(coarse_indices),
            "coarse_steps": args.coarse_steps,
            "refine_steps": args.refine_steps,
            "coarse_lr": args.coarse_lr,
            "refine_lr": args.refine_lr,
            "thickness_mm": args.thickness_mm,
            "seed": args.seed,
            "device": str(device),
        },
        "mean_metrics": mean_mse,
        "oracle_mse_reduction_vs_reference_percent": 100.0
        * (mean_mse["reference_mse"] - mean_mse["oracle_mse"])
        / mean_mse["reference_mse"],
        "reference_to_oracle_mse_ratio": mean_mse["reference_mse"]
        / mean_mse["oracle_mse"],
        "median_oracle_sample_mse": float(np.median(per_sample["oracle_mse"])),
        "selected_plot_examples": selected_examples,
        "seconds": time.time() - start_time,
        "artifacts": {
            "fits": str((args.output_root / "fits.npz").resolve()),
            "sample_metrics": str(samples_path.resolve()),
            "optimization_history": str(history_path.resolve()),
            "plots": [str(path.resolve()) for path in plot_paths],
        },
    }
    write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
