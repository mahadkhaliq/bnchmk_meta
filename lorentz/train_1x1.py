"""Train and evaluate the finite-slab Lorentz model on the v3 1x1 dataset.

Run from the repository root:

    python -m lorentz.train_1x1 --epochs 200

The supervised target is the stored power transmittance T. The model still
computes complex S11 and S21 internally, with T_pred = |S21|^2.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .lorentz import Model
from .losses import beta2_loss


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "power_tx_data" / "version_3" / "preprocessed_1x1.npz"
DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parent
    / "artifacts"
    / "best_1x1_silu_beta2_512_500ep.pt"
)
CONSTRAINT_NAMES = ("wp", "w0", "gamma", "epsilon_inf", "mu_inf")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--activation", choices=("silu", "relu"), default="silu")
    parser.add_argument("--n-e", type=int, default=1)
    parser.add_argument("--n-m", type=int, default=1)
    parser.add_argument("--thickness-mm", type=float, default=0.2)
    parser.add_argument("--wp-scale", type=float, default=0.5)
    parser.add_argument("--wp-floor", type=float, default=1e-5)
    parser.add_argument("--gamma-scale", type=float, default=0.1)
    parser.add_argument("--gamma-floor", type=float, default=1e-4)
    parser.add_argument("--epsilon-inf-offset", type=float, default=1.0)
    parser.add_argument("--mu-inf-offset", type=float, default=1.0)
    parser.add_argument(
        "--raw-physics-parameters",
        action="store_true",
        help=(
            "Feed F1 outputs directly to the Lorentz equations without "
            "softplus, floors, resonance bounds, or constrained backgrounds."
        ),
    )
    parser.add_argument(
        "--only-constraint",
        choices=CONSTRAINT_NAMES,
        default=None,
        help="Enable only this parameter mapping and leave the other four raw.",
    )
    parser.add_argument(
        "--constraints",
        type=str,
        default=None,
        help=(
            "Comma-separated parameter mappings to enable. Use 'all' or 'none' "
            "for the two endpoints. This is mutually exclusive with "
            "--raw-physics-parameters and --only-constraint."
        ),
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limit samples for a quick pipeline check; 0 uses all 2,000.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--history",
        type=Path,
        default=None,
        help="CSV path; defaults to <checkpoint stem>.history.csv.",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="JSON path; defaults to <checkpoint stem>.metrics.json.",
    )
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_constraint_flags(args):
    selectors = (
        int(args.raw_physics_parameters)
        + int(args.only_constraint is not None)
        + int(args.constraints is not None)
    )
    if selectors > 1:
        raise ValueError(
            "--raw-physics-parameters, --only-constraint, and --constraints "
            "are mutually exclusive."
        )

    if args.raw_physics_parameters:
        enabled = set()
    elif args.only_constraint is not None:
        enabled = {args.only_constraint}
    elif args.constraints is None or args.constraints.strip().lower() == "all":
        enabled = set(CONSTRAINT_NAMES)
    elif args.constraints.strip().lower() in {"none", "raw"}:
        enabled = set()
    else:
        enabled = {
            name.strip() for name in args.constraints.split(",") if name.strip()
        }
        unknown = enabled.difference(CONSTRAINT_NAMES)
        if unknown:
            raise ValueError(
                f"Unknown constraints {sorted(unknown)}; choose from "
                f"{list(CONSTRAINT_NAMES)}."
            )

    flags = {name: name in enabled for name in CONSTRAINT_NAMES}
    if not enabled:
        profile = "raw"
    elif len(enabled) == len(CONSTRAINT_NAMES):
        profile = "all"
    elif len(enabled) == 1:
        profile = f"{next(iter(enabled))}_only"
    elif len(enabled) == len(CONSTRAINT_NAMES) - 1:
        disabled = next(name for name in CONSTRAINT_NAMES if name not in enabled)
        profile = f"drop_{disabled}"
    else:
        profile = "custom_" + "_".join(
            name for name in CONSTRAINT_NAMES if name in enabled
        )
    return flags, profile


def load_data(path, seed, max_samples=0):
    with np.load(path, allow_pickle=True) as data:
        required = {"atoms", "T", "freq_GHz", "feat_names"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} is missing keys: {sorted(missing)}")

        atoms = np.asarray(data["atoms"], dtype=np.float32)
        target = np.asarray(data["T"], dtype=np.float32)
        freq = np.asarray(data["freq_GHz"], dtype=np.float32)
        feature_names = [str(name) for name in data["feat_names"]]

    if atoms.ndim != 3 or atoms.shape[1:] != (1, 4):
        raise ValueError(f"Expected 1x1 atoms shaped (B,1,4), got {atoms.shape}.")
    if target.shape != (len(atoms), len(freq)):
        raise ValueError(f"Target/frequency mismatch: T={target.shape}, f={freq.shape}.")
    if feature_names != ["d", "l", "w", "g"]:
        raise ValueError(f"Unexpected feature order: {feature_names}")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(atoms))
    if max_samples:
        if max_samples < 20:
            raise ValueError("--max-samples must be at least 20.")
        order = order[: min(max_samples, len(order))]
    atoms = atoms[order]
    target = target[order]

    n = len(atoms)
    n_test = max(1, round(0.15 * n))
    n_fit = n - n_test
    n_val = max(1, round(0.20 * n_fit))
    n_train = n_fit - n_val
    train = slice(0, n_train)
    val = slice(n_train, n_train + n_val)
    test = slice(n_train + n_val, n)

    x_train = atoms[train]
    x_min = x_train.min(axis=(0, 1), keepdims=True)
    x_max = x_train.max(axis=(0, 1), keepdims=True)
    span = np.maximum(x_max - x_min, 1e-8)

    def normalize(values):
        return (2.0 * (values - x_min) / span - 1.0).astype(np.float32)

    splits = {
        "train": (normalize(atoms[train]), target[train]),
        "val": (normalize(atoms[val]), target[val]),
        "test": (normalize(atoms[test]), target[test]),
    }
    normalization = {"min": x_min.squeeze().tolist(), "max": x_max.squeeze().tolist()}
    return freq, feature_names, splits, normalization


def make_loader(split, batch_size, shuffle, seed):
    x, y = split
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def predict_power(model, geometry):
    return model.power_transmittance(geometry)


def train_epoch(model, loader, optimizer, device):
    model.train()
    squared_error = 0.0
    count = 0
    gradient_norm_sum = 0.0
    gradient_steps = 0
    max_gradient_norm = 0.0
    for geometry, target in loader:
        geometry = geometry.to(device)
        target = target.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = predict_power(model, geometry)
        loss = beta2_loss(prediction, target)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss.")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0, error_if_nonfinite=True
        )
        gradient_norm = float(gradient_norm.detach())
        optimizer.step()
        squared_error += loss.item() * target.numel()
        count += target.numel()
        gradient_norm_sum += gradient_norm
        gradient_steps += 1
        max_gradient_norm = max(max_gradient_norm, gradient_norm)
    return {
        "beta2": squared_error / count,
        "mean_gradient_norm": gradient_norm_sum / gradient_steps,
        "max_gradient_norm": max_gradient_norm,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    squared_error = 0.0
    absolute_error = 0.0
    beta2_error = 0.0
    count = 0
    prediction_min = float("inf")
    prediction_max = float("-inf")
    max_energy_sum = float("-inf")
    prediction_above_one = 0
    energy_above_one = 0
    for geometry, target in loader:
        geometry = geometry.to(device)
        target = target.to(device)
        reflection, transmission = model(geometry)
        prediction = transmission.abs().square()
        reflected_power = reflection.abs().square()
        if not torch.isfinite(prediction).all() or not torch.isfinite(
            reflected_power
        ).all():
            raise FloatingPointError("Non-finite S-parameter power during evaluation.")
        error = prediction - target
        weight = 1.0 + 2.0 * (1.0 - target).clamp(min=0.0).square()
        energy_sum = reflected_power + prediction
        squared_error += error.square().sum().item()
        absolute_error += error.abs().sum().item()
        beta2_error += (weight * error.square()).sum().item()
        count += target.numel()
        prediction_min = min(prediction_min, prediction.min().item())
        prediction_max = max(prediction_max, prediction.max().item())
        max_energy_sum = max(max_energy_sum, energy_sum.max().item())
        prediction_above_one += (prediction > 1.0 + 1e-6).sum().item()
        energy_above_one += (energy_sum > 1.0 + 1e-6).sum().item()
    return {
        "mse": squared_error / count,
        "beta2": beta2_error / count,
        "mae": absolute_error / count,
        "prediction_min": prediction_min,
        "prediction_max": prediction_max,
        "max_R_plus_T": max_energy_sum,
        "fraction_T_above_one": prediction_above_one / count,
        "fraction_R_plus_T_above_one": energy_above_one / count,
    }


def tensor_statistics(value, *, lower=None, upper=None):
    value = torch.cat([part.reshape(-1).cpu() for part in value])
    finite = torch.isfinite(value)
    finite_value = value[finite]
    result = {
        "count": int(value.numel()),
        "finite_fraction": float(finite.float().mean()),
    }
    if not finite_value.numel():
        return result
    quantiles = torch.quantile(
        finite_value.float(), torch.tensor([0.01, 0.5, 0.99])
    )
    result.update(
        {
            "min": float(finite_value.min()),
            "p01": float(quantiles[0]),
            "mean": float(finite_value.float().mean()),
            "median": float(quantiles[1]),
            "p99": float(quantiles[2]),
            "max": float(finite_value.max()),
            "negative_fraction": float((finite_value < 0).float().mean()),
            "below_one_fraction": float((finite_value < 1).float().mean()),
        }
    )
    if lower is not None or upper is not None:
        outside = torch.zeros_like(finite_value, dtype=torch.bool)
        if lower is not None:
            outside |= finite_value < lower
        if upper is not None:
            outside |= finite_value > upper
        result["outside_mapping_bounds_fraction"] = float(outside.float().mean())
    return result


@torch.no_grad()
def collect_parameter_diagnostics(model, loader, device):
    model.eval()
    raw_values = {}
    physical_values = {}

    def append(target, name, value):
        target.setdefault(name, []).append(value.detach())

    for geometry, _ in loader:
        theta = model.f1(geometry.to(device))
        batch, cells, _ = theta.shape
        n_e = model.physics.n_e
        n_m = model.physics.n_m
        n_ep = 3 * n_e
        n_mp = 3 * n_m

        if n_e:
            electric = theta[..., :n_ep].reshape(batch, cells, n_e, 3)
            electric_physical = model.physics._oscillator_parameters(electric)
            for index, name in enumerate(("wp_e", "w0_e", "gamma_e")):
                append(raw_values, name, electric[..., index])
                append(physical_values, name, electric_physical[index])

        if n_m:
            magnetic = theta[..., n_ep : n_ep + n_mp].reshape(
                batch, cells, n_m, 3
            )
            magnetic_physical = model.physics._oscillator_parameters(magnetic)
            for index, name in enumerate(("wp_m", "w0_m", "gamma_m")):
                append(raw_values, name, magnetic[..., index])
                append(physical_values, name, magnetic_physical[index])

        for offset, name, constrained, background_offset in (
            (
                -2,
                "epsilon_inf",
                model.physics.constrain_epsilon_inf,
                model.physics.epsilon_inf_offset,
            ),
            (
                -1,
                "mu_inf",
                model.physics.constrain_mu_inf,
                model.physics.mu_inf_offset,
            ),
        ):
            background_raw = theta[..., offset].mean(dim=1, keepdim=True)
            background = model.physics._background(
                theta[..., offset], constrained, background_offset
            )
            append(raw_values, name, background_raw)
            append(physical_values, name, background)

    physical_statistics = {}
    for name, parts in physical_values.items():
        bounds = {}
        if name in {"w0_e", "w0_m"}:
            bounds = {
                "lower": float(model.physics.w0_min),
                "upper": float(model.physics.w0_max),
            }
        physical_statistics[name] = tensor_statistics(parts, **bounds)
    return {
        "raw": {name: tensor_statistics(parts) for name, parts in raw_values.items()},
        "physical": physical_statistics,
    }


def write_history(path, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "epoch",
                "train_beta2",
                "val_mse",
                "val_beta2",
                "mean_gradient_norm",
                "max_gradient_norm",
                "lr",
            ),
        )
        writer.writeheader()
        writer.writerows(history)


def main():
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be positive.")
    if args.wp_scale <= 0:
        raise ValueError("--wp-scale must be positive.")
    if args.wp_floor < 0:
        raise ValueError("--wp-floor must be non-negative.")
    if args.gamma_scale <= 0:
        raise ValueError("--gamma-scale must be positive.")
    if args.gamma_floor < 0:
        raise ValueError("--gamma-floor must be non-negative.")
    if args.epsilon_inf_offset < 0 or args.mu_inf_offset < 0:
        raise ValueError("Background offsets must be non-negative.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if args.history is None:
        args.history = args.checkpoint.with_suffix(".history.csv")
    if args.metrics is None:
        args.metrics = args.checkpoint.with_suffix(".metrics.json")

    seed_everything(args.seed)
    device = torch.device(args.device)
    freq, feature_names, splits, normalization = load_data(
        args.dataset, args.seed, args.max_samples
    )
    loaders = {
        name: make_loader(
            split,
            args.batch_size,
            shuffle=name == "train",
            seed=args.seed,
        )
        for name, split in splits.items()
    }

    constraint_flags, constraint_profile = resolve_constraint_flags(args)

    physics_config = {
        "parameterize": all(constraint_flags.values()),
        "wp_scale": args.wp_scale,
        "wp_floor": args.wp_floor,
        "gamma_scale": args.gamma_scale,
        "gamma_floor": args.gamma_floor,
        "epsilon_inf_offset": args.epsilon_inf_offset,
        "mu_inf_offset": args.mu_inf_offset,
        "constrain_wp": constraint_flags["wp"],
        "constrain_w0": constraint_flags["w0"],
        "constrain_gamma": constraint_flags["gamma"],
        "constrain_epsilon_inf": constraint_flags["epsilon_inf"],
        "constrain_mu_inf": constraint_flags["mu_inf"],
    }

    model = Model(
        n_geom=4,
        n_e=args.n_e,
        n_m=args.n_m,
        hidden=args.hidden,
        depth=args.depth,
        activation=args.activation,
        freq_GHz=freq,
        thickness_mm=args.thickness_mm,
        **physics_config,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(3, args.epochs // 20)
    )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_epoch = -1
    history = []
    checkpoint_written = False
    failure = None
    failure_exception = None
    start = time.time()
    print(
        f"1x1 finite-slab Lorentz | samples "
        f"{len(splits['train'][0])}/{len(splits['val'][0])}/{len(splits['test'][0])} "
        f"| {args.hidden}x{args.depth} {args.activation.upper()} + beta2 "
        f"| oscillators e={args.n_e}, m={args.n_m} "
        f"| scales wp={args.wp_scale:g}, gamma={args.gamma_scale:g} "
        f"| floors wp={args.wp_floor:g}, gamma={args.gamma_floor:g} "
        f"| offsets eps={args.epsilon_inf_offset:g}, "
        f"mu={args.mu_inf_offset:g} "
        f"| constraints={constraint_profile} "
        f"| device={device}"
    )

    try:
        for epoch in range(1, args.epochs + 1):
            train_metrics = train_epoch(model, loaders["train"], optimizer, device)
            val_metrics = evaluate(model, loaders["val"], device)
            scheduler.step(val_metrics["mse"])
            lr = optimizer.param_groups[0]["lr"]
            history.append(
                {
                    "epoch": epoch,
                    "train_beta2": train_metrics["beta2"],
                    "val_mse": val_metrics["mse"],
                    "val_beta2": val_metrics["beta2"],
                    "mean_gradient_norm": train_metrics["mean_gradient_norm"],
                    "max_gradient_norm": train_metrics["max_gradient_norm"],
                    "lr": lr,
                }
            )

            if val_metrics["mse"] < best_val:
                best_val = val_metrics["mse"]
                best_epoch = epoch
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "model_config": {
                            "n_geom": 4,
                            "n_e": args.n_e,
                            "n_m": args.n_m,
                            "hidden": args.hidden,
                            "depth": args.depth,
                            "activation": args.activation,
                            "thickness_mm": args.thickness_mm,
                            **physics_config,
                        },
                        "freq_GHz": freq,
                        "feature_names": feature_names,
                        "normalization": normalization,
                        "training_loss": "beta2",
                        "seed": args.seed,
                        "best_val_mse": best_val,
                        "best_epoch": best_epoch,
                        "constraint_profile": constraint_profile,
                        "constraint_flags": constraint_flags,
                    },
                    args.checkpoint,
                )
                checkpoint_written = True

            report_every = max(1, args.epochs // 10)
            if epoch == 1 or epoch == args.epochs or epoch % report_every == 0:
                print(
                    f"epoch {epoch:4d} | train beta2 "
                    f"{train_metrics['beta2']:.6f} | val MSE "
                    f"{val_metrics['mse']:.6f} | grad "
                    f"{train_metrics['mean_gradient_norm']:.3e}/"
                    f"{train_metrics['max_gradient_norm']:.3e} | lr {lr:.2e}",
                    flush=True,
                )
                write_history(args.history, history)
    except Exception as exc:
        failure_exception = exc
        failure = {
            "type": type(exc).__name__,
            "message": str(exc),
            "failed_during_epoch": len(history) + 1,
        }
        print(f"training failed: {failure}", flush=True)
    finally:
        write_history(args.history, history)

    test_metrics = None
    parameter_diagnostics = None
    evaluation_failure = None
    if checkpoint_written:
        try:
            checkpoint = torch.load(
                args.checkpoint, map_location=device, weights_only=False
            )
            model.load_state_dict(checkpoint["model_state"])
            test_metrics = evaluate(model, loaders["test"], device)
            parameter_diagnostics = collect_parameter_diagnostics(
                model, loaders["test"], device
            )
        except Exception as exc:
            evaluation_failure = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            if failure is None:
                failure_exception = exc
                failure = evaluation_failure

    result = {
        "status": "completed" if failure is None else "failed",
        "constraint_profile": constraint_profile,
        "constraint_flags": constraint_flags,
        "requested_epochs": args.epochs,
        "completed_epochs": len(history),
        "best_val_mse": best_val,
        "best_epoch": best_epoch,
        "test": test_metrics,
        "parameter_diagnostics": parameter_diagnostics,
        "failure": failure,
        "evaluation_failure": evaluation_failure,
        "seconds": time.time() - start,
        "checkpoint": str(args.checkpoint),
        "history": str(args.history),
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if failure_exception is not None:
        raise failure_exception


if __name__ == "__main__":
    main()
