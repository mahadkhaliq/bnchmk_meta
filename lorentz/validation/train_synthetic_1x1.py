"""Train F1 end-to-end on self-consistent synthetic unary Lorentz data."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..lorentz import Model
from ..losses import beta2_loss
from .common import (
    PARAMETER_NAMES,
    ROOT,
    parameter_metrics,
    seed_everything,
    sha256_file,
    write_json,
)


DEFAULT_DATASET = (
    ROOT / "lorentz" / "experiments" / "unary_validation_1x1_20260806"
    / "data" / "synthetic_unary_1x1.npz"
)
DEFAULT_OUTPUT = (
    ROOT / "lorentz" / "experiments" / "unary_validation_1x1_20260806"
    / "synthetic_recovery" / "seed_0"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--activation", choices=("silu", "relu"), default="silu")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--parameter-loss-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--thickness-mm", type=float, default=0.2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def load_dataset(path):
    with np.load(path, allow_pickle=True) as data:
        required = {
            "freq_GHz",
            "atoms",
            "split",
            "theta_raw",
            "theta_physical",
            "theta_names",
            "T",
            "S11",
            "S21",
            "feat_names",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} is missing keys: {sorted(missing)}")
        values = {name: np.asarray(data[name]) for name in required}

    names = tuple(str(name) for name in values["theta_names"])
    if names != PARAMETER_NAMES:
        raise ValueError(f"Unexpected parameter order: {names}")
    atoms = values["atoms"].astype(np.float32)
    theta_raw = values["theta_raw"].astype(np.float32)
    theta_physical = values["theta_physical"].astype(np.float32)
    target = values["T"].astype(np.float32)
    freq = values["freq_GHz"].astype(np.float32)
    expected_theta = (len(atoms), 1, len(PARAMETER_NAMES))
    if atoms.shape != (len(atoms), 1, 4):
        raise ValueError(f"Expected atoms shaped (B,1,4), got {atoms.shape}.")
    if theta_raw.shape != expected_theta or theta_physical.shape != expected_theta:
        raise ValueError("Synthetic parameter arrays have unexpected shapes.")
    if target.shape != (len(atoms), len(freq)):
        raise ValueError("Synthetic T and frequency shapes do not match.")

    split_labels = values["split"].astype(str)
    indices = {name: np.flatnonzero(split_labels == name) for name in ("train", "val", "test")}
    if any(not len(index) for index in indices.values()):
        raise ValueError("Synthetic dataset must contain train, val, and test splits.")

    train_atoms = atoms[indices["train"]]
    x_min = train_atoms.min(axis=(0, 1), keepdims=True)
    x_max = train_atoms.max(axis=(0, 1), keepdims=True)
    x_span = np.maximum(x_max - x_min, 1e-8)
    raw_mean = theta_raw[indices["train"]].mean(axis=(0, 1), keepdims=True)
    raw_std = np.maximum(
        theta_raw[indices["train"]].std(axis=(0, 1), keepdims=True), 1e-6
    )
    physical_mean = theta_physical[indices["train"]].mean(
        axis=(0, 1), keepdims=True
    )
    physical_std = np.maximum(
        theta_physical[indices["train"]].std(axis=(0, 1), keepdims=True), 1e-6
    )

    splits = {}
    for name, index in indices.items():
        splits[name] = {
            "indices": index,
            "x": (2.0 * (atoms[index] - x_min) / x_span - 1.0).astype(np.float32),
            "T": target[index],
            "theta_raw": theta_raw[index],
            "theta_physical": theta_physical[index],
            "S11": values["S11"][index].astype(np.complex64),
            "S21": values["S21"][index].astype(np.complex64),
        }
    return {
        "freq_GHz": freq,
        "feature_names": [str(name) for name in values["feat_names"]],
        "splits": splits,
        "normalization": {
            "min": x_min.squeeze().tolist(),
            "max": x_max.squeeze().tolist(),
        },
        "raw_mean": raw_mean.astype(np.float32),
        "raw_std": raw_std.astype(np.float32),
        "physical_mean": physical_mean.astype(np.float32),
        "physical_std": physical_std.astype(np.float32),
    }


def make_loader(split, batch_size, shuffle, seed):
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        TensorDataset(
            torch.from_numpy(split["x"]),
            torch.from_numpy(split["T"]),
            torch.from_numpy(split["theta_raw"]),
            torch.from_numpy(split["theta_physical"]),
            torch.from_numpy(split["S11"]),
            torch.from_numpy(split["S21"]),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def train_epoch(model, loader, optimizer, raw_std, parameter_weight, device):
    model.train()
    sums = {"total": 0.0, "beta2": 0.0, "parameter": 0.0}
    samples = 0
    gradient_sum = 0.0
    gradient_max = 0.0
    steps = 0
    for geometry, target, theta_raw, _, _, _ in loader:
        geometry = geometry.to(device)
        target = target.to(device)
        theta_raw = theta_raw.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction_raw = model.f1(geometry)
        _, transmission = model.physics(prediction_raw)
        prediction = transmission.abs().square()
        spectrum_loss = beta2_loss(prediction, target)
        parameter_loss = (((prediction_raw - theta_raw) / raw_std) ** 2).mean()
        loss = spectrum_loss + parameter_weight * parameter_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite synthetic training loss.")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0, error_if_nonfinite=True
        )
        optimizer.step()
        batch = len(geometry)
        sums["total"] += float(loss.detach()) * batch
        sums["beta2"] += float(spectrum_loss.detach()) * batch
        sums["parameter"] += float(parameter_loss.detach()) * batch
        samples += batch
        gradient_value = float(gradient.detach())
        gradient_sum += gradient_value
        gradient_max = max(gradient_max, gradient_value)
        steps += 1
    return {
        "total": sums["total"] / samples,
        "beta2": sums["beta2"] / samples,
        "parameter_standardized_mse": sums["parameter"] / samples,
        "mean_gradient_norm": gradient_sum / steps,
        "max_gradient_norm": gradient_max,
    }


@torch.no_grad()
def evaluate(model, loader, raw_std, physical_std, device, collect=False):
    model.eval()
    totals = {
        "mse": 0.0,
        "beta2": 0.0,
        "mae": 0.0,
        "complex_s_mse": 0.0,
        "raw_parameter_standardized_mse": 0.0,
        "physical_parameter_standardized_mse": 0.0,
    }
    samples = 0
    collected = {
        "indices": [],
        "target_T": [],
        "prediction_T": [],
        "target_raw": [],
        "prediction_raw": [],
        "target_physical": [],
        "prediction_physical": [],
    }
    position = 0
    for geometry, target, theta_raw, theta_physical, s11, s21 in loader:
        geometry = geometry.to(device)
        target = target.to(device)
        theta_raw = theta_raw.to(device)
        theta_physical = theta_physical.to(device)
        s11 = s11.to(device)
        s21 = s21.to(device)
        prediction_raw = model.f1(geometry)
        reflection, transmission = model.physics(prediction_raw)
        prediction = transmission.abs().square()
        prediction_physical = model.physics.physical_parameters(prediction_raw)
        if not all(
            torch.isfinite(value).all()
            for value in (prediction, prediction_raw, prediction_physical)
        ):
            raise FloatingPointError("Non-finite synthetic evaluation value.")

        error = prediction - target
        weight = 1.0 + 2.0 * (1.0 - target).clamp(min=0.0).square()
        batch = len(geometry)
        totals["mse"] += float(error.square().mean(dim=1).sum())
        totals["beta2"] += float((weight * error.square()).mean(dim=1).sum())
        totals["mae"] += float(error.abs().mean(dim=1).sum())
        complex_error = 0.5 * (
            (reflection - s11).abs().square().mean(dim=1)
            + (transmission - s21).abs().square().mean(dim=1)
        )
        totals["complex_s_mse"] += float(complex_error.sum())
        raw_error = ((prediction_raw - theta_raw) / raw_std).square().mean(dim=(1, 2))
        physical_error = (
            (prediction_physical - theta_physical) / physical_std
        ).square().mean(dim=(1, 2))
        totals["raw_parameter_standardized_mse"] += float(raw_error.sum())
        totals["physical_parameter_standardized_mse"] += float(physical_error.sum())
        samples += batch

        if collect:
            collected["indices"].append(np.arange(position, position + batch))
            collected["target_T"].append(target.cpu().numpy())
            collected["prediction_T"].append(prediction.cpu().numpy())
            collected["target_raw"].append(theta_raw.cpu().numpy())
            collected["prediction_raw"].append(prediction_raw.cpu().numpy())
            collected["target_physical"].append(theta_physical.cpu().numpy())
            collected["prediction_physical"].append(prediction_physical.cpu().numpy())
        position += batch

    metrics = {name: value / samples for name, value in totals.items()}
    if not collect:
        return metrics, None
    return metrics, {
        name: np.concatenate(parts) for name, parts in collected.items()
    }


def write_history(path, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch size must be positive.")
    if args.parameter_loss_weight < 0:
        raise ValueError("--parameter-loss-weight must be non-negative.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    seed_everything(args.seed)
    device = torch.device(args.device)
    data = load_dataset(args.dataset)
    loaders = {
        name: make_loader(
            split,
            args.batch_size,
            shuffle=name == "train",
            seed=args.seed,
        )
        for name, split in data["splits"].items()
    }
    raw_std = torch.from_numpy(data["raw_std"]).to(device)
    physical_std = torch.from_numpy(data["physical_std"]).to(device)

    model_config = {
        "n_geom": 4,
        "n_e": 1,
        "n_m": 1,
        "hidden": args.hidden,
        "depth": args.depth,
        "activation": args.activation,
        "thickness_mm": args.thickness_mm,
    }
    model = Model(freq_GHz=data["freq_GHz"], **model_config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(3, args.epochs // 20)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "model.pt"
    history_path = args.output_dir / "history.csv"
    metrics_path = args.output_dir / "metrics.json"
    best_val = float("inf")
    best_epoch = -1
    history = []
    start_time = time.time()
    print(
        "Synthetic unary recovery | samples "
        + "/".join(str(len(data["splits"][name]["x"])) for name in ("train", "val", "test"))
        + f" | {args.hidden}x{args.depth} {args.activation.upper()} | "
        f"parameter loss weight={args.parameter_loss_weight:g} | seed={args.seed} | device={device}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        train = train_epoch(
            model,
            loaders["train"],
            optimizer,
            raw_std,
            args.parameter_loss_weight,
            device,
        )
        val, _ = evaluate(
            model, loaders["val"], raw_std, physical_std, device
        )
        scheduler.step(val["mse"])
        row = {
            "epoch": epoch,
            "train_total": train["total"],
            "train_beta2": train["beta2"],
            "train_parameter_standardized_mse": train["parameter_standardized_mse"],
            "val_mse": val["mse"],
            "val_beta2": val["beta2"],
            "val_complex_s_mse": val["complex_s_mse"],
            "val_raw_parameter_standardized_mse": val["raw_parameter_standardized_mse"],
            "val_physical_parameter_standardized_mse": val["physical_parameter_standardized_mse"],
            "mean_gradient_norm": train["mean_gradient_norm"],
            "max_gradient_norm": train["max_gradient_norm"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        if val["mse"] < best_val:
            best_val = val["mse"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": model_config,
                    "freq_GHz": data["freq_GHz"],
                    "feature_names": data["feature_names"],
                    "parameter_names": PARAMETER_NAMES,
                    "normalization": data["normalization"],
                    "raw_parameter_mean": data["raw_mean"],
                    "raw_parameter_std": data["raw_std"],
                    "physical_parameter_mean": data["physical_mean"],
                    "physical_parameter_std": data["physical_std"],
                    "training_loss": "beta2"
                    if args.parameter_loss_weight == 0
                    else "beta2_plus_standardized_raw_parameter_mse",
                    "parameter_loss_weight": args.parameter_loss_weight,
                    "seed": args.seed,
                    "best_val_mse": best_val,
                    "best_epoch": best_epoch,
                    "dataset_sha256": sha256_file(args.dataset),
                },
                checkpoint_path,
            )
        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.epochs // 10) == 0:
            print(
                f"epoch {epoch:4d} | train beta2 {train['beta2']:.3e} | "
                f"val T MSE {val['mse']:.3e} | val P std-MSE "
                f"{val['physical_parameter_standardized_mse']:.3e} | "
                f"lr {optimizer.param_groups[0]['lr']:.2e}",
                flush=True,
            )
            write_history(history_path, history)
    write_history(history_path, history)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test, predictions = evaluate(
        model, loaders["test"], raw_std, physical_std, device, collect=True
    )
    test["raw_parameter_metrics"] = parameter_metrics(
        predictions["prediction_raw"], predictions["target_raw"]
    )
    test["physical_parameter_metrics"] = parameter_metrics(
        predictions["prediction_physical"], predictions["target_physical"]
    )
    train_mean = data["splits"]["train"]["T"].mean(axis=0)
    target_test = predictions["target_T"]
    test["training_target_mean_mse"] = float(
        np.mean((target_test - train_mean[None, :]) ** 2)
    )
    test["per_sample_mse_mean"] = float(
        np.mean((predictions["prediction_T"] - target_test) ** 2, axis=1).mean()
    )
    test["per_sample_mse_median"] = float(
        np.median(np.mean((predictions["prediction_T"] - target_test) ** 2, axis=1))
    )

    np.savez_compressed(
        args.output_dir / "test_predictions.npz",
        freq_GHz=data["freq_GHz"],
        test_indices=data["splits"]["test"]["indices"],
        target_T=predictions["target_T"],
        prediction_T=predictions["prediction_T"],
        target_raw=predictions["target_raw"],
        prediction_raw=predictions["prediction_raw"],
        target_physical=predictions["target_physical"],
        prediction_physical=predictions["prediction_physical"],
        parameter_names=np.asarray(PARAMETER_NAMES),
    )
    result = {
        "status": "completed",
        "experiment": "self-consistent synthetic unary recovery",
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256_file(args.dataset),
        "seed": args.seed,
        "requested_epochs": args.epochs,
        "completed_epochs": len(history),
        "best_epoch": best_epoch,
        "best_val_mse": best_val,
        "parameter_loss_weight": args.parameter_loss_weight,
        "model_config": model_config,
        "optimizer": {
            "name": "Adam",
            "initial_lr": args.lr,
            "weight_decay": args.weight_decay,
        },
        "scheduler": {
            "name": "ReduceLROnPlateau",
            "factor": 0.5,
            "patience": max(3, args.epochs // 20),
            "selection_metric": "validation plain T MSE",
        },
        "split_sizes": {
            name: len(split["x"]) for name, split in data["splits"].items()
        },
        "test": test,
        "seconds": time.time() - start_time,
        "artifacts": {
            "checkpoint": str(checkpoint_path.resolve()),
            "history": str(history_path.resolve()),
            "test_predictions": str((args.output_dir / "test_predictions.npz").resolve()),
        },
    }
    write_json(metrics_path, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
