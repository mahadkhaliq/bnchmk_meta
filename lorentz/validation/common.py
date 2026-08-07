"""Shared data, metric, and serialization helpers for unary validation."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from ..lorentz import Model


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_V3_DATASET = ROOT / "power_tx_data" / "version_3" / "preprocessed_1x1.npz"
DEFAULT_REFERENCE_CHECKPOINT = (
    ROOT / "lorentz" / "artifacts" / "best_1x1_silu_beta2_512_500ep.pt"
)
PARAMETER_NAMES = (
    "wp_e",
    "w0_e",
    "gamma_e",
    "wp_m",
    "w0_m",
    "gamma_m",
    "epsilon_inf",
    "mu_inf",
)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def load_v3_1x1(path=DEFAULT_V3_DATASET, seed=0, max_samples=0):
    """Load v3 using the exact split and train-only normalization convention."""
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        required = {"atoms", "T", "S11", "S21", "freq_GHz", "feat_names"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} is missing keys: {sorted(missing)}")
        arrays = {
            "atoms": np.asarray(data["atoms"], dtype=np.float32),
            "T": np.asarray(data["T"], dtype=np.float32),
            "S11": np.asarray(data["S11"], dtype=np.complex64),
            "S21": np.asarray(data["S21"], dtype=np.complex64),
        }
        freq = np.asarray(data["freq_GHz"], dtype=np.float32)
        feature_names = [str(name) for name in data["feat_names"]]

    if arrays["atoms"].ndim != 3 or arrays["atoms"].shape[1:] != (1, 4):
        raise ValueError(f"Expected atoms shaped (B,1,4), got {arrays['atoms'].shape}.")
    if feature_names != ["d", "l", "w", "g"]:
        raise ValueError(f"Unexpected feature order: {feature_names}")
    for name in ("T", "S11", "S21"):
        if arrays[name].shape != (len(arrays["atoms"]), len(freq)):
            raise ValueError(f"{name}/frequency mismatch: {arrays[name].shape}.")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(arrays["atoms"]))
    if max_samples:
        if max_samples < 20:
            raise ValueError("max_samples must be at least 20.")
        order = order[: min(max_samples, len(order))]

    n = len(order)
    n_test = max(1, round(0.15 * n))
    n_fit = n - n_test
    n_val = max(1, round(0.20 * n_fit))
    n_train = n_fit - n_val
    positions = {
        "train": order[:n_train],
        "val": order[n_train : n_train + n_val],
        "test": order[n_train + n_val :],
    }

    train_atoms = arrays["atoms"][positions["train"]]
    x_min = train_atoms.min(axis=(0, 1), keepdims=True)
    x_max = train_atoms.max(axis=(0, 1), keepdims=True)
    span = np.maximum(x_max - x_min, 1e-8)

    splits = {}
    for split, indices in positions.items():
        atoms = arrays["atoms"][indices]
        splits[split] = {
            "indices": indices,
            "atoms": atoms,
            "x": (2.0 * (atoms - x_min) / span - 1.0).astype(np.float32),
            "T": arrays["T"][indices],
            "S11": arrays["S11"][indices],
            "S21": arrays["S21"][indices],
        }

    return {
        "path": path,
        "freq_GHz": freq,
        "feature_names": feature_names,
        "normalization": {
            "min": x_min.squeeze().tolist(),
            "max": x_max.squeeze().tolist(),
        },
        "splits": splits,
    }


def normalize_with(values, normalization):
    lower = np.asarray(normalization["min"], dtype=np.float32).reshape(1, 1, -1)
    upper = np.asarray(normalization["max"], dtype=np.float32).reshape(1, 1, -1)
    return (2.0 * (values - lower) / np.maximum(upper - lower, 1e-8) - 1.0).astype(
        np.float32
    )


def load_model_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = Model(freq_GHz=checkpoint["freq_GHz"], **checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return checkpoint, model


@torch.no_grad()
def batched_model_outputs(model, geometry, batch_size=128, device="cpu"):
    raw_parts = []
    physical_parts = []
    reflection_parts = []
    transmission_parts = []
    for start in range(0, len(geometry), batch_size):
        x = torch.as_tensor(
            geometry[start : start + batch_size], dtype=torch.float32, device=device
        )
        raw = model.f1(x)
        reflection, transmission = model.physics(raw)
        raw_parts.append(raw.cpu())
        physical_parts.append(model.physics.physical_parameters(raw).cpu())
        reflection_parts.append(reflection.cpu())
        transmission_parts.append(transmission.cpu())
    return {
        "raw": torch.cat(raw_parts).numpy(),
        "physical": torch.cat(physical_parts).numpy(),
        "S11": torch.cat(reflection_parts).numpy(),
        "S21": torch.cat(transmission_parts).numpy(),
    }


def regression_metrics(prediction, target):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    error = prediction - target
    mse = float(np.mean(error**2))
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(mse))
    target_std = float(np.std(target))
    target_variance = float(np.var(target))
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "nrmse_by_target_std": rmse / target_std if target_std > 0 else None,
        "r2": 1.0 - mse / target_variance if target_variance > 0 else None,
    }


def parameter_metrics(prediction, target, names=PARAMETER_NAMES):
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    if prediction.shape != target.shape or prediction.shape[-1] != len(names):
        raise ValueError(
            f"Parameter arrays must share (...,{len(names)}) shape; got "
            f"{prediction.shape} and {target.shape}."
        )
    return {
        name: regression_metrics(prediction[..., index], target[..., index])
        for index, name in enumerate(names)
    }
