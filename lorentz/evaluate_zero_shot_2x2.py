"""Evaluate the 1x1 Lorentz checkpoint on v3 2x2 without retraining."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from .lorentz import Model
from .train_1x1 import DEFAULT_CHECKPOINT


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "power_tx_data" / "version_3" / "preprocessed_2x2.npz"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent / "artifacts" / "zero_shot_1x1_to_2x2.json"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    with np.load(args.dataset, allow_pickle=True) as data:
        atoms = np.asarray(data["atoms"], dtype=np.float32)
        target = np.asarray(data["T"], dtype=np.float32)
        freq = np.asarray(data["freq_GHz"], dtype=np.float32)
        feature_names = [str(name) for name in data["feat_names"]]

    if atoms.ndim != 3 or atoms.shape[1:] != (4, 4):
        raise ValueError(f"Expected 2x2 atoms shaped (B,4,4), got {atoms.shape}.")
    if target.shape != (len(atoms), len(freq)):
        raise ValueError(f"Target/frequency mismatch: T={target.shape}, f={freq.shape}.")
    if feature_names != checkpoint["feature_names"]:
        raise ValueError(
            f"Feature order differs: data={feature_names}, checkpoint={checkpoint['feature_names']}"
        )
    if not np.allclose(freq, checkpoint["freq_GHz"]):
        raise ValueError("The 1x1 checkpoint and 2x2 data use different frequency grids.")

    indices = np.arange(len(atoms))
    _, test_indices = train_test_split(
        indices, test_size=0.15, random_state=checkpoint["seed"]
    )
    atoms = atoms[test_indices]
    target = target[test_indices]

    normalization = checkpoint["normalization"]
    x_min = np.asarray(normalization["min"], dtype=np.float32).reshape(1, 1, 4)
    x_max = np.asarray(normalization["max"], dtype=np.float32).reshape(1, 1, 4)
    geometry = 2.0 * (atoms - x_min) / np.maximum(x_max - x_min, 1e-8) - 1.0
    geometry = geometry.astype(np.float32)

    model = Model(freq_GHz=freq, **checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    squared_error = 0.0
    weighted_error = 0.0
    absolute_error = 0.0
    count = 0
    prediction_min = float("inf")
    prediction_max = float("-inf")
    max_energy = 0.0
    with torch.no_grad():
        for start in range(0, len(geometry), args.batch_size):
            x = torch.from_numpy(geometry[start : start + args.batch_size])
            y = torch.from_numpy(target[start : start + args.batch_size])
            reflection, transmission = model(x)
            prediction = transmission.abs().square()
            error = prediction - y
            weight = 1.0 + 2.0 * (1.0 - y).clamp(min=0.0).square()

            squared_error += error.square().sum().item()
            weighted_error += (weight * error.square()).sum().item()
            absolute_error += error.abs().sum().item()
            count += y.numel()
            prediction_min = min(prediction_min, prediction.min().item())
            prediction_max = max(prediction_max, prediction.max().item())
            energy = reflection.abs().square() + prediction
            max_energy = max(max_energy, energy.max().item())

        probe = torch.from_numpy(geometry[:1])
        original = model.power_transmittance(probe)
        permuted = model.power_transmittance(probe[:, [2, 0, 3, 1], :])
        permutation_delta = (original - permuted).abs().max().item()

    result = {
        "experiment": "1x1-trained finite-slab Lorentz -> 2x2 v3 zero-shot",
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "test_samples": len(geometry),
        "input_grid_shape_per_sample": [2, 2, 4],
        "input_cell_shape_per_sample": [4, 4],
        "f1_output_shape_per_sample": [4, 8],
        "output_shape_per_sample": [len(freq)],
        "test_mse": squared_error / count,
        "test_beta2": weighted_error / count,
        "test_mae": absolute_error / count,
        "prediction_min": prediction_min,
        "prediction_max": prediction_max,
        "max_R_plus_T": max_energy,
        "cell_permutation_max_delta": permutation_delta,
        "fine_tuned_on_2x2": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
