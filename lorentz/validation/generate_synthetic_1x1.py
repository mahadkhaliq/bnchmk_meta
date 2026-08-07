"""Generate self-consistent 1x1 spectra with known unary Lorentz parameters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..lorentz import LorentzPhysics
from .common import DEFAULT_V3_DATASET, PARAMETER_NAMES, ROOT, sha256_file
from .synthetic import teacher_raw_parameters


DEFAULT_OUTPUT = (
    ROOT / "lorentz" / "experiments" / "unary_validation_1x1_20260806"
    / "data" / "synthetic_unary_1x1.npz"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dataset", type=Path, default=DEFAULT_V3_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-val", type=int, default=2000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--thickness-mm", type=float, default=0.2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    counts = (args.n_train, args.n_val, args.n_test)
    if any(count < 1 for count in counts):
        raise ValueError("Every split size must be positive.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    with np.load(args.reference_dataset, allow_pickle=True) as data:
        atoms_reference = np.asarray(data["atoms"], dtype=np.float32)
        freq = np.asarray(data["freq_GHz"], dtype=np.float32)
        feature_names = np.asarray(data["feat_names"])

    geometry_min = atoms_reference.min(axis=(0, 1))
    geometry_max = atoms_reference.max(axis=(0, 1))
    rng = np.random.default_rng(args.seed)
    total = sum(counts)
    normalized_teacher_geometry = rng.uniform(
        -1.0, 1.0, size=(total, 1, 4)
    ).astype(np.float32)
    atoms = (
        geometry_min.reshape(1, 1, 4)
        + 0.5
        * (normalized_teacher_geometry + 1.0)
        * (geometry_max - geometry_min).reshape(1, 1, 4)
    ).astype(np.float32)

    split = np.empty(total, dtype="U5")
    split[: args.n_train] = "train"
    split[args.n_train : args.n_train + args.n_val] = "val"
    split[args.n_train + args.n_val :] = "test"

    device = torch.device(args.device)
    physics = LorentzPhysics(
        freq,
        thickness_mm=args.thickness_mm,
        n_e=1,
        n_m=1,
    ).to(device)
    raw_parts = []
    physical_parts = []
    reflection_parts = []
    transmission_parts = []
    with torch.no_grad():
        for start in range(0, total, args.batch_size):
            geometry = torch.from_numpy(
                normalized_teacher_geometry[start : start + args.batch_size]
            ).to(device)
            raw = teacher_raw_parameters(geometry)
            reflection, transmission = physics(raw)
            raw_parts.append(raw.cpu())
            physical_parts.append(physics.physical_parameters(raw).cpu())
            reflection_parts.append(reflection.cpu())
            transmission_parts.append(transmission.cpu())

    theta_raw = torch.cat(raw_parts).numpy().astype(np.float32)
    theta_physical = torch.cat(physical_parts).numpy().astype(np.float32)
    s11 = torch.cat(reflection_parts).numpy().astype(np.complex64)
    s21 = torch.cat(transmission_parts).numpy().astype(np.complex64)
    transmittance = np.abs(s21).astype(np.float32) ** 2

    arrays = (theta_raw, theta_physical, s11, s21, transmittance)
    if not all(np.isfinite(value).all() for value in arrays):
        raise FloatingPointError("Synthetic generator produced non-finite values.")

    metadata = {
        "experiment": "self-consistent unary Lorentz parameter recovery",
        "teacher": "lorentz.validation.synthetic.teacher_raw_parameters",
        "reference_dataset": str(args.reference_dataset.resolve()),
        "reference_dataset_sha256": sha256_file(args.reference_dataset),
        "seed": args.seed,
        "counts": {
            "train": args.n_train,
            "val": args.n_val,
            "test": args.n_test,
        },
        "thickness_mm": args.thickness_mm,
        "geometry_generation_bounds": {
            "min": geometry_min.tolist(),
            "max": geometry_max.tolist(),
        },
        "physics_mapping": {
            "wp_scale": 0.5,
            "wp_floor": 1e-5,
            "gamma_scale": 0.1,
            "gamma_floor": 1e-4,
            "epsilon_inf_offset": 1.0,
            "mu_inf_offset": 1.0,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        freq_GHz=freq,
        atoms=atoms,
        geometry_teacher_normalized=normalized_teacher_geometry,
        split=split,
        theta_raw=theta_raw,
        theta_physical=theta_physical,
        theta_names=np.asarray(PARAMETER_NAMES),
        T=transmittance,
        S11=s11,
        S21=s21,
        feat_names=feature_names,
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    metadata["output"] = str(args.output.resolve())
    metadata["output_sha256"] = sha256_file(args.output)
    metadata["shapes"] = {
        "atoms": list(atoms.shape),
        "theta_raw": list(theta_raw.shape),
        "theta_physical": list(theta_physical.shape),
        "S11": list(s11.shape),
        "S21": list(s21.shape),
        "T": list(transmittance.shape),
    }
    metadata["ranges"] = {
        "T": [float(transmittance.min()), float(transmittance.max())],
        "theta_physical_min": theta_physical.min(axis=(0, 1)).tolist(),
        "theta_physical_max": theta_physical.max(axis=(0, 1)).tolist(),
    }
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
