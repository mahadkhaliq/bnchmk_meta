#!/usr/bin/env python3
"""Validate an assembled coupled-Lorentz synthetic dataset release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    args = parser.parse_args()

    manifest_path = args.dataset_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    expected_band = tuple(manifest["common_band_GHz"])
    expected_freq = int(manifest["n_freq"])
    total_samples = 0
    total_bytes = 0

    for tag, details in sorted(
        manifest["scales"].items(), key=lambda item: int(item[0].split("x", 1)[0])
    ):
        side = int(tag.split("x", 1)[0])
        path = args.dataset_dir / details["file"]
        with np.load(path) as data:
            assert data["atoms"].shape == (20_000, side * side, 4)
            assert data["geom"].shape == (20_000, side * side * 4)
            assert data["T"].shape == (20_000, expected_freq)
            assert data["freq_GHz"].shape == (expected_freq,)
            assert np.isfinite(data["geom"]).all()
            assert np.isfinite(data["T"]).all()
            assert np.all((data["T"] >= 0.0) & (data["T"] <= 1.0))
            assert np.isclose(data["freq_GHz"][0], expected_band[0])
            assert np.isclose(data["freq_GHz"][-1], expected_band[1])

            size_mib = path.stat().st_size / 2**20
            total_bytes += path.stat().st_size
            total_samples += len(data["T"])
            print(
                f"{tag}: geom={data['geom'].shape}, T={data['T'].shape}, "
                f"range=[{data['T'].min():.6f}, {data['T'].max():.6f}], "
                f"{size_mib:.1f} MiB"
            )

    print(f"PASS: {total_samples} samples, {total_bytes / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
