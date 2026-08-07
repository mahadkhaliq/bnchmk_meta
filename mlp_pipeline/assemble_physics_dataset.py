#!/usr/bin/env python3
"""Assemble independently generated physics-data scales into one dataset release."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def scale_key(path: Path) -> int:
    return int(path.name.split("x", 1)[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    scale_dirs = sorted(
        (path for path in args.staging.iterdir() if path.is_dir() and "x" in path.name),
        key=scale_key,
    )
    if not scale_dirs:
        raise SystemExit(f"No scale directories found in {args.staging}")

    args.output.mkdir(parents=True, exist_ok=True)
    metadata_dir = args.output / "metadata" / "per_scale_manifests"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    common = None
    scales: dict[str, dict] = {}
    for scale_dir in scale_dirs:
        manifest_path = scale_dir / "MANIFEST.json"
        if not manifest_path.is_file():
            raise SystemExit(f"Missing manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if len(manifest.get("sizes", {})) != 1:
            raise SystemExit(f"Expected one scale in {manifest_path}")

        tag, details = next(iter(manifest["sizes"].items()))
        source = scale_dir / details["file"]
        destination = args.output / source.name
        if not source.is_file():
            raise SystemExit(f"Missing generated dataset: {source}")
        if destination.exists():
            raise SystemExit(f"Refusing to overwrite existing dataset: {destination}")

        os.replace(source, destination)
        shutil.copy2(manifest_path, metadata_dir / f"{tag}.json")
        scales[tag] = {
            **details,
            "file": destination.name,
            "source_manifest": f"metadata/per_scale_manifests/{tag}.json",
        }

        shared = {
            key: manifest.get(key)
            for key in (
                "common_band_GHz",
                "n_freq",
                "f0_dist",
                "gamma_floor",
                "transmission",
                "model_file",
                "model_sha256",
                "model",
                "na_operator",
            )
        }
        if common is None:
            common = shared
        elif common != shared:
            raise SystemExit(f"Generation settings differ for {tag}")

    assembled = {
        "dataset_name": args.name,
        "schema": "coupled_lorentz_synthetic_v1",
        "feature_order": ["d", "l", "w", "g"],
        "geometry_layout": "atom-major, row-major flattened geom",
        "target": "power transmission T = |S21|^2",
        "scales": scales,
        **(common or {}),
    }
    (args.output / "MANIFEST.json").write_text(json.dumps(assembled, indent=2) + "\n")
    print(f"assembled {len(scales)} scales in {args.output}")


if __name__ == "__main__":
    main()
