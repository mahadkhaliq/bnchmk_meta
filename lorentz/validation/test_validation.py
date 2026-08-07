"""Focused shape, split, reconstruction, and gradient checks."""

import numpy as np
import torch

from ..lorentz import LorentzPhysics
from ..train_1x1 import load_data as load_legacy_data
from .common import DEFAULT_V3_DATASET, load_v3_1x1
from .synthetic import teacher_raw_parameters


def main():
    current = load_v3_1x1(DEFAULT_V3_DATASET, seed=7, max_samples=64)
    freq, names, legacy, normalization = load_legacy_data(
        DEFAULT_V3_DATASET, seed=7, max_samples=64
    )
    assert names == current["feature_names"]
    assert np.array_equal(freq, current["freq_GHz"])
    assert normalization == current["normalization"]
    for split in ("train", "val", "test"):
        legacy_x, legacy_t = legacy[split]
        assert np.array_equal(legacy_x, current["splits"][split]["x"])
        assert np.array_equal(legacy_t, current["splits"][split]["T"])

    physics = LorentzPhysics(
        current["freq_GHz"], thickness_mm=0.2, n_e=1, n_m=1
    )
    geometry = torch.linspace(-1.0, 1.0, 32).reshape(8, 1, 4)
    raw = teacher_raw_parameters(geometry).requires_grad_()
    physical = physics.physical_parameters(raw)
    reflection, transmission = physics(raw)
    transmittance = transmission.abs().square()
    loss = transmittance.mean() + physical.mean()
    loss.backward()

    assert raw.shape == (8, 1, 8)
    assert physical.shape == (8, 1, 8)
    assert reflection.shape == (8, 2001)
    assert transmission.shape == (8, 2001)
    assert torch.isfinite(physical).all()
    assert torch.isfinite(reflection).all()
    assert torch.isfinite(transmission).all()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()

    print("legacy/new split and train-only normalization match")
    print("geometry -> raw P -> physical P -> complex S -> T shapes pass")
    print("teacher/decoder backward pass is finite")


if __name__ == "__main__":
    main()
