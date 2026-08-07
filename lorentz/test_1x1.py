"""Dataset-backed forward/backward smoke test for the 1x1 Lorentz model."""

from pathlib import Path

import numpy as np
import torch

from .lorentz import LorentzPhysics, Model
from .losses import beta2_loss


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "power_tx_data" / "version_3" / "preprocessed_1x1.npz"


def main():
    with np.load(DATASET, allow_pickle=True) as data:
        geometry = np.asarray(data["atoms"][:8], dtype=np.float32)
        target = np.asarray(data["T"][:8], dtype=np.float32)
        freq = np.asarray(data["freq_GHz"], dtype=np.float32)

    lower = geometry.min(axis=(0, 1), keepdims=True)
    upper = geometry.max(axis=(0, 1), keepdims=True)
    geometry = 2.0 * (geometry - lower) / np.maximum(upper - lower, 1e-8) - 1.0

    model = Model(
        n_geom=4,
        n_e=1,
        n_m=1,
        hidden=512,
        depth=2,
        activation="silu",
        freq_GHz=freq,
        thickness_mm=0.2,
    )
    x = torch.from_numpy(geometry.astype(np.float32))
    y = torch.from_numpy(target)

    reflection, transmission = model(x)
    prediction = transmission.abs().square()
    loss = beta2_loss(prediction, y)
    loss.backward()

    assert reflection.shape == (8, 2001)
    assert transmission.shape == (8, 2001)
    assert reflection.is_complex() and transmission.is_complex()
    assert torch.isfinite(reflection).all()
    assert torch.isfinite(transmission).all()
    assert torch.isfinite(loss)
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)

    raw_oscillator = torch.zeros(1, 1, 1, 3)
    wp_values = []
    for wp_scale, wp_floor in ((0.5, 1e-5), (0.5, 0.0), (1.0, 0.0)):
        physics = LorentzPhysics(
            freq,
            thickness_mm=0.2,
            n_e=1,
            n_m=1,
            wp_scale=wp_scale,
            wp_floor=wp_floor,
        )
        wp_values.append(physics._oscillator_parameters(raw_oscillator)[0])
    expected_wp = torch.tensor(
        [0.5 * np.log(2.0) + 1e-5, 0.5 * np.log(2.0), np.log(2.0)],
        dtype=torch.float32,
    )
    assert torch.allclose(torch.stack(wp_values).flatten(), expected_wp)

    print(f"geometry: {tuple(x.shape)}")
    print(f"S11/S21: {tuple(reflection.shape)} complex")
    print(f"T prediction: {tuple(prediction.shape)}")
    print(f"untrained beta2 loss: {loss.item():.6f}")
    print("wp baseline/no-floor/softplus-only mappings passed")
    print("forward/backward test passed")


if __name__ == "__main__":
    main()
