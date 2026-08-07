"""Plot a representative held-out 1x1 Lorentz prediction."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .lorentz import Model
from .train_1x1 import DEFAULT_CHECKPOINT, DEFAULT_DATASET, load_data


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "artifacts" / "test_1x1_spectrum.png"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    freq, feature_names, splits, normalization = load_data(
        args.dataset, checkpoint["seed"]
    )
    test_x, test_y = splits["test"]

    model_config = dict(checkpoint["model_config"])
    # Checkpoints created before the SiLU change used ReLU.
    model_config.setdefault("activation", "relu")
    model = Model(freq_GHz=freq, **model_config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        prediction = model.power_transmittance(torch.from_numpy(test_x)).numpy()

    sample_mse = np.mean((prediction - test_y) ** 2, axis=1)
    order = np.argsort(sample_mse)
    sample = int(order[len(order) // 2])

    x_min = np.asarray(normalization["min"], dtype=np.float32)
    x_max = np.asarray(normalization["max"], dtype=np.float32)
    geometry = 0.5 * (test_x[sample, 0] + 1.0) * (x_max - x_min) + x_min
    geometry_text = ", ".join(
        f"{name}={value:.4f}" for name, value in zip(feature_names, geometry)
    )

    target = test_y[sample]
    pred = prediction[sample]
    residual = pred - target

    fig, (ax, err_ax) = plt.subplots(
        2,
        1,
        figsize=(9, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True,
    )
    ax.plot(freq, target, color="#d1495b", linewidth=2.0, label="CST target")
    ax.plot(freq, pred, color="#2166ac", linewidth=1.6, label="Lorentz prediction")
    ax.set_ylabel(r"Power transmittance $T$")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    ax.set_title(
        f"V3 1x1 held-out median-error sample | MSE={sample_mse[sample]:.6f}\n"
        f"{geometry_text}"
    )

    err_ax.axhline(0.0, color="#333333", linewidth=0.8)
    err_ax.plot(freq, residual, color="#3a7d44", linewidth=1.2)
    err_ax.set_xlabel("Frequency (GHz)")
    err_ax.set_ylabel("Pred. - target")
    err_ax.grid(alpha=0.22)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
