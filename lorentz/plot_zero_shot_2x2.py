"""Plot a representative zero-shot 1x1-to-2x2 prediction."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from .evaluate_zero_shot_2x2 import DEFAULT_DATASET
from .lorentz import Model
from .train_1x1 import DEFAULT_CHECKPOINT


DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent / "artifacts" / "zero_shot_2x2_spectrum.png"
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

    model = Model(freq_GHz=freq, **checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(geometry), args.batch_size):
            x = torch.from_numpy(geometry[start : start + args.batch_size])
            predictions.append(model.power_transmittance(x).numpy())
    prediction = np.concatenate(predictions)

    sample_mse = np.mean((prediction - target) ** 2, axis=1)
    order = np.argsort(sample_mse)
    sample = int(order[len(order) // 2])
    pred = prediction[sample]
    truth = target[sample]

    cell_lines = []
    for index, values in enumerate(atoms[sample]):
        row, col = divmod(index, 2)
        features = ", ".join(
            f"{name}={value:.3f}" for name, value in zip(feature_names, values)
        )
        cell_lines.append(f"({row},{col}): {features}")

    fig, (ax, err_ax) = plt.subplots(
        2,
        1,
        figsize=(9, 6.6),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True,
    )
    ax.plot(freq, truth, color="#d1495b", linewidth=2.0, label="CST target")
    ax.plot(freq, pred, color="#2166ac", linewidth=1.6, label="1x1 zero-shot")
    ax.set_ylabel(r"Power transmittance $T$")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    ax.set_title(
        f"V3 2x2 zero-shot median-error sample | MSE={sample_mse[sample]:.5f}"
    )
    ax.text(
        0.015,
        0.035,
        "\n".join(cell_lines),
        transform=ax.transAxes,
        fontsize=8,
        family="monospace",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.88},
    )

    residual = pred - truth
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
