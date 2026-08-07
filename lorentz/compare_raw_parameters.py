"""Compare constrained and raw-parameter 1x1 Lorentz experiments."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .lorentz import Model
from .train_1x1 import DEFAULT_DATASET, load_data


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
OUTPUT_DIR = ARTIFACTS / "raw_parameters_100ep"
RUNS = (
    (
        "constrained",
        ARTIFACTS / "scale_ablation_100ep" / "wp0p5_gamma0p1.pt",
        "completed",
    ),
    (
        "raw, lr=3e-4",
        OUTPUT_DIR / "raw_1e1m.pt",
        "failed with a non-finite loss between epochs 51 and 59",
    ),
    (
        "raw, lr=3e-5",
        OUTPUT_DIR / "raw_1e1m_lr3e-5.pt",
        "completed",
    ),
)


def statistics(value):
    value = value.detach()
    return {
        "min": float(value.min()),
        "mean": float(value.mean()),
        "max": float(value.max()),
        "negative_fraction": float((value < 0).float().mean()),
    }


@torch.no_grad()
def evaluate_run(label, checkpoint_path, status, freq, split):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = Model(freq_GHz=freq, **checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    geometry = torch.from_numpy(split[0])
    target = torch.from_numpy(split[1])
    theta = model.f1(geometry)
    reflection, transmission = model(geometry)
    prediction = transmission.abs().square()
    error = prediction - target
    weight = 1.0 + 2.0 * (1.0 - target).clamp(min=0.0).square()
    energy = reflection.abs().square() + prediction

    electric = theta[..., :3].reshape(len(theta), 1, 1, 3)
    magnetic = theta[..., 3:6].reshape(len(theta), 1, 1, 3)
    wp_e, w0_e, gamma_e = model.physics._oscillator_parameters(electric)
    wp_m, w0_m, gamma_m = model.physics._oscillator_parameters(magnetic)
    eps_inf = model.physics._background(
        theta[..., -2],
        model.physics.constrain_epsilon_inf,
        model.physics.epsilon_inf_offset,
    )
    mu_inf = model.physics._background(
        theta[..., -1],
        model.physics.constrain_mu_inf,
        model.physics.mu_inf_offset,
    )

    return {
        "label": label,
        "status": status,
        "checkpoint": str(checkpoint_path),
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_val_mse": float(checkpoint["best_val_mse"]),
        "parameterize": bool(
            checkpoint["model_config"].get("parameterize", True)
        ),
        "test": {
            "mse": float(error.square().mean()),
            "beta2": float((weight * error.square()).mean()),
            "mae": float(error.abs().mean()),
            "prediction_min": float(prediction.min()),
            "prediction_max": float(prediction.max()),
            "max_R_plus_T": float(energy.max()),
        },
        "parameters": {
            "wp_e": statistics(wp_e),
            "w0_e": statistics(w0_e),
            "gamma_e": statistics(gamma_e),
            "wp_m": statistics(wp_m),
            "w0_m": statistics(w0_m),
            "gamma_m": statistics(gamma_m),
            "epsilon_inf": statistics(eps_inf),
            "mu_inf": statistics(mu_inf),
        },
        "prediction": prediction.numpy(),
        "target": target.numpy(),
    }


def main():
    first_checkpoint = torch.load(RUNS[0][1], map_location="cpu", weights_only=False)
    freq, _, splits, _ = load_data(DEFAULT_DATASET, first_checkpoint["seed"])
    runs = [evaluate_run(*spec, freq, splits["test"]) for spec in RUNS]

    serializable = []
    for run in runs:
        serializable.append(
            {
                key: value
                for key, value in run.items()
                if key not in {"prediction", "target"}
            }
        )
    summary = {
        "experiment": "constrained versus raw Lorentz parameters",
        "raw_definition": (
            "F1 outputs directly define wp, w0, gamma, epsilon_inf, and mu_inf. "
            "Frequency normalization and passive square-root branch selection remain."
        ),
        "runs": serializable,
        "conclusion": (
            "Raw training either became non-finite or learned negative damping/"
            "backgrounds and power above one. The parameter mappings are materially "
            "important for stability and passivity."
        ),
    }
    summary_path = OUTPUT_DIR / "raw_parameter_comparison.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    target_mean = runs[0]["target"].mean(axis=0)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    axes[0].plot(freq, target_mean, color="#cc4455", linewidth=2.0, label="CST")
    for run, color in zip(runs, ("#4477aa", "#ddaa33", "#228833")):
        axes[0].plot(
            freq,
            run["prediction"].mean(axis=0),
            color=color,
            linewidth=1.35,
            label=run["label"],
        )
    axes[0].set_xlabel("Frequency (GHz)")
    axes[0].set_ylabel("Mean power transmittance")
    axes[0].set_title("Mean held-out spectrum")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    labels = [run["label"] for run in runs]
    mse = [run["test"]["mse"] for run in runs]
    axes[1].bar(labels, mse, color=("#4477aa", "#ddaa33", "#228833"))
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Test MSE")
    axes[1].set_title("Prediction error")
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].grid(axis="y", alpha=0.25)

    x = np.arange(len(runs))
    width = 0.36
    axes[2].bar(
        x - width / 2,
        [run["test"]["prediction_max"] for run in runs],
        width,
        label="Maximum T",
    )
    axes[2].bar(
        x + width / 2,
        [run["test"]["max_R_plus_T"] for run in runs],
        width,
        label="Maximum R+T",
    )
    axes[2].axhline(1.0, color="#333333", linestyle="--", linewidth=1.0)
    axes[2].set_xticks(x, labels, rotation=18, ha="right")
    axes[2].set_ylabel("Maximum on test set")
    axes[2].set_title("Passivity diagnostic")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend(fontsize=8)

    fig.suptitle("1x1 constrained versus raw Lorentz parameters")
    fig.tight_layout()
    figure_path = OUTPUT_DIR / "raw_parameter_comparison.png"
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"figure": str(figure_path), **summary}, indent=2))


if __name__ == "__main__":
    main()
