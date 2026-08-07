"""Plot how mapping-constant ablations change Lorentz decoder outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .lorentz import LorentzPhysics
from .train_1x1 import DEFAULT_DATASET


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "artifacts" / "mapping_effects"
COLORS = {
    "baseline": "#27647b",
    "no_scale": "#d28b26",
    "no_floor": "#8c6d31",
    "softplus_only": "#4c78a8",
    "no_epsilon_offset": "#319e8f",
    "no_mu_offset": "#c94f46",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--thickness-mm", type=float, default=0.2)
    return parser.parse_args()


def softplus(value):
    return np.logaddexp(0.0, value)


def decoder_response(freq, thickness_mm, *, raw_wp, raw_gamma, **constants):
    raw_epsilon = constants.pop("raw_epsilon", -6.0)
    raw_mu = constants.pop("raw_mu", -6.0)
    physics = LorentzPhysics(
        freq,
        thickness_mm=thickness_mm,
        n_e=1,
        n_m=0,
        **constants,
    )
    theta = torch.tensor(
        [[[raw_wp, 0.0, raw_gamma, raw_epsilon, raw_mu]]],
        dtype=torch.float32,
    )
    electric = theta[..., :3].reshape(1, 1, 1, 3)
    wp, w0, gamma = physics._oscillator_parameters(electric)
    epsilon_inf = physics._background(
        theta[..., -2], True, physics.epsilon_inf_offset
    )
    mu_inf = physics._background(theta[..., -1], True, physics.mu_inf_offset)
    susceptibility = physics._susceptibility(electric)
    reflection, transmission = physics(theta)
    return {
        "wp": float(wp.item()),
        "w0": float(w0.item()),
        "gamma": float(gamma.item()),
        "epsilon_inf": float(epsilon_inf.item()),
        "mu_inf": float(mu_inf.item()),
        "chi": susceptibility.squeeze(0).detach().numpy(),
        "T": transmission.abs().square().squeeze(0).detach().numpy(),
        "R_plus_T": (
            reflection.abs().square() + transmission.abs().square()
        ).squeeze(0).detach().numpy(),
    }


def plot_mappings(output):
    raw = np.linspace(-15.0, 5.0, 800)
    sp = softplus(raw)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    wp_curves = {
        "Reference: 0.5 softplus + 1e-5": (0.5 * sp + 1e-5, "baseline", "-"),
        "No 0.5 scale: softplus + 1e-5": (sp + 1e-5, "no_scale", "--"),
        "No floor: 0.5 softplus": (0.5 * sp, "no_floor", ":"),
        "Plain softplus": (sp, "softplus_only", "-."),
    }
    for label, (value, color, style) in wp_curves.items():
        axes[0].plot(raw, value, color=COLORS[color], linestyle=style, label=label)
    axes[0].axvline(0.0, color="#333333", linewidth=1, alpha=0.6)
    axes[0].set_yscale("log")
    axes[0].set_title("Plasma-frequency mapping")
    axes[0].set_ylabel(r"Mapped $\omega_p$")

    gamma_curves = {
        "Reference: 0.1 softplus + 1e-4": (0.1 * sp + 1e-4, "baseline", "-"),
        "No 0.1 scale: softplus + 1e-4": (sp + 1e-4, "no_scale", "--"),
        "No floor: 0.1 softplus": (0.1 * sp, "no_floor", ":"),
    }
    for label, (value, color, style) in gamma_curves.items():
        axes[1].plot(raw, value, color=COLORS[color], linestyle=style, label=label)
    axes[1].axvline(-8.0, color="#333333", linewidth=1, alpha=0.6)
    axes[1].set_yscale("log")
    axes[1].set_title("Damping mapping")
    axes[1].set_ylabel(r"Mapped $\gamma$")

    axes[2].plot(
        raw,
        1.0 + sp,
        color=COLORS["baseline"],
        label="Reference: 1 + softplus",
    )
    axes[2].plot(
        raw,
        sp,
        color=COLORS["no_epsilon_offset"],
        linestyle="--",
        label="No +1: softplus",
    )
    axes[2].axvline(0.0, color="#333333", linewidth=1, alpha=0.6)
    axes[2].axhline(1.0, color="#333333", linewidth=1, linestyle=":")
    axes[2].set_title("Background mapping")
    axes[2].set_ylabel(r"Mapped $\epsilon_\infty$ or $\mu_\infty$")

    for axis in axes:
        axis.set_xlabel("Raw neural-network output x")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("From raw F1 outputs to Lorentz parameters")
    fig.tight_layout()
    path = output / "01_parameter_mapping_curves.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_outputs(freq, thickness_mm, output):
    common = {
        "gamma_scale": 0.1,
        "gamma_floor": 1e-4,
        "epsilon_inf_offset": 1.0,
        "mu_inf_offset": 1.0,
    }
    wp_profiles = {
        "Reference": decoder_response(
            freq, thickness_mm, raw_wp=0.0, raw_gamma=-2.0,
            wp_scale=0.5, wp_floor=1e-5, **common
        ),
        "No wp floor": decoder_response(
            freq, thickness_mm, raw_wp=0.0, raw_gamma=-2.0,
            wp_scale=0.5, wp_floor=0.0, **common
        ),
        "No 0.5 scale": decoder_response(
            freq, thickness_mm, raw_wp=0.0, raw_gamma=-2.0,
            wp_scale=1.0, wp_floor=1e-5, **common
        ),
        "wp = softplus(x)": decoder_response(
            freq, thickness_mm, raw_wp=0.0, raw_gamma=-2.0,
            wp_scale=1.0, wp_floor=0.0, **common
        ),
    }
    gamma_profiles = {
        "Reference": decoder_response(
            freq, thickness_mm, raw_wp=0.0, raw_gamma=-8.0,
            wp_scale=0.5, wp_floor=1e-5, **common
        ),
        "No gamma floor": decoder_response(
            freq, thickness_mm, raw_wp=0.0, raw_gamma=-8.0,
            wp_scale=0.5, wp_floor=1e-5, gamma_scale=0.1,
            gamma_floor=0.0, epsilon_inf_offset=1.0, mu_inf_offset=1.0
        ),
        "No 0.1 scale": decoder_response(
            freq, thickness_mm, raw_wp=0.0, raw_gamma=-8.0,
            wp_scale=0.5, wp_floor=1e-5, gamma_scale=1.0,
            gamma_floor=1e-4, epsilon_inf_offset=1.0, mu_inf_offset=1.0
        ),
    }
    background_profiles = {
        "Reference": decoder_response(
            freq, thickness_mm, raw_wp=-20.0, raw_gamma=-2.0,
            wp_scale=0.5, wp_floor=1e-5, raw_epsilon=0.0, raw_mu=0.0,
            **common
        ),
        "No epsilon +1": decoder_response(
            freq, thickness_mm, raw_wp=-20.0, raw_gamma=-2.0,
            wp_scale=0.5, wp_floor=1e-5, raw_epsilon=0.0, raw_mu=0.0,
            gamma_scale=0.1, gamma_floor=1e-4,
            epsilon_inf_offset=0.0, mu_inf_offset=1.0
        ),
        "No mu +1": decoder_response(
            freq, thickness_mm, raw_wp=-20.0, raw_gamma=-2.0,
            wp_scale=0.5, wp_floor=1e-5, raw_epsilon=0.0, raw_mu=0.0,
            gamma_scale=0.1, gamma_floor=1e-4,
            epsilon_inf_offset=1.0, mu_inf_offset=0.0
        ),
    }

    color_order = ["baseline", "no_floor", "no_scale", "softplus_only"]
    fig, axes = plt.subplots(3, 2, figsize=(15, 13))
    for index, (label, response) in enumerate(wp_profiles.items()):
        color = COLORS[color_order[index]]
        axes[0, 0].plot(freq, np.abs(response["chi"]), color=color, label=label)
        axes[0, 1].plot(freq, response["T"], color=color, label=label)
    axes[0, 0].set_title(r"Wp ablations at raw x = 0: susceptibility $|\chi_e|$")
    axes[0, 1].set_title(r"Wp ablations at raw x = 0: $T=|S_{21}|^2$")

    gamma_colors = ("baseline", "no_floor", "no_scale")
    for color_name, (label, response) in zip(gamma_colors, gamma_profiles.items()):
        color = COLORS[color_name]
        axes[1, 0].plot(freq, np.abs(response["chi"]), color=color, label=label)
        axes[1, 1].plot(freq, response["T"], color=color, label=label)
    axes[1, 0].set_title(r"Gamma ablations at raw x = -8: susceptibility $|\chi_e|$")
    axes[1, 1].set_title(r"Gamma ablations at raw x = -8: $T=|S_{21}|^2$")
    axes[1, 0].set_xlim(18.7, 19.3)
    axes[1, 1].set_xlim(18.7, 19.3)

    background_colors = ("baseline", "no_epsilon_offset", "no_mu_offset")
    x = np.arange(len(background_profiles))
    width = 0.34
    eps_values = [response["epsilon_inf"] for response in background_profiles.values()]
    mu_values = [response["mu_inf"] for response in background_profiles.values()]
    axes[2, 0].bar(x - width / 2, eps_values, width, color="#319e8f", label=r"$\epsilon_\infty$")
    axes[2, 0].bar(x + width / 2, mu_values, width, color="#c94f46", label=r"$\mu_\infty$")
    axes[2, 0].set_xticks(x, background_profiles, rotation=18, ha="right")
    axes[2, 0].set_title("Background values at raw x = 0")
    for color_name, (label, response) in zip(
        background_colors, background_profiles.items()
    ):
        axes[2, 1].plot(freq, response["T"], color=COLORS[color_name], label=label)
    axes[2, 1].set_title(r"Background ablations: $T=|S_{21}|^2$")

    for row in range(3):
        axes[row, 0].set_ylabel("Magnitude" if row < 2 else "Background value")
        axes[row, 1].set_ylabel("Power transmittance T")
        axes[row, 0].grid(alpha=0.25)
        axes[row, 1].grid(alpha=0.25)
        axes[row, 0].legend(fontsize=8)
        axes[row, 1].legend(fontsize=8)
    axes[0, 0].set_yscale("log")
    axes[1, 0].set_yscale("log")
    axes[2, 0].set_ylim(bottom=0.0)
    for row, column in ((0, 0), (0, 1), (1, 0), (1, 1), (2, 1)):
        axes[row, column].set_xlabel("Frequency (GHz)")
    fig.suptitle(
        "Effect of mapping constants on one Lorentz oscillator and finite-slab output"
    )
    fig.tight_layout()
    path = output / "02_lorentz_output_curves.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path, wp_profiles, gamma_profiles, background_profiles


def write_values(path, groups):
    fields = (
        "group", "profile", "wp", "w0", "gamma", "epsilon_inf", "mu_inf",
        "T_min", "T_at_resonance", "T_max", "max_R_plus_T"
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for group, profiles in groups.items():
            for profile, response in profiles.items():
                midpoint = len(response["T"]) // 2
                writer.writerow(
                    {
                        "group": group,
                        "profile": profile,
                        "wp": response["wp"],
                        "w0": response["w0"],
                        "gamma": response["gamma"],
                        "epsilon_inf": response["epsilon_inf"],
                        "mu_inf": response["mu_inf"],
                        "T_min": float(response["T"].min()),
                        "T_at_resonance": float(response["T"][midpoint]),
                        "T_max": float(response["T"].max()),
                        "max_R_plus_T": float(response["R_plus_T"].max()),
                    }
                )
    return path


def main():
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with np.load(args.dataset, allow_pickle=True) as data:
        freq = np.asarray(data["freq_GHz"], dtype=np.float32)
    mapping_path = plot_mappings(output)
    output_path, wp_profiles, gamma_profiles, background_profiles = plot_outputs(
        freq, args.thickness_mm, output
    )
    values_path = write_values(
        output / "selected_values.csv",
        {
            "wp": wp_profiles,
            "gamma": gamma_profiles,
            "background": background_profiles,
        },
    )
    print(mapping_path)
    print(output_path)
    print(values_path)


if __name__ == "__main__":
    main()
