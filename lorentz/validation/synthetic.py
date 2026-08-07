"""Self-consistent synthetic teacher for unary Lorentz recovery."""

from __future__ import annotations

import math

import torch


def teacher_raw_parameters(normalized_geometry):
    """Map ``(B,1,4)`` geometry to known raw ``(B,1,8)`` parameters.

    This smooth fixed function is deliberately nonlinear but modest in range.
    It is not trained and therefore provides exact intermediate labels for the
    unary recovery experiment.
    """
    if normalized_geometry.ndim != 3 or normalized_geometry.shape[1:] != (1, 4):
        raise ValueError(
            "Expected normalized geometry shaped (B,1,4), got "
            f"{tuple(normalized_geometry.shape)}."
        )
    d, length, width, gap = normalized_geometry.unbind(-1)

    wp_e = -0.45 + 0.65 * d - 0.30 * gap + 0.20 * torch.sin(
        math.pi * length * width
    )
    w0_e = 1.20 * length - 0.45 * width + 0.25 * d * gap
    gamma_e = -2.10 + 0.35 * gap + 0.18 * width.square()

    wp_m = -0.75 + 0.45 * width + 0.30 * d - 0.20 * length * gap
    w0_m = -1.05 * length + 0.55 * gap + 0.20 * d * width
    gamma_m = -1.85 - 0.25 * gap + 0.20 * length.square()

    epsilon_inf = -0.65 + 0.30 * d + 0.20 * width - 0.12 * length * gap
    mu_inf = -0.85 + 0.25 * length - 0.18 * gap + 0.10 * d * width

    return torch.stack(
        (
            wp_e,
            w0_e,
            gamma_e,
            wp_m,
            w0_m,
            gamma_m,
            epsilon_inf,
            mu_inf,
        ),
        dim=-1,
    )
