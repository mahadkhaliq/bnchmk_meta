"""Geometry-to-Lorentz-parameter network."""

import torch.nn as nn


class F1(nn.Module):
    """Map each cell geometry to its isolated-cell Lorentz parameters.

    The raw output layout for every cell is

        [wp, w0, gamma] * n_e
        [wp, w0, gamma] * n_m
        [eps_inf, mu_inf]

    ``nn.Linear`` acts on the final axis, so one shared network is applied to
    every cell independently.
    """

    def __init__(self, n_geom, n_e, n_m, hidden, depth, activation="silu"):
        super().__init__()
        if n_geom < 1:
            raise ValueError("n_geom must be positive.")
        if n_e < 0 or n_m < 0 or n_e + n_m < 1:
            raise ValueError("At least one electric or magnetic oscillator is required.")
        if hidden < 1 or depth < 0:
            raise ValueError("hidden must be positive and depth cannot be negative.")
        activations = {"relu": nn.ReLU, "silu": nn.SiLU}
        if activation not in activations:
            raise ValueError(f"activation must be one of {sorted(activations)}, got {activation!r}.")
        activation_cls = activations[activation]

        self.n_e = n_e
        self.n_m = n_m
        self.n_osc = 3 * (n_e + n_m)
        self.theta_dim = self.n_osc + 2

        layers = []
        width = n_geom
        for _ in range(depth):
            layers.extend((nn.Linear(width, hidden), activation_cls()))
            width = hidden
        layers.append(nn.Linear(width, self.theta_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Return raw parameters for ``x`` shaped ``(B, M, n_geom)``."""
        return self.net(x)
