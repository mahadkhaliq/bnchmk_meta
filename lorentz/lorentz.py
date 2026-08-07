"""Differentiable finite-slab Lorentz decoder and geometry model.

The physics follows the shared slide equations:

    chi = wp^2 / (w0^2 - w^2 - i gamma w)
    eps_r = eps_inf + sum chi_e
    mu_r  = mu_inf  + sum chi_m
    n = sqrt(eps_r mu_r),  Z = sqrt(mu_r / eps_r)

    t = 1 / [cos(n k0 d) - i/2 (1/Z + Z) sin(n k0 d)]
    r = [i/2 (1/Z - Z) sin(n k0 d)] / same denominator

The MLP predicts dimensionless raw values. They are mapped to positive,
frequency-normalized oscillator parameters before entering the equations.
This keeps the equations unchanged while avoiding neural outputs of order
1e11 rad/s.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .f1 import F1


C_MM_PER_S = 2.99792458e11


class LorentzPhysics(nn.Module):
    """Convert per-cell Lorentz parameters to complex reflection/transmission."""

    def __init__(
        self,
        freq_GHz,
        thickness_mm,
        n_e,
        n_m,
        *,
        parameterize=True,
        wp_scale=0.5,
        wp_floor=1e-5,
        gamma_scale=0.1,
        gamma_floor=1e-4,
        epsilon_inf_offset=1.0,
        mu_inf_offset=1.0,
        constrain_wp=None,
        constrain_w0=None,
        constrain_gamma=None,
        constrain_epsilon_inf=None,
        constrain_mu_inf=None,
    ):
        super().__init__()
        if thickness_mm <= 0:
            raise ValueError("thickness_mm must be positive.")
        if n_e < 0 or n_m < 0 or n_e + n_m < 1:
            raise ValueError("At least one oscillator is required.")
        if wp_floor < 0 or gamma_floor < 0:
            raise ValueError("Oscillator floors must be non-negative.")
        if epsilon_inf_offset < 0 or mu_inf_offset < 0:
            raise ValueError("Background offsets must be non-negative.")

        freq = torch.as_tensor(freq_GHz, dtype=torch.float32)
        if freq.ndim != 1 or freq.numel() < 2:
            raise ValueError("freq_GHz must be a one-dimensional frequency grid.")
        if not torch.all(freq[1:] > freq[:-1]):
            raise ValueError("freq_GHz must be strictly increasing.")

        self.n_e = n_e
        self.n_m = n_m
        self.parameterize = parameterize
        self.wp_scale = float(wp_scale)
        self.wp_floor = float(wp_floor)
        self.gamma_scale = float(gamma_scale)
        self.gamma_floor = float(gamma_floor)
        self.epsilon_inf_offset = float(epsilon_inf_offset)
        self.mu_inf_offset = float(mu_inf_offset)
        self.constrain_wp = (
            bool(parameterize) if constrain_wp is None else bool(constrain_wp)
        )
        self.constrain_w0 = (
            bool(parameterize) if constrain_w0 is None else bool(constrain_w0)
        )
        self.constrain_gamma = (
            bool(parameterize)
            if constrain_gamma is None
            else bool(constrain_gamma)
        )
        self.constrain_epsilon_inf = (
            bool(parameterize)
            if constrain_epsilon_inf is None
            else bool(constrain_epsilon_inf)
        )
        self.constrain_mu_inf = (
            bool(parameterize)
            if constrain_mu_inf is None
            else bool(constrain_mu_inf)
        )

        f_hz = freq * 1e9
        omega = 2 * math.pi * f_hz
        omega_ref = omega.mean()

        # Lorentz ratios are unchanged when every frequency-like quantity is
        # divided by the same reference angular frequency.
        self.register_buffer("w", omega / omega_ref)
        self.register_buffer("k0d", omega * float(thickness_mm) / C_MM_PER_S)

        w_min = max(0.05, float(self.w.min()) - 0.15)
        w_max = float(self.w.max()) + 0.15
        self.register_buffer("w0_min", torch.tensor(w_min, dtype=torch.float32))
        self.register_buffer("w0_max", torch.tensor(w_max, dtype=torch.float32))

    def _oscillator_parameters(self, raw):
        raw_wp, raw_w0, raw_gamma = raw.unbind(-1)
        if self.constrain_wp:
            wp = self.wp_scale * F.softplus(raw_wp) + self.wp_floor
        else:
            wp = raw_wp
        if self.constrain_w0:
            w0 = self.w0_min + (self.w0_max - self.w0_min) * torch.sigmoid(
                raw_w0
            )
        else:
            w0 = raw_w0
        if self.constrain_gamma:
            gamma = (
                self.gamma_scale * F.softplus(raw_gamma) + self.gamma_floor
            )
        else:
            gamma = raw_gamma
        return wp, w0, gamma

    def _susceptibility(self, raw):
        """Sum oscillators, then average cells, returning ``(B, F)``."""
        if raw.shape[2] == 0:
            shape = (raw.shape[0], self.w.numel())
            return torch.zeros(shape, dtype=torch.complex64, device=raw.device)

        wp, w0, gamma = self._oscillator_parameters(raw)
        w = self.w
        numerator = wp.square().unsqueeze(-1)
        denominator = (
            w0.square().unsqueeze(-1)
            - w.square()
            - 1j * gamma.unsqueeze(-1) * w
        )
        chi = numerator / denominator
        return chi.sum(dim=2).mean(dim=1)

    @staticmethod
    def _background(raw, constrained, offset=1.0):
        if constrained:
            raw = float(offset) + F.softplus(raw)
        return raw.mean(dim=1, keepdim=True)

    @staticmethod
    def _passive_sqrt(value, *, impedance=False):
        root = torch.sqrt(value)
        if impedance:
            return torch.where(root.real < 0, -root, root)
        return torch.where(root.imag < 0, -root, root)

    def _split_theta(self, theta):
        expected = 3 * (self.n_e + self.n_m) + 2
        if theta.ndim != 3 or theta.shape[-1] != expected:
            raise ValueError(
                f"Expected theta shape (B,M,{expected}), got {tuple(theta.shape)}."
            )

        batch, cells, _ = theta.shape
        n_ep = 3 * self.n_e
        n_mp = 3 * self.n_m
        electric = theta[..., :n_ep].reshape(batch, cells, self.n_e, 3)
        magnetic = theta[..., n_ep : n_ep + n_mp].reshape(
            batch, cells, self.n_m, 3
        )
        return electric, magnetic

    def physical_parameters(self, theta):
        """Convert raw ``theta`` to constrained per-cell physical parameters.

        The returned tensor has the same final-axis layout as ``theta``. Unlike
        the slab forward pass, background values are not averaged across cells;
        this method exposes every cell's intermediate parameter vector ``P_i``.
        """
        electric, magnetic = self._split_theta(theta)
        pieces = []
        if self.n_e:
            pieces.append(
                torch.stack(self._oscillator_parameters(electric), dim=-1).flatten(2)
            )
        if self.n_m:
            pieces.append(
                torch.stack(self._oscillator_parameters(magnetic), dim=-1).flatten(2)
            )

        epsilon_inf = theta[..., -2]
        if self.constrain_epsilon_inf:
            epsilon_inf = self.epsilon_inf_offset + F.softplus(epsilon_inf)
        mu_inf = theta[..., -1]
        if self.constrain_mu_inf:
            mu_inf = self.mu_inf_offset + F.softplus(mu_inf)
        pieces.extend((epsilon_inf.unsqueeze(-1), mu_inf.unsqueeze(-1)))
        return torch.cat(pieces, dim=-1)

    def forward(self, theta):
        """Map ``theta: (B,M,3*(n_e+n_m)+2)`` to complex ``(r,t)``."""
        electric, magnetic = self._split_theta(theta)

        eps = self._background(
            theta[..., -2],
            self.constrain_epsilon_inf,
            self.epsilon_inf_offset,
        ) + self._susceptibility(electric)
        mu = self._background(
            theta[..., -1],
            self.constrain_mu_inf,
            self.mu_inf_offset,
        ) + self._susceptibility(magnetic)

        refractive_index = self._passive_sqrt(eps * mu)
        impedance = self._passive_sqrt(mu / eps, impedance=True)

        phase = refractive_index * self.k0d
        sin_phase = torch.sin(phase)
        cos_phase = torch.cos(phase)
        denominator = cos_phase - 0.5j * (
            impedance.reciprocal() + impedance
        ) * sin_phase

        reflection = (
            0.5j * (impedance.reciprocal() - impedance) * sin_phase
        ) / denominator
        transmission = denominator.reciprocal()
        return reflection, transmission


class Model(nn.Module):
    """Shared F1 network followed by the fixed finite-slab decoder."""

    def __init__(
        self,
        n_geom,
        n_e,
        n_m,
        hidden,
        depth,
        freq_GHz,
        thickness_mm,
        activation="silu",
        **physics_kwargs,
    ):
        super().__init__()
        self.f1 = F1(n_geom, n_e, n_m, hidden, depth, activation=activation)
        self.physics = LorentzPhysics(
            freq_GHz,
            thickness_mm,
            n_e,
            n_m,
            **physics_kwargs,
        )

    def forward(self, x):
        """Return complex ``(S11, S21)`` for ``x: (B,M,n_geom)``."""
        return self.physics(self.f1(x))

    def power_transmittance(self, x):
        """Return ``T = |S21|^2``."""
        _, transmission = self(x)
        return transmission.abs().square()
