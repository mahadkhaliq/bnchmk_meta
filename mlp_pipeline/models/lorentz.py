"""Full physics-informed model from the heterogeneous-metasurface slides.

This is the NON-relaxed version: the trainable decoder D of concept_one.py is
replaced by the fixed Lorentz slab physics f_r (Eq. 1-2 of the slides).

    Theta_i = U(g_i) + sum_j V(g_i, g_j, rel_ij)          (Eq. 4)
    eps_r, mu_r = Lorentz oscillator sums over ALL cells   (slides: 4*Ne terms)
    n, Z, t  = slab transfer-matrix formulas               (fixed, no params)
    y_hat    = |t|^2   (power transmission)

Frequency axis: pass freqs=<the npz freq_GHz array>; it is normalised by its
mean so the physics sees a dimensionless axis near 1, and the trainable
KD_SCALE absorbs c, the thickness d, and that normalisation constant.
Open items to confirm with Dr. Malof:
  * KD_SCALE stays trainable; fix it if the physical thickness d is known.
  * reduce="mean": slides write a raw sum over 4*Ne oscillators; mean is used
    so a 2x2 of identical cells reproduces the 1x1 (unitary) spectrum exactly,
    which a raw sum would not. Set reduce="sum" for the literal slide version.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mlp import MLP
from models.rel_encoding import RelEncoding


class LorentzPhysics(nn.Module):
    """Fixed f_r: raw per-cell Theta -> power-transmission spectrum.

    theta: (B, M, n_osc, 6) unconstrained reals; last dim =
           (wp_e, w0_e, g_e, wp_m, w0_m, g_m). Softplus enforces positivity.
    """

    def __init__(self, n_freq, freqs=None, w_min=0.5, w_max=1.5, kd_scale=3.14,
                 train_kd=True, reduce="mean"):
        super().__init__()
        assert reduce in ("mean", "sum")
        self.reduce = reduce
        if freqs is not None:
            w = torch.as_tensor(np.asarray(freqs), dtype=torch.float32)
            if w.numel() != n_freq:
                raise ValueError(f"freqs has {w.numel()} points, expected n_freq={n_freq}.")
            w = w / w.mean()   # dimensionless axis centred near 1; kd_scale absorbs c*d
        else:
            w = torch.linspace(w_min, w_max, n_freq)
        self.register_buffer("w", w)
        self.eps_inf_raw = nn.Parameter(torch.zeros(1))   # eps_inf = 1 + softplus
        self.mu_inf_raw = nn.Parameter(torch.zeros(1))
        kd = torch.tensor(float(kd_scale))
        if train_kd:
            self.kd_scale = nn.Parameter(kd)
        else:
            self.register_buffer("kd_scale", kd)

    def _chi(self, wp, w0, g):
        # wp/w0/g: (B, M, O) -> susceptibility (B, F)
        w = self.w                                        # (F,)
        num = (wp ** 2).unsqueeze(-1)                     # (B, M, O, 1)
        den = (w0 ** 2).unsqueeze(-1) - w ** 2 - 1j * g.unsqueeze(-1) * w
        chi = num / den                                   # (B, M, O, F) complex
        return chi.mean(dim=(1, 2)) if self.reduce == "mean" else chi.sum(dim=(1, 2))

    def forward(self, theta):
        p = F.softplus(theta) + 1e-4                      # positive, avoid 0
        wp_e, w0_e, g_e, wp_m, w0_m, g_m = p.unbind(-1)
        eps = 1.0 + F.softplus(self.eps_inf_raw) + self._chi(wp_e, w0_e, g_e)
        mu = 1.0 + F.softplus(self.mu_inf_raw) + self._chi(wp_m, w0_m, g_m)
        n = torch.sqrt(eps * mu)
        n = torch.where(n.imag < 0, -n, n)                # passive branch Im(n)>=0
        z = torch.sqrt(mu / eps)
        z = torch.where(z.real < 0, -z, z)                # Re(Z)>=0
        nkd = n * (self.kd_scale * self.w)                # n * k0 d, k0 prop. to w
        t = 1.0 / (torch.cos(nkd) - 0.5j * (1.0 / z + z) * torch.sin(nkd))
        return (t.abs() ** 2).clamp(max=1.0)              # (B, F)


class LorentzMetasurface(nn.Module):
    """U + V nets predicting Theta, decoded by the FIXED LorentzPhysics block."""

    def __init__(self, K=3, C=4, n_freq=2001, n_osc=2, hidden=256, n_hidden=4,
                 rel_encoding="offset", rel_emb_dim=8,
                 use_relative_position=None, **phys_kw):
        super().__init__()
        if K % 2 != 1:
            raise ValueError("K must be odd.")
        if use_relative_position is False:
            rel_encoding = "none"
        self.K, self.C, self.pad = K, C, K // 2
        self.n_osc = n_osc
        self.theta_dim = n_osc * 6
        self.rel = RelEncoding(rel_encoding, K, rel_emb_dim)
        pair_in = 2 * C + self.rel.extra_dim
        self.unitary = MLP([C] + [hidden] * n_hidden + [self.theta_dim])
        self.interaction = MLP([pair_in] + [hidden] * n_hidden + [self.theta_dim])
        self.physics = LorentzPhysics(n_freq, **phys_kw)

    def forward(self, grid):
        B, N, _, C = grid.shape
        cells = grid.reshape(B * N * N, C)
        theta = self.unitary(cells).reshape(B, N * N, self.theta_dim)

        if N > 1 and self.K > 1:
            src = (torch.arange(N + 2 * self.pad, device=grid.device) - self.pad) % N
            padded = grid[:, src[:, None], src[None, :], :]
            feats, targets = [], []
            for idx in range(N * N):
                i, j = idx // N, idx % N
                center = grid[:, i, j, :]
                for di in range(-self.pad, self.pad + 1):
                    for dj in range(-self.pad, self.pad + 1):
                        if di == 0 and dj == 0:
                            continue
                        neigh = padded[:, i + di + self.pad, j + dj + self.pad, :]
                        pieces = [center, neigh]
                        rel = self.rel(di, dj, B, grid.device, grid.dtype)
                        if rel is not None:
                            pieces.append(rel)
                        feats.append(torch.cat(pieces, dim=1))
                        targets.append(idx)
            delta = self.interaction(torch.cat(feats, dim=0))
            delta = delta.reshape(len(targets), B, self.theta_dim)
            for k, idx in enumerate(targets):
                theta[:, idx, :] = theta[:, idx, :] + delta[k]

        theta = theta.reshape(B, N * N, self.n_osc, 6)
        return self.physics(theta)
