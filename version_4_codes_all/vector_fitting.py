"""Vector-fitting variant of Concept #1 (ConceptOneVF).

Same Eq. 4 trunk as concept_one.py (unitary U + interaction V, additive per
cell), but the decoder is replaced by a pole-residue rational in the Laplace
variable s = i*omega (classical vector-fitting form, Gustavsen-Semlyen):

    t(s) = sum_k [ r_k/(s - p_k) + conj(r_k)/(s - conj(p_k)) ]
         + sum_m q_m/(s - rho_m)  +  d  +  h*s,           T = |t|^2

Design points carried over from the VF notebook (vf_concept1_staged.ipynb):
  * vf_head is a SINGLE nn.Linear with no activation, so Eq. 4's additivity
    carries exactly into raw parameter space.
  * Head init: weights * 0.05 (nearly constant t at step 0); beta bias seeded
    linspace(0.2, 2.0) so resonances start pre-scattered across the normalised
    band; d-slot bias 1.0 (start as a transparent slab, T ~ 1); h-slot 0.
  * Stability: p = -(softplus(alpha) + DELTA) + i*beta guarantees Re p < 0,
    DELTA = 0.01 caps sharpness.
  * Union pooling with the unitary identity: residues (and d, h) are divided /
    averaged over the U = N^2 cells, so a 2x2 of identical cells reproduces
    the 1x1 spectrum exactly: N^2 * (r/N^2)/(s-p) = r/(s-p).
  * omega is the dataset freq axis normalised by its mean (freqs=...), or a
    linspace fallback.

Packed per-cell layout (this file's convention; if your notebook slices in a
different order the behaviour is identical, only the slot order differs):
    [alpha(n) | beta(n) | Re r(n) | Im r(n) | q(n_real) | rho_raw(n_real) | d | h]
    width = 4*n_pole + 2*n_real + 2
"""
import numpy as np
import torch
import torch.nn as nn

from models.mlp import MLP
from models.rel_encoding import RelEncoding

DELTA = 0.01


def _vf_head(d_in, n_pole, n_real=0):
    head = nn.Linear(d_in, 4 * n_pole + 2 * n_real + 2)
    with torch.no_grad():
        head.weight.mul_(0.05)
        head.bias.zero_()
        head.bias[n_pole:2 * n_pole] = torch.linspace(0.2, 2.0, n_pole)  # beta
        head.bias[-2] = 1.0                                              # d
    return head


class ConceptOneVF(nn.Module):
    def __init__(self, K=3, C=4, n_freq=2001, latent_dim=64, hidden=256,
                 n_hidden=4, n_pole=8, n_real=0, freqs=None,
                 w_min=0.5, w_max=1.5, rel_encoding="offset", rel_emb_dim=8):
        super().__init__()
        if K % 2 != 1:
            raise ValueError("K must be odd.")
        self.K, self.C, self.pad = K, C, K // 2
        self.latent_dim, self.n_pole, self.n_real = latent_dim, n_pole, n_real
        self.rel = RelEncoding(rel_encoding, K, rel_emb_dim)
        pair_in = 2 * C + self.rel.extra_dim
        self.unitary = MLP([C] + [hidden] * n_hidden + [latent_dim])
        self.interaction = MLP([pair_in] + [hidden] * n_hidden + [latent_dim])
        self.vf_head = _vf_head(latent_dim, n_pole, n_real)
        if freqs is not None:
            w = torch.as_tensor(np.asarray(freqs), dtype=torch.float32)
            if w.numel() != n_freq:
                raise ValueError(f"freqs has {w.numel()} points, expected {n_freq}.")
            w = w / w.mean()
        else:
            w = torch.linspace(w_min, w_max, n_freq)
        self.register_buffer("w", w)

    def _compose_vf(self, raw, U):
        """raw: (B, U, 4n+2m+2) packed params -> T: (B, F). Fixed math, no weights."""
        n, m = self.n_pole, self.n_real
        a  = raw[..., 0:n]
        b  = raw[..., n:2 * n]
        rr = raw[..., 2 * n:3 * n]
        ri = raw[..., 3 * n:4 * n]
        d  = raw[..., -2].mean(dim=1)                       # (B,)
        h  = raw[..., -1].mean(dim=1)
        p = torch.complex(-(torch.nn.functional.softplus(a) + DELTA), b)  # Re p < 0
        r = torch.complex(rr, ri) / U                        # unitary identity
        B = raw.shape[0]
        p = p.reshape(B, U * n, 1)                           # union over cells
        r = r.reshape(B, U * n, 1)
        s = torch.complex(torch.zeros_like(self.w), self.w)  # s = i*omega, (F,)
        t = (r / (s - p) + r.conj() / (s - p.conj())).sum(dim=1)          # (B, F)
        if m:
            q   = raw[..., 4 * n:4 * n + m] / U
            rho = -(torch.nn.functional.softplus(raw[..., 4 * n + m:4 * n + 2 * m]) + DELTA)
            q = q.reshape(B, U * m, 1)
            rho = rho.reshape(B, U * m, 1)
            t = t + (q / (s - torch.complex(rho, torch.zeros_like(rho)))).sum(dim=1)
        t = t + torch.complex(d, torch.zeros_like(d)).unsqueeze(-1) + h.unsqueeze(-1) * s
        return t.real ** 2 + t.imag ** 2

    def forward(self, grid):
        B, N, _, C = grid.shape
        cells = grid.reshape(B * N * N, C)
        latent = self.unitary(cells).reshape(B, N * N, self.latent_dim)

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
            delta = delta.reshape(len(targets), B, self.latent_dim)
            for k, idx in enumerate(targets):
                latent[:, idx, :] = latent[:, idx, :] + delta[k]

        raw = self.vf_head(latent.reshape(B * N * N, self.latent_dim))
        raw = raw.reshape(B, N * N, -1)
        return self._compose_vf(raw, N * N)
