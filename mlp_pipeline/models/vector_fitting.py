"""Vector-fitting head on the Concept #1 trunk (ConceptOneVF).

Same Eq. 4 trunk as concept_one.py (unitary U + interaction V, additive per
cell), but the decoder is a pole-residue rational in s = i*omega:

    t(s) = sum_k [ r_k/(s - p_k) + conj(r_k)/(s - conj(p_k)) ]   (conjugate pairs)
         + sum_m  q_m/(s - rho_m)                                (real poles, n_real)
         + d  +  h*s ,      T = |t|^2

Trunk uses SiLU activations and NO BatchNorm, so each cell's latent depends only
on that cell -> the unitary identity (residues / U) holds exactly and a supercell
of identical cells reproduces the 1x1 spectrum.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rel_encoding import RelEncoding

DELTA = 0.01


def _silu_mlp(dims):
    """Linear+SiLU stack, no BatchNorm, no activation on the final layer."""
    layers = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b), nn.SiLU()]
    layers = layers[:-1]                       # drop last SiLU
    return nn.Sequential(*layers)


def _vf_head(d_in, n_pole, n_real=0):
    head = nn.Linear(d_in, 4 * n_pole + 2 * n_real + 2)
    with torch.no_grad():
        head.weight.mul_(0.05)
        head.bias.zero_()
        head.bias[n_pole:2 * n_pole] = torch.linspace(0.2, 2.0, n_pole)  # beta seed
        head.bias[-2] = 1.0                                              # d -> transparent slab
    return head


class ConceptOneVF(nn.Module):
    def __init__(self, K=3, C=4, n_freq=2001, latent_dim=64, hidden=512,
                 n_hidden=4, n_pole=8, n_real=0, freqs=None,
                 w_min=0.5, w_max=1.5, rel_encoding="offset", rel_emb_dim=8):
        super().__init__()
        if K % 2 != 1:
            raise ValueError("K must be odd.")
        self.K, self.C, self.pad = K, C, K // 2
        self.latent_dim, self.n_pole, self.n_real = latent_dim, n_pole, n_real
        self.rel = RelEncoding(rel_encoding, K, rel_emb_dim)
        pair_in = 2 * C + self.rel.extra_dim
        self.unitary = _silu_mlp([C] + [hidden] * n_hidden + [latent_dim])
        self.interaction = _silu_mlp([pair_in] + [hidden] * n_hidden + [latent_dim])
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
        n, m = self.n_pole, self.n_real
        a = raw[..., 0:n]
        b = raw[..., n:2 * n]
        rr = raw[..., 2 * n:3 * n]
        ri = raw[..., 3 * n:4 * n]
        d = raw[..., -2].mean(dim=1)
        h = raw[..., -1].mean(dim=1)
        p = torch.complex(-(F.softplus(a) + DELTA), b)   # Re p < 0
        r = torch.complex(rr, ri) / U                    # unitary identity
        B = raw.shape[0]
        p = p.reshape(B, U * n, 1)
        r = r.reshape(B, U * n, 1)
        s = torch.complex(torch.zeros_like(self.w), self.w)
        t = (r / (s - p) + r.conj() / (s - p.conj())).sum(dim=1)
        if m:
            q = raw[..., 4 * n:4 * n + m] / U
            rho = -(F.softplus(raw[..., 4 * n + m:4 * n + 2 * m]) + DELTA)
            q = q.reshape(B, U * m, 1)
            rho = rho.reshape(B, U * m, 1)
            t = t + (q / (s - torch.complex(rho, torch.zeros_like(rho)))).sum(dim=1)
        t = t + torch.complex(d, torch.zeros_like(d)).unsqueeze(-1) + h.unsqueeze(-1) * s
        # power transmission is physically in [0, 1]; clamp bounds the stiff
        # pole-residue rational (mirrors the Lorentz head's clamp(max=1.0)).
        return (t.real ** 2 + t.imag ** 2).clamp(0.0, 1.0)

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
