"""Concept #1 model from the heterogeneous-metasurface slides.

This is the "relaxed Lorentz" version: it keeps the physics-inspired structure

    base cell response + summed neighbour perturbations -> per-cell decoder

but replaces the fixed Lorentzian function with a trainable neural decoder.

Note (per Dr. Malof): the "Lookup Table" label on the Concept #1 slide is a
typo. f_theta1 (here `unitary`) is an ordinary neural network, and the
composition f_theta_r(f_theta1(x)) — here `decoder(unitary(x))`, the N == 1
path below — is the model pretrained to predict 1x1 spectra (equivalently a
2x2 whose four cells are identical).
That makes the model easier to optimize while preserving the important
size-invariant structure: the same unitary, interaction, and decoder networks
are reused for every cell, and the final spectrum is averaged over cells.
"""
import torch
import torch.nn as nn

from models.mlp import MLP
from models.rel_encoding import RelEncoding


class ConceptOneMetasurface(nn.Module):
    """Additive-interaction metasurface model with a neural spectrum decoder.

    For each cell i with geometry g_i:

        z_i = U(g_i) + sum_j V(g_i, g_j, dx_ij, dy_ij)
        y_i = D(z_i)
        y_hat = mean_i y_i

    U is the single-cell/unitary network, V is the pairwise interaction network,
    and D is the trainable replacement for the hard Lorentzian physics block.
    """

    def __init__(
        self,
        K=3,
        C=4,
        n_freq=2001,
        latent_dim=64,
        hidden=256,
        n_hidden=4,
        rel_encoding="offset",
        rel_emb_dim=8,
        use_relative_position=None,   # legacy: False maps to rel_encoding="none"
    ):
        super().__init__()
        if K % 2 != 1:
            raise ValueError("K must be odd so each window has a center cell.")
        if use_relative_position is False:
            rel_encoding = "none"

        self.K = K
        self.C = C
        self.pad = K // 2
        self.n_freq = n_freq
        self.latent_dim = latent_dim
        self.rel = RelEncoding(rel_encoding, K, rel_emb_dim)

        pair_in = 2 * C + self.rel.extra_dim
        self.unitary = MLP([C] + [hidden] * n_hidden + [latent_dim])
        self.interaction = MLP([pair_in] + [hidden] * n_hidden + [latent_dim])
        self.decoder = MLP([latent_dim] + [hidden] * n_hidden + [n_freq])

    def _relative_position(self, B, di, dj, device, dtype):
        return self.rel(di, dj, B, device, dtype)

    def forward(self, grid):
        B, N, _, C = grid.shape
        if C != self.C:
            raise ValueError(f"Expected {self.C} channels per cell, got {C}.")

        cells = grid.reshape(B * N * N, C)
        base = self.unitary(cells).reshape(B, N * N, self.latent_dim)

        # A 1x1 metasurface has no distinct neighbours; this is the pretraining
        # path U(g) -> D(z), matching Concept #1's single-cell warm start.
        if N == 1 or self.K == 1:
            return self.decoder(base.reshape(B * N * N, self.latent_dim)).reshape(
                B, N * N, self.n_freq
            ).mean(dim=1)

        src = (torch.arange(N + 2 * self.pad, device=grid.device) - self.pad) % N
        padded = grid[:, src[:, None], src[None, :], :]

        perturb = torch.zeros(B, N * N, self.latent_dim, device=grid.device, dtype=grid.dtype)
        pair_features = []
        pair_targets = []

        for idx in range(N * N):
            i, j = idx // N, idx % N
            center = grid[:, i, j, :]
            for di in range(-self.pad, self.pad + 1):
                for dj in range(-self.pad, self.pad + 1):
                    if di == 0 and dj == 0:
                        continue
                    neigh = padded[:, i + di + self.pad, j + dj + self.pad, :]
                    pieces = [center, neigh]
                    rel = self._relative_position(B, di, dj, grid.device, grid.dtype)
                    if rel is not None:
                        pieces.append(rel)
                    pair_features.append(torch.cat(pieces, dim=1))
                    pair_targets.append(idx)

        all_pairs = torch.cat(pair_features, dim=0)
        all_delta = self.interaction(all_pairs).reshape(len(pair_targets), B, self.latent_dim)
        for k, idx in enumerate(pair_targets):
            perturb[:, idx, :] = perturb[:, idx, :] + all_delta[k]

        latent = base + perturb
        spectra = self.decoder(latent.reshape(B * N * N, self.latent_dim))
        spectra = spectra.reshape(B, N * N, self.n_freq)
        return spectra.mean(dim=1)
