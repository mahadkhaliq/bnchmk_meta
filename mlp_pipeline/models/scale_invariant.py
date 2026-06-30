"""The MLP WITH the neighbouring mechanism (the "scale-invariant" model).

Instead of feeding the whole supercell to one big MLP, this model:

    grid (B, N, N, C)
      1. periodic (wrap-around) padding        -> (B, N+2p, N+2p, C)
      2. extract the K x K window around each   -> (B, N*N, K*K*C)
         of the N*N cells and flatten it
      3. run a SHARED MLP f_theta on every cell  -> (B, N*N, n_freq)
         (the SAME weights are applied to each cell -> permutation/"scale"
          invariance; the cell count can change without changing the model)
      4. average the per-cell spectra            -> y_hat (B, n_freq)

This mirrors the data flow documented in the original notebook:
    (n, N*N, K*K*C) -> reshape (n*N*N, K*K*C) -> f_theta -> (n*N*N, n_freq)
                    -> reshape (n, N*N, n_freq) -> mean over cells -> (n, n_freq)
"""
import torch
import torch.nn as nn

from models.mlp import MLP


class ScaleInvariantMetasurface(nn.Module):
    def __init__(self, N=2, K=3, C=5, n_freq=2001, hidden=2000, n_hidden=10):
        super().__init__()
        self.N, self.K, self.C = N, K, C
        self.pad = K // 2
        self.n_freq = n_freq
        # shared per-cell network: K*K*C inputs -> n_freq outputs
        layers = [K * K * C] + [hidden] * n_hidden + [n_freq]
        self.f_theta = MLP(layers)

    def forward(self, grid):
        B = grid.shape[0]
        N, K, C, pad = self.N, self.K, self.C, self.pad

        # 1. periodic padding via wrap-around source indices
        src = (torch.arange(N + 2 * pad, device=grid.device) - pad) % N
        padded = grid[:, src[:, None], src[None, :], :]

        # 2. extract each cell's K x K neighbourhood and flatten
        x = torch.zeros(B, N * N, K * K * C, device=grid.device)
        for idx in range(N * N):
            i, j = idx // N, idx % N
            window = padded[:, i:i + K, j:j + K, :]
            x[:, idx, :] = window.reshape(B, K * K * C)

        # 3. shared MLP over all cells at once
        x_flat = x.reshape(B * N * N, K * K * C)
        z_flat = self.f_theta(x_flat)

        # 4. reshape back and average over the N*N cells
        z = z_flat.reshape(B, N * N, self.n_freq)
        return z.mean(dim=1)
