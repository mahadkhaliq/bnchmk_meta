"""Scale-invariant neighbourhood model with SiLU/Sigmoid shared MLP.

This is the SiLU/beta2 counterpart of scale_invariant.py:
    grid -> K x K wrap-around windows -> shared MLPSiLU -> average spectra.
"""
import torch
import torch.nn as nn

from models.mlp_silu import MLPSiLU


class ScaleInvariantSiLU(nn.Module):
    def __init__(self, N=2, K=3, C=4, n_freq=2001, hidden=512, n_layers=4):
        super().__init__()
        self.N, self.K, self.C = N, K, C
        self.pad = K // 2
        self.n_freq = n_freq
        self.f_theta = MLPSiLU(
            d_in=K * K * C,
            d_out=n_freq,
            hidden=hidden,
            n_layers=n_layers,
        )

    def forward(self, grid):
        B = grid.shape[0]
        N, K, C, pad = self.N, self.K, self.C, self.pad

        src = (torch.arange(N + 2 * pad, device=grid.device) - pad) % N
        padded = grid[:, src[:, None], src[None, :], :]

        x = torch.zeros(B, N * N, K * K * C, device=grid.device)
        for idx in range(N * N):
            i, j = idx // N, idx % N
            window = padded[:, i:i + K, j:j + K, :]
            x[:, idx, :] = window.reshape(B, K * K * C)

        x_flat = x.reshape(B * N * N, K * K * C)
        z_flat = self.f_theta(x_flat)
        z = z_flat.reshape(B, N * N, self.n_freq)
        return z.mean(dim=1)
