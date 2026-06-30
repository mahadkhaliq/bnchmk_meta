"""The core MLP, used by BOTH variants.

Architecture (Deng et al. 2021 baseline):
    - a stack of fully-connected layers whose widths are given by `layers`
    - every HIDDEN layer is:  Linear -> BatchNorm1d -> ReLU
    - the OUTPUT layer is a bare Linear (no BN, no activation)

It operates on 2D input of shape (batch, features). In the baseline the input is
the flat 14-D geometry; in the neighbourhood model the SAME class is reused as
the shared per-cell network f_theta (input = K*K*C flattened window).
"""
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, layers):
        """`layers` is the full width list, e.g. [14, 2000, ..., 2000, 2001]."""
        super().__init__()
        self.linears = nn.ModuleList()
        self.bn_linears = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.linears.append(nn.Linear(layers[i], layers[i + 1]))
            self.bn_linears.append(nn.BatchNorm1d(layers[i + 1]))

    def forward(self, g):
        out = g
        for i, (fc, bn) in enumerate(zip(self.linears, self.bn_linears)):
            if i < len(self.linears) - 1:
                out = F.relu(bn(fc(out)))   # hidden layer
            else:
                out = fc(out)               # output layer: plain Linear
        return out
