"""Pure MLP with SiLU activations and a Sigmoid output (predictions in [0, 1]).

This is a DIFFERENT architecture from models/mlp.py::MLP:
    models/mlp.py   : Linear -> BatchNorm1d -> ReLU  (... bare Linear output)
    this one        : Linear -> SiLU                 (... Linear -> Sigmoid)

The final Sigmoid constrains the output to [0, 1] = power transmission T, and
there is no BatchNorm. Pairs with the dip-weighted `beta2` loss (losses.py).
"""
import torch.nn as nn


class MLPSiLU(nn.Module):
    def __init__(self, d_in=16, d_out=2001, hidden=512, n_layers=4):
        super().__init__()
        layers, prev = [], d_in
        for _ in range(n_layers):
            layers += [nn.Linear(prev, hidden), nn.SiLU()]
            prev = hidden
        layers += [nn.Linear(prev, d_out), nn.Sigmoid()]   # output in [0,1] = power T
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
