"""Loss functions for the power-transmission task.

`beta2_loss` is a dip-weighted MSE: it up-weights frequencies where the target
transmission is low (the resonance dips), which a plain MSE tends to under-fit
(it floors the predicted dips above the true near-zero nulls). Weight grows
quadratically as T -> 0:  w = 1 + 2 * (1 - T)^2.
"""
import torch


def beta2_loss(pred, tgt):
    w = 1.0 + 2.0 * (1.0 - tgt).clamp(min=0.0).square()
    return (w * (pred - tgt).square()).mean()


def plain_mse(pred, tgt):
    """Unweighted MSE — used for validation / model selection / reporting."""
    return (pred - tgt).square().mean()
