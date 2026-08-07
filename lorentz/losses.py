"""Losses for fitting power-transmittance spectra."""


def beta2_loss(prediction, target):
    """Dip-weighted MSE with weights from 1 at T=1 to 3 at T=0."""
    weight = 1.0 + 2.0 * (1.0 - target).clamp(min=0.0).square()
    return (weight * (prediction - target).square()).mean()


def plain_mse(prediction, target):
    """Unweighted MSE used for validation and model selection."""
    return (prediction - target).square().mean()
