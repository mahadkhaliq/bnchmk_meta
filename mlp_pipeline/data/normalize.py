"""Feature normalisation: min-max scaling to [-1, 1] per column.

This is the exact scheme used throughout the original notebook:

    x_range = (max - min) / 2
    x_avg   = (max + min) / 2
    x_norm  = (x - x_avg) / x_range

Always FIT on the training split (x_max / x_min computed there) and APPLY the
same x_max / x_min to the test split — never fit normalisation on test data.
"""
import numpy as np


def normalize(x, x_max=None, x_min=None):
    """Scale each column of `x` to [-1, 1].

    Args:
        x: array of shape (n_samples, n_features).
        x_max, x_min: per-column statistics. If None they are computed from `x`
            (use this for the training split). Pass the training values back in
            to scale the test split with the same range.

    Returns:
        (x_normalized, x_max, x_min)
    """
    if x_max is None:
        x_max = x.max(axis=0)   # axis=0 -> per column (top-to-bottom)
        x_min = x.min(axis=0)

    x_range = (x_max - x_min) / 2.0
    x_avg   = (x_max + x_min) / 2.0
    x_range = np.where(x_range == 0, 1e-8, x_range)  # avoid divide-by-zero on constant columns

    x_normalized = (x - x_avg) / x_range
    return x_normalized, x_max, x_min
