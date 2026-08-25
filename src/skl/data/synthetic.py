"""Synthetic spectra and data generators."""

import numpy as np


def make_spectrum(kind, n, rate):
    """Generate a normalized, strictly decreasing eigenvalue spectrum.

    The spectrum is normalized so that the largest eigenvalue equals 1,
    which keeps lambda comparable across decay rates.

    Parameters
    ----------
    kind : {'polynomial', 'exponential'}
        Decay family. Polynomial gives sigma_j = j ** (-rate);
        exponential gives sigma_j = exp(-rate * (j - 1)).
    n : int
        Number of eigenvalues. Must be positive.
    rate : float
        Decay rate. Must be positive; larger means faster decay.

    Returns
    -------
    np.ndarray
        Array of length n, strictly decreasing, with spectrum[0] == 1.
    """
    if not np.issubdtype(type(n), np.integer) or n <= 0:
        raise ValueError(f"n must be a positive integer, n = {n}")

    if rate <= 0:
        raise ValueError(f"rate must be positive, rate = {rate}")

    # Integer arrays cannot be raised to a negative power, so index in float.
    j = np.arange(1, n + 1, dtype=np.float64)

    if kind == "polynomial":
        spectrum = j ** (-rate)
    elif kind == "exponential":
        spectrum = np.exp(-rate * (j - 1))
    else:
        raise ValueError(f"Unknown spectrum kind: {kind}")

    return spectrum / spectrum[0]