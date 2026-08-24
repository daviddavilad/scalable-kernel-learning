"""Effective dimension of a kernel ridge regression problem."""

import numpy as np


def effective_dimension(eigenvalues, lam, n=None, tol=1e-10):
    """Compute d_eff(lambda) = sum_i sigma_i / (sigma_i + lambda * n).

    Convention: the regularized system is (K + lambda * n * I) alpha = y,
    so lambda is scaled by n. Papers that write (K + lambda * I) use a
    lambda that is n times larger.

    Parameters
    ----------
    eigenvalues : array_like
        Eigenvalues of the kernel matrix K. Must be non-empty.
    lam : float or array_like
        Regularization parameter(s) lambda. Must be positive. If an array
        is given, d_eff is evaluated at each value.
    n : int, optional
        Number of observations. If None, inferred from len(eigenvalues).
    tol : float, optional
        Relative tolerance for negative eigenvalues, scaled by the largest
        eigenvalue. Values above -tol * max(eigenvalues) are treated as
        floating-point noise and clipped to zero; anything more negative
        raises ValueError.

    Returns
    -------
    float or np.ndarray
        Effective dimension. A float if lam is scalar, otherwise an array
        with one entry per lambda.
    """
    lam = np.atleast_1d(np.asarray(lam, dtype=float))       # (n_lam,)
    if np.any(lam <= 0):
        raise ValueError(f"lam must be positive, lam = {lam}")

    eigenvalues = np.asarray(eigenvalues, dtype=float)

    if eigenvalues.size == 0:
        raise ValueError("eigenvalues must not be empty")

    if n is None:
        n = len(eigenvalues)

    if n <= 0:
        raise ValueError(f"n must be positive, n = {n}")

    if eigenvalues.size and np.any(eigenvalues < -tol * eigenvalues.max()):
        raise ValueError(
            f"eigenvalues contains a significant negative value: "
            f"min = {eigenvalues.min()}"
        )

    eigenvalues = np.clip(eigenvalues, 0.0, None)

    num = eigenvalues[None, :]                              # (1, n_eig)
    den = eigenvalues[None, :] + lam[:, None] * n           # (n_lam, n_eig)
    result = np.sum(num / den, axis=-1)                     # (n_lam,)

    return result.item() if result.size == 1 else result