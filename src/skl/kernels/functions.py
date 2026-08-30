import numpy as np


def _squared_distances(X, Z):
    """
    Compute the squared Euclidean distances between each pair of rows in X and Z.

    Parameters
    ----------
    X : array-like of shape (n_samples_X, n_features)
        First set of input vectors.
    Z : array-like of shape (n_samples_Z, n_features)
        Second set of input vectors.

    Returns
    -------
    sq_dists : array-like of shape (n_samples_X, n_samples_Z)
        Squared Euclidean distance matrix.
    """

    X = np.asarray(X, dtype=float)
    Z = np.asarray(Z, dtype=float)

    if X.ndim != 2:
        raise ValueError(
            f"X must be 2-D of shape (n_samples, n_features), got shape {X.shape}"
        )

    if Z.ndim != 2:
        raise ValueError(
            f"Z must be 2-D of shape (n_samples, n_features), got shape {Z.shape}"
        )

    if X.shape[1] != Z.shape[1]:
        raise ValueError(
            f"X and Z must have the same number of features: "
            f"X has {X.shape[1]}, Z has {Z.shape[1]}"
        )

    # ||x - z||^2 = ||x||^2 - 2 x.z + ||z||^2. Cheaper than broadcasting an
    # (n, m, d) intermediate, but subtracting near-equal quantities can give
    # small negative values, so clip before exponentiating.

    sq_dists = (
        np.sum(X**2, axis=1)[:, None]
        - 2.0 * (X @ Z.T)
        + np.sum(Z**2, axis=1)[None, :]
    )

    return np.clip(sq_dists, 0.0, None)


def rbf_kernel(X, Z=None, lengthscale=1.0):
    """
    Compute the Radial Basis Function (RBF) kernel between two sets of input vectors.

    Parameters
    ----------
    X : array-like of shape (n_samples_X, n_features)
        First set of input vectors.
    Z : array-like of shape (n_samples_Z, n_features), optional
        Second set of input vectors. If None, Z is set to X.
    lengthscale : float, default=1.0
        Lengthscale parameter for the RBF kernel.

    Returns
    -------
    K : array-like of shape (n_samples_X, n_samples_Z)
        Kernel matrix.
    """

    X = np.asarray(X, dtype=float)

    if Z is None:
        Z = X
    
    if lengthscale <= 0:
        raise ValueError(f"lengthscale must be positive, lengthscale = {lengthscale}")

    sq_dists = _squared_distances(X, Z)

    return np.exp(-sq_dists / (2.0 * lengthscale**2))