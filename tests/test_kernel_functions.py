import numpy as np
import pytest

from skl.kernels.functions import matern_kernel, rbf_kernel

NUS = [0.5, 1.5, 2.5]

def test_rbf_kernel_shape():
    rng = np.random.default_rng(0)

    X = rng.normal(size=(3, 2))
    Y = rng.normal(size=(2, 2))

    K = rbf_kernel(X, Y)

    assert K.shape == (3, 2)


def test_rbf_kernel_self_shape():
    rng = np.random.default_rng(0)

    X = rng.normal(size=(3, 2))

    K = rbf_kernel(X)

    assert K.shape == (3, 3)


def test_rbf_kernel_symmetry():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 3))

    K = rbf_kernel(X)

    np.testing.assert_allclose(K, K.T)


def test_rbf_kernel_unit_diagonal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 3))

    K = rbf_kernel(X)

    np.testing.assert_allclose(np.diag(K), 1.0)


def test_rbf_kernel_is_bounded():
    """RBF values lie in (0, 1] mathematically, but underflow to 0 at distance."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 3))
    Z = rng.normal(size=(4, 3))

    K = rbf_kernel(X, Z)

    assert np.all(K >= 0.0)
    assert np.all(K <= 1.0)


def test_rbf_kernel_underflows_to_zero_at_distance():
    """Far-apart points underflow rather than erroring or going negative."""
    X = np.array([[0.0]])
    Z = np.array([[1e4]])

    K = rbf_kernel(X, Z, lengthscale=1.0)

    assert K[0, 0] == 0.0


def test_rbf_kernel_matches_naive():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 3))
    Z = rng.normal(size=(4, 3))
    lengthscale = 2.0

    K = rbf_kernel(X, Z, lengthscale)

    dists = np.sum((X[:, None, :] - Z[None, :, :]) ** 2, axis=-1)
    K_naive = np.exp(-0.5 * dists / (lengthscale**2))

    np.testing.assert_allclose(K, K_naive)


def test_rbf_kernel_increases_with_lengthscale():
    """Off-diagonal entries only: the diagonal is 1 at every lengthscale."""
    X = np.array([[0.0], [1.0], [2.0]])

    K_small = rbf_kernel(X, lengthscale=0.5)
    K_large = rbf_kernel(X, lengthscale=2.0)

    off_diagonal = ~np.eye(3, dtype=bool)

    assert np.all(K_large[off_diagonal] > K_small[off_diagonal])


def test_rbf_kernel_one_dimensional_input_raises():
    with pytest.raises(ValueError):
        rbf_kernel(np.array([1.0, 2.0, 3.0]))


def test_rbf_kernel_is_positive_semidefinite():
    """The defining property of a kernel: K must be PSD."""
    rng = np.random.default_rng(0)

    X = rng.normal(size=(8, 3))

    K = rbf_kernel(X)
    eigenvalues = np.linalg.eigvalsh(K)

    assert np.all(eigenvalues >= -1e-10)


def test_rbf_kernel_dimension_mismatch_raises():
    rng = np.random.default_rng(0)

    X = rng.normal(size=(5, 3))
    Z = rng.normal(size=(4, 5))

    with pytest.raises(ValueError):
        rbf_kernel(X, Z)


@pytest.mark.parametrize("nu", NUS)
def test_matern_kernel_shape(nu):
    rng = np.random.default_rng(0)

    X = rng.normal(size=(3, 2))
    Z = rng.normal(size=(4, 2))

    K = matern_kernel(X, Z, nu=nu)

    assert K.shape == (3, 4)


@pytest.mark.parametrize("nu", NUS)
def test_matern_kernel_symmetry(nu):
    rng = np.random.default_rng(0)

    X = rng.normal(size=(5, 3))

    K = matern_kernel(X, nu=nu)

    np.testing.assert_allclose(K, K.T)


@pytest.mark.parametrize("nu", NUS)
def test_matern_kernel_unit_diagonal(nu):
    rng = np.random.default_rng(0)

    X = rng.normal(size=(5, 3))

    K = matern_kernel(X, nu=nu)

    np.testing.assert_allclose(np.diag(K), np.ones(X.shape[0]))


@pytest.mark.parametrize("nu", NUS)
def test_matern_kernel_is_bounded(nu):
    rng = np.random.default_rng(0)

    X = rng.normal(size=(5, 3))

    K = matern_kernel(X, nu=nu)

    assert np.all(K >= 0.0)
    assert np.all(K <= 1.0)


@pytest.mark.parametrize("nu", NUS)
def test_matern_kernel_is_positive_semidefinite(nu):
    rng = np.random.default_rng(0)

    X = rng.normal(size=(8, 3))

    K = matern_kernel(X, nu=nu)

    eigenvalues = np.linalg.eigvalsh(K)

    assert np.all(eigenvalues >= -1e-10)


@pytest.mark.parametrize("nu", [1.0, 0.0, -1.0])
def test_matern_kernel_invalid_nu_raises(nu):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 3))

    with pytest.raises(ValueError):
        matern_kernel(X, nu=nu)