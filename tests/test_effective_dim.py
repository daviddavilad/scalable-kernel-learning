import numpy as np
import pytest

from skl.diagnostics.effective_dim import effective_dimension


def test_flat_spectrum_matches_closed_form():
    """k eigenvalues equal to a, rest zero: d_eff = k*a / (a + lam*n)."""
    n = 100
    k = 10
    a = 3.0
    lam = 1e-2

    eigenvalues = np.zeros(n)
    eigenvalues[:k] = a

    expected = k * a / (a + lam * n)
    actual = effective_dimension(eigenvalues, lam, n)

    np.testing.assert_allclose(actual, expected, rtol=1e-12)


def test_effective_dimension_approaches_rank_as_lambda_goes_to_zero():
    """As lambda -> 0, d_eff approaches the rank of K."""
    n = 100
    k = 10
    lam = 1e-12

    eigenvalues = np.zeros(n)
    eigenvalues[:k] = 3.0

    actual = effective_dimension(eigenvalues, lam, n)

    np.testing.assert_allclose(actual, k, rtol=1e-6)


def test_effective_dimension_approaches_zero_as_lambda_goes_to_infinity():
    """As lambda -> infinity, d_eff approaches zero."""
    n = 100
    k = 10
    lam = 1e12

    eigenvalues = np.zeros(n)
    eigenvalues[:k] = 3.0

    actual = effective_dimension(eigenvalues, lam, n)

    np.testing.assert_allclose(actual, 0.0, atol=1e-9)


def test_effective_dimension_is_strictly_decreasing_in_lambda():
    """d_eff(lambda) must strictly decrease as lambda increases."""
    n = 100

    eigenvalues = np.zeros(n)
    eigenvalues[:10] = 3.0

    lambdas = np.logspace(-6, 2, 20)

    values = np.array([
        effective_dimension(eigenvalues, lam, n)
        for lam in lambdas
    ])

    assert np.all(np.diff(values) < 0)


def test_nonpositive_lambda_raises():
    eigenvalues = np.ones(10)

    with pytest.raises(ValueError):
        effective_dimension(eigenvalues, 0.0, 10)

    with pytest.raises(ValueError):
        effective_dimension(eigenvalues, -1.0, 10)


def test_tiny_negative_eigenvalues_are_clipped():
    """eigvalsh returns small negatives for near-singular matrices."""
    eigenvalues = np.array([3.0, 3.0, -1e-17, -2e-16])
    expected = effective_dimension(np.array([3.0, 3.0, 0.0, 0.0]), 1e-2, 4)

    actual = effective_dimension(eigenvalues, 1e-2, 4)

    np.testing.assert_allclose(actual, expected, rtol=1e-12)


def test_large_negative_eigenvalues_raise():
    with pytest.raises(ValueError):
        effective_dimension(np.array([3.0, -0.5]), 1e-2, 2)


def test_array_lambda_matches_scalar_calls():
    eigenvalues = np.zeros(100)
    eigenvalues[:10] = 3.0
    lambdas = np.logspace(-6, 2, 5)

    expected = np.array([effective_dimension(eigenvalues, l, 100) for l in lambdas])
    actual = effective_dimension(eigenvalues, lambdas, 100)

    np.testing.assert_allclose(actual, expected, rtol=1e-12)


def test_nonpositive_n_raises():
    eigenvalues = np.ones(10)

    with pytest.raises(ValueError):
        effective_dimension(eigenvalues, 1e-2, 0)

    with pytest.raises(ValueError):
        effective_dimension(eigenvalues, 1e-2, -5)


def test_empty_eigenvalues_raises():
    with pytest.raises(ValueError):
        effective_dimension(np.array([]), 1e-2)