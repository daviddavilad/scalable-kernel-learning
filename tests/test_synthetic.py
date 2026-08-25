import numpy as np
import pytest

from skl.data.synthetic import make_spectrum
from skl.diagnostics.effective_dim import effective_dimension

KINDS = ["polynomial", "exponential"]


@pytest.mark.parametrize("kind", KINDS)
def test_spectrum_has_correct_length(kind):
    assert len(make_spectrum(kind, 10, 2.0)) == 10


@pytest.mark.parametrize("kind", KINDS)
def test_spectrum_is_positive(kind):
    assert np.all(make_spectrum(kind, 10, 2.0) > 0)


@pytest.mark.parametrize("kind", KINDS)
def test_spectrum_is_strictly_decreasing(kind):
    assert np.all(np.diff(make_spectrum(kind, 10, 2.0)) < 0)


@pytest.mark.parametrize("kind", KINDS)
def test_spectrum_is_normalized(kind):
    assert make_spectrum(kind, 10, 2.0)[0] == 1.0


def test_polynomial_matches_hand_computed_values():
    """sigma_j = j ** -2 for j = 1..4."""
    expected = np.array([1.0, 0.25, 1.0 / 9.0, 0.0625])

    np.testing.assert_allclose(make_spectrum("polynomial", 4, 2.0), expected, rtol=1e-12)


def test_exponential_matches_hand_computed_values():
    """sigma_j = exp(-2 * (j - 1)) for j = 1..3."""
    expected = np.array([1.0, np.e ** -2, np.e ** -4])

    np.testing.assert_allclose(make_spectrum("exponential", 3, 2.0), expected, rtol=1e-12)


@pytest.mark.parametrize("kind", KINDS)
def test_spectrum_matches_formula(kind):
    n, rate = 10, 2.0
    j = np.arange(1, n + 1, dtype=float)
    expected = j ** (-rate) if kind == "polynomial" else np.exp(-rate * (j - 1))

    np.testing.assert_allclose(make_spectrum(kind, n, rate), expected, rtol=1e-12)


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        make_spectrum("unknown", 10, 2.0)


@pytest.mark.parametrize("bad_n", [0, -5])
def test_nonpositive_n_raises(bad_n):
    with pytest.raises(ValueError):
        make_spectrum("polynomial", bad_n, 2.0)


@pytest.mark.parametrize("bad_rate", [0.0, -1.0])
def test_nonpositive_rate_raises(bad_rate):
    with pytest.raises(ValueError):
        make_spectrum("polynomial", 10, bad_rate)


def test_numpy_integer_n_is_accepted():
    """n often arrives as np.int64 from sweeps and configs."""
    assert len(make_spectrum("polynomial", np.int64(10), 2.0)) == 10


def test_exponential_has_lower_effective_dimension():
    """Exponential spectrum should have lower effective dimension than polynomial."""
    n = 100
    rate = 2.0
    lam = 1e-2

    poly_spectrum = make_spectrum("polynomial", n, rate)
    exp_spectrum = make_spectrum("exponential", n, rate)

    poly_eff_dim = effective_dimension(poly_spectrum, lam)
    exp_eff_dim = effective_dimension(exp_spectrum, lam)

    assert exp_eff_dim < poly_eff_dim