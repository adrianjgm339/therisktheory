"""Tests for lognormal claim severity."""

import numpy as np
import pytest
from scipy.stats import lognorm

from therisktheory.severity_aggregate.lognormal import (
    lognormal_severity_cdf,
    lognormal_severity_moments,
    lognormal_severity_pdf,
    lognormal_severity_ppf,
)


def test_reference_moments():
    m = lognormal_severity_moments(
        mu=np.log(100),
        sigma=0.5,
    )

    expected_mean = 100 * np.exp(0.125)
    expected_second = 100**2 * np.exp(0.5)
    expected_var = 100**2 * np.exp(0.25) * np.expm1(0.25)

    np.testing.assert_allclose(m["mean"], expected_mean, rtol=1e-13)
    np.testing.assert_allclose(
        m["second_raw_moment"],
        expected_second,
        rtol=1e-13,
    )
    np.testing.assert_allclose(m["variance"], expected_var, rtol=1e-13)

    assert all(v.shape == () for v in m.values())


@pytest.mark.parametrize(
    "name,func",
    [
        ("pdf", lognormal_severity_pdf),
        ("cdf", lognormal_severity_cdf),
    ],
)
def test_agreement_with_scipy(name, func):
    x = np.array([-1, 0, 10, 100, 200])

    actual = func(x, mu=np.log(100), sigma=0.5)

    expected = getattr(lognorm, name)(x, s=0.5, scale=100)

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=0)
    assert actual.shape == x.shape


def test_quantiles_and_roundtrip():
    q = np.array([0.001, 0.2, 0.5, 0.8, 0.999])

    x = lognormal_severity_ppf(q, mu=np.log(100), sigma=0.5)

    np.testing.assert_allclose(
        x,
        lognorm.ppf(q, s=0.5, scale=100),
        rtol=1e-13,
    )

    recovered = lognormal_severity_cdf(x, mu=np.log(100), sigma=0.5)

    np.testing.assert_allclose(recovered, q, rtol=1e-12, atol=1e-15)


def test_support_boundaries():
    assert lognormal_severity_pdf(0, mu=0, sigma=1).item() == 0

    assert lognormal_severity_cdf(-4, mu=0, sigma=1).item() == 0

    assert lognormal_severity_ppf(0, mu=0, sigma=1).item() == 0

    assert np.isposinf(lognormal_severity_ppf(1, mu=0, sigma=1))


def test_broadcasting():
    x = np.array([[1.0], [2.0]])
    mu = np.array([0.0, 1.0, 2.0])

    actual = lognormal_severity_cdf(x, mu=mu, sigma=0.5)

    assert actual.shape == (2, 3)

    np.testing.assert_allclose(
        actual,
        lognorm.cdf(x, s=0.5, scale=np.exp(mu)),
    )

    moments = lognormal_severity_moments(mu=mu, sigma=0.5)

    assert moments["mean"].shape == (3,)


@pytest.mark.parametrize(
    "value",
    [None, "2", True, 1 + 1j, [1, "2"], [None]],
)
def test_invalid_mu_types(value):
    with pytest.raises(TypeError):
        lognormal_severity_pdf(1, mu=value, sigma=1)


@pytest.mark.parametrize(
    "value",
    [None, True, 1j, "3", [None]],
)
def test_invalid_observation_types(value):
    with pytest.raises(TypeError):
        lognormal_severity_cdf(value, mu=0, sigma=1)


@pytest.mark.parametrize(
    "value",
    [0, -1, np.inf, np.nan],
)
def test_invalid_sigma(value):
    with pytest.raises(ValueError):
        lognormal_severity_moments(mu=0, sigma=value)


@pytest.mark.parametrize(
    "value",
    [np.inf, -np.inf, np.nan],
)
def test_invalid_mu_values(value):
    with pytest.raises(ValueError):
        lognormal_severity_moments(mu=value, sigma=1)


@pytest.mark.parametrize(
    "value",
    [-0.1, 1.1, np.inf, np.nan],
)
def test_invalid_probabilities(value):
    with pytest.raises(ValueError):
        lognormal_severity_ppf(value, mu=0, sigma=1)


def test_nonfinite_observations():
    with pytest.raises(ValueError):
        lognormal_severity_pdf(np.inf, mu=0, sigma=1)


def test_incompatible_broadcasting():
    with pytest.raises(ValueError, match="broadcast"):
        lognormal_severity_pdf(
            np.ones(2),
            mu=np.ones(3),
            sigma=1,
        )


def test_empty_arrays():
    funcs = (
        lognormal_severity_pdf,
        lognormal_severity_cdf,
        lognormal_severity_ppf,
    )

    for func in funcs:
        result = func(np.array([]), mu=0, sigma=1)
        assert result.shape == (0,)

    moments = lognormal_severity_moments(mu=np.array([]), sigma=1)

    assert moments["mean"].size == 0


def test_explicit_overflow():
    with pytest.raises(OverflowError):
        lognormal_severity_moments(mu=700, sigma=1)

    with pytest.raises(OverflowError):
        lognormal_severity_ppf(0.5, mu=900, sigma=1)

    x = np.nextafter(0.0, 1.0)

    with pytest.raises(OverflowError):
        lognormal_severity_pdf(x, mu=np.log(x), sigma=1e-10)


def test_small_sigma_variance():
    moments = lognormal_severity_moments(
        mu=340,
        sigma=1e-200,
    )

    assert moments["variance"].item() > 0


def test_scalar_array_contract():
    funcs = (
        lognormal_severity_pdf,
        lognormal_severity_cdf,
        lognormal_severity_ppf,
    )

    for func in funcs:
        result = func(0.5, mu=0, sigma=1)

        assert isinstance(result, np.ndarray)
        assert result.shape == ()


def test_reject_ragged_input():
    with pytest.raises(TypeError):
        lognormal_severity_pdf(
            [[1], [1, 2]],
            mu=0,
            sigma=1,
        )
