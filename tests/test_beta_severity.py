"""Reference and stress tests for scaled Beta severity."""

import numpy as np
import pytest
from scipy.stats import beta as scipy_beta

import therisktheory.severity_aggregate.beta as beta_module
from therisktheory.severity_aggregate.beta import (
    beta_severity_cdf,
    beta_severity_moments,
    beta_severity_pdf,
    beta_severity_ppf,
)


@pytest.mark.parametrize(
    "a,b,m",
    [
        (2, 3, 1000),
        (0.5, 2, 100),
        (2, 0.5, 100),
        (1, 1, 1),
        (3, 4, 10),
    ],
)
def test_pdf_cdf_scipy(a, b, m):
    x = np.array([-1, 0, m * 0.1, m * 0.4, m * 0.8, m, m + 1])

    with np.errstate(divide="ignore", invalid="ignore"):
        expected_pdf = scipy_beta.pdf(x, a, b, scale=m)

    np.testing.assert_allclose(
        beta_severity_pdf(x, alpha=a, beta=b, maximum_loss=m),
        expected_pdf,
        rtol=1e-12,
        atol=0,
    )

    np.testing.assert_allclose(
        beta_severity_cdf(x, alpha=a, beta=b, maximum_loss=m),
        scipy_beta.cdf(x, a, b, scale=m),
        rtol=1e-12,
        atol=1e-15,
    )


@pytest.mark.parametrize(
    "a,b,m",
    [
        (2, 3, 1000),
        (0.5, 0.75, 100),
        (1, 1, 1),
        (3, 10, 100),
    ],
)
def test_ppf_roundtrip(a, b, m):
    q = np.array([0, 1e-8, 0.01, 0.2, 0.5, 0.8, 0.99, 1])

    quantiles = beta_severity_ppf(q, alpha=a, beta=b, maximum_loss=m)

    np.testing.assert_allclose(
        quantiles,
        scipy_beta.ppf(q, a, b, scale=m),
        rtol=1e-11,
        atol=1e-13,
    )

    np.testing.assert_allclose(
        beta_severity_cdf(
            quantiles[1:-1],
            alpha=a,
            beta=b,
            maximum_loss=m,
        ),
        q[1:-1],
        rtol=1e-10,
        atol=1e-12,
    )

    assert quantiles[0] == 0
    assert quantiles[-1] == m


@pytest.mark.parametrize(
    "a,b,m",
    [
        (2, 3, 1000),
        (0.5, 2, 100),
        (2, 0.5, 100),
        (1, 1, 1),
        (3, 4, 10),
    ],
)
def test_moments_scipy(a, b, m):
    result = beta_severity_moments(alpha=a, beta=b, maximum_loss=m)

    mean, variance = scipy_beta.stats(a, b, scale=m, moments="mv")

    np.testing.assert_allclose(result["mean"], mean, rtol=1e-12)

    np.testing.assert_allclose(result["variance"], variance, rtol=1e-12)

    np.testing.assert_allclose(
        result["second_raw_moment"],
        variance + mean**2,
        rtol=1e-12,
    )

    assert all(
        isinstance(v, np.ndarray) and v.shape == () for v in result.values()
    )


def test_analytical_moments():
    result = beta_severity_moments(alpha=2, beta=3, maximum_loss=1000)

    np.testing.assert_allclose(
        [
            result["mean"],
            result["variance"],
            result["second_raw_moment"],
        ],
        [400, 40000, 200000],
        rtol=1e-12,
    )


def test_endpoints():
    assert np.isposinf(beta_severity_pdf(0, alpha=0.5, beta=2))

    assert np.isposinf(beta_severity_pdf(1, alpha=2, beta=0.5))

    assert beta_severity_pdf(0, alpha=2, beta=2).item() == 0

    assert beta_severity_pdf(1, alpha=2, beta=2).item() == 0

    assert beta_severity_pdf(0, alpha=1, beta=3).item() == pytest.approx(3)

    assert beta_severity_pdf(1, alpha=3, beta=1).item() == pytest.approx(3)

    assert beta_severity_pdf(-1, alpha=0.5, beta=0.5).item() == 0

    assert beta_severity_pdf(2, alpha=0.5, beta=0.5).item() == 0

    assert beta_severity_cdf(-1, alpha=2, beta=2).item() == 0

    assert beta_severity_cdf(2, alpha=2, beta=2).item() == 1


def test_broadcasting():
    x = np.array([[0.2], [0.8]])
    alpha = np.array([1, 2, 3])

    result = beta_severity_pdf(x, alpha=alpha, beta=2)

    assert result.shape == (2, 3)

    np.testing.assert_allclose(
        result,
        scipy_beta.pdf(x, alpha, 2),
        rtol=1e-13,
    )

    assert beta_severity_cdf(x, alpha=alpha, beta=2).shape == (2, 3)

    assert beta_severity_ppf(0.5, alpha=alpha, beta=2).shape == (3,)

    assert beta_severity_moments(alpha=alpha, beta=2)["mean"].shape == (3,)


@pytest.mark.parametrize(
    "func",
    [
        beta_severity_pdf,
        beta_severity_cdf,
        beta_severity_ppf,
    ],
)
def test_scalar_array_contract(func):
    assert func(0.5, alpha=2, beta=3).shape == ()


@pytest.mark.parametrize(
    "bad",
    [
        None,
        "2",
        True,
        False,
        2j,
        [None],
        [[1], [1, 2]],
    ],
)
def test_invalid_types(bad):
    with pytest.raises(TypeError):
        beta_severity_pdf(0.5, alpha=bad, beta=2)

    with pytest.raises(TypeError):
        beta_severity_cdf(0.5, alpha=2, beta=bad)

    with pytest.raises(TypeError):
        beta_severity_ppf(
            0.5,
            alpha=2,
            beta=2,
            maximum_loss=bad,
        )

    with pytest.raises(TypeError):
        beta_severity_pdf(bad, alpha=2, beta=2)


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf])
def test_invalid_parameter_values(bad):
    with pytest.raises(ValueError):
        beta_severity_moments(alpha=bad, beta=2)

    with pytest.raises(ValueError):
        beta_severity_moments(alpha=2, beta=bad)

    with pytest.raises(ValueError):
        beta_severity_moments(
            alpha=2,
            beta=3,
            maximum_loss=bad,
        )


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_invalid_observations(bad):
    with pytest.raises(ValueError):
        beta_severity_pdf(bad, alpha=2, beta=3)


@pytest.mark.parametrize("bad", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_probabilities(bad):
    with pytest.raises(ValueError):
        beta_severity_ppf(bad, alpha=2, beta=3)


def test_incompatible_shapes():
    with pytest.raises(ValueError, match="broadcast"):
        beta_severity_pdf(
            np.ones(2),
            alpha=np.ones(3),
            beta=1,
        )

    with pytest.raises(ValueError, match="broadcast"):
        beta_severity_moments(
            alpha=np.ones(2),
            beta=np.ones(3),
        )


def test_empty_arrays():
    empty = np.array([])

    for fn in (
        beta_severity_pdf,
        beta_severity_cdf,
        beta_severity_ppf,
    ):
        assert fn(empty, alpha=2, beta=3).shape == (0,)

    assert beta_severity_moments(alpha=empty, beta=3)["mean"].shape == (0,)


def test_moment_overflow():
    with pytest.raises(OverflowError):
        beta_severity_moments(
            alpha=2,
            beta=3,
            maximum_loss=1e308,
        )


def test_pdf_overflow():
    with pytest.raises(OverflowError):
        beta_severity_pdf(
            0,
            alpha=1,
            beta=1e308,
            maximum_loss=1e-308,
        )


def test_extreme_shape_sum():
    result = beta_severity_moments(
        alpha=1e308,
        beta=1e308,
        maximum_loss=1,
    )

    np.testing.assert_allclose(result["mean"], 0.5, rtol=2e-13)

    assert np.isfinite(result["variance"]).all()
    assert np.isfinite(result["second_raw_moment"]).all()


def test_large_scale_second_moment_overflow():
    with pytest.raises(OverflowError):
        beta_severity_moments(
            alpha=2,
            beta=3,
            maximum_loss=1e200,
        )


def test_tiny_scale():
    result = beta_severity_moments(
        alpha=2,
        beta=3,
        maximum_loss=1e-200,
    )

    assert result["mean"].item() > 0
    assert result["second_raw_moment"].item() == 0


def test_near_endpoints():
    near_zero = np.nextafter(0.0, 1.0)
    near_one = np.nextafter(1.0, 0.0)

    for x in (near_zero, near_one):
        result = beta_severity_pdf(x, alpha=2, beta=3)

        assert np.isfinite(result).all()
        assert result.item() >= 0


def test_default_maximum_loss():
    np.testing.assert_allclose(
        beta_severity_cdf(0.5, alpha=2, beta=3),
        scipy_beta.cdf(0.5, 2, 3),
    )


def test_scipy_cdf_numerical_failure(monkeypatch):
    monkeypatch.setattr(
        beta_module,
        "betainc",
        lambda *_: np.nan,
    )

    with pytest.raises(FloatingPointError, match="CDF"):
        beta_severity_cdf(0.5, alpha=2, beta=3)


def test_scipy_ppf_numerical_failure(monkeypatch):
    monkeypatch.setattr(
        beta_module,
        "betaincinv",
        lambda *_: np.nan,
    )

    with pytest.raises(FloatingPointError, match="quantile"):
        beta_severity_ppf(0.5, alpha=2, beta=3)
