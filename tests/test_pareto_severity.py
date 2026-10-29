"""Tests for Pareto Type I severity."""

import numpy as np
import pytest
from scipy.stats import pareto

from therisktheory.severity_aggregate.pareto import (
    pareto_severity_cdf,
    pareto_severity_moments,
    pareto_severity_pdf,
    pareto_severity_ppf,
)


def test_reference_moments():
    result = pareto_severity_moments(alpha=3.0, minimum=100.0)

    np.testing.assert_allclose(
        [
            result[k]
            for k in (
                "mean",
                "variance",
                "second_raw_moment",
            )
        ],
        [150.0, 7500.0, 30000.0],
        rtol=1e-13,
    )

    assert all(
        isinstance(v, np.ndarray) and v.shape == () for v in result.values()
    )


@pytest.mark.parametrize(
    "name,fn",
    [
        ("pdf", pareto_severity_pdf),
        ("cdf", pareto_severity_cdf),
    ],
)
def test_scipy_density_cdf(name, fn):
    x = np.array([-100.0, 0.0, 50.0, 100.0, 150.0, 1000.0])

    got = fn(x, alpha=3.0, minimum=100.0)

    expected = getattr(pareto, name)(x, b=3.0, scale=100.0)

    np.testing.assert_allclose(got, expected, rtol=1e-13, atol=1e-15)


def test_quantiles_and_roundtrip():
    qs = np.array([0.0, 0.001, 0.2, 0.5, 0.9, 0.999, 1.0])

    got = pareto_severity_ppf(qs, alpha=3, minimum=100)

    np.testing.assert_allclose(
        got,
        pareto.ppf(qs, b=3, scale=100),
        rtol=1e-13,
    )

    np.testing.assert_allclose(
        pareto_severity_cdf(got[1:-1], alpha=3, minimum=100),
        qs[1:-1],
        rtol=1e-11,
        atol=2e-15,
    )

    assert np.isposinf(got[-1])
    assert got[0] == 100


def test_moment_thresholds():
    shapes = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    got = pareto_severity_moments(alpha=shapes, minimum=1.0)

    assert np.all(np.isposinf(got["mean"][:2]))
    assert np.all(np.isfinite(got["mean"][2:]))

    assert np.all(np.isposinf(got["variance"][:4]))

    assert np.all(np.isposinf(got["second_raw_moment"][:4]))

    np.testing.assert_allclose(
        got["variance"][4:],
        pareto.var(shapes[4:]),
        rtol=1e-13,
    )

    np.testing.assert_allclose(
        got["second_raw_moment"][4:],
        [5.0, 3.0],
        rtol=1e-13,
    )


def test_scalar_and_parameter_broadcast():
    x = np.array([[100.0], [200.0]])
    shape = np.array([1.5, 2.5, 3.5])

    result = pareto_severity_pdf(x, alpha=shape, minimum=100)

    assert result.shape == (2, 3)

    np.testing.assert_allclose(
        result,
        pareto.pdf(x, b=shape, scale=100),
    )

    moments = pareto_severity_moments(alpha=shape, minimum=100)

    assert moments["mean"].shape == (3,)

    for fn in (
        pareto_severity_pdf,
        pareto_severity_cdf,
        pareto_severity_ppf,
    ):
        assert fn(0.5, alpha=3, minimum=100).shape == ()


def test_boundaries():
    assert pareto_severity_pdf(
        100, alpha=2, minimum=100
    ).item() == pytest.approx(0.02)

    assert pareto_severity_cdf(100, alpha=2, minimum=100).item() == 0.0

    assert pareto_severity_pdf(50, alpha=2, minimum=100).item() == 0.0

    assert pareto_severity_cdf(50, alpha=2, minimum=100).item() == 0.0

    assert np.isposinf(pareto_severity_ppf(1, alpha=2, minimum=100).item())


@pytest.mark.parametrize(
    "value",
    [None, "2", True, 3 + 4j, [1, "2"], [None]],
)
def test_invalid_alpha_type(value):
    with pytest.raises(TypeError):
        pareto_severity_pdf(2, alpha=value, minimum=1)


@pytest.mark.parametrize(
    "value",
    [None, "2", False, 3 + 4j, [None]],
)
def test_invalid_x_type(value):
    with pytest.raises(TypeError):
        pareto_severity_cdf(value, alpha=2, minimum=1)


@pytest.mark.parametrize(
    "value",
    [None, "2", False, 3 + 4j, [None]],
)
def test_invalid_minimum_type(value):
    with pytest.raises(TypeError):
        pareto_severity_moments(alpha=2, minimum=value)


@pytest.mark.parametrize(
    "bad",
    [0, -1, np.inf, -np.inf, np.nan],
)
def test_invalid_alpha_value(bad):
    with pytest.raises(ValueError):
        pareto_severity_moments(alpha=bad, minimum=1)


@pytest.mark.parametrize(
    "bad",
    [0, -1, np.inf, np.nan],
)
def test_invalid_minimum_value(bad):
    with pytest.raises(ValueError):
        pareto_severity_moments(alpha=3, minimum=bad)


@pytest.mark.parametrize(
    "bad",
    [np.nan, np.inf, -np.inf],
)
def test_invalid_x_value(bad):
    with pytest.raises(ValueError):
        pareto_severity_pdf(bad, alpha=3, minimum=1)


@pytest.mark.parametrize(
    "bad",
    [-1e-3, 1.01, np.nan, np.inf],
)
def test_invalid_q(bad):
    with pytest.raises(ValueError):
        pareto_severity_ppf(bad, alpha=3, minimum=1)


def test_incompatible_shapes():
    with pytest.raises(ValueError, match="broadcast"):
        pareto_severity_pdf(
            np.ones(2),
            alpha=np.ones(3),
            minimum=1,
        )


def test_empty_arrays():
    for fn in (
        pareto_severity_pdf,
        pareto_severity_cdf,
        pareto_severity_ppf,
    ):
        assert fn(np.array([]), alpha=3, minimum=100).shape == (0,)

    assert (
        pareto_severity_moments(alpha=np.array([]), minimum=100)["mean"].size
        == 0
    )


def test_finite_quantile_overflow():
    with pytest.raises(OverflowError):
        pareto_severity_ppf(0.9, alpha=0.01, minimum=1e308)


def test_finite_density_overflow():
    with pytest.raises(OverflowError):
        pareto_severity_pdf(
            1e-300,
            alpha=1e100,
            minimum=1e-300,
        )


def test_finite_moment_overflow():
    with pytest.raises(OverflowError):
        pareto_severity_moments(alpha=3, minimum=1e308)


def test_numerical_cdf_support_edge():
    x = np.nextafter(1.0, np.inf)

    y = pareto_severity_cdf(x, alpha=3, minimum=1.0).item()

    assert y > 0

    np.testing.assert_allclose(y, 3 * (x - 1), rtol=1e-12, atol=0)


def test_ragged_input():
    with pytest.raises(TypeError):
        pareto_severity_pdf(
            [[1], [1, 2]],
            alpha=3,
            minimum=1,
        )


def test_underflow_density_is_zero():
    result = pareto_severity_pdf(1e300, alpha=1000, minimum=1)

    assert result.item() == 0
