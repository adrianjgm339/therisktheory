"""Analytical and numerical tests for Weibull severity."""

import numpy as np
import pytest
from scipy.stats import weibull_min

from therisktheory.severity_aggregate.weibull import (
    weibull_severity_cdf,
    weibull_severity_moments,
    weibull_severity_pdf,
    weibull_severity_ppf,
)


@pytest.mark.parametrize("shape", [0.5, 1.0, 2.0, 3.0, 10.0])
def test_pdf_cdf_scipy(shape):
    x = np.array([-2.0, 0, 0.2, 2.0, 10.0, 300.0])

    with np.errstate(divide="ignore", invalid="ignore"):
        expected_pdf = weibull_min.pdf(x, c=shape, scale=10)

    np.testing.assert_allclose(
        weibull_severity_pdf(x, shape=shape, scale=10),
        expected_pdf,
        rtol=1e-12,
    )

    np.testing.assert_allclose(
        weibull_severity_cdf(x, shape=shape, scale=10),
        weibull_min.cdf(x, c=shape, scale=10),
        rtol=1e-12,
        atol=1e-15,
    )


@pytest.mark.parametrize("shape", [0.5, 1, 2, 3, 5])
def test_ppf_roundtrip(shape):
    q = np.array([0.0, 1e-10, 0.001, 0.1, 0.5, 0.9, 0.999, 1.0])

    result = weibull_severity_ppf(q, shape=shape, scale=100)

    np.testing.assert_allclose(
        result,
        weibull_min.ppf(q, c=shape, scale=100),
        rtol=1e-12,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        weibull_severity_cdf(
            result[1:-1],
            shape=shape,
            scale=100,
        ),
        q[1:-1],
        rtol=1e-10,
        atol=1e-14,
    )

    assert result[0] == 0
    assert np.isposinf(result[-1])


@pytest.mark.parametrize("shape", [0.5, 1.0, 2.0, 3.0, 10.0])
def test_moments_scipy(shape):
    result = weibull_severity_moments(shape=shape, scale=100)

    mean, variance = weibull_min.stats(c=shape, scale=100, moments="mv")

    np.testing.assert_allclose(
        [result["mean"], result["variance"]],
        [mean, variance],
        rtol=1e-12,
    )

    np.testing.assert_allclose(
        result["second_raw_moment"],
        variance + mean**2,
        rtol=1e-12,
    )

    assert all(
        isinstance(value, np.ndarray) and value.shape == ()
        for value in result.values()
    )


def test_exponential_special_case():
    result = weibull_severity_moments(shape=1, scale=100)

    np.testing.assert_allclose(
        [
            result["mean"],
            result["variance"],
            result["second_raw_moment"],
        ],
        [100, 10000, 20000],
        rtol=1e-12,
    )

    assert weibull_severity_pdf(0, shape=1, scale=100).item() == pytest.approx(
        0.01
    )


def test_singular_and_zero_boundaries():
    assert np.isposinf(weibull_severity_pdf(0, shape=0.5, scale=1))

    assert weibull_severity_pdf(0, shape=2, scale=1).item() == 0

    assert weibull_severity_pdf(-1, shape=0.5, scale=1).item() == 0

    assert weibull_severity_cdf(0, shape=0.5, scale=1).item() == 0

    assert weibull_severity_cdf(-1, shape=2, scale=1).item() == 0


def test_large_shape_variance_precision():
    shape = 1e8

    result = weibull_severity_moments(shape=shape, scale=100)

    expected = 100**2 * (np.pi**2 / 6) / shape**2

    assert result["variance"].item() > 0

    np.testing.assert_allclose(
        result["variance"],
        expected,
        rtol=1e-7,
    )


def test_vectorized_inputs():
    x = np.array([[1.0], [10.0]])
    shape = np.array([0.5, 1.0, 2.0])
    scale = np.array([1.0, 10.0, 100.0])

    result = weibull_severity_pdf(x, shape=shape, scale=scale)

    assert result.shape == (2, 3)

    np.testing.assert_allclose(
        result,
        weibull_min.pdf(x, c=shape, scale=scale),
    )

    np.testing.assert_allclose(
        weibull_severity_cdf(x, shape=shape, scale=scale),
        weibull_min.cdf(x, c=shape, scale=scale),
    )

    assert weibull_severity_ppf(0.5, shape=shape, scale=scale).shape == (3,)

    assert weibull_severity_moments(shape=shape, scale=scale)["mean"].shape == (
        3,
    )


@pytest.mark.parametrize(
    "fn",
    [
        weibull_severity_pdf,
        weibull_severity_cdf,
        weibull_severity_ppf,
    ],
)
def test_scalar_array_type(fn):
    assert fn(0.5, shape=2, scale=10).shape == ()


@pytest.mark.parametrize(
    "bad",
    [
        None,
        "2",
        True,
        False,
        1 + 2j,
        [None],
        [[1], [1, 2]],
    ],
)
def test_bad_types(bad):
    with pytest.raises(TypeError):
        weibull_severity_pdf(1, shape=bad, scale=2)

    with pytest.raises(TypeError):
        weibull_severity_cdf(1, shape=2, scale=bad)

    with pytest.raises(TypeError):
        weibull_severity_pdf(bad, shape=2, scale=2)


@pytest.mark.parametrize("bad", [0, -1, np.inf, np.nan])
def test_invalid_shape(bad):
    with pytest.raises(ValueError):
        weibull_severity_moments(shape=bad, scale=1)


@pytest.mark.parametrize("bad", [0, -1, np.inf, np.nan])
def test_invalid_scale(bad):
    with pytest.raises(ValueError):
        weibull_severity_moments(shape=1, scale=bad)


@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
def test_invalid_x(bad):
    with pytest.raises(ValueError):
        weibull_severity_pdf(bad, shape=1, scale=1)


@pytest.mark.parametrize("bad", [-1e-5, 1.001, np.nan, np.inf])
def test_invalid_probability(bad):
    with pytest.raises(ValueError):
        weibull_severity_ppf(bad, shape=2, scale=1)


def test_broadcast_error():
    with pytest.raises(ValueError, match="broadcast"):
        weibull_severity_pdf(
            np.ones(2),
            shape=np.ones(3),
            scale=1,
        )

    with pytest.raises(ValueError, match="broadcast"):
        weibull_severity_moments(
            shape=np.ones(2),
            scale=np.ones(3),
        )


def test_empty_arrays():
    for fn in (
        weibull_severity_pdf,
        weibull_severity_cdf,
        weibull_severity_ppf,
    ):
        assert fn(np.array([]), shape=1, scale=1).shape == (0,)

    assert (
        weibull_severity_moments(shape=np.array([]), scale=1)["mean"].size == 0
    )


def test_density_overflows():
    with pytest.raises(OverflowError):
        weibull_severity_pdf(
            1e-308,
            shape=1e308,
            scale=1e-308,
        )

    with pytest.raises(OverflowError):
        weibull_severity_pdf(0, shape=1, scale=1e-320)


def test_quantile_overflow():
    with pytest.raises(OverflowError):
        weibull_severity_ppf(0.99, shape=0.01, scale=1e308)


def test_moment_overflow():
    with pytest.raises(OverflowError):
        weibull_severity_moments(shape=0.01, scale=100)

    with pytest.raises(OverflowError):
        weibull_severity_moments(shape=1, scale=1e308)


def test_close_to_scale_large_shape():
    shape = 1e12
    x = np.nextafter(1.0, np.inf)

    result = weibull_severity_pdf(x, shape=shape, scale=1).item()

    expected = weibull_min.pdf(x, c=shape, scale=1)

    np.testing.assert_allclose(result, expected, rtol=3e-4)


def test_far_tail_and_underflow():
    assert weibull_severity_pdf(1e300, shape=1000, scale=1).item() == 0

    assert weibull_severity_cdf(1e300, shape=1000, scale=1).item() == 1

    assert weibull_severity_ppf(1e-300, shape=0.5, scale=1).item() == 0


def test_shapes_and_raw_moments_agree():
    shapes = np.array([1e3, 1e4, 1e8])

    result = weibull_severity_moments(shape=shapes, scale=1)

    assert np.all(result["variance"] > 0)

    np.testing.assert_allclose(
        result["mean"],
        weibull_min.mean(shapes),
        rtol=1e-12,
    )
