"""Weibull claim severity with shape k and scale lambda.

Validation status: experimental; independent mathematical review pending.
Reference: scipy.stats.weibull_min(c=shape, loc=0, scale=scale).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.special import gammaln, zeta


def _finite_real(value: npt.ArrayLike, name: str) -> npt.NDArray[np.float64]:
    """Convert finite numeric input to float64, rejecting bool and complex."""
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain real numeric values.") from exc

    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values.")

    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(raw, dtype=np.float64)

    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")

    return result


def _parameters(
    shape: npt.ArrayLike,
    scale: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Check strictly positive finite Weibull shape and scale."""
    k = _finite_real(shape, "shape")
    lam = _finite_real(scale, "scale")

    if np.any(k <= 0):
        raise ValueError("shape must be strictly positive.")

    if np.any(lam <= 0):
        raise ValueError("scale must be strictly positive.")

    return k, lam


def _broadcast(
    *values: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], ...]:
    """Broadcast or raise a descriptive shape mismatch error."""
    try:
        return tuple(np.broadcast_arrays(*values))
    except ValueError as exc:
        raise ValueError("Input shapes are not broadcast-compatible.") from exc


def _log_ratio(
    x: npt.NDArray[np.float64],
    scale: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Evaluate log(x/scale) accurately near the scale boundary."""
    safe_x = np.where(x > 0, x, scale)

    with np.errstate(
        over="ignore",
        divide="ignore",
        invalid="ignore",
    ):
        far = np.log(safe_x) - np.log(scale)
        near = np.log1p((safe_x - scale) / scale)

    return np.where(
        np.abs(safe_x - scale) <= scale * 0.5,
        near,
        far,
    )


def weibull_severity_pdf(
    x: npt.ArrayLike,
    *,
    shape: npt.ArrayLike,
    scale: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Evaluate Weibull probability density.

    Args:
        x: Finite real severity, including values below support.
        shape: Strictly positive Weibull shape k.
        scale: Strictly positive Weibull scale lambda.

    Returns:
        numpy.ndarray: Broadcast float64 PDF, 0-D for scalar inputs.

    Notes:
        At x=0, the density is infinite for shape<1,
        equal to 1/scale for shape=1, and zero for shape>1.

    Raises:
        OverflowError: A mathematically finite density exceeds float64 range.
    """
    values = _finite_real(x, "x")
    k, lam = _parameters(shape, scale)
    values, k, lam = _broadcast(values, k, lam)

    z = _log_ratio(values, lam)

    with np.errstate(
        over="ignore",
        under="ignore",
        divide="ignore",
        invalid="ignore",
    ):
        log_t = k * z
        t = np.exp(log_t)

        log_density = np.log(k) - np.log(lam) + (k - 1.0) * z - t

        positive = np.exp(np.where(log_t > 50, -np.inf, log_density))

        boundary = np.where(
            k < 1,
            np.inf,
            np.where(k == 1, 1.0 / lam, 0.0),
        )

        density = np.where(
            values > 0,
            positive,
            np.where(values == 0, boundary, 0.0),
        )

    singular = (values == 0) & (k < 1)

    if np.any(~np.isfinite(density) & ~singular):
        raise OverflowError("A finite Weibull density exceeds float64 range.")

    return np.asarray(density, dtype=np.float64)


def weibull_severity_cdf(
    x: npt.ArrayLike,
    *,
    shape: npt.ArrayLike,
    scale: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Evaluate Weibull cumulative probabilities.

    Args:
        x: Finite real severity.
        shape: Strictly positive Weibull shape.
        scale: Strictly positive Weibull scale.

    Returns:
        numpy.ndarray: Broadcast CDF, zero when x <= 0.
    """
    values = _finite_real(x, "x")
    k, lam = _parameters(shape, scale)
    values, k, lam = _broadcast(values, k, lam)

    z = _log_ratio(values, lam)

    with np.errstate(
        over="ignore",
        under="ignore",
        invalid="ignore",
    ):
        cdf = np.where(
            values > 0,
            -np.expm1(-np.exp(k * z)),
            0.0,
        )

    return np.asarray(cdf, dtype=np.float64)


def weibull_severity_ppf(
    q: npt.ArrayLike,
    *,
    shape: npt.ArrayLike,
    scale: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Evaluate Weibull quantiles.

    Args:
        q: Finite probabilities in [0, 1].
        shape: Strictly positive Weibull shape.
        scale: Strictly positive Weibull scale.

    Returns:
        numpy.ndarray: Float64 quantiles: q=0 returns zero;
            q=1 returns positive infinity.

    Raises:
        OverflowError: A finite quantile exceeds float64 range.
    """
    probability = _finite_real(q, "q")

    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("q must lie in [0, 1].")

    k, lam = _parameters(shape, scale)

    probability, k, lam = _broadcast(probability, k, lam)

    with np.errstate(
        over="ignore",
        under="ignore",
        divide="ignore",
        invalid="ignore",
    ):
        log_q = np.log(lam) + np.log(-np.log1p(-probability)) / k

        quantile = np.exp(log_q)

    if np.any(~np.isfinite(quantile) & (probability != 1)):
        raise OverflowError("A finite Weibull quantile exceeds float64 range.")

    return np.asarray(quantile, dtype=np.float64)


def weibull_severity_moments(
    *,
    shape: npt.ArrayLike,
    scale: npt.ArrayLike,
) -> dict[str, npt.NDArray[np.float64]]:
    """Return mean, variance, and second raw moment.

    Args:
        shape: Strictly positive Weibull shape k.
        scale: Strictly positive Weibull scale lambda.

    Returns:
        dict[str, numpy.ndarray]: Arrays named mean, variance,
            second_raw_moment.

    Notes:
        All moments are mathematically finite for
        positive finite shape and scale.

        Log-Gamma evaluation avoids direct Gamma overflow.
        For very large shape, a series is used to reduce
        cancellation in the variance calculation.

        Mathematically finite but unrepresentable values
        raise OverflowError.
    """
    k, lam = _parameters(shape, scale)
    k, lam = _broadcast(k, lam)

    with np.errstate(
        over="ignore",
        under="ignore",
        divide="ignore",
        invalid="ignore",
    ):
        u = 1.0 / k

        log_g1 = gammaln(1.0 + u)
        log_g2 = gammaln(1.0 + 2.0 * u)

        delta_direct = log_g2 - 2.0 * log_g1

        # Log-Gamma ratio expansion for small u.
        delta_series = (
            u
            * u
            * (
                np.pi**2 / 6.0
                - 2.0 * zeta(3.0, 1.0) * u
                + 3.5 * (np.pi**4 / 90.0) * u * u
            )
        )

        delta = np.where(
            k > 1000.0,
            delta_series,
            delta_direct,
        )

        log_lam = np.log(lam)

        log_results = {
            "mean": log_lam + log_g1,
            "second_raw_moment": (2.0 * log_lam + log_g2),
            "variance": (
                2.0 * log_lam + 2.0 * log_g1 + np.log(np.expm1(delta))
            ),
        }

        results = {
            key: np.asarray(
                np.exp(value),
                dtype=np.float64,
            )
            for key, value in log_results.items()
        }

    if any(np.any(~np.isfinite(value)) for value in results.values()):
        raise OverflowError("A Weibull moment exceeds float64 range.")

    return results
