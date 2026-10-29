"""Lognormal claim severity functions.

Mathematical validation: experimental; independent peer review pending.

For X > 0, log(X) follows Normal(mu, sigma**2).
Reference: scipy.stats.lognorm, shape=sigma, scale=exp(mu), loc=0.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.special import ndtr, ndtri

_LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)


def _real_finite(
    value: npt.ArrayLike,
    name: str,
) -> npt.NDArray[np.float64]:
    """Validate real, finite numeric inputs, including empty arrays."""
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain real numeric values.") from exc

    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values.")

    values = np.asarray(raw, dtype=np.float64)

    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")

    return values


def _parameters(
    mu: npt.ArrayLike,
    sigma: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Validate log-location and log-standard deviation."""
    location = _real_finite(mu, "mu")
    spread = _real_finite(sigma, "sigma")

    if np.any(spread <= 0):
        raise ValueError("sigma must be strictly positive.")

    return location, spread


def _broadcast(
    *values: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], ...]:
    """Broadcast inputs or raise a descriptive error."""
    try:
        return tuple(np.broadcast_arrays(*values))
    except ValueError as exc:
        raise ValueError("Input shapes are not broadcast-compatible.") from exc


def lognormal_severity_pdf(
    x: npt.ArrayLike,
    *,
    mu: npt.ArrayLike,
    sigma: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Evaluate the lognormal probability density.

    Args:
        x: Finite real severity. Density is zero when x <= 0.
        mu: Finite mean of log severity.
        sigma: Finite, strictly positive standard deviation of log severity.

    Returns:
        numpy.ndarray: Broadcast float64 density; 0-D for scalar inputs.

    Raises:
        TypeError: Input contains non-real or non-numeric values.
        ValueError: Invalid parameters or incompatible input shapes.
        OverflowError: A finite density exceeds float64 range.
    """
    values = _real_finite(x, "x")
    location, spread = _parameters(mu, sigma)

    values, location, spread = _broadcast(values, location, spread)

    safe_values = np.where(values > 0, values, 1.0)
    log_x = np.log(safe_values)

    with np.errstate(
        over="ignore",
        under="ignore",
        invalid="ignore",
    ):
        z = (log_x - location) / spread

        log_density = -0.5 * z * z - log_x - np.log(spread) - _LOG_SQRT_2PI

        density = np.where(
            values > 0,
            np.exp(log_density),
            0.0,
        )

    if np.any(~np.isfinite(density)):
        raise OverflowError("The lognormal density exceeds float64 range.")

    return np.asarray(density, dtype=np.float64)


def lognormal_severity_cdf(
    x: npt.ArrayLike,
    *,
    mu: npt.ArrayLike,
    sigma: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Evaluate the cumulative lognormal distribution.

    Args:
        x: Finite real severity; CDF is zero for x <= 0.
        mu: Mean of log severity.
        sigma: Strictly positive standard deviation of log severity.

    Returns:
        numpy.ndarray: Broadcast float64 probabilities.
    """
    values = _real_finite(x, "x")
    location, spread = _parameters(mu, sigma)

    values, location, spread = _broadcast(values, location, spread)

    safe_values = np.where(values > 0, values, 1.0)

    with np.errstate(over="ignore", invalid="ignore"):
        z = (np.log(safe_values) - location) / spread

        probability = np.where(
            values > 0,
            ndtr(z),
            0.0,
        )

    return np.asarray(probability, dtype=np.float64)


def lognormal_severity_ppf(
    q: npt.ArrayLike,
    *,
    mu: npt.ArrayLike,
    sigma: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Calculate lognormal quantiles for probabilities in [0, 1].

    Args:
        q: Finite probability between zero and one, inclusive.
        mu: Mean of log severity.
        sigma: Strictly positive standard deviation of log severity.

    Returns:
        numpy.ndarray: Broadcast float64 quantiles. q=0 returns zero and
            q=1 returns positive infinity.

    Notes:
        Very small positive quantiles may underflow to zero.
        Unrepresentable finite upper quantiles raise OverflowError.
    """
    probabilities = _real_finite(q, "q")

    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("q must lie in [0, 1].")

    location, spread = _parameters(mu, sigma)

    probabilities, location, spread = _broadcast(
        probabilities, location, spread
    )

    with np.errstate(
        over="ignore",
        under="ignore",
        invalid="ignore",
    ):
        log_quantile = location + spread * ndtri(probabilities)

        quantile = np.exp(log_quantile)

    invalid = ~np.isfinite(quantile) & (probabilities != 1.0)

    if np.any(invalid):
        raise OverflowError(
            "A finite lognormal quantile exceeds float64 range."
        )

    return np.asarray(quantile, dtype=np.float64)


def lognormal_severity_moments(
    *,
    mu: npt.ArrayLike,
    sigma: npt.ArrayLike,
) -> dict[str, npt.NDArray[np.float64]]:
    """Return the first two raw moments and the variance.

    Args:
        mu: Mean of log severity.
        sigma: Strictly positive standard deviation of log severity.

    Returns:
        dict: Three float64 NumPy arrays:
            mean, variance, second_raw_moment.

    Notes:
        Every lognormal moment is mathematically finite for
        finite mu and finite positive sigma.

        Variance uses a log-domain evaluation to reduce loss
        of precision when sigma is small.

        Unrepresentable moments raise OverflowError.
    """
    location, spread = _parameters(mu, sigma)

    location, spread = _broadcast(location, spread)

    with np.errstate(
        over="ignore",
        under="ignore",
        divide="ignore",
        invalid="ignore",
    ):
        spread_sq = spread * spread

        log_expm1 = np.where(
            spread_sq < 1e-4,
            (
                2.0 * np.log(spread)
                + np.log1p(0.5 * spread_sq + spread_sq**2 / 6.0)
            ),
            spread_sq + np.log(-np.expm1(-spread_sq)),
        )

        logarithms = {
            "mean": location + 0.5 * spread_sq,
            "variance": (2.0 * location + spread_sq + log_expm1),
            "second_raw_moment": (2.0 * location + 2.0 * spread_sq),
        }

        result = {
            key: np.asarray(
                np.exp(value),
                dtype=np.float64,
            )
            for key, value in logarithms.items()
        }

    if any(np.any(~np.isfinite(value)) for value in result.values()):
        raise OverflowError("A lognormal moment exceeds float64 range.")

    return result
