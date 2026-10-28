"""Scaled Beta claim severity distribution.

Mathematical validation: experimental; peer review pending.

Let Y follow Beta(alpha, beta). Then X = maximum_loss * Y.
Reference: scipy.stats.beta and scipy.special.betainc.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.special import betainc, betaincinv, betaln


def _finite_real(
    value: npt.ArrayLike,
    name: str,
) -> npt.NDArray[np.float64]:
    """Convert finite, real numeric inputs to float64 arrays."""
    try:
        original = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain real numeric values.") from exc

    if original.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values.")

    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(original, dtype=np.float64)

    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")

    return result


def _parameters(
    alpha: npt.ArrayLike,
    beta: npt.ArrayLike,
    maximum_loss: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], ...]:
    """Validate both shapes and the finite positive maximum."""
    a = _finite_real(alpha, "alpha")
    b = _finite_real(beta, "beta")
    maximum = _finite_real(maximum_loss, "maximum_loss")

    if np.any(a <= 0):
        raise ValueError("alpha must be strictly positive.")

    if np.any(b <= 0):
        raise ValueError("beta must be strictly positive.")

    if np.any(maximum <= 0):
        raise ValueError("maximum_loss must be strictly positive.")

    return a, b, maximum


def _broadcast(
    *values: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], ...]:
    """Broadcast arrays with a descriptive error on mismatch."""
    try:
        return tuple(np.broadcast_arrays(*values))
    except ValueError as exc:
        raise ValueError("Input shapes are not broadcast-compatible.") from exc


def beta_severity_pdf(
    x: npt.ArrayLike,
    *,
    alpha: npt.ArrayLike,
    beta: npt.ArrayLike,
    maximum_loss: npt.ArrayLike = 1.0,
) -> npt.NDArray[np.float64]:
    """Evaluate the scaled Beta probability density.

    Parameters
    ----------
    x : array_like
        Finite real severity.
    alpha, beta : array_like
        Strictly positive finite shape parameters.
    maximum_loss : array_like, optional
        Strictly positive upper severity bound, default 1.

    Returns
    -------
    numpy.ndarray
        Broadcast float64 density, zero outside the support.
        Scalar inputs produce zero-dimensional arrays.

    Notes
    -----
    The support is [0, maximum_loss].
    Endpoint densities may legitimately be infinite when
    the corresponding shape parameter is below one.
    A finite density exceeding float64 raises OverflowError.
    """
    values = _finite_real(x, "x")
    a, b, maximum = _parameters(alpha, beta, maximum_loss)

    values, a, b, maximum = _broadcast(values, a, b, maximum)

    interior = (values > 0) & (values < maximum)
    left = values == 0
    right = values == maximum

    with np.errstate(
        over="ignore",
        under="ignore",
        invalid="ignore",
        divide="ignore",
    ):
        log_maximum = np.log(maximum)

        safe_x = np.where(interior, values, 1.0)
        safe_gap = np.where(interior, maximum - values, 1.0)

        log_pdf = (
            (a - 1.0) * (np.log(safe_x) - log_maximum)
            + (b - 1.0) * (np.log(safe_gap) - log_maximum)
            - betaln(a, b)
            - log_maximum
        )

        interior_pdf = np.exp(log_pdf)

        left_pdf = np.where(
            a < 1.0,
            np.inf,
            np.where(
                a == 1.0,
                np.exp(np.log(b) - log_maximum),
                0.0,
            ),
        )

        right_pdf = np.where(
            b < 1.0,
            np.inf,
            np.where(
                b == 1.0,
                np.exp(np.log(a) - log_maximum),
                0.0,
            ),
        )

        density = np.where(
            interior,
            interior_pdf,
            np.where(
                left,
                left_pdf,
                np.where(right, right_pdf, 0.0),
            ),
        )

    singular = (left & (a < 1.0)) | (right & (b < 1.0))

    if np.any(~np.isfinite(density) & ~singular):
        raise OverflowError("A finite Beta density exceeds float64 range.")

    return np.asarray(density, dtype=np.float64)


def beta_severity_cdf(
    x: npt.ArrayLike,
    *,
    alpha: npt.ArrayLike,
    beta: npt.ArrayLike,
    maximum_loss: npt.ArrayLike = 1.0,
) -> npt.NDArray[np.float64]:
    """Evaluate the scaled Beta cumulative distribution.

    Parameters
    ----------
    x : array_like
        Finite severity; values outside support are allowed.
    alpha, beta : array_like
        Strictly positive finite shape parameters.
    maximum_loss : array_like, optional
        Positive upper bound, default 1.

    Returns
    -------
    numpy.ndarray
        Float64 probabilities between zero and one.

    Raises
    ------
    FloatingPointError
        Numerical evaluation of the incomplete Beta fails.
    """
    values = _finite_real(x, "x")
    a, b, maximum = _parameters(alpha, beta, maximum_loss)

    values, a, b, maximum = _broadcast(values, a, b, maximum)

    with np.errstate(
        over="ignore",
        invalid="ignore",
        divide="ignore",
    ):
        y = np.clip(values / maximum, 0.0, 1.0)

        probabilities = np.asarray(
            betainc(a, b, y),
            dtype=np.float64,
        )

    if np.any(~np.isfinite(probabilities)):
        raise FloatingPointError("Beta CDF evaluation failed numerically.")

    return probabilities


def beta_severity_ppf(
    q: npt.ArrayLike,
    *,
    alpha: npt.ArrayLike,
    beta: npt.ArrayLike,
    maximum_loss: npt.ArrayLike = 1.0,
) -> npt.NDArray[np.float64]:
    """Evaluate scaled Beta quantiles.

    Parameters
    ----------
    q : array_like
        Finite probabilities in [0, 1].
    alpha, beta : array_like
        Strictly positive finite shape parameters.
    maximum_loss : array_like, optional
        Positive upper severity bound, default 1.

    Returns
    -------
    numpy.ndarray
        Quantiles between zero and maximum_loss.
        q=0 gives zero; q=1 gives maximum_loss.

    Raises
    ------
    ValueError
        Probabilities outside [0, 1].
    FloatingPointError
        Numerical inverse Beta evaluation fails.
    """
    probabilities = _finite_real(q, "q")

    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("q must lie in [0, 1].")

    a, b, maximum = _parameters(alpha, beta, maximum_loss)

    probabilities, a, b, maximum = _broadcast(probabilities, a, b, maximum)

    with np.errstate(
        over="ignore",
        under="ignore",
        invalid="ignore",
    ):
        fraction = betaincinv(a, b, probabilities)

        quantile = maximum * fraction

    if np.any(~np.isfinite(quantile)):
        raise FloatingPointError("Beta quantile evaluation failed numerically.")

    return np.asarray(quantile, dtype=np.float64)


def beta_severity_moments(
    *,
    alpha: npt.ArrayLike,
    beta: npt.ArrayLike,
    maximum_loss: npt.ArrayLike = 1.0,
) -> dict[str, npt.NDArray[np.float64]]:
    """Calculate mean, variance and second raw moment.

    Parameters
    ----------
    alpha, beta : array_like
        Strictly positive finite shape parameters.
    maximum_loss : array_like, optional
        Strictly positive upper severity bound.

    Returns
    -------
    dict[str, numpy.ndarray]
        Keys: mean, variance, second_raw_moment.
        Values are broadcast float64 NumPy arrays.

    Notes
    -----
    All three moments are mathematically finite.
    Log-domain calculations reduce intermediate overflow
    for large scales and large shape parameters.
    Unrepresentable finite moments raise OverflowError.
    """
    a, b, maximum = _parameters(alpha, beta, maximum_loss)

    a, b, maximum = _broadcast(a, b, maximum)

    with np.errstate(
        over="ignore",
        under="ignore",
        invalid="ignore",
    ):
        log_a = np.log(a)
        log_b = np.log(b)
        log_m = np.log(maximum)

        log_total = np.logaddexp(log_a, log_b)

        log_total_plus_one = np.logaddexp(log_total, 0.0)

        log_a_plus_one = np.logaddexp(log_a, 0.0)

        logs = {
            "mean": (log_m + log_a - log_total),
            "variance": (2.0 * log_m + log_a + log_b - 2.0 * log_total - log_total_plus_one),
            "second_raw_moment": (
                2.0 * log_m + log_a + log_a_plus_one - log_total - log_total_plus_one
            ),
        }

        results = {
            key: np.asarray(
                np.exp(value),
                dtype=np.float64,
            )
            for key, value in logs.items()
        }

    if any(np.any(~np.isfinite(value)) for value in results.values()):
        raise OverflowError("A finite Beta moment exceeds float64 range.")

    return results
