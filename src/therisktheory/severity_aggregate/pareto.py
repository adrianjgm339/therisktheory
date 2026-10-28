"""Pareto Type I claim severity.

Mathematical status: experimental; independent review pending.
The shape parameter is alpha and support begins at minimum.
See scipy.stats.pareto (b=alpha, loc=0, scale=minimum).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def _finite_real(value: npt.ArrayLike, name: str) -> npt.NDArray[np.float64]:
    """Return a float64 array from finite real numeric input."""
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain real numeric values.") from exc

    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values.")

    with np.errstate(over="ignore", invalid="ignore"):
        values = np.asarray(raw, dtype=np.float64)

    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")

    return values


def _parameters(
    alpha: npt.ArrayLike,
    minimum: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Validate shape and minimum severity."""
    shape = _finite_real(alpha, "alpha")
    lower = _finite_real(minimum, "minimum")

    if np.any(shape <= 0.0):
        raise ValueError("alpha must be strictly positive.")

    if np.any(lower <= 0.0):
        raise ValueError("minimum must be strictly positive.")

    return shape, lower


def _broadcast(
    *values: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], ...]:
    """Broadcast input arrays or raise a descriptive error."""
    try:
        return tuple(np.broadcast_arrays(*values))
    except ValueError as exc:
        raise ValueError("Input shapes are not broadcast-compatible.") from exc


def _support_log_ratio(
    x: npt.NDArray[np.float64],
    minimum: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute log(x/minimum) accurately near the support edge."""
    safe_x = np.where(x >= minimum, x, minimum)

    with np.errstate(
        divide="ignore",
        over="ignore",
        invalid="ignore",
    ):
        return np.log1p((safe_x - minimum) / minimum)


def pareto_severity_pdf(
    x: npt.ArrayLike,
    *,
    alpha: npt.ArrayLike,
    minimum: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Evaluate the Pareto Type I probability density.

    Parameters
    ----------
    x : array_like
        Finite real severity; density is zero below minimum.
    alpha : array_like
        Finite strictly positive shape.
    minimum : array_like
        Finite strictly positive minimum severity.

    Returns
    -------
    numpy.ndarray
        Broadcast float64 density; scalar results are 0-D.

    Raises
    ------
    TypeError
        An input contains non-real or nonnumeric values.
    ValueError
        Invalid parameters or incompatible shapes.
    OverflowError
        A finite density exceeds float64 range.
    """
    values = _finite_real(x, "x")
    shape, lower = _parameters(alpha, minimum)
    values, shape, lower = _broadcast(values, shape, lower)

    log_ratio = _support_log_ratio(values, lower)

    with np.errstate(
        over="ignore",
        under="ignore",
        invalid="ignore",
    ):
        log_pdf = np.log(shape) - np.log(lower) - (shape + 1.0) * log_ratio

        pdf = np.where(
            values >= lower,
            np.exp(log_pdf),
            0.0,
        )

    if np.any(~np.isfinite(pdf)):
        raise OverflowError("A finite Pareto density exceeds float64 range.")

    return np.asarray(pdf, dtype=np.float64)


def pareto_severity_cdf(
    x: npt.ArrayLike,
    *,
    alpha: npt.ArrayLike,
    minimum: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Evaluate Pareto Type I cumulative probabilities.

    Parameters
    ----------
    x : array_like
        Finite severity; CDF is zero at or below minimum.
    alpha : array_like
        Strictly positive shape.
    minimum : array_like
        Strictly positive minimum severity.

    Returns
    -------
    numpy.ndarray
        Broadcast float64 cumulative probabilities.
    """
    values = _finite_real(x, "x")
    shape, lower = _parameters(alpha, minimum)
    values, shape, lower = _broadcast(values, shape, lower)

    log_ratio = _support_log_ratio(values, lower)

    with np.errstate(
        over="ignore",
        under="ignore",
        invalid="ignore",
    ):
        cdf = np.where(
            values > lower,
            -np.expm1(-shape * log_ratio),
            0.0,
        )

    return np.asarray(cdf, dtype=np.float64)


def pareto_severity_ppf(
    q: npt.ArrayLike,
    *,
    alpha: npt.ArrayLike,
    minimum: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Evaluate Pareto Type I quantiles.

    Parameters
    ----------
    q : array_like
        Finite probabilities in [0, 1].
    alpha : array_like
        Strictly positive shape.
    minimum : array_like
        Strictly positive minimum severity.

    Returns
    -------
    numpy.ndarray
        Float64 quantiles: q=0 gives minimum;
        q=1 gives positive infinity.

    Raises
    ------
    ValueError
        Probabilities outside [0, 1].
    OverflowError
        A finite quantile exceeds float64 range.
    """
    probabilities = _finite_real(q, "q")

    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("q must lie in [0, 1].")

    shape, lower = _parameters(alpha, minimum)

    probabilities, shape, lower = _broadcast(probabilities, shape, lower)

    with np.errstate(
        divide="ignore",
        over="ignore",
        under="ignore",
        invalid="ignore",
    ):
        log_q = np.log(lower) - np.log1p(-probabilities) / shape

        quantile = np.where(
            probabilities == 0.0,
            lower,
            np.exp(log_q),
        )

    if np.any(~np.isfinite(quantile) & (probabilities != 1.0)):
        raise OverflowError("A finite Pareto quantile exceeds float64 range.")

    return np.asarray(quantile, dtype=np.float64)


def pareto_severity_moments(
    *,
    alpha: npt.ArrayLike,
    minimum: npt.ArrayLike,
) -> dict[str, npt.NDArray[np.float64]]:
    """Return Pareto mean, variance, and second raw moment.

    Parameters
    ----------
    alpha : array_like
        Strictly positive Pareto shape.
    minimum : array_like
        Strictly positive minimum severity.

    Returns
    -------
    dict[str, numpy.ndarray]
        Arrays named mean, variance, second_raw_moment.

        Mean is infinite for alpha <= 1.
        Variance and second raw moment are infinite
        for alpha <= 2.

    Raises
    ------
    OverflowError
        A mathematically finite moment exceeds float64 range.
    """
    shape, lower = _parameters(alpha, minimum)
    shape, lower = _broadcast(shape, lower)

    valid_mean = shape > 1.0
    valid_second = shape > 2.0

    a_minus_1 = np.where(valid_mean, shape - 1.0, 1.0)
    a_minus_2 = np.where(valid_second, shape - 2.0, 1.0)

    with np.errstate(
        over="ignore",
        under="ignore",
        invalid="ignore",
    ):
        log_shape = np.log(shape)
        log_minimum = np.log(lower)
        log_a1 = np.log(a_minus_1)
        log_a2 = np.log(a_minus_2)

        mean = np.where(
            valid_mean,
            np.exp(log_shape + log_minimum - log_a1),
            np.inf,
        )

        second = np.where(
            valid_second,
            np.exp(log_shape + 2.0 * log_minimum - log_a2),
            np.inf,
        )

        variance = np.where(
            valid_second,
            np.exp(log_shape + 2.0 * log_minimum - 2.0 * log_a1 - log_a2),
            np.inf,
        )

    for value, defined in (
        (mean, valid_mean),
        (second, valid_second),
        (variance, valid_second),
    ):
        if np.any(~np.isfinite(value) & defined):
            raise OverflowError("A finite Pareto moment exceeds float64 range.")

    return {
        "mean": np.asarray(mean, dtype=np.float64),
        "variance": np.asarray(variance, dtype=np.float64),
        "second_raw_moment": np.asarray(second, dtype=np.float64),
    }
