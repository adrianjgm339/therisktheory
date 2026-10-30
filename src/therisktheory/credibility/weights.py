"""Buhlmann credibility weights and balanced-data component estimators.

Reference:
    Loss Data Analytics, 2nd ed., Chapter 12, Sections 12.2 and 12.5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

_REAL_KINDS = frozenset("iuf")
_COUNT_KINDS = frozenset("iu")


def _real_array(value: npt.ArrayLike, name: str) -> npt.NDArray[np.float64]:
    """Validate and convert finite real numeric inputs."""
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must contain real numbers") from error

    if raw.dtype.kind not in _REAL_KINDS:
        raise TypeError(f"{name} must contain real numbers")

    with np.errstate(over="ignore", invalid="ignore"):
        array = raw.astype(np.float64, copy=False)

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite within float64 range")

    return array


def _balanced_observations(
    observations: npt.ArrayLike, *, minimum_risks: int
) -> npt.NDArray[np.float64]:
    """Validate equal-period observations with risks in rows."""
    array = _real_array(observations, "observations")

    if array.ndim != 2:
        raise ValueError("observations must be a two-dimensional matrix")

    if array.shape[0] < minimum_risks:
        raise ValueError(f"observations require at least {minimum_risks} risks")

    if array.shape[1] < 2:
        raise ValueError("observations require at least two periods")

    return array


def _finite_computation(
    value: npt.ArrayLike, name: str
) -> npt.NDArray[np.float64]:
    """Reject nonfinite numerical results after a computation."""
    result = np.asarray(value, dtype=np.float64)

    if not np.all(np.isfinite(result)):
        raise OverflowError(f"{name} exceeded the float64 range")

    return result


def credibility_weight(
    n: npt.ArrayLike,
    epv: npt.ArrayLike,
    vhm: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Compute the classical Buhlmann credibility weight.

    Computes Z = n * VHM / (n * VHM + EPV) using scaled arithmetic.

    Args:
        n: Non-negative integer number of periods, scalar or array.
        epv: Non-negative finite expected process variance.
        vhm: Non-negative finite variance of hypothetical means.

    Returns:
        Float64 array of weights in [0, 1], 0-D for scalar inputs.
        Zero periods yield zero weight by convention. If EPV is zero
        and VHM is positive, positive periods yield weight one.
        If VHM is zero and EPV is positive, the weight is zero.

    Raises:
        TypeError: If n is not an integer dtype, or variance inputs
            are not real numeric values.
        ValueError: If inputs are negative, nonfinite, incompatible
            in shape, or both variance components are zero.

    Notes:
        A zero EPV and zero VHM define an indeterminate model and
        are rejected, including when n is zero. This API uses the
        balanced Buhlmann model, not Buhlmann-Straub exposure.
        Validation status: experimental.
    """
    try:
        raw_n = np.asarray(n)
    except (TypeError, ValueError) as error:
        raise TypeError("n must contain non-negative integers") from error

    if raw_n.dtype.kind not in _COUNT_KINDS:
        raise TypeError("n must contain non-negative integers")

    if np.any(raw_n < 0):
        raise ValueError("n must be non-negative")

    count = _real_array(raw_n, "n")
    process = _real_array(epv, "epv")
    hypothetical = _real_array(vhm, "vhm")

    if np.any(process < 0) or np.any(hypothetical < 0):
        raise ValueError("epv and vhm must be non-negative")

    try:
        count, process, hypothetical = np.broadcast_arrays(
            count, process, hypothetical
        )
    except ValueError as error:
        raise ValueError("n, epv, and vhm have incompatible shapes") from error

    if np.any((process == 0) & (hypothetical == 0)):
        raise ValueError("epv and vhm cannot both be zero")

    # For n > 0, Z = VHM / (VHM + EPV/n).
    # Scale both terms before adding to avoid overflow.
    with np.errstate(divide="ignore", invalid="ignore", under="ignore"):
        per_period = np.divide(
            process,
            count,
            out=np.zeros(count.shape, dtype=np.float64),
            where=count > 0,
        )

        scale = np.maximum(hypothetical, per_period)

        a = np.divide(
            hypothetical,
            scale,
            out=np.zeros(count.shape, dtype=np.float64),
            where=scale > 0,
        )

        b = np.divide(
            per_period,
            scale,
            out=np.zeros(count.shape, dtype=np.float64),
            where=scale > 0,
        )

        denominator = a + b

        result = np.divide(
            a,
            denominator,
            out=np.zeros(count.shape, dtype=np.float64),
            where=denominator > 0,
        )

    return np.asarray(np.where(count == 0, 0.0, result))


def estimate_process_variance(
    observations: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Estimate EPV from balanced Buhlmann observations.

    Computes the average within-risk sample variance. Each risk is
    a row and each period is a column. Sample variances use ddof=1.

    Args:
        observations: Finite real matrix containing at least one risk
            and two periods. Signed observations are permitted.

    Returns:
        Zero-dimensional float64 array containing estimated EPV.

    Raises:
        TypeError: If observations are not real numeric values.
        ValueError: If dimensions or values are invalid.
        OverflowError: If the estimator exceeds float64 range.

    Notes:
        Assumes equal exposure and conditionally iid observations
        within risks. Validation status: experimental.
    """
    data = _balanced_observations(observations, minimum_risks=1)

    with np.errstate(over="ignore", invalid="ignore"):
        result = np.var(data, axis=1, ddof=1).mean()

    return _finite_computation(result, "EPV")


def estimate_hypothetical_mean_variance(
    observations: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Estimate VHM from balanced Buhlmann observations.

    Computes the between-risk sample variance of individual means,
    minus the estimated EPV divided by the number of periods.

    Args:
        observations: Finite real matrix containing at least two
            risks and at least two equal-exposure periods.

    Returns:
        Zero-dimensional float64 array with the signed VHM estimate.
        A negative estimate is retained rather than silently clipped.

    Raises:
        TypeError: If observations are not real numeric values.
        ValueError: If dimensions or values are invalid.
        OverflowError: If the estimator exceeds float64 range.

    Notes:
        This is the untruncated method-of-moments estimator.
        Negative estimates cannot directly enter credibility_weight.
        Validation status: experimental.
    """
    data = _balanced_observations(observations, minimum_risks=2)
    n = data.shape[1]

    with np.errstate(over="ignore", invalid="ignore"):
        risk_means = data.mean(axis=1)
        between = np.var(risk_means, ddof=1)
        epv = np.var(data, axis=1, ddof=1).mean()
        result = between - epv / n

    return _finite_computation(result, "VHM")
