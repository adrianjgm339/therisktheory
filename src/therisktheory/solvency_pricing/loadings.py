"""Safety loading calculations for actuarial premium principles."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from therisktheory import _validation
from therisktheory.solvency_pricing.premiums import expected_value_premium

if TYPE_CHECKING:
    import numpy.typing as npt


def safety_loading(
    expected_loss: npt.ArrayLike,
    premium: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Compute the monetary safety loading of an insurance premium.

    The safety loading is the excess of the charged premium over
    the expected aggregate loss:

        L = premium - E[S]

    Mathematical conditions:
        Both inputs must be finite and non-negative.
        The charged premium must not be below the expected loss.

    Args:
        expected_loss: Expected aggregate loss E[S].
        premium: Total premium charged by the insurer.

    Returns:
        Non-negative monetary loading as a float64 array.
        NumPy broadcasting rules apply.

    Raises:
        TypeError: If an input is not numeric.
        ValueError: If an input is outside the permitted domain
            or the input shapes are incompatible.

    Examples:
        >>> safety_loading(600.0, 660.0)
        array(60.)
        >>> safety_loading(600.0, 600.0)
        array(0.)

    Note:
        Validation status: experimental.
        This function returns a monetary amount, not the
        relative loading coefficient.
    """
    expected = _validation.as_nonnegative_finite_array(
        expected_loss, "expected_loss"
    )
    charged = _validation.as_nonnegative_finite_array(premium, "premium")

    _validation.check_broadcastable(
        expected_loss=expected,
        premium=charged,
    )

    if np.any(charged < expected):
        raise ValueError("premium must not be below expected_loss")

    result = np.asarray(charged - expected)
    return _validation.check_finite_result(result, "safety_loading")


def loaded_premium(
    expected_loss: npt.ArrayLike,
    loading: npt.ArrayLike,
    *,
    principle: Literal[
        "expected_value", "variance", "standard_deviation"
    ] = "expected_value",
    variance: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float64]:
    """Compute a loaded premium using a classical premium principle.

    Supported principles:
        expected_value: E[S] * (1 + theta).
        variance: E[S] + alpha * Var(S).
        standard_deviation: E[S] + alpha * sqrt(Var(S)).

    Mathematical conditions:
        Expected loss, loading and variance must be finite and
        non-negative. Variance is required for the variance and
        standard deviation principles.

        The moments must refer to the same aggregate loss S.
        Statistical consistency of supplied moments is assumed.

    Args:
        expected_loss: Expected aggregate loss E[S].
        loading: Non-negative principle coefficient.
        principle: Premium calculation principle.
        variance: Aggregate loss variance, when required.

    Returns:
        Loaded premium as a float64 NumPy array.
        Scalar inputs produce zero-dimensional arrays.

    Raises:
        TypeError: If an input is not numeric or principle is
            not a string.
        ValueError: If the principle or input domain is invalid,
            required variance is missing, or shapes conflict.
        OverflowError: If the calculated premium is not finite.

    Examples:
        >>> loaded_premium(600.0, 0.10)
        array(660.)
        >>> loaded_premium(
        ...     600.0, 0.0005,
        ...     principle="variance", variance=240000.0
        ... )
        array(720.)

    Note:
        Validation status: experimental.

    Reference:
        Kaas et al. (2008), Section 5.3, p. 119.
    """
    if not isinstance(principle, str):
        raise TypeError("principle must be a string")

    valid_principles = (
        "expected_value",
        "variance",
        "standard_deviation",
    )

    if principle not in valid_principles:
        raise ValueError(f"Unknown premium principle: {principle}")

    if principle == "expected_value":
        if variance is not None:
            raise ValueError(
                "variance is not used by the expected_value principle"
            )
        return expected_value_premium(expected_loss, loading)

    if variance is None:
        raise ValueError(f"variance is required for the {principle} principle")

    mean = _validation.as_nonnegative_finite_array(
        expected_loss, "expected_loss"
    )
    alpha = _validation.as_nonnegative_finite_array(loading, "loading")
    var = _validation.as_nonnegative_finite_array(variance, "variance")

    _validation.check_broadcastable(
        expected_loss=mean,
        loading=alpha,
        variance=var,
    )

    with np.errstate(over="ignore", invalid="ignore"):
        extra = alpha * var if principle == "variance" else alpha * np.sqrt(var)

        result = np.asarray(mean + extra, dtype=np.float64)

    return _validation.check_finite_result(result, "loaded_premium")


def premium_principle_checks(
    expected_loss: npt.ArrayLike,
    loading: npt.ArrayLike,
    *,
    principle: Literal[
        "expected_value", "variance", "standard_deviation"
    ] = "expected_value",
    variance: npt.ArrayLike | None = None,
    maximum_loss: npt.ArrayLike | None = None,
    cash_shift: npt.ArrayLike = 1.0,
) -> dict[str, npt.NDArray[np.bool_] | None]:
    """Evaluate premium-principle properties for specified risk scenarios.

    Translation consistency and independent-copy additivity are
    evaluated algebraically for the selected premium principle.
    The no-rip-off diagnostic compares monetary loading with the
    available margin up to the supplied maximum loss.
    Non-negative loading compares the computed premium and mean.
    These checks do not validate the supplied risk distribution.

    Independent additivity is checked using an independent copy of the
    supplied risk, with twice the mean and twice the variance.
    The supplied moments and maximum loss are assumed consistent.

    Args:
        expected_loss: Mean aggregate loss.
        loading: Non-negative coefficient of the selected principle.
        principle: Premium calculation principle.
        variance: Aggregate loss variance, when required.
        maximum_loss: Known essential supremum of the loss distribution.
        cash_shift: Strictly positive deterministic claim increment.

    Returns:
        A dictionary containing arrays of boolean diagnostics:
        non_negative_loading, no_rip_off, translation_consistency,
        and independent_additivity. The no_rip_off result is None
        when maximum_loss is not supplied.

    Raises:
        TypeError: If a supplied numeric input has an invalid type.
        ValueError: If parameters, shapes or bounds are invalid.
        OverflowError: If a calculated premium overflows.

    Note:
        Validation status: experimental.
        Algebraic property checks do not use floating-point
        closeness tolerances.
        Iterativity is not evaluated by this function.

    Reference:
        Kaas et al. (2008), Section 5.3.1, pp. 120-122.
    """
    base = loaded_premium(
        expected_loss,
        loading,
        principle=principle,
        variance=variance,
    )

    mean = _validation.as_nonnegative_finite_array(
        expected_loss, "expected_loss"
    )
    shift = _validation.as_nonnegative_finite_array(cash_shift, "cash_shift")

    if np.any(shift == 0):
        raise ValueError("cash_shift must be strictly positive")

    _validation.check_broadcastable(
        expected_loss=mean,
        premium=base,
        cash_shift=shift,
    )

    var = (
        None
        if variance is None
        else _validation.as_nonnegative_finite_array(variance, "variance")
    )

    maximum = None
    if maximum_loss is not None:
        maximum = _validation.as_nonnegative_finite_array(
            maximum_loss, "maximum_loss"
        )
        _validation.check_broadcastable(
            expected_loss=mean,
            premium=base,
            maximum_loss=maximum,
        )
        if np.any(maximum < mean):
            raise ValueError("maximum_loss must not be below expected_loss")

    theta = _validation.as_nonnegative_finite_array(loading, "loading")

    # All diagnostics must share compatible NumPy dimensions.
    arrays = {
        "expected_loss": mean,
        "premium": base,
        "loading": theta,
        "cash_shift": shift,
    }
    if var is not None:
        arrays["variance"] = var
    if maximum is not None:
        arrays["maximum_loss"] = maximum

    _validation.check_broadcastable(**arrays)

    shape = np.broadcast_shapes(*(array.shape for array in arrays.values()))

    mean_b = np.broadcast_to(mean, shape)
    premium_b = np.broadcast_to(base, shape)
    theta_b = np.broadcast_to(theta, shape)

    # For a positive shift c:
    # Expected value: pi(S+c) - pi(S) - c = theta*c.
    # Variance and standard deviation: translation invariant.
    translation_consistency = (
        theta_b == 0
        if principle == "expected_value"
        else np.ones(shape, dtype=bool)
    )

    # For two independent, identically distributed risks:
    # E[S+S'] = 2*E[S] and Var(S+S') = 2*Var(S).
    # Standard deviation loading is additive only if
    # alpha = 0 or Var(S) = 0.
    independent_additivity = (
        (theta_b == 0) | (np.broadcast_to(var, shape) == 0)
        if principle == "standard_deviation"
        else np.ones(shape, dtype=bool)
    )

    # Compare the monetary loading with the available margin.
    # Prevent rounding in the computed premium from hiding
    # a positive loading when no margin is available.
    no_rip_off = None
    if maximum is not None:
        allowance = np.broadcast_to(maximum, shape) - mean_b
        risk_measure = mean_b

        if principle == "variance":
            risk_measure = np.broadcast_to(var, shape)
        elif principle == "standard_deviation":
            risk_measure = np.sqrt(np.broadcast_to(var, shape))

        with np.errstate(over="ignore", under="ignore"):
            monetary_loading = theta_b * risk_measure

        no_rip_off = np.asarray(
            (monetary_loading <= allowance)
            & ((allowance > 0) | (theta_b == 0) | (risk_measure == 0))
        )

    return {
        "non_negative_loading": np.asarray(premium_b >= mean_b),
        "no_rip_off": no_rip_off,
        "translation_consistency": np.asarray(translation_consistency),
        "independent_additivity": np.asarray(independent_additivity),
    }
