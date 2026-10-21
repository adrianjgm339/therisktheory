"""Adjustment coefficient for favorable unit-step risk games."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from therisktheory import _validation

if TYPE_CHECKING:
    import numpy.typing as npt


def unit_step_adjustment_coefficient(
    win_probability: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Calculate the adjustment coefficient for a unit-step game.

    A player gains one unit with probability p and loses one
    unit with probability q = 1 - p.

    The nontrivial root of

        p * exp(-R) + q * exp(R) = 1

    is R = log(p / q), positive when 0.5 < p < 1.

    The fair-game boundary p = 0.5 gives R = 0.
    The case p = 1 is excluded because the nontrivial-root
    expression has no finite limit there.

    Args:
        win_probability: Probability p, with 0.5 <= p < 1.

    Returns:
        Non-negative, finite adjustment coefficient as a
        float64 NumPy array. Scalars yield zero-dimensional
        arrays.

    Raises:
        TypeError: If the input is not real numeric data.
        ValueError: If the probability is invalid or outside
            the supported mathematical domain.
        OverflowError: If a computed result is not finite.

    Examples:
        >>> unit_step_adjustment_coefficient(0.5)
        array(0.)
        >>> round(float(unit_step_adjustment_coefficient(0.75)), 6)
        1.098612

    Note:
        Validation status: experimental.
        This coefficient applies only to unit-step games.
        R = 0 is a boundary value, not a positive
        adjustment coefficient.

    Reference:
        Kaas et al. (2008), Section 4.5, equation (4.34).
    """
    p = _validation.as_probability_array(win_probability, "win_probability")

    if np.any(p < 0.5):
        raise ValueError("win_probability must be at least 0.5")

    if np.any(p == 1.0):
        raise ValueError(
            "win_probability must be below 1 "
            "for a finite nontrivial-root coefficient"
        )

    # log(p) - log1p(-p) avoids forming p / (1-p).
    # This is numerically stable near the permitted boundaries.
    result = np.asarray(
        np.log(p) - np.log1p(-p),
        dtype=np.float64,
    )

    return _validation.check_finite_result(
        result, "unit_step_adjustment_coefficient"
    )
