"""Safety loading for favorable unit-step games."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from therisktheory import _validation

if TYPE_CHECKING:
    import numpy.typing as npt


def round_safety_loading(
    win_probability: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Calculate the non-negative expected gain per unit-bet round.

    A player gains one unit with probability p and loses one
    unit with probability q = 1 - p. The expected gain is:

        E[G] = p - q = 2*p - 1.

    This function covers fair and favorable games only.

    Args:
        win_probability: Probability p of winning, with
            0.5 <= p <= 1.

    Returns:
        Expected gain per round as a float64 NumPy array.
        Scalar inputs produce zero-dimensional arrays.

    Raises:
        TypeError: If the input is not numeric.
        ValueError: If the probability is non-finite or
            outside the permitted domain.

    Examples:
        >>> round_safety_loading(0.5)
        array(0.)
        >>> round_safety_loading(0.75)
        array(0.5)

    Note:
        Validation status: experimental.
        This is a unit-step expected gain, not an insurance
        premium loading coefficient.

    Reference:
        Kaas et al. (2008), Section 4.5.
    """
    p = _validation.as_probability_array(win_probability, "win_probability")

    if np.any(p < 0.5):
        raise ValueError(
            "win_probability must be at least 0.5 "
            "for non-negative round safety loading"
        )

    return np.asarray(p - (1.0 - p), dtype=np.float64)
