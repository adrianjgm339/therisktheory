"""Restricted safety measures for the unit-step ruin model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from therisktheory.finetti_ruin.adjustment import (
    unit_step_adjustment_coefficient,
)
from therisktheory.finetti_ruin.round_loading import round_safety_loading

if TYPE_CHECKING:
    import numpy.typing as npt


def safety_index(
    win_probability: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Calculate the safety index in a favorable unit-step model.

    For independent increments of +1 with probability p and -1
    with probability q = 1-p, this is the adjustment coefficient:

        R = log(p/q).

    Mathematical conditions:
        The supported domain is 0.5 <= p < 1.
        At p = 0.5, R = 0 is a boundary value.
        A positive finite index exists for 0.5 < p < 1.

    Args:
        win_probability: Winning probability in the unit-step game.

    Returns:
        Non-negative float64 NumPy array, with scalar inputs
        represented as zero-dimensional arrays.

    Raises:
        TypeError: If the input is not real numeric data.
        ValueError: If the probability is outside the domain.
        OverflowError: If the computed value is non-finite.

    Examples:
        >>> round(float(safety_index(0.75)), 6)
        1.098612

    Note:
        Validation status: experimental.
        This is not a general safety-index solver for arbitrary
        claim-size distributions or stochastic reserve models.
    """
    return unit_step_adjustment_coefficient(win_probability)


def de_finetti_safety_loading(
    win_probability: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Calculate expected surplus gain per unit-step round.

    For G = +1 with probability p and G = -1 with probability
    q = 1-p, the expected surplus increment is:

        E[G] = p-q = 2p-1.

    Mathematical conditions:
        The supported domain is 0.5 <= p <= 1.
        Independence of repeated increments is assumed when
        applying this value to a reserve process.

    Args:
        win_probability: Winning probability in the unit-step game.

    Returns:
        Non-negative expected surplus increment as a float64
        NumPy array.

    Raises:
        TypeError: If the input is not real numeric data.
        ValueError: If the probability is outside the domain.

    Examples:
        >>> de_finetti_safety_loading(0.75)
        array(0.5)

    Note:
        Validation status: experimental.
        This is a unit-step surplus drift, not an insurance
        premium loading percentage or a general De Finetti
        insurance premium calculation.
    """
    return round_safety_loading(win_probability)
