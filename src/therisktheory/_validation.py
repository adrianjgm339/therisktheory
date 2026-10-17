"""Input and output checks shared by the package."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

# Signed integer, unsigned integer and real floating dtypes. Booleans,
# complex numbers, strings and objects are rejected.
_NUMERIC_KINDS = frozenset("iuf")


def as_real_array(
    value: npt.ArrayLike, name: str, *, allow_inf: bool = False
) -> npt.NDArray[np.float64]:
    """Returns ``value`` as a float64 array of real numbers.

    Args:
        value: Scalar or array-like of real numbers.
        name: Parameter name used in the error messages.
        allow_inf: Whether infinite values are accepted.

    Returns:
        The input as a float64 array (0-d for scalar inputs).

    Raises:
        TypeError: If ``value`` is not made of real numbers.
        ValueError: If any element is NaN, or infinite when ``allow_inf`` is
            false.
    """
    try:
        array = np.asarray(value)
    except ValueError as error:  # e.g. ragged nested lists
        raise TypeError(f"{name} must be a real number or array") from error
    if array.dtype.kind not in _NUMERIC_KINDS:
        raise TypeError(
            f"{name} must be a real number or array, got dtype {array.dtype}"
        )
    array = array.astype(np.float64, copy=False)
    if np.any(np.isnan(array)):
        raise ValueError(f"{name} must not be NaN")
    if not allow_inf and np.any(np.isinf(array)):
        raise ValueError(f"{name} must be finite")
    return array


def as_nonnegative_finite_array(
    value: npt.ArrayLike, name: str
) -> npt.NDArray[np.float64]:
    """Returns ``value`` as a float64 array after checking its domain.

    Args:
        value: Scalar or array-like of real numbers.
        name: Parameter name used in the error messages.

    Returns:
        The input as a float64 array (0-d for scalar inputs).

    Raises:
        TypeError: If ``value`` is not made of real numbers.
        ValueError: If any element is NaN, infinite or negative.
    """
    array = as_real_array(value, name)
    if np.any(array < 0):
        raise ValueError(f"{name} must be non-negative")
    return array


def as_probability_array(
    value: npt.ArrayLike, name: str
) -> npt.NDArray[np.float64]:
    """Returns ``value`` as a float64 array of probabilities in ``[0, 1]``.

    Args:
        value: Scalar or array-like of real numbers.
        name: Parameter name used in the error messages.

    Returns:
        The input as a float64 array (0-d for scalar inputs).

    Raises:
        TypeError: If ``value`` is not made of real numbers.
        ValueError: If any element is NaN or outside ``[0, 1]``.
    """
    array = as_real_array(value, name)
    if np.any((array < 0) | (array > 1)):
        raise ValueError(f"{name} must be between 0 and 1")
    return array


def check_broadcastable(**arrays: npt.NDArray[np.float64]) -> None:
    """Raises ValueError if the arrays cannot be broadcast together.

    Args:
        **arrays: Arrays keyed by parameter name.

    Raises:
        ValueError: If the shapes are incompatible.
    """
    try:
        np.broadcast_shapes(*(a.shape for a in arrays.values()))
    except ValueError as error:
        shapes = ", ".join(f"{k} {a.shape}" for k, a in arrays.items())
        raise ValueError(f"incompatible shapes: {shapes}") from error


def check_finite_result(
    result: npt.NDArray[np.float64], name: str
) -> npt.NDArray[np.float64]:
    """Raises OverflowError if a computed result is not finite.

    With finite, non-negative inputs a non-finite result can only come from
    floating-point overflow.

    Args:
        result: Computed values.
        name: Name of the computed quantity, used in the error message.

    Returns:
        ``result`` unchanged.

    Raises:
        OverflowError: If any element is NaN or infinite.
    """
    if not np.all(np.isfinite(result)):
        raise OverflowError(
            f"{name} overflowed: the result exceeds the float64 range"
        )
    return result
