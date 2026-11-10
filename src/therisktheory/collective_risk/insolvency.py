"""Finite-horizon insolvency observed at discrete period ends.

The underlying surplus is U(t) = u + c*t - S(t), but this module only
observes U(0), U(1), ..., U(T). A negative surplus between observation times
is not detected unless it is still negative at a period end.

Reference:
    Kaas, R., Goovaerts, M., Dhaene, J., and Denuit, M. (2008).
    Modern Actuarial Risk Theory: Using R, 2nd ed., Chapter 4.
    https://doi.org/10.1007/978-3-540-70998-5
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np

from therisktheory import _validation
from therisktheory.collective_risk.process import (
    CollectiveRiskProcess,
    simulate_surplus,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt


@dataclasses.dataclass(frozen=True, slots=True)
class FiniteHorizonInsolvency:
    """Empirical insolvency probabilities over the observed periods.

    Attributes:
        cumulative_probability: Fraction of paths with U(j) < 0 for any
            observed period 1 <= j <= k, for each k = 1, ..., T.
        terminal_probability: Fraction of paths with U(k) < 0 at each k.
        first_insolvency_probability: Fraction first observed negative at k.
        cumulative_standard_error: Plug-in Bernoulli standard errors for
            cumulative_probability, not a confidence interval.
        terminal_standard_error: Plug-in Bernoulli standard errors for
            terminal_probability, not a confidence interval.
        n_paths: Number of independent simulated or observed paths.
        n_periods: Number T of observed periods after the initial capital.
    """

    cumulative_probability: npt.NDArray[np.float64]
    terminal_probability: npt.NDArray[np.float64]
    first_insolvency_probability: npt.NDArray[np.float64]
    cumulative_standard_error: npt.NDArray[np.float64]
    terminal_standard_error: npt.NDArray[np.float64]
    n_paths: int
    n_periods: int


def finite_horizon_insolvency(
    surplus_paths: npt.ArrayLike,
) -> FiniteHorizonInsolvency:
    """Estimate insolvency at and through each observed period end.

    For independent paths U_i(0), ..., U_i(T), the cumulative estimator
    averages indicators of min(U_i(1), ..., U_i(k)) < 0. The terminal
    estimator averages indicators of U_i(k) < 0. The first-insolvency
    probability is the increment of the cumulative probability.

    Only strictly negative surplus counts as insolvency; zero is solvent.
    All paths must share the same non-negative initial capital.

    Args:
        surplus_paths: Finite real matrix of shape (n_paths, T + 1),
            with n_paths >= 1 and T >= 1; column 0 holds initial capital.

    Returns:
        Frozen result holding independent, read-only probability arrays.

    Raises:
        TypeError: If entries are not real numbers.
        ValueError: If dimensions, finite values or initial capital are invalid.

    Examples:
        >>> paths = [[10, 5, -1, 2], [10, 9, 4, 3],
        ...          [10, -2, -4, -1], [10, 3, 0, 1]]
        >>> result = finite_horizon_insolvency(paths)
        >>> result.cumulative_probability.tolist()
        [0.25, 0.5, 0.5]
        >>> result.terminal_probability.tolist()
        [0.25, 0.5, 0.25]
    """
    values = _validation.as_real_array(surplus_paths, "surplus_paths")
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError("surplus_paths must have shape (n_paths, T + 1)")
    if np.any(values[:, 0] < 0):
        raise ValueError("initial capital must be non-negative")
    if np.any(values[:, 0] != values[0, 0]):
        raise ValueError("all paths must share the same initial capital")

    n_paths, columns = values.shape
    negative = values[:, 1:] < 0
    by_period = np.logical_or.accumulate(negative, axis=1)
    cumulative = by_period.mean(axis=0)
    terminal = negative.mean(axis=0)
    first = cumulative.copy()
    first[1:] -= cumulative[:-1]

    cum_se = np.sqrt(cumulative * (1.0 - cumulative) / n_paths)
    term_se = np.sqrt(terminal * (1.0 - terminal) / n_paths)

    return FiniteHorizonInsolvency(
        cumulative_probability=_read_only(cumulative),
        terminal_probability=_read_only(terminal),
        first_insolvency_probability=_read_only(first),
        cumulative_standard_error=_read_only(cum_se),
        terminal_standard_error=_read_only(term_se),
        n_paths=n_paths,
        n_periods=columns - 1,
    )


def simulate_finite_horizon_insolvency(
    process: CollectiveRiskProcess,
    *,
    severity_sampler: Callable[
        [int, np.random.Generator], npt.NDArray[np.float64]
    ],
    n_periods: int,
    n_paths: int,
    rng: np.random.Generator | int | None = None,
) -> FiniteHorizonInsolvency:
    """Simulate discrete-end surplus and estimate insolvency probabilities.

    Delegates the simulation to Bladimir's ``simulate_surplus`` so the
    compound Poisson process and severity sampling are not duplicated.

    Args:
        process: Valid classical compound Poisson risk process.
        severity_sampler: Callable drawing non-negative claim severities.
        n_periods: Positive integer number of observed periods.
        n_paths: Positive integer number of independent paths.
        rng: NumPy random generator or seed.

    Returns:
        Probabilities and plug-in Monte Carlo standard errors by period.

    Raises:
        TypeError: If the process or sampler has an invalid type.
        ValueError: If the simulation arguments are outside their domains.
    """
    if not isinstance(process, CollectiveRiskProcess):
        raise TypeError("process must be a CollectiveRiskProcess")
    if not callable(severity_sampler):
        raise TypeError("severity_sampler must be callable")
    paths = simulate_surplus(
        process,
        severity_sampler=severity_sampler,
        n_periods=n_periods,
        n_paths=n_paths,
        rng=rng,
    )
    return finite_horizon_insolvency(paths)


def _read_only(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return a detached, read-only float64 probability vector."""
    encoded = np.asarray(values, dtype=np.float64).tobytes()
    return np.frombuffer(encoded, dtype=np.float64)
