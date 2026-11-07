"""The collective risk process and the generating function of the annual result.

Collective risk theory follows the capital of an insurer over time instead of
a single period. The surplus process (Kaas et al., 2008, eq. (4.1), p. 89) is

    U(t) = u + c t - S(t),   S(t) = X_1 + ... + X_{N(t)},

with initial capital ``u``, premium income ``c`` per unit of time, claims
``X_i >= 0`` and a claim number process ``N(t)``. In the classical model
``N(t)`` is a Poisson process with intensity ``lambda`` (Definition 4.2.1,
p. 90), so ``S(t)`` is compound Poisson, and the premium contains a safety
loading ``theta`` with ``c = (1 + theta) lambda mu_1`` (eq. (4.9), p. 91). The
main objective is to measure how stable the insurer is, through the
probability ``psi(u)`` that ``U(t)`` ever becomes negative (eq. (4.3), p. 90).

The result of a period of length ``t`` is ``G(t) = c t - S(t)``. Its
generating function

    E[exp(r G(t))] = exp(r c t) E[exp(-r S(t))] = exp(r c t) P_N(m_X(-r))

is the object behind the adjustment coefficient, which solves
``E[exp(-R G(1))] = 1`` (eq. (4.11), p. 91).

This module builds the process once (``collective_risk_process``) so that the
other Topic 5 functions (insolvency over several periods, ruin probabilities,
Lundberg's equation, reinsurance effects) can work on the same object. It
reuses Topic 1 (``expected_aggregate_loss``) and Topic 3 (the compound
Poisson cumulants, the Poisson pgf and the compound Poisson simulation).

Reference:
    Kaas, R., Goovaerts, M., Dhaene, J. and Denuit, M. (2008). *Modern
    Actuarial Risk Theory: Using R*, 2nd ed. Springer. Chapter 4.
    https://doi.org/10.1007/978-3-540-70998-5
"""

from __future__ import annotations

import dataclasses
import operator
from typing import TYPE_CHECKING

import numpy as np

from therisktheory import _validation, frequency
from therisktheory.solvency_pricing import premiums

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numpy.typing as npt

# Relative tolerance for the moment inequalities checked on construction.
_MOMENT_TOLERANCE = 1e-12


@dataclasses.dataclass(frozen=True, slots=True)
class CollectiveRiskProcess:
    """Classical compound Poisson surplus process ``U(t) = u + c t - S(t)``.

    Build it with :func:`collective_risk_process`, which validates the
    parameters.

    Attributes:
        initial_capital: ``u = U(0) >= 0``.
        claim_rate: Poisson intensity ``lambda > 0`` (claims per unit time).
        severity_moments: Raw moments ``(mu_1, mu_2, ...)`` of the claims,
            at least two.
        safety_loading: ``theta``, with ``c = (1 + theta) lambda mu_1``.
        premium_rate: ``c``, the premium income per unit of time.
    """

    initial_capital: float
    claim_rate: float
    severity_moments: tuple[float, ...]
    safety_loading: float
    premium_rate: float

    @property
    def expected_claims_per_unit_time(self) -> float:
        """Returns ``lambda mu_1``, the expected claims per unit of time."""
        return self.claim_rate * self.severity_moments[0]

    def annual_result_moments(
        self, time: npt.ArrayLike = 1.0
    ) -> dict[str, npt.NDArray[np.float64]]:
        """Returns the moments of the result ``G(t) = c t - S(t)``.

        With the compound Poisson cumulants ``kappa_j = lambda t mu_j``:
        ``E[G(t)] = (c - lambda mu_1) t = theta lambda mu_1 t``,
        ``Var[G(t)] = lambda t mu_2`` and, if ``mu_3`` is known,
        ``E[(G(t) - E G(t))**3] = -lambda t mu_3``.

        Args:
            time: Length ``t >= 0`` of the period; arrays are accepted.

        Returns:
            A dict with ``mean`` and ``variance``, and
            ``third_central_moment`` when three claim moments were given.

        Raises:
            TypeError: If ``time`` is not made of real numbers.
            ValueError: If ``time`` is negative, NaN or infinite.
            OverflowError: If a moment exceeds the float64 range.
        """
        t = _validation.as_nonnegative_finite_array(time, "time")
        kappa = frequency.compound_poisson_cumulants(
            rate=self.claim_rate,
            severity_raw_moments=self.severity_moments[:3],
            time=t,
        )
        with np.errstate(over="ignore", invalid="ignore"):
            mean = np.asarray(self.premium_rate * t - kappa[..., 0])
        result = {
            "mean": _validation.check_finite_result(mean, "mean"),
            "variance": np.asarray(kappa[..., 1], dtype=np.float64),
        }
        if len(self.severity_moments) >= 3:
            result["third_central_moment"] = np.asarray(
                -kappa[..., 2], dtype=np.float64
            )
        return result

    def surplus_moments(
        self, time: npt.ArrayLike
    ) -> dict[str, npt.NDArray[np.float64]]:
        """Returns the moments of the surplus ``U(t) = u + G(t)``.

        Args:
            time: Time ``t >= 0``; arrays are accepted.

        Returns:
            The moments of :meth:`annual_result_moments` with the mean
            shifted by ``u``.

        Raises:
            TypeError: If ``time`` is not made of real numbers.
            ValueError: If ``time`` is negative, NaN or infinite.
            OverflowError: If a moment exceeds the float64 range.
        """
        moments = self.annual_result_moments(time)
        with np.errstate(over="ignore"):
            mean = np.asarray(self.initial_capital + moments["mean"])
        moments["mean"] = _validation.check_finite_result(mean, "mean")
        return moments


def collective_risk_process(
    *,
    initial_capital: float,
    claim_rate: float,
    severity_moments: Sequence[float] | npt.ArrayLike,
    safety_loading: float | None = None,
    premium_rate: float | None = None,
) -> CollectiveRiskProcess:
    """Builds and validates a classical collective risk process.

    Give exactly one of ``safety_loading`` and ``premium_rate``; the other
    follows from ``c = (1 + theta) lambda mu_1`` (eq. (4.9), p. 91), where
    ``lambda mu_1`` is the expected aggregate loss of Topic 1.

    Mathematical conditions:
        * ``u >= 0`` and ``lambda > 0``, finite.
        * ``mu_1 > 0``; ``mu_2 >= mu_1**2`` (the variance is not negative);
          if ``mu_3`` is given, ``mu_1 mu_3 >= mu_2**2`` (Cauchy-Schwarz,
          claims are non-negative).
        * ``c > 0``, that is ``theta > -1``. A positive loading
          ``theta > 0`` is needed to avoid certain ruin, but it is reported
          by :func:`collective_model_assumptions` rather than enforced here.

    Args:
        initial_capital: ``u``.
        claim_rate: ``lambda``.
        severity_moments: ``(mu_1, mu_2[, mu_3, ...])``.
        safety_loading: ``theta``.
        premium_rate: ``c``.

    Returns:
        A frozen :class:`CollectiveRiskProcess`.

    Raises:
        TypeError: If an input has the wrong type.
        ValueError: If the conditions above do not hold, or both or neither
            of ``safety_loading`` and ``premium_rate`` are given.

    Examples:
        >>> proc = collective_risk_process(
        ...     initial_capital=1000.0, claim_rate=3.0,
        ...     severity_moments=[200.0, 80000.0], safety_loading=0.1,
        ... )
        >>> round(proc.premium_rate, 6)
        660.0
    """
    u = _scalar(initial_capital, "initial_capital", minimum=0.0)
    lam = _scalar(claim_rate, "claim_rate", minimum=0.0, strict=True)
    mu = _severity_moments(severity_moments)
    expected = float(premiums.expected_aggregate_loss(lam, mu[0]))
    if (safety_loading is None) == (premium_rate is None):
        raise ValueError("give exactly one of safety_loading and premium_rate")
    if safety_loading is not None:
        theta = _scalar(safety_loading, "safety_loading")
        if theta <= -1.0:
            raise ValueError("safety_loading must be greater than -1")
        c = (1.0 + theta) * expected
    else:
        c = _scalar(premium_rate, "premium_rate", minimum=0.0, strict=True)
        theta = c / expected - 1.0
    if not np.isfinite(c):
        raise OverflowError("premium_rate overflowed")
    return CollectiveRiskProcess(
        initial_capital=u,
        claim_rate=lam,
        severity_moments=mu,
        safety_loading=theta,
        premium_rate=c,
    )


def annual_result_generator(
    r: npt.ArrayLike,
    *,
    process: CollectiveRiskProcess,
    severity_mgf: Callable[[npt.NDArray[np.float64]], npt.ArrayLike],
    time: float = 1.0,
) -> npt.NDArray[np.float64]:
    """Computes ``E[exp(r G(t))]`` for the result ``G(t) = c t - S(t)``.

    Since ``S(t)`` is compound Poisson with parameter ``lambda t``, eq. (3.5)
    gives ``E[exp(-r S(t))] = P_N(m_X(-r))`` with the Poisson pgf of Topic 3,
    so::

        E[exp(r G(t))] = exp(r c t) exp(lambda t (m_X(-r) - 1)).

    For ``r >= 0`` the claims mgf is evaluated at ``-r <= 0``, where it always
    exists because the claims are non-negative. The adjustment coefficient
    ``R`` is the positive root of ``E[exp(-R G(1))] = 1`` (eq. (4.11), p. 91).

    Args:
        r: Real argument; arrays are accepted.
        process: The collective risk process.
        severity_mgf: Function ``s -> E[exp(s X)]`` of the claims, evaluated
            element-wise on an array.
        time: Length ``t >= 0`` of the period.

    Returns:
        The generating function as a float64 array with the shape of ``r``.

    Raises:
        TypeError: If ``r`` or ``time`` is not made of real numbers.
        ValueError: If ``r`` or ``time`` is outside its domain, or
            ``severity_mgf`` returns the wrong shape or a non-positive value.
        OverflowError: If the claims mgf or the result is not finite.

    Examples:
        >>> proc = collective_risk_process(
        ...     initial_capital=0.0, claim_rate=3.0,
        ...     severity_moments=[200.0, 80000.0], safety_loading=0.1,
        ... )
        >>> def exp_mgf(s):
        ...     return 1.0 / (1.0 - 200.0 * s)
        >>> big_r = 0.1 / (1.1 * 200.0)  # eq. (4.13)
        >>> round(float(annual_result_generator(
        ...     -big_r, process=proc, severity_mgf=exp_mgf)), 12)
        1.0
    """
    s = _validation.as_real_array(r, "r")
    t = _scalar(time, "time", minimum=0.0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        m_x = np.asarray(severity_mgf(-s), dtype=np.float64)
    if m_x.shape != s.shape:
        raise ValueError(
            "severity_mgf must return an array shaped like its input"
        )
    if not np.all(np.isfinite(m_x)):
        raise OverflowError("severity_mgf is not finite at -r")
    if np.any(m_x <= 0):
        raise ValueError("severity_mgf must be positive")
    claims_part = frequency.poisson_count_pgf(m_x, rate=process.claim_rate * t)
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(
            np.exp(s * process.premium_rate * t) * claims_part, dtype=np.float64
        )
    return _validation.check_finite_result(result, "annual_result_generator")


@dataclasses.dataclass(frozen=True, slots=True)
class ModelAssumption:
    """One hypothesis of the classical collective risk model.

    Attributes:
        key: Short identifier.
        statement: The hypothesis in words.
        reference: Where it is stated in Kaas et al. (2008).
        holds: ``True`` or ``False`` when it can be checked on a process,
            ``None`` when it cannot be checked from the parameters.
    """

    key: str
    statement: str
    reference: str
    holds: bool | None = None


def collective_model_assumptions(
    process: CollectiveRiskProcess | None = None,
) -> tuple[ModelAssumption, ...]:
    """Lists the hypotheses of the classical model and checks those it can.

    Without a process all ``holds`` are ``None``. With a process, the
    hypotheses that depend only on its parameters are checked: non-negative
    initial capital, positive intensity, consistent claim moments and,
    above all, a positive safety loading, without which ruin is certain
    (Corollary 4.4.2, p. 97). The others (Poisson arrivals, independence,
    constant premium income, no interest or dividends) are properties of the
    real portfolio and cannot be read from the parameters.

    Args:
        process: Optional process to check.

    Returns:
        A tuple of :class:`ModelAssumption`.

    Examples:
        >>> proc = collective_risk_process(
        ...     initial_capital=0.0, claim_rate=3.0,
        ...     severity_moments=[200.0, 80000.0], safety_loading=0.0,
        ... )
        >>> checks = collective_model_assumptions(proc)
        >>> [a.key for a in checks if a.holds is False]
        ['positive_safety_loading']
    """
    checks: dict[str, bool | None] = dict.fromkeys(_ASSUMPTIONS)
    if process is not None:
        mu = process.severity_moments
        moments_ok = mu[0] > 0 and mu[1] >= mu[0] ** 2 * (1 - _MOMENT_TOLERANCE)
        if len(mu) >= 3:
            moments_ok = moments_ok and mu[0] * mu[2] >= mu[1] ** 2 * (
                1 - _MOMENT_TOLERANCE
            )
        checks["non_negative_initial_capital"] = process.initial_capital >= 0
        checks["positive_claim_rate"] = process.claim_rate > 0
        checks["consistent_claim_moments"] = bool(moments_ok)
        checks["positive_safety_loading"] = process.safety_loading > 0
    return tuple(
        ModelAssumption(key, statement, reference, checks[key])
        for key, (statement, reference) in _ASSUMPTIONS.items()
    )


def simulate_surplus(
    process: CollectiveRiskProcess,
    *,
    severity_sampler: Callable[
        [int, np.random.Generator], npt.NDArray[np.float64]
    ],
    n_periods: int,
    n_paths: int,
    rng: np.random.Generator | int | None = None,
) -> npt.NDArray[np.float64]:
    """Simulates the surplus at the end of each period, ``U(0), ..., U(n)``.

    ``U(k) = u + G_1 + ... + G_k`` with independent period results
    ``G_k = c - S_k`` (eq. (4.33), p. 98). The claims of all periods and
    paths are drawn with the compound Poisson sampler of Topic 3 in one call.
    Ruin between two period ends is not observed; this is the discrete-time
    view of Section 4.5.

    Args:
        process: The collective risk process.
        severity_sampler: Function ``(n, generator) -> array`` returning
            ``n`` independent claim amounts.
        n_periods: Number of periods ``n >= 1``.
        n_paths: Number of simulated paths ``>= 1``.
        rng: NumPy generator or seed, for reproducible results.

    Returns:
        A float64 array of shape ``(n_paths, n_periods + 1)``; column 0 is
        ``u``.

    Raises:
        TypeError: If a count is not an integer.
        ValueError: If a count is smaller than 1 or the sampler fails.
    """
    periods = _count(n_periods, "n_periods")
    paths = _count(n_paths, "n_paths")
    claims = frequency.compound_poisson_sample(
        rate=process.claim_rate,
        severity_sampler=severity_sampler,
        sample_shape=periods * paths,
        rng=rng,
    ).reshape(paths, periods)
    results = process.premium_rate - claims
    surplus = np.empty((paths, periods + 1), dtype=np.float64)
    surplus[:, 0] = process.initial_capital
    np.cumsum(results, axis=1, out=surplus[:, 1:])
    surplus[:, 1:] += process.initial_capital
    return surplus


_ASSUMPTIONS: dict[str, tuple[str, str]] = {
    "poisson_claim_arrivals": (
        "Claims arrive as a Poisson process with constant intensity "
        "lambda: independent, stationary increments "
        "N(t + h) - N(t) ~ Poisson(lambda h).",
        "Definition 4.2.1 and eq. (4.4), p. 90",
    ),
    "iid_claims_independent_of_counts": (
        "Claim amounts X_i >= 0 are iid and independent of the claim number "
        "process, so S(t) is compound Poisson.",
        "Section 4.2, eq. (4.1), p. 89, and p. 90",
    ),
    "constant_premium_income": (
        "Premiums are earned continuously at a constant rate c per unit "
        "of time.",
        "Section 4.2, eq. (4.1), p. 89",
    ),
    "no_interest_dividends_or_interventions": (
        "Premiums and claims do not change over time; inflation and return on "
        "capital cancel out; no dividends or interventions.",
        "Section 4.1, p. 87",
    ),
    "non_negative_initial_capital": (
        "The initial capital u = U(0) is non-negative.",
        "Section 4.2, eq. (4.1), p. 89",
    ),
    "positive_claim_rate": (
        "The intensity lambda is positive.",
        "Definition 4.2.1, p. 90",
    ),
    "consistent_claim_moments": (
        "The given claim moments can belong to a non-negative random variable "
        "(mu_1 > 0, mu_2 >= mu_1**2, mu_1 mu_3 >= mu_2**2).",
        "Eq. (4.8), p. 91",
    ),
    "positive_safety_loading": (
        "The safety loading theta = c / (lambda mu_1) - 1 is positive; if "
        "theta <= 0 ruin is certain.",
        "Eq. (4.9), p. 91; Corollary 4.4.2, p. 97",
    ),
}


def _scalar(
    value: float | None,
    name: str,
    *,
    minimum: float | None = None,
    strict: bool = False,
) -> float:
    """Validates a finite real scalar, optionally bounded below."""
    array = _validation.as_real_array(value, name)
    if array.ndim != 0:
        raise ValueError(f"{name} must be a scalar")
    number = float(array)
    if minimum is not None and (
        number < minimum or (strict and number == minimum)
    ):
        relation = "greater than" if strict else "at least"
        raise ValueError(f"{name} must be {relation} {minimum}")
    return number


def _severity_moments(
    value: Sequence[float] | npt.ArrayLike,
) -> tuple[float, ...]:
    """Validates the raw claim moments and returns them as a tuple."""
    mu = _validation.as_real_array(value, "severity_moments")
    if mu.ndim != 1 or mu.size < 2:
        raise ValueError("severity_moments must list at least mu_1 and mu_2")
    if mu[0] <= 0:
        raise ValueError("the mean claim mu_1 must be positive")
    if mu[1] < mu[0] ** 2 * (1 - _MOMENT_TOLERANCE):
        raise ValueError("mu_2 must be at least mu_1**2")
    if mu.size >= 3 and mu[0] * mu[2] < mu[1] ** 2 * (1 - _MOMENT_TOLERANCE):
        raise ValueError("mu_3 must satisfy mu_1 mu_3 >= mu_2**2")
    return tuple(float(m) for m in mu)


def _count(value: int, name: str) -> int:
    """Returns a positive integer count."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        count = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if count < 1:
        raise ValueError(f"{name} must be at least 1")
    return count
