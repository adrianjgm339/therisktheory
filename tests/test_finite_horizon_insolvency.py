"""Tests for observed finite-horizon insolvency (Topic V, Garcia)."""

import dataclasses
import doctest

import numpy as np
import pytest

from therisktheory import collective_risk
from therisktheory.collective_risk import insolvency

EXAMPLE = np.array(
    [
        [10.0, 5.0, -1.0, 2.0],
        [10.0, 9.0, 4.0, 3.0],
        [10.0, -2.0, -4.0, -1.0],
        [10.0, 3.0, 0.0, 1.0],
    ]
)


def _make_process():
    return collective_risk.collective_risk_process(
        initial_capital=1000.0,
        claim_rate=3.0,
        severity_moments=[200.0, 80000.0],
        safety_loading=0.1,
    )


def _exponential_sampler(n, rng):
    return rng.exponential(200.0, size=n)


def test_doctest() -> None:
    check = doctest.testmod(insolvency)
    assert check.attempted >= 2
    assert check.failed == 0


def test_explicit_four_paths() -> None:
    result = insolvency.finite_horizon_insolvency(EXAMPLE)
    assert result.n_paths == 4
    assert result.n_periods == 3
    np.testing.assert_array_equal(
        result.cumulative_probability, [0.25, 0.5, 0.5]
    )
    np.testing.assert_array_equal(
        result.terminal_probability, [0.25, 0.5, 0.25]
    )
    np.testing.assert_array_equal(
        result.first_insolvency_probability, [0.25, 0.25, 0.0]
    )
    assert result.cumulative_standard_error[0] == pytest.approx(
        np.sqrt(0.25 * 0.75 / 4)
    )
    assert result.terminal_standard_error[1] == pytest.approx(0.25)
    assert np.all(np.diff(result.cumulative_probability) >= 0)
    assert np.all(result.cumulative_probability >= result.terminal_probability)
    assert result.first_insolvency_probability.sum() == pytest.approx(
        result.cumulative_probability[-1]
    )


def test_one_period_and_zero_is_not_insolvent() -> None:
    paths = [[5, 0], [5, 3], [5, -2]]
    result = insolvency.finite_horizon_insolvency(paths)
    np.testing.assert_allclose(result.cumulative_probability, [1 / 3])
    np.testing.assert_allclose(result.terminal_probability, [1 / 3])
    np.testing.assert_allclose(result.first_insolvency_probability, [1 / 3])


def test_no_insolvency_and_certain_insolvency() -> None:
    none = insolvency.finite_horizon_insolvency([[0, 0, 1], [0, 2, 1]])
    all_ = insolvency.finite_horizon_insolvency([[0, -1, -2], [0, -2, -3]])
    np.testing.assert_array_equal(none.cumulative_probability, [0, 0])
    np.testing.assert_array_equal(none.cumulative_standard_error, [0, 0])
    np.testing.assert_array_equal(all_.cumulative_probability, [1, 1])
    np.testing.assert_array_equal(all_.first_insolvency_probability, [1, 0])
    np.testing.assert_array_equal(all_.terminal_standard_error, [0, 0])


def test_result_is_frozen_and_arrays_detached() -> None:
    paths = EXAMPLE.copy()
    result = insolvency.finite_horizon_insolvency(paths)
    paths[0, 2] = 100.0
    assert result.cumulative_probability[-1] == 0.5
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.n_paths = 9
    with pytest.raises(ValueError):
        result.cumulative_probability[0] = 0.0
    for array in (
        result.cumulative_probability,
        result.terminal_probability,
        result.first_insolvency_probability,
        result.cumulative_standard_error,
        result.terminal_standard_error,
    ):
        with pytest.raises(ValueError):
            array.setflags(write=True)


@pytest.mark.parametrize(
    "bad",
    [
        [0.0, -1.0],
        np.empty((0, 2)),
        np.empty((2, 0)),
        np.empty((2, 1)),
        np.empty((1, 2, 2)),
    ],
)
def test_invalid_dimensions(bad) -> None:
    with pytest.raises(ValueError, match="shape"):
        insolvency.finite_horizon_insolvency(bad)


@pytest.mark.parametrize(
    ("bad", "error"),
    [
        ([[10, np.nan]], ValueError),
        ([[10, np.inf]], ValueError),
        ([[10, -np.inf]], ValueError),
        ([[True, False]], TypeError),
        ([["10", "-1"]], TypeError),
        ([[1 + 0j, -1 + 0j]], TypeError),
        ([[1, 2], [1]], TypeError),
        ([[-1, 0]], ValueError),
        ([[0, 1], [1, -1]], ValueError),
    ],
)
def test_invalid_values(bad, error) -> None:
    with pytest.raises(error):
        insolvency.finite_horizon_insolvency(bad)


def test_simulation_matches_underlying_process() -> None:
    process = _make_process()
    kwargs = {
        "severity_sampler": _exponential_sampler,
        "n_periods": 4,
        "n_paths": 500,
        "rng": 20261110,
    }
    observed = insolvency.simulate_finite_horizon_insolvency(process, **kwargs)
    repeated = insolvency.simulate_finite_horizon_insolvency(process, **kwargs)
    trajectories = collective_risk.simulate_surplus(process, **kwargs)
    expected = insolvency.finite_horizon_insolvency(trajectories)
    np.testing.assert_array_equal(
        observed.cumulative_probability, expected.cumulative_probability
    )
    np.testing.assert_array_equal(
        observed.terminal_probability, repeated.terminal_probability
    )
    assert observed.n_paths == 500
    assert observed.n_periods == 4


def test_single_period_matches_analytical_poisson_tail() -> None:
    process = collective_risk.collective_risk_process(
        initial_capital=0.0,
        claim_rate=1.0,
        severity_moments=[1.0, 1.0],
        safety_loading=0.0,
    )

    def deterministic_claims(n, rng):
        return np.ones(n)

    count = 50_000
    estimate = insolvency.simulate_finite_horizon_insolvency(
        process,
        severity_sampler=deterministic_claims,
        n_periods=1,
        n_paths=count,
        rng=20261110,
    )
    # Here U(1) = 1 - N, N~Poisson(1): P(U(1)<0) = 1 - 2/e.
    exact = 1 - 2 / np.e
    standard_error = np.sqrt(exact * (1 - exact) / count)
    assert abs(estimate.terminal_probability[0] - exact) < 5 * standard_error
    assert estimate.terminal_probability[0] == pytest.approx(
        estimate.cumulative_probability[0]
    )


def test_simulation_rejects_invalid_process_and_sampler() -> None:
    with pytest.raises(TypeError, match="CollectiveRiskProcess"):
        insolvency.simulate_finite_horizon_insolvency(
            None,
            severity_sampler=_exponential_sampler,
            n_periods=1,
            n_paths=2,
        )
    with pytest.raises(TypeError, match="callable"):
        insolvency.simulate_finite_horizon_insolvency(
            _make_process(),
            severity_sampler=None,
            n_periods=1,
            n_paths=2,
        )


@pytest.mark.parametrize(
    ("bad", "error"),
    [
        ({"n_periods": 0}, ValueError),
        ({"n_paths": 0}, ValueError),
        ({"n_periods": 1.5}, TypeError),
        ({"n_paths": True}, TypeError),
    ],
)
def test_simulation_rejects_invalid_counts(bad, error) -> None:
    kwargs = {
        "severity_sampler": _exponential_sampler,
        "n_periods": 2,
        "n_paths": 5,
        **bad,
    }
    with pytest.raises(error):
        insolvency.simulate_finite_horizon_insolvency(_make_process(), **kwargs)
