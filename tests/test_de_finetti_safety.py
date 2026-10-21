"""Tests for restricted De Finetti unit-step safety measures."""

import doctest

import numpy as np
import pytest

import therisktheory.finetti_ruin.de_finetti_safety as safety_module
from therisktheory.finetti_ruin import (
    de_finetti_safety_loading,
    gambler_ruin_probability,
    round_safety_loading,
    safety_index,
    unit_step_adjustment_coefficient,
)


@pytest.mark.parametrize("p", [0.5, 0.6, 0.75, 0.9])
def test_index_matches_adjustment_coefficient(p: float) -> None:
    np.testing.assert_allclose(
        safety_index(p),
        unit_step_adjustment_coefficient(p),
        rtol=1e-13,
        atol=0.0,
    )


@pytest.mark.parametrize("p", [0.5, 0.6, 0.75, 1.0])
def test_loading_matches_expected_drift(p: float) -> None:
    np.testing.assert_allclose(
        de_finetti_safety_loading(p),
        round_safety_loading(p),
        rtol=1e-13,
        atol=0.0,
    )


def test_vectorized_interfaces() -> None:
    p = np.array([0.5, 0.6, 0.75])

    np.testing.assert_allclose(
        safety_index(p),
        np.log(p / (1 - p)),
    )

    np.testing.assert_allclose(
        de_finetti_safety_loading(p),
        2 * p - 1,
    )


def test_relation_with_ruin_probability() -> None:
    p = 0.6
    initial_reserve = np.arange(0, 12)
    r = safety_index(p)

    # Ruin at U < 0 corresponds to gambler capital u + 1.
    exact = gambler_ruin_probability(initial_reserve + 1, np.inf, p)

    expected = np.exp(-r * (initial_reserve + 1))

    np.testing.assert_allclose(exact, expected, rtol=1e-12, atol=0.0)

    assert np.all(exact <= np.exp(-r * initial_reserve))


@pytest.mark.parametrize("bad", [0.49, 1.0, np.nan, "0.6", True])
def test_index_rejects_invalid_domain(bad) -> None:
    with pytest.raises((TypeError, ValueError)):
        safety_index(bad)


@pytest.mark.parametrize("bad", [0.49, np.nan, "0.6", True])
def test_loading_rejects_invalid_domain(bad) -> None:
    with pytest.raises((TypeError, ValueError)):
        de_finetti_safety_loading(bad)


def test_empty_arrays() -> None:
    assert safety_index([]).shape == (0,)
    assert de_finetti_safety_loading([]).shape == (0,)


def test_docstrings() -> None:
    result = doctest.testmod(safety_module)
    assert result.failed == 0
    assert result.attempted > 0
