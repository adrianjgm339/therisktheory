"""Tests for the unit-step adjustment coefficient."""

import doctest

import numpy as np
import pytest

import therisktheory.finetti_ruin.adjustment as adjustment
from therisktheory.finetti_ruin import (
    gambler_ruin_probability,
    round_safety_loading,
    unit_step_adjustment_coefficient,
)


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.5, 0.0),
        (0.6, np.log(1.5)),
        (0.75, np.log(3.0)),
        (0.9, np.log(9.0)),
    ],
)
def test_adjustment_scalar(probability, expected) -> None:
    result = unit_step_adjustment_coefficient(probability)

    assert result.shape == ()
    assert result.item() == pytest.approx(expected)


def test_adjustment_vectorized() -> None:
    probabilities = np.array([0.5, 0.6, 0.75, 0.9])
    result = unit_step_adjustment_coefficient(probabilities)

    expected = np.log(probabilities / (1 - probabilities))
    np.testing.assert_allclose(result, expected)


def test_adjustment_empty_array() -> None:
    result = unit_step_adjustment_coefficient(np.array([]))

    assert result.shape == (0,)


def test_adjustment_near_fair_game() -> None:
    p = 0.5 + 1e-14
    result = unit_step_adjustment_coefficient(p)

    assert result.item() > 0
    assert result.item() == pytest.approx(
        2 * np.arctanh(2 * p - 1),
        rel=1e-10,
        abs=1e-15,
    )


def test_adjustment_near_one() -> None:
    p = np.nextafter(1.0, 0.0)
    result = unit_step_adjustment_coefficient(p)

    assert np.isfinite(result.item())
    assert result.item() > 30.0


def test_adjustment_satisfies_root_equation() -> None:
    p = np.array([0.5, 0.6, 0.75, 0.9])
    r = unit_step_adjustment_coefficient(p)

    lhs = p * np.exp(-r) + (1 - p) * np.exp(r)

    np.testing.assert_allclose(lhs, np.ones_like(p), rtol=1e-13, atol=1e-13)


def test_adjustment_matches_gambler_ruin() -> None:
    p = np.array([0.5, 0.6, 0.75])
    z = np.array([1, 2, 3])

    r = unit_step_adjustment_coefficient(p)
    ruin = gambler_ruin_probability(z, np.inf, p)

    np.testing.assert_allclose(ruin, np.exp(-r * z), rtol=1e-12, atol=1e-12)


def test_adjustment_agrees_with_round_loading() -> None:
    p = np.array([0.5, 0.6, 0.75])
    drift = round_safety_loading(p)
    result = unit_step_adjustment_coefficient(p)

    expected = np.log1p(drift) - np.log1p(-drift)
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "probability",
    [-0.1, 0.49, 1.0, 1.1, np.nan, np.inf, -np.inf],
)
def test_adjustment_invalid_domain(probability) -> None:
    with pytest.raises(ValueError):
        unit_step_adjustment_coefficient(probability)


@pytest.mark.parametrize(
    "probability",
    ["0.6", True, 1 + 0j, None],
)
def test_adjustment_invalid_type(probability) -> None:
    with pytest.raises(TypeError):
        unit_step_adjustment_coefficient(probability)


def test_adjustment_docstring() -> None:
    result = doctest.testmod(adjustment)

    assert result.attempted > 0
    assert result.failed == 0
