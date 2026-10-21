"""Tests for unit-step round safety loading."""

import numpy as np
import pytest

from therisktheory.finetti_ruin import round_safety_loading


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.5, 0.0),
        (0.6, 0.2),
        (0.75, 0.5),
        (1.0, 1.0),
    ],
)
def test_round_loading_scalar(probability: float, expected: float) -> None:
    result = round_safety_loading(probability)

    assert result.shape == ()
    assert result.item() == pytest.approx(expected)


def test_round_loading_vectorized() -> None:
    result = round_safety_loading([0.5, 0.6, 0.75, 1.0])

    np.testing.assert_allclose(result, [0.0, 0.2, 0.5, 1.0])


def test_round_loading_empty_array() -> None:
    result = round_safety_loading(np.array([]))

    assert result.shape == (0,)


@pytest.mark.parametrize(
    "probability",
    [-0.1, 0.49, 1.1, np.nan, np.inf],
)
def test_round_loading_invalid_domain(probability: float) -> None:
    with pytest.raises(ValueError):
        round_safety_loading(probability)


@pytest.mark.parametrize(
    "probability",
    ["0.6", True, 1 + 0j, None],
)
def test_round_loading_invalid_type(probability: object) -> None:
    with pytest.raises(TypeError):
        round_safety_loading(probability)
