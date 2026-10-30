"""Tests for the balanced Buhlmann credibility component APIs."""

import numpy as np
import pytest

from therisktheory.credibility import (
    credibility_weight,
    estimate_hypothetical_mean_variance,
    estimate_process_variance,
)

REFERENCE = np.array([[0.0, 1.0, 0.0], [2.0, 1.0, 2.0]])
NEGATIVE = np.array([[3.0, 0.0, 0.0], [3.0, 0.0, 3.0]])


def test_exact_positive_estimators() -> None:
    epv = estimate_process_variance(REFERENCE)
    vhm = estimate_hypothetical_mean_variance(REFERENCE)
    z = credibility_weight(3, epv, vhm)

    assert epv.item() == pytest.approx(1 / 3)
    assert vhm.item() == pytest.approx(7 / 9)
    assert z.item() == pytest.approx(7 / 8)


def test_exact_negative_vhm_is_preserved() -> None:
    epv = estimate_process_variance(NEGATIVE)
    vhm = estimate_hypothetical_mean_variance(NEGATIVE)

    assert epv.item() == pytest.approx(3)
    assert vhm.item() == pytest.approx(-0.5)

    with pytest.raises(ValueError, match="non-negative"):
        credibility_weight(3, epv, vhm)


def test_independent_numpy_estimators() -> None:
    rng = np.random.default_rng(410)
    data = rng.normal(size=(8, 4))

    epv = np.var(data, axis=1, ddof=1).mean()
    vhm = np.var(data.mean(axis=1), ddof=1)
    vhm -= epv / data.shape[1]

    np.testing.assert_allclose(estimate_process_variance(data), epv)
    np.testing.assert_allclose(estimate_hypothetical_mean_variance(data), vhm)


def test_one_risk_for_epv_only() -> None:
    result = estimate_process_variance([[2.0, 4.0]])
    assert result.item() == pytest.approx(2.0)

    with pytest.raises(ValueError, match="at least 2 risks"):
        estimate_hypothetical_mean_variance([[2.0, 4.0]])


def test_constant_observations() -> None:
    data = [[7, 7, 7], [7, 7, 7]]

    assert estimate_process_variance(data).item() == 0
    assert estimate_hypothetical_mean_variance(data).item() == 0


def test_signed_data() -> None:
    data = [[-1.0, 1.0], [1.0, 3.0]]

    assert estimate_process_variance(data).item() == pytest.approx(2)
    assert estimate_hypothetical_mean_variance(data).item() == pytest.approx(1)


@pytest.mark.parametrize(
    "invalid",
    [
        [1.0, 2.0],
        [[[0.0, 1.0]]],
        np.empty((0, 2)),
        np.empty((1, 0)),
        np.array([[1.0], [2.0]]),
        [[1.0, 2.0], [np.inf, 1.0]],
    ],
)
def test_invalid_observations(invalid) -> None:
    with pytest.raises(ValueError):
        estimate_process_variance(invalid)

    with pytest.raises(ValueError):
        estimate_hypothetical_mean_variance(invalid)


@pytest.mark.parametrize(
    "invalid",
    [
        [["bad", 2], [2, 3]],
        [[1], [1, 2]],
        [[True, False], [True, False]],
        [[1 + 1j, 2], [1, 2]],
        None,
    ],
)
def test_nonreal_observations(invalid) -> None:
    with pytest.raises(TypeError):
        estimate_process_variance(invalid)

    with pytest.raises(TypeError):
        estimate_hypothetical_mean_variance(invalid)


def test_invalid_risk_count_vhm() -> None:
    with pytest.raises(ValueError, match="at least 2 risks"):
        estimate_hypothetical_mean_variance([[1.0, 2.0]])


def test_nonfinite_computed_variance() -> None:
    data = [[0.0, 1e200], [1e200, 0.0]]

    with pytest.raises(OverflowError):
        estimate_process_variance(data)

    with pytest.raises(OverflowError):
        estimate_hypothetical_mean_variance(data)


def test_weight_broadcasting_and_limits() -> None:
    z = credibility_weight([0, 1, 3, 10], 1.0, 1.0)

    np.testing.assert_allclose(z, [0, 1 / 2, 3 / 4, 10 / 11])

    assert z.dtype == np.float64

    assert credibility_weight(3, 0.0, 1.0).item() == 1.0
    assert credibility_weight(3, 1.0, 0.0).item() == 0.0
    assert credibility_weight(0, 0.0, 1.0).item() == 0.0
    assert credibility_weight(0, 1.0, 0.0).item() == 0.0

    result = credibility_weight(3, 1e308, 1e308)
    assert result.item() == pytest.approx(0.75)


def test_weight_multi_axis_broadcast() -> None:
    n = np.array([1, 3])
    epv = np.array([[1.0], [2.0]])

    result = credibility_weight(n, epv, 1.0)

    np.testing.assert_allclose(result, [[1 / 2, 3 / 4], [1 / 3, 3 / 5]])


@pytest.mark.parametrize(
    "bad",
    [-1, True, 1.5, "3", None, [1, -1]],
)
def test_bad_counts(bad) -> None:
    with pytest.raises((TypeError, ValueError)):
        credibility_weight(bad, 1.0, 1.0)


@pytest.mark.parametrize(
    "bad",
    [-1.0, np.nan, np.inf, "1", None, 1 + 2j, True],
)
def test_bad_epv_or_vhm(bad) -> None:
    with pytest.raises((TypeError, ValueError)):
        credibility_weight(3, bad, 1.0)

    with pytest.raises((TypeError, ValueError)):
        credibility_weight(3, 1.0, bad)


def test_undefined_weight_and_incompatible_shapes() -> None:
    with pytest.raises(ValueError, match="cannot both be zero"):
        credibility_weight(0, 0, 0)

    with pytest.raises(ValueError, match="cannot both be zero"):
        credibility_weight(3, 0, 0)

    with pytest.raises(ValueError, match="incompatible shapes"):
        credibility_weight([1, 2, 3], [1.0, 2.0], 1.0)


def test_empty_weight_broadcast() -> None:
    result = credibility_weight(np.array([], dtype=int), 1.0, 1.0)
    assert result.shape == (0,)


def test_huge_count_and_stable_weight() -> None:
    n = np.array([np.iinfo(np.int64).max])
    result = credibility_weight(n, 1.0, 1.0)

    assert result[0] == pytest.approx(1.0)


def test_ragged_count_is_rejected() -> None:
    with pytest.raises(TypeError, match="non-negative integers"):
        credibility_weight([[1, 2], [1]], 1.0, 1.0)
