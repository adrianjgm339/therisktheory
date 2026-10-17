"""Tests for actuarial safety loading calculations."""

import numpy as np
import pytest

from therisktheory.solvency_pricing import (
    expected_value_premium,
    loaded_premium,
    premium_principle_checks,
    pure_premium,
    safety_loading,
)


def test_safety_loading_scalar() -> None:
    result = safety_loading(600.0, 660.0)
    assert result.item() == pytest.approx(60.0)


def test_zero_safety_loading() -> None:
    result = safety_loading(600.0, 600.0)
    assert result.item() == pytest.approx(0.0)


def test_safety_loading_vectorized() -> None:
    result = safety_loading(
        [600.0, 1000.0],
        [660.0, 1150.0],
    )
    np.testing.assert_allclose(result, [60.0, 150.0])


def test_safety_loading_broadcasting() -> None:
    result = safety_loading(
        [100.0, 200.0, 300.0],
        350.0,
    )
    np.testing.assert_allclose(result, [250.0, 150.0, 50.0])


def test_premium_below_expected_loss() -> None:
    with pytest.raises(ValueError):
        safety_loading(600.0, 550.0)


@pytest.mark.parametrize(
    ("expected_loss", "premium"),
    [
        (-1.0, 100.0),
        (100.0, -1.0),
        (np.nan, 100.0),
        (100.0, np.inf),
    ],
)
def test_invalid_values(expected_loss, premium) -> None:
    with pytest.raises(ValueError):
        safety_loading(expected_loss, premium)


def test_incompatible_shapes() -> None:
    with pytest.raises(ValueError, match="incompatible shapes"):
        safety_loading([100.0, 200.0], [110.0, 220.0, 330.0])


def test_non_numeric_input() -> None:
    with pytest.raises(TypeError):
        safety_loading("600", 660.0)


def test_loaded_premium_expected_value() -> None:
    net = pure_premium(3.0, 200.0)
    result = loaded_premium(net, 0.10)

    np.testing.assert_array_equal(
        result,
        expected_value_premium(net, 0.10),
    )
    assert result.item() == pytest.approx(660.0)


def test_loaded_premium_variance() -> None:
    result = loaded_premium(
        600.0,
        0.0005,
        principle="variance",
        variance=240_000.0,
    )
    assert result.item() == pytest.approx(720.0)


def test_loaded_premium_standard_deviation() -> None:
    result = loaded_premium(
        600.0,
        0.10,
        principle="standard_deviation",
        variance=240_000.0,
    )
    assert result.item() == pytest.approx(600.0 + 0.10 * np.sqrt(240_000.0))


@pytest.mark.parametrize(
    "principle",
    ["expected_value", "variance", "standard_deviation"],
)
def test_loaded_premium_zero_loading(principle: str) -> None:
    kwargs = {}
    if principle != "expected_value":
        kwargs["variance"] = 240_000.0

    result = loaded_premium(600.0, 0.0, principle=principle, **kwargs)
    assert result.item() == pytest.approx(600.0)


@pytest.mark.parametrize(
    "principle",
    ["variance", "standard_deviation"],
)
def test_loaded_premium_vectorization(principle: str) -> None:
    result = loaded_premium(
        [100.0, 200.0],
        0.10,
        principle=principle,
        variance=[0.0, 100.0],
    )

    expected = [100.0, 210.0] if principle == "variance" else [100.0, 201.0]

    np.testing.assert_allclose(result, expected)


def test_loaded_premium_broadcasting() -> None:
    result = loaded_premium(
        [100.0, 200.0],
        0.10,
        principle="variance",
        variance=[[0.0], [100.0]],
    )
    np.testing.assert_allclose(
        result,
        [[100.0, 200.0], [110.0, 210.0]],
    )


def test_loaded_premium_unknown_principle() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        loaded_premium(600.0, 0.10, principle="invalid")


def test_loaded_premium_non_string_principle() -> None:
    with pytest.raises(TypeError, match="principle"):
        loaded_premium(600.0, 0.10, principle=123)


def test_loaded_premium_missing_variance() -> None:
    with pytest.raises(ValueError, match="variance is required"):
        loaded_premium(600.0, 0.10, principle="variance")


def test_loaded_premium_unexpected_variance() -> None:
    with pytest.raises(ValueError, match="variance is not used"):
        loaded_premium(
            600.0,
            0.10,
            principle="expected_value",
            variance=100.0,
        )


@pytest.mark.parametrize(
    "bad_variance",
    [-1.0, np.nan, np.inf],
)
def test_loaded_premium_invalid_variance(bad_variance) -> None:
    with pytest.raises(ValueError):
        loaded_premium(
            600.0,
            0.10,
            principle="variance",
            variance=bad_variance,
        )


def test_loaded_premium_non_numeric_variance() -> None:
    with pytest.raises(TypeError):
        loaded_premium(
            600.0,
            0.10,
            principle="variance",
            variance="100",
        )


def test_loaded_premium_incompatible_shapes() -> None:
    with pytest.raises(ValueError, match="incompatible shapes"):
        loaded_premium(
            [100.0, 200.0],
            0.10,
            principle="variance",
            variance=[10.0, 20.0, 30.0],
        )


@pytest.mark.parametrize(
    "principle",
    ["variance", "standard_deviation"],
)
def test_loaded_premium_overflow(principle: str) -> None:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(OverflowError):
            loaded_premium(
                1e308,
                1e308,
                principle=principle,
                variance=1e308,
            )


def test_loaded_premium_rejects_invalid_loading() -> None:
    with pytest.raises(ValueError, match="loading"):
        loaded_premium(
            600.0,
            -0.1,
            principle="variance",
            variance=100.0,
        )


def test_loaded_premium_rejects_invalid_mean() -> None:
    with pytest.raises(ValueError, match="expected_loss"):
        loaded_premium(
            np.nan,
            0.1,
            principle="standard_deviation",
            variance=100.0,
        )


def test_loaded_premium_docstring_examples() -> None:
    import doctest

    import therisktheory.solvency_pricing.loadings as module

    result = doctest.testmod(module)
    assert result.attempted > 0
    assert result.failed == 0


@pytest.mark.parametrize(
    ("principle", "cash_consistent", "additive"),
    [
        ("expected_value", False, True),
        ("variance", True, True),
        ("standard_deviation", True, False),
    ],
)
def test_premium_principle_properties(
    principle: str,
    cash_consistent: bool,
    additive: bool,
) -> None:
    # X = 0 or 2, each with probability 0.5.
    # Therefore E[X] = 1, Var(X) = 1, max(X) = 2.
    variance = None if principle == "expected_value" else 1.0

    result = premium_principle_checks(
        1.0,
        0.10,
        principle=principle,
        variance=variance,
        maximum_loss=2.0,
        cash_shift=1.0,
    )

    assert result["non_negative_loading"].item() is True
    assert result["no_rip_off"].item() is True
    assert result["translation_consistency"].item() is cash_consistent
    assert result["independent_additivity"].item() is additive


@pytest.mark.parametrize(
    ("principle", "loading", "variance", "maximum"),
    [
        ("expected_value", 0.10, None, 1.0),
        ("variance", 2.0, 1.0, 2.0),
        ("standard_deviation", 2.0, 1.0, 2.0),
    ],
)
def test_no_rip_off_counterexamples(
    principle, loading, variance, maximum
) -> None:
    result = premium_principle_checks(
        1.0,
        loading,
        principle=principle,
        variance=variance,
        maximum_loss=maximum,
    )
    assert result["no_rip_off"].item() is False


def test_no_rip_off_is_unknown_without_maximum() -> None:
    result = premium_principle_checks(100.0, 0.10)
    assert result["no_rip_off"] is None


def test_zero_loading_properties() -> None:
    result = premium_principle_checks(
        1.0,
        0.0,
        principle="standard_deviation",
        variance=1.0,
        maximum_loss=2.0,
    )
    assert all(value.item() for value in result.values())


def test_principle_checks_vectorized() -> None:
    result = premium_principle_checks(
        [1.0, 2.0],
        0.10,
        principle="variance",
        variance=[1.0, 4.0],
        maximum_loss=[2.0, 4.0],
    )
    np.testing.assert_array_equal(
        result["translation_consistency"],
        [True, True],
    )
    np.testing.assert_array_equal(
        result["independent_additivity"],
        [True, True],
    )


def test_principle_checks_reject_zero_shift() -> None:
    with pytest.raises(ValueError, match="cash_shift"):
        premium_principle_checks(1.0, 0.10, cash_shift=0.0)


def test_principle_checks_reject_invalid_maximum() -> None:
    with pytest.raises(ValueError, match="maximum_loss"):
        premium_principle_checks(
            2.0,
            0.10,
            maximum_loss=1.0,
        )


def test_principle_checks_reject_incompatible_shapes() -> None:
    with pytest.raises(ValueError, match="incompatible shapes"):
        premium_principle_checks(
            [1.0, 2.0],
            0.10,
            maximum_loss=[2.0, 3.0, 4.0],
        )


def test_expected_value_tiny_loading_translation() -> None:
    result = premium_principle_checks(
        1.0,
        1e-12,
        principle="expected_value",
        cash_shift=1.0,
    )
    assert result["translation_consistency"].item() is False


def test_stddev_tiny_loading_independent_additivity() -> None:
    result = premium_principle_checks(
        1.0,
        1e-12,
        principle="standard_deviation",
        variance=1.0,
    )
    assert result["independent_additivity"].item() is False


def test_stddev_zero_variance_is_additive() -> None:
    result = premium_principle_checks(
        1.0,
        0.5,
        principle="standard_deviation",
        variance=0.0,
    )
    assert result["independent_additivity"].item() is True


def test_principle_checks_broadcast_shapes() -> None:
    result = premium_principle_checks(
        [1.0, 2.0],
        [0.0, 1e-12],
        principle="standard_deviation",
        variance=[1.0, 4.0],
        maximum_loss=[[2.0, 4.0]],
        cash_shift=[[1.0], [2.0]],
    )

    np.testing.assert_array_equal(
        result["independent_additivity"],
        [[True, False], [True, False]],
    )
    assert result["translation_consistency"].shape == (2, 2)


def test_no_rip_off_tiny_loading_for_constant_risk() -> None:
    """Detect positive loadings hidden by float64 rounding."""
    result = premium_principle_checks(
        expected_loss=[1.0, 1.0],
        loading=[0.0, 1e-17],
        principle="expected_value",
        maximum_loss=1.0,
    )

    np.testing.assert_array_equal(
        result["no_rip_off"],
        [True, False],
    )
