import doctest
import itertools
import warnings

import numpy as np
import pytest

from therisktheory.solvency_pricing import premiums

expected_aggregate_loss = premiums.expected_aggregate_loss
expected_value_premium = premiums.expected_value_premium
pure_premium = premiums.pure_premium

ALL_FUNCTIONS = (expected_aggregate_loss, pure_premium, expected_value_premium)


def test_pure_premium_scalar() -> None:
    assert pure_premium(2.0, 500.0).item() == pytest.approx(1000.0)


def test_expected_value_premium_scalar() -> None:
    assert expected_value_premium(1000.0, 0.2).item() == pytest.approx(1200.0)


def test_docstring_examples() -> None:
    result = doctest.testmod(premiums)
    assert result.attempted > 0
    assert result.failed == 0


def test_expected_loss_by_enumeration() -> None:
    # Small model where every outcome can be listed: E[N] = 0.7, E[X] = 180.
    count_pmf = {0: 0.5, 1: 0.3, 2: 0.2}
    size_pmf = {100.0: 0.6, 300.0: 0.4}

    mean_s = 0.0
    for n, p_n in count_pmf.items():
        for claims in itertools.product(size_pmf, repeat=n):
            prob = p_n * np.prod([size_pmf[x] for x in claims])
            mean_s += prob * sum(claims)

    assert mean_s == pytest.approx(126.0)
    assert expected_aggregate_loss(0.7, 180.0).item() == pytest.approx(mean_s)


def test_expected_loss_by_simulation() -> None:
    # Compound Poisson(3) with exponential claims of mean 200, so E[S] = 600.
    # Var(S) = 3 * 2 * 200**2, which gives a standard error of about 1.1 for
    # the sample mean; 5 standard errors is a safe tolerance.
    rng = np.random.default_rng(20261010)
    n_sims, lam, mean_x = 200_000, 3.0, 200.0

    counts = rng.poisson(lam, size=n_sims)
    amounts = rng.exponential(mean_x, size=counts.sum())
    owners = np.repeat(np.arange(n_sims), counts)
    totals = np.bincount(owners, weights=amounts, minlength=n_sims)

    std_error = np.sqrt(lam * 2 * mean_x**2 / n_sims)
    expected = expected_aggregate_loss(lam, mean_x).item()
    assert abs(totals.mean() - expected) < 5 * std_error


def test_pure_premium_arrays() -> None:
    result = pure_premium([0.5, 1.0, 2.0], [1000.0, 750.0, 500.0])
    np.testing.assert_allclose(result, [500.0, 750.0, 1000.0])


def test_pure_premium_broadcasting() -> None:
    np.testing.assert_allclose(
        pure_premium([1.0, 2.0, 3.0], 100.0), [100.0, 200.0, 300.0]
    )


def test_expected_value_premium_broadcasting() -> None:
    result = expected_value_premium([100.0, 200.0], [[0.0], [0.1], [0.25]])
    np.testing.assert_allclose(
        result, [[100.0, 200.0], [110.0, 220.0], [125.0, 250.0]]
    )


def test_integer_inputs_are_accepted() -> None:
    assert pure_premium(np.int64(3), np.uint8(200)).item() == 600.0


@pytest.mark.parametrize("function", ALL_FUNCTIONS)
def test_scalars_give_0d_float_arrays(function) -> None:
    result = function(1, 2)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 0
    assert result.dtype == np.float64


@pytest.mark.parametrize("function", ALL_FUNCTIONS)
def test_empty_arrays_give_empty_result(function) -> None:
    result = function(np.array([]), np.array([]))
    assert result.shape == (0,)
    assert result.dtype == np.float64


def test_empty_array_broadcasts_with_scalar() -> None:
    assert expected_value_premium([], 0.1).shape == (0,)


@pytest.mark.parametrize(
    ("frequency", "severity"), [(0.0, 1000.0), (5.0, 0.0), (0.0, 0.0)]
)
def test_zero_frequency_or_severity(frequency: float, severity: float) -> None:
    assert pure_premium(frequency, severity).item() == 0.0


def test_zero_loading_gives_pure_premium() -> None:
    net = pure_premium([0.1, 1.5, 4.0], [2500.0, 800.0, 120.0])
    np.testing.assert_array_equal(expected_value_premium(net, 0.0), net)


def test_largest_finite_result_is_accepted() -> None:
    big = np.finfo(np.float64).max
    assert expected_aggregate_loss(1.0, big).item() == big


def test_pure_premium_scales_with_currency() -> None:
    np.testing.assert_allclose(
        pure_premium(1.7, 3.5 * 640.0), 3.5 * pure_premium(1.7, 640.0)
    )


def test_premium_increases_with_loading() -> None:
    premiums_ = expected_value_premium(1000.0, np.linspace(0.0, 1.0, 11))
    assert np.all(np.diff(premiums_) > 0)


def test_premium_not_below_expected_loss() -> None:
    losses = np.array([0.0, 10.0, 1e6])
    assert np.all(expected_value_premium(losses, 0.3) >= losses)


@pytest.mark.parametrize(
    ("function", "args"),
    [
        (expected_aggregate_loss, (1e200, 1e200)),
        (pure_premium, ([1.0, 1e300], [1.0, 1e10])),
        (expected_value_premium, (np.finfo(np.float64).max, 1.0)),
    ],
)
def test_overflow_raises_without_runtime_warning(function, args) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # NumPy's RuntimeWarning would fail
        with pytest.raises(OverflowError, match="overflowed"):
            function(*args)


@pytest.mark.parametrize(
    ("frequency", "severity"),
    [
        (-1.0, 100.0),
        (1.0, -100.0),
        ([1.0, -0.1], 100.0),
        (np.nan, 100.0),
        (1.0, np.nan),
        ([1.0, np.nan], 100.0),
        (np.inf, 100.0),
        (1.0, -np.inf),
    ],
)
def test_invalid_values(frequency, severity) -> None:
    with pytest.raises(ValueError):
        expected_aggregate_loss(frequency, severity)
    with pytest.raises(ValueError):
        pure_premium(frequency, severity)
    with pytest.raises(ValueError):
        expected_value_premium(frequency, severity)


@pytest.mark.parametrize("function", ALL_FUNCTIONS)
@pytest.mark.parametrize(
    ("left", "right"),
    [([1.0, 2.0], [1.0, 2.0, 3.0]), (np.ones((2, 3)), np.ones((3, 2)))],
)
def test_incompatible_shapes(function, left, right) -> None:
    with pytest.raises(ValueError, match="incompatible shapes"):
        function(left, right)


@pytest.mark.parametrize("function", ALL_FUNCTIONS)
@pytest.mark.parametrize(
    "bad",
    ["100", None, ["a", "b"], 1 + 2j, True, [[1.0, 2.0], [3.0]], object()],
)
def test_non_numeric_inputs(function, bad) -> None:
    with pytest.raises(TypeError):
        function(bad, 1.0)
    with pytest.raises(TypeError):
        function(1.0, bad)


def test_error_names_the_parameter() -> None:
    with pytest.raises(ValueError, match="loading"):
        expected_value_premium(100.0, -0.5)
    with pytest.raises(TypeError, match="mean_severity"):
        pure_premium(1.0, "100")
