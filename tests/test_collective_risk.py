import dataclasses
import doctest
import warnings

import numpy as np
import pytest

from therisktheory import collective_risk
from therisktheory.collective_risk import process as process_module
from therisktheory.solvency_pricing import premiums

# Exponential claims with mean 200: mu_j = j! 200**j, m_X(s) = beta/(beta - s).
MEAN = 200.0
BETA = 1.0 / MEAN
MU = [MEAN, 2 * MEAN**2, 6 * MEAN**3]


def exp_mgf(s):
    return BETA / (BETA - s)


def exp_sampler(n, gen):
    return gen.exponential(MEAN, size=n)


def make(theta=0.1, u=1000.0, lam=3.0, mu=MU):
    return collective_risk.collective_risk_process(
        initial_capital=u,
        claim_rate=lam,
        severity_moments=mu,
        safety_loading=theta,
    )


def test_docstring_examples() -> None:
    result = doctest.testmod(process_module)
    assert result.attempted > 0
    assert result.failed == 0


# Construction.


def test_premium_from_loading_and_back() -> None:
    proc = make(theta=0.1)
    assert proc.premium_rate == pytest.approx(660.0)
    # Topic 1: c is the expected value premium of the expected annual claims.
    expected = premiums.expected_aggregate_loss(3.0, MEAN)
    assert proc.premium_rate == pytest.approx(
        premiums.expected_value_premium(expected, 0.1).item()
    )
    back = collective_risk.collective_risk_process(
        initial_capital=1000.0,
        claim_rate=3.0,
        severity_moments=MU,
        premium_rate=660.0,
    )
    assert back.safety_loading == pytest.approx(0.1)  # eq. (4.9)
    assert proc.expected_claims_per_unit_time == 600.0


def test_process_is_frozen() -> None:
    proc = make()
    with pytest.raises(dataclasses.FrozenInstanceError):
        proc.premium_rate = 1.0


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"initial_capital": -1.0}, ValueError),
        ({"claim_rate": 0.0}, ValueError),
        ({"claim_rate": np.inf}, ValueError),
        ({"severity_moments": [200.0]}, ValueError),
        ({"severity_moments": [[200.0, 8e4]]}, ValueError),
        ({"severity_moments": [0.0, 1.0]}, ValueError),
        ({"severity_moments": [200.0, 1e4]}, ValueError),  # mu_2 < mu_1**2
        ({"severity_moments": [200.0, 8e4, 1e7]}, ValueError),  # mu_3 too small
        ({"safety_loading": -1.0}, ValueError),
        ({"safety_loading": None}, ValueError),  # neither given
        ({"premium_rate": 600.0}, ValueError),  # both given
        ({"initial_capital": [1.0, 2.0]}, ValueError),
        ({"initial_capital": "1000"}, TypeError),
        ({"claim_rate": True}, TypeError),
    ],
)
def test_invalid_process(kwargs, error) -> None:
    args = {
        "initial_capital": 1000.0,
        "claim_rate": 3.0,
        "severity_moments": MU,
        "safety_loading": 0.1,
        **kwargs,
    }
    with pytest.raises(error):
        collective_risk.collective_risk_process(**args)


def test_invalid_premium_rate() -> None:
    with pytest.raises(ValueError):
        collective_risk.collective_risk_process(
            initial_capital=0.0,
            claim_rate=3.0,
            severity_moments=MU,
            premium_rate=0.0,
        )


def test_premium_overflow() -> None:
    # E[S] finite, but (1 + theta) E[S] is not.
    with pytest.raises(OverflowError, match="premium_rate"):
        collective_risk.collective_risk_process(
            initial_capital=0.0,
            claim_rate=1e300,
            severity_moments=[1.0, 1.0],
            safety_loading=1e10,
        )
    # E[S] itself overflows (Topic 1 check).
    with pytest.raises(OverflowError):
        collective_risk.collective_risk_process(
            initial_capital=0.0,
            claim_rate=1e300,
            severity_moments=[1e10, 1e20],
            safety_loading=1e10,
        )


# Moments of the result and the surplus.


def test_annual_result_moments() -> None:
    proc = make(theta=0.1)
    m = proc.annual_result_moments([1.0, 2.0])
    np.testing.assert_allclose(m["mean"], [60.0, 120.0])  # theta lambda mu_1 t
    np.testing.assert_allclose(m["variance"], [3 * MU[1], 6 * MU[1]])
    np.testing.assert_allclose(
        m["third_central_moment"], [-3 * MU[2], -6 * MU[2]]
    )
    two = make(mu=MU[:2]).annual_result_moments()
    assert "third_central_moment" not in two


def test_surplus_moments() -> None:
    m = make(theta=0.1, u=1000.0).surplus_moments(5.0)
    assert m["mean"].item() == pytest.approx(1000.0 + 5 * 60.0)
    assert m["variance"].item() == pytest.approx(5 * 3 * MU[1])


def test_surplus_moment_overflow() -> None:
    proc = make()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(OverflowError):
            proc.annual_result_moments(1e307)


def test_simulated_surplus_matches_moments() -> None:
    proc = make(theta=0.1, u=1000.0)
    n_paths = 100_000
    paths = collective_risk.simulate_surplus(
        proc,
        severity_sampler=exp_sampler,
        n_periods=4,
        n_paths=n_paths,
        rng=20261107,
    )
    assert paths.shape == (n_paths, 5)
    assert np.all(paths[:, 0] == 1000.0)
    results = np.diff(paths, axis=1).ravel()
    m = proc.annual_result_moments()
    se = np.sqrt(m["variance"].item() / results.size)
    assert abs(results.mean() - m["mean"].item()) < 5 * se
    assert results.var() == pytest.approx(m["variance"].item(), rel=0.02)
    third = np.mean((results - results.mean()) ** 3)
    assert third == pytest.approx(m["third_central_moment"].item(), rel=0.1)
    final = proc.surplus_moments(4.0)
    assert paths[:, 4].mean() == pytest.approx(final["mean"].item(), rel=0.01)


def test_simulation_reproducible_and_invalid_counts() -> None:
    proc = make()
    a = collective_risk.simulate_surplus(
        proc, severity_sampler=exp_sampler, n_periods=3, n_paths=10, rng=1
    )
    b = collective_risk.simulate_surplus(
        proc,
        severity_sampler=exp_sampler,
        n_periods=3,
        n_paths=10,
        rng=np.random.default_rng(1),
    )
    np.testing.assert_array_equal(a, b)
    for kwargs, error in [
        ({"n_periods": 0, "n_paths": 1}, ValueError),
        ({"n_periods": 1, "n_paths": 0}, ValueError),
        ({"n_periods": 1.5, "n_paths": 1}, TypeError),
        ({"n_periods": True, "n_paths": 1}, TypeError),
    ]:
        with pytest.raises(error):
            collective_risk.simulate_surplus(
                proc, severity_sampler=exp_sampler, **kwargs
            )


# Generating function of the annual result.


def test_generator_at_zero_and_closed_form() -> None:
    proc = make(theta=0.1)
    r = np.array([-0.002, -0.0005, 0.0, 0.0005, 0.003])
    got = collective_risk.annual_result_generator(
        r, process=proc, severity_mgf=exp_mgf
    )
    closed = np.exp(r * 660.0 + 3.0 * (BETA / (BETA + r) - 1.0))
    np.testing.assert_allclose(got, closed, rtol=1e-12)
    assert got[2] == 1.0


def test_adjustment_coefficient_equations() -> None:
    # Example 4.3.3, eq. (4.13): R = theta beta / (1 + theta) for exponential
    # claims. It solves eq. (4.10), 1 + (1 + theta) mu_1 R = m_X(R), and the
    # equivalent form (4.11), E[exp(-R (c - S))] = 1.
    for theta in (0.05, 0.1, 0.5, 2.0):
        proc = make(theta=theta)
        big_r = theta * BETA / (1 + theta)
        assert 1 + (1 + theta) * MEAN * big_r == pytest.approx(exp_mgf(big_r))
        value = collective_risk.annual_result_generator(
            -big_r, process=proc, severity_mgf=exp_mgf
        )
        assert value.item() == pytest.approx(1.0, abs=1e-12)
        # The generating function is convex in r: below 1 between 0 and R,
        # above 1 beyond R.
        inside, beyond = collective_risk.annual_result_generator(
            np.array([-0.5, -1.2]) * big_r, process=proc, severity_mgf=exp_mgf
        )
        assert inside < 1.0 < beyond


def test_generator_derivative_is_the_mean() -> None:
    proc = make(theta=0.1)
    h = 1e-7
    value = collective_risk.annual_result_generator(
        np.array([-h, h]), process=proc, severity_mgf=exp_mgf
    )
    assert (value[1] - value[0]) / (2 * h) == pytest.approx(60.0, rel=1e-6)


def test_generator_over_time_is_a_power() -> None:
    # Independent stationary increments: E[exp(r G(t))] = E[exp(r G(1))]**t.
    proc = make(theta=0.1)
    r = np.array([-0.001, 0.0008])
    one = collective_risk.annual_result_generator(
        r, process=proc, severity_mgf=exp_mgf
    )
    three = collective_risk.annual_result_generator(
        r, process=proc, severity_mgf=exp_mgf, time=3.0
    )
    np.testing.assert_allclose(three, one**3, rtol=1e-12)


def test_generator_matches_simulation() -> None:
    proc = make(theta=0.1, u=0.0)
    results = np.diff(
        collective_risk.simulate_surplus(
            proc,
            severity_sampler=exp_sampler,
            n_periods=1,
            n_paths=400_000,
            rng=11,
        ),
        axis=1,
    ).ravel()
    for r in (-0.0008, 0.0004):
        empirical = np.exp(r * results)
        se = empirical.std() / np.sqrt(empirical.size)
        exact = collective_risk.annual_result_generator(
            r, process=proc, severity_mgf=exp_mgf
        ).item()
        assert abs(empirical.mean() - exact) < 5 * se


def test_generator_errors() -> None:
    proc = make()
    with pytest.raises(ValueError, match="shaped"):
        collective_risk.annual_result_generator(
            [0.1, 0.2], process=proc, severity_mgf=lambda s: 1.0
        )
    with pytest.raises(OverflowError):  # m_X(beta) does not exist
        collective_risk.annual_result_generator(
            -BETA, process=proc, severity_mgf=exp_mgf
        )
    with pytest.raises(ValueError, match="positive"):
        collective_risk.annual_result_generator(
            -2 * BETA, process=proc, severity_mgf=exp_mgf
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(OverflowError):
            collective_risk.annual_result_generator(
                10.0, process=proc, severity_mgf=exp_mgf
            )
    with pytest.raises(TypeError):
        collective_risk.annual_result_generator(
            "0.1", process=proc, severity_mgf=exp_mgf
        )
    with pytest.raises(ValueError):
        collective_risk.annual_result_generator(
            0.1, process=proc, severity_mgf=exp_mgf, time=-1.0
        )


# Hypotheses of the model.


def test_assumptions_without_a_process() -> None:
    items = collective_risk.collective_model_assumptions()
    assert len(items) == 8
    assert all(item.holds is None for item in items)
    assert all("p." in item.reference for item in items)
    keys = {item.key for item in items}
    assert {"poisson_claim_arrivals", "positive_safety_loading"} <= keys


def test_assumptions_for_a_sound_process() -> None:
    checked = {
        a.key: a.holds
        for a in collective_risk.collective_model_assumptions(make(theta=0.1))
    }
    assert checked["positive_safety_loading"] is True
    assert checked["consistent_claim_moments"] is True
    assert checked["non_negative_initial_capital"] is True
    assert checked["positive_claim_rate"] is True
    assert checked["poisson_claim_arrivals"] is None  # not checkable


@pytest.mark.parametrize("theta", [0.0, -0.2])
def test_no_positive_loading_is_flagged(theta: float) -> None:
    # Corollary 4.4.2: theta <= 0 means certain ruin.
    checked = {
        a.key: a.holds
        for a in collective_risk.collective_model_assumptions(make(theta=theta))
    }
    assert checked["positive_safety_loading"] is False


def test_assumptions_with_two_moments() -> None:
    checked = {
        a.key: a.holds
        for a in collective_risk.collective_model_assumptions(make(mu=MU[:2]))
    }
    assert checked["consistent_claim_moments"] is True
