"""Joint tests: collective risk process (#19) with Pérez's Lundberg code (#25).

These tests use ``collective_risk.safety_equation`` from #25. They are skipped
when that module is not in the branch, so this PR does not fail because of
another PR; once both PRs are merged they run against the real code.
"""

import numpy as np
import pytest

from therisktheory import collective_risk

safety_equation = pytest.importorskip(
    "therisktheory.collective_risk.safety_equation",
    reason="needs Pérez's safety equation (#25)",
)

MEAN = 200.0
BETA = 1.0 / MEAN
MU = [MEAN, 2 * MEAN**2, 6 * MEAN**3]


def exp_mgf(s):
    return BETA / (BETA - s)


@pytest.mark.parametrize("theta", [0.05, 0.1, 0.5, 2.0])
def test_adjustment_coefficient_agrees_with_annual_result_generator(
    theta: float,
) -> None:
    # Pérez solves Lundberg's equation (4.10); the same R must make the
    # generating function of the annual result equal to 1 at -R (4.11), and
    # for exponential claims R = theta beta / (1 + theta) (4.13).
    proc = collective_risk.collective_risk_process(
        initial_capital=1000.0,
        claim_rate=3.0,
        severity_moments=MU,
        safety_loading=theta,
    )
    big_r = safety_equation.adjustment_coefficient(
        exp_mgf, MEAN, proc.safety_loading, r_max=0.99 * BETA
    )
    assert big_r == pytest.approx(theta * BETA / (1 + theta), rel=1e-8)
    value = collective_risk.annual_result_generator(
        -big_r, process=proc, severity_mgf=exp_mgf
    )
    assert value.item() == pytest.approx(1.0, abs=1e-9)
    residual = safety_equation.lundberg_equation(
        big_r, exp_mgf, MEAN, proc.safety_loading
    )
    assert residual == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("theta", [0.05, 0.1, 0.5])
def test_full_retention_bound_from_the_annual_result(theta: float) -> None:
    # Kaas Example 4.5.2, p. 98: 2 E[G] / Var[G] = 2 theta mu_1 / mu_2 is an
    # upper bound for R. Pérez's full_retention_adjustment computes it from
    # the claim moments; here it is rebuilt from the moments of G.
    proc = collective_risk.collective_risk_process(
        initial_capital=0.0,
        claim_rate=3.0,
        severity_moments=MU,
        safety_loading=theta,
    )
    moments = proc.annual_result_moments()
    from_process = 2 * moments["mean"].item() / moments["variance"].item()
    approx = safety_equation.full_retention_adjustment(
        (MU[0], MU[1]), proc.safety_loading
    )
    assert approx == pytest.approx(from_process, rel=1e-12)
    exact = theta * BETA / (1 + theta)
    assert exact <= approx
    assert np.isclose(approx / exact, 1 + theta)  # exponential claims
