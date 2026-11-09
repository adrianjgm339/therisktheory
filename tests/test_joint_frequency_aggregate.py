"""Joint tests: frequency (#14), aggregate model (#20), severity (#10).

These tests use Monticelli's ``collective_risk.aggregate_loss`` and Garcia's
``severity_aggregate.lognormal``. They are skipped when those modules are not
in the branch, so this PR does not fail because of another PR; once the
three PRs are merged they run against the real code.
"""

import numpy as np
import pytest

from therisktheory import frequency

aggregate_loss = pytest.importorskip(
    "therisktheory.collective_risk.aggregate_loss",
    reason="needs Monticelli's aggregate model (#20)",
)
lognormal = pytest.importorskip(
    "therisktheory.severity_aggregate.lognormal",
    reason="needs Garcia's severity distributions (#10)",
)


def test_negative_binomial_lognormal_aggregate_matches_simulation() -> None:
    # Frequency moments from #14, severity moments from #10, aggregate
    # moments from #20 (Kaas eq. (3.4)); checked against a simulation that
    # draws the counts with #14 and the claims from the same lognormal.
    size, prob, mu, sigma = 2.0, 0.4, 5.0, 0.8
    n = frequency.negative_binomial_count_moments(size=size, probability=prob)
    x = lognormal.lognormal_severity_moments(mu=mu, sigma=sigma)
    mean_s, var_s, std_s = aggregate_loss.aggregate_loss_moments(
        n["mean"], n["variance"], x["mean"], x["variance"]
    )

    rng = np.random.default_rng(20261109)
    n_sims = 300_000
    counts = frequency.negative_binomial_count_sample(
        size=size, probability=prob, sample_shape=n_sims, rng=rng
    )
    amounts = rng.lognormal(mu, sigma, size=counts.sum())
    owners = np.repeat(np.arange(n_sims), counts)
    totals = np.bincount(owners, weights=amounts, minlength=n_sims)

    se = float(std_s) / np.sqrt(n_sims)
    assert abs(totals.mean() - float(mean_s)) < 5 * se
    assert totals.var() == pytest.approx(float(var_s), rel=0.03)


def test_poisson_aggregate_moments_match_cumulants() -> None:
    lam = 3.0
    x = lognormal.lognormal_severity_moments(mu=5.0, sigma=0.8)
    n = frequency.poisson_count_moments(rate=lam)
    mean_s, var_s, _ = aggregate_loss.aggregate_loss_moments(
        n["mean"], n["variance"], x["mean"], x["variance"]
    )
    kappa = frequency.compound_poisson_cumulants(
        rate=lam, severity_raw_moments=[x["mean"], x["second_raw_moment"]]
    )
    np.testing.assert_allclose([mean_s, var_s], kappa, rtol=1e-12)


def test_aggregate_mgf_and_characteristic_function_match_pgf() -> None:
    # m_S(t) = P_N(m_X(t)) and phi_S(u) = P_N(phi_X(u)) (Kaas eq. (3.5)),
    # with exponential claims of mean 200.
    lam, mean_x = 3.0, 200.0
    t = np.array([-0.002, 0.0, 0.001, 0.004])
    m_x = 1.0 / (1.0 - mean_x * t)
    np.testing.assert_allclose(
        aggregate_loss.aggregate_mgf(t, lam, m_x),
        frequency.poisson_count_pgf(m_x, rate=lam),
        rtol=1e-12,
    )
    u = np.array([0.001, 0.01, 0.05])
    phi_x = 1.0 / (1.0 - 1j * mean_x * u)
    np.testing.assert_allclose(
        aggregate_loss.aggregate_characteristic_function(u, lam, phi_x),
        frequency.poisson_count_pgf(phi_x, rate=lam),
        rtol=1e-12,
    )
