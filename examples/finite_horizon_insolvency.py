"""Example of observed insolvency over multiple periods (Topic V)."""

import numpy as np

from therisktheory.collective_risk import collective_risk_process
from therisktheory.collective_risk.insolvency import (
    simulate_finite_horizon_insolvency,
)


def exponential_claims(n, rng):
    """Draw n i.i.d. exponential claims with mean 200."""
    return rng.exponential(200.0, size=n)


def main() -> None:
    """Print cumulative and terminal period-end insolvency estimates."""
    process = collective_risk_process(
        initial_capital=1000.0,
        claim_rate=3.0,
        severity_moments=[200.0, 80000.0],
        safety_loading=0.1,
    )
    results = simulate_finite_horizon_insolvency(
        process,
        severity_sampler=exponential_claims,
        n_periods=5,
        n_paths=10_000,
        rng=20261110,
    )
    print("Period | cumulative | terminal | MC SE cumulative")
    for k in range(results.n_periods):
        print(
            f"{k + 1:>6} | {results.cumulative_probability[k]:>10.4f} | "
            f"{results.terminal_probability[k]:>8.4f} | "
            f"{results.cumulative_standard_error[k]:>16.4f}"
        )
    assert np.all(np.diff(results.cumulative_probability) >= 0)


if __name__ == "__main__":
    main()
