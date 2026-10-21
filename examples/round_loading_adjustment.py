"""Demonstrate unit-step expected gain and adjustment coefficient."""

import numpy as np

from therisktheory.finetti_ruin import (
    gambler_ruin_probability,
    round_safety_loading,
    unit_step_adjustment_coefficient,
)

initial_capital = 3
probabilities = [0.50, 0.60, 0.75]

print("=== TOPIC 2: UNIT-STEP RISK GAME ===")
print("Initial gambler capital:", initial_capital)
print("Target: infinity")
print()
print("   p     E[G]        R       Ruin probability")

for p in probabilities:
    gain = round_safety_loading(p).item()
    coefficient = unit_step_adjustment_coefficient(p).item()

    ruin = gambler_ruin_probability(
        initial_capital,
        np.inf,
        p,
    ).item()

    independent_result = np.exp(-coefficient * initial_capital)

    np.testing.assert_allclose(
        ruin,
        independent_result,
        rtol=1e-12,
        atol=1e-12,
    )

    print(f"{p:5.2f}  {gain:7.2f}  {coefficient:8.6f}  {ruin:16.6f}")

print()
print("All analytical comparisons passed.")
