"""Demonstrate safety loading and classical premium principles."""

from therisktheory.solvency_pricing import (
    loaded_premium,
    premium_principle_checks,
    pure_premium,
    safety_loading,
)

# Compound Poisson model:
# N ~ Poisson(3), X ~ Exponential(mean=200).
# E[S] = E[N] * E[X] = 600.
# Var(S) = E[N] * Var(X) + Var(N) * E[X]**2.
# Var(S) = 3 * 200**2 + 3 * 200**2 = 240000.

frequency = 3.0
mean_severity = 200.0
aggregate_variance = 240_000.0

net_premium = pure_premium(frequency, mean_severity)

expected_value = loaded_premium(
    net_premium,
    0.10,
    principle="expected_value",
)

variance_premium = loaded_premium(
    net_premium,
    0.0005,
    principle="variance",
    variance=aggregate_variance,
)

stddev_premium = loaded_premium(
    net_premium,
    0.10,
    principle="standard_deviation",
    variance=aggregate_variance,
)

print("=== ACTUARIAL PREMIUM PRINCIPLES ===")
print(f"Pure premium: {net_premium.item():.2f}")

for label, premium in (
    ("Expected value", expected_value),
    ("Variance", variance_premium),
    ("Standard deviation", stddev_premium),
):
    monetary_loading = safety_loading(net_premium, premium)

    print(
        f"{label}: premium={premium.item():.2f}, "
        f"safety loading={monetary_loading.item():.2f}"
    )

print("\n=== MATHEMATICAL PROPERTY DIAGNOSTICS ===")

checks = premium_principle_checks(
    net_premium,
    0.10,
    principle="standard_deviation",
    variance=aggregate_variance,
)

print(
    "Translation consistency:",
    checks["translation_consistency"].item(),
)
print(
    "Independent-copy additivity:",
    checks["independent_additivity"].item(),
)
print("No-rip-off diagnostic:", checks["no_rip_off"])
