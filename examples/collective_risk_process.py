import numpy as np

from therisktheory import collective_risk

# 3 claims per year, exponential claims with mean 200, 10 % loading, u = 1000.
mean_claim = 200.0
proc = collective_risk.collective_risk_process(
    initial_capital=1000.0,
    claim_rate=3.0,
    severity_moments=[mean_claim, 2 * mean_claim**2, 6 * mean_claim**3],
    safety_loading=0.1,
)
print(proc.premium_rate)  # 660.0
print(proc.annual_result_moments())  # mean 60, variance 240 000, third -1.44e8

# Hypotheses of the model: which ones hold for this process.
for item in collective_risk.collective_model_assumptions(proc):
    print(f"{item.key:40} {item.holds}")


# Generating function of the annual result; it equals 1 at -R.
def exponential_mgf(s):
    """Moment generating function of the exponential claims."""
    return 1.0 / (1.0 - mean_claim * s)


big_r = 0.1 / (1.1 * mean_claim)  # Kaas eq. (4.13)
print(
    collective_risk.annual_result_generator(
        np.array([0.0, -big_r]), process=proc, severity_mgf=exponential_mgf
    )
)  # [1. 1.]

# Five years of surplus for three paths.
paths = collective_risk.simulate_surplus(
    proc,
    severity_sampler=lambda n, gen: gen.exponential(mean_claim, size=n),
    n_periods=5,
    n_paths=3,
    rng=1,
)
print(paths.round(1))
