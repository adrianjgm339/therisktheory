# Topic V — Insolvency over multiple periods

Package: `therisktheory.collective_risk.insolvency`.
Validation state: **experimental**, pending human mathematical peer review and
maintainer authorization.

## Mathematical model and scope

Use the compound Poisson surplus `U(t) = u + ct - S(t)` built by the Topic V
process module in Bladimir's PR #19. The observed annual sequence is
`U(0), U(1), ..., U(T)` for positive integer `T`. We estimate two distinct
probabilities at each period end `k`:

- **Cumulative observed insolvency:**
  `psi_obs(k) = P(min(U(1), ..., U(k)) < 0)`.
- **Terminal insolvency:** `q(k) = P(U(k) < 0)`.

The *first observed insolvency* at k is the event that all earlier observed
surpluses are nonnegative and `U(k) < 0`; its probability equals
`psi_obs(k) - psi_obs(k-1)`, with `psi_obs(0) = 0`.

**Important distinction:** this is not necessarily the continuous-time ruin
probability `P(inf_{0 <= t <= T} U(t) < 0)`. A path may become negative after a
claim and recover through subsequent premium income before the next observed
period end. The underlying `simulate_surplus` only records period-end values.
No theoretical equality with continuous-time ruin is claimed.

Strict `< 0` is intentional: `U(k) == 0` is *not* insolvent in this model.
A negative initial capital is not permitted by the process contract.

## Monte Carlo estimators

For n independently simulated trajectories `U_i(0..T)`:

```text
I_i(k)  = 1{U_i(k) < 0}
C_i(k)  = 1{there is j in 1..k with U_i(j) < 0}
q_hat(k)   = sum_i I_i(k) / n
psi_hat(k) = sum_i C_i(k) / n
first_hat(k) = psi_hat(k) - psi_hat(k-1)
SE_hat(p) = sqrt(p_hat * (1 - p_hat) / n)
```

The standard errors are plug-in Bernoulli Monte Carlo estimates. They are
not confidence intervals; at probabilities exactly 0 or 1 the plug-in
standard error is zero, which **does not establish absence of model risk**.
Trajectories must be independent, generated from a common model and share
one non-negative initial capital. For the same observed periods,
`psi_hat(k)` is nondecreasing, `psi_hat(k) >= q_hat(k)`, and
`sum_k first_hat(k) = psi_hat(T)`.

## Public API

- `finite_horizon_insolvency(surplus_paths)` calculates all estimators from
  a finite real 2D trajectory matrix `(n_paths, T + 1)` with `T >= 1`.
- `simulate_finite_horizon_insolvency(process, *, severity_sampler,
  n_periods, n_paths, rng=None)` delegates simulation to Bladimir's
  `collective_risk.simulate_surplus`; it does not duplicate the process.
- `FiniteHorizonInsolvency` stores read-only vectors for the cumulative,
  terminal and first-observed probabilities, Bernoulli standard errors,
  and sample sizes.

To avoid a shared-file ownership conflict, this contribution uses the direct
module import and does **not** edit Bladimir's
`collective_risk/__init__.py` while his PR #19 remains open.

## Independently checkable examples

Four observed trajectories:

```text
          U(0) U(1) U(2) U(3)
path 1      10    5   -1    2
path 2      10    9    4    3
path 3      10   -2   -4   -1
path 4      10    3    0    1
```

The cumulative probabilities for k=1,2,3 are `[0.25, 0.50, 0.50]`;
the terminal probabilities are `[0.25, 0.50, 0.25]`; and the first-observed
probabilities are `[0.25, 0.25, 0.0]`.

Independent one-period analytical check: when `u=0`, `lambda=1`, claims
have certain severity `X=1`, and `theta=0`, then the premium rate is `c=1`.
As `N(1) ~ Poisson(1)`, `U(1)=1-N(1)` and insolvency requires `N(1)>=2`.
Hence `P(U(1)<0) = 1 - 2/e`, approximately `0.2642411177`.

## Assumptions, references and integration

The process assumes nonnegative i.i.d. severities independent of Poisson
counts, stationary independent increments, and constant premium income.
The independent-path assumption is needed for the Bernoulli standard-error
expression, though the observed sample fractions remain defined without it.

Kaas, R., Goovaerts, M., Dhaene, J., and Denuit, M. (2008), *Modern Actuarial
Risk Theory: Using R*, 2nd ed., Springer, Chapter 4 (collective surplus,
ruin and discrete-time risk process),
https://doi.org/10.1007/978-3-540-70998-5.

The period-end Monte Carlo estimators above are direct empirical-indicator
calculations, not a quotation of an analytical finite-horizon formula from
that reference. Their derivation and implementation require independent
mathematical review before instructor validation.

Dependencies: PR #19 (which incorporates PR #14, #3 and #1). No imports
from the currently failing PR #25. No changes to teammates' modules.
