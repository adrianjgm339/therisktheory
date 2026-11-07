# Topic 5: the collective risk process

Module: `therisktheory.collective_risk.process`. Status: experimental.

Page numbers refer to Kaas, Goovaerts, Dhaene and Denuit (2008), *Modern
Actuarial Risk Theory: Using R*, 2nd edition, Springer,
https://doi.org/10.1007/978-3-540-70998-5.

## Objectives of collective risk theory

Topics 1 to 3 look at one period. Collective risk theory follows the
insurer's capital over time (Section 4.1, p. 87). The capital grows with the
premiums and drops at each claim, and the insurer is ruined if it ever
becomes negative. The ruin probability `ψ(u)` measures how sound the
combination of premiums, claims and initial capital is. A high value says
that reinsurance, higher premiums or more capital are needed.

It is a comparative tool rather than a forecast of bankruptcy. The model
leaves out interest, inflation, dividends and any reaction of the insurer,
so the exact value matters less than how it changes with the parameters.

## The collective risk process

The surplus process (eq. (4.1), p. 89) is

```text
U(t) = u + c t − S(t),     S(t) = X_1 + ... + X_{N(t)},
```

with:

- `u = U(0)`, the initial capital;
- `c`, the premium income per unit of time;
- `X_i ≥ 0`, the claims;
- `N(t)`, the number of claims up to `t`.

Ruin is the first time `T` with `U(T) < 0`, and `ψ(u) = Pr[T < ∞]`
(eqs. (4.2)–(4.3), pp. 89–90).

## Hypotheses of the model

`collective_model_assumptions()` lists them with their source. It checks
those that depend only on the parameters and marks the rest as
not checkable (`holds=None`).

| Hypothesis | Source | Checked |
|---|---|---|
| Claims arrive as a Poisson process with intensity `λ` | Def. 4.2.1, eq. (4.4), p. 90 | no |
| Claims `X_i ≥ 0` iid and independent of `N(t)` | eq. (4.1), p. 89; p. 90 | no |
| Constant premium income `c` | eq. (4.1), p. 89 | no |
| No interest, inflation, dividends or interventions | Section 4.1, p. 87 | no |
| `u ≥ 0` | eq. (4.1), p. 89 | yes |
| `λ > 0` | Def. 4.2.1, p. 90 | yes |
| Moments of a non-negative claim: `μ_1 > 0`, `μ_2 ≥ μ_1²`, `μ_1 μ_3 ≥ μ_2²` | eq. (4.8), p. 91 | yes |
| Positive safety loading `θ > 0` | eq. (4.9), p. 91; Corollary 4.4.2, p. 97 | yes |

The safety loading is defined by `c = (1 + θ) λ μ_1` (eq. (4.9)). If `θ ≤ 0`,
ruin is certain, `ψ(u) ≡ 1` (Corollary 4.4.2, p. 97). This is the same
conclusion as for the pure premium (Topic 1) and the fair game (Topic 2).
The constructor accepts `θ > −1` so that such a process can still be built
and studied, and the assumptions function reports it.

## The process object

`collective_risk_process(initial_capital=, claim_rate=, severity_moments=,
safety_loading= | premium_rate=)` returns a frozen `CollectiveRiskProcess`.
It computes `c` from `θ` with `expected_aggregate_loss` (Topic 1), or `θ`
from `c` by eq. (4.9). Its methods give the moments of the result of a
period `G(t) = c t − S(t)` and of the surplus `U(t) = u + G(t)`. They use the
compound Poisson cumulants `κ_j = λ t μ_j` of Topic 3:

```text
E[G(t)] = θ λ μ_1 t,   Var[G(t)] = λ t μ_2,   E[(G − E G)³] = −λ t μ_3.
```

## Generating function of the annual result

The result of a period is `G(t) = c t − S(t)`. Since `S(t)` is compound
Poisson with parameter `λt`, eq. (3.5) gives
`E[e^{−rS(t)}] = P_N(m_X(−r))` with the Poisson pgf of Topic 3, so

```text
E[e^{r G(t)}] = e^{r c t} exp(λ t (m_X(−r) − 1)).
```

For `r ≥ 0`, `m_X(−r) ≤ 1` always exists because the claims are
non-negative. Three properties follow:

- **Independent increments:** `E[e^{r G(t)}] = E[e^{r G(1)}]^t`.
- **Mean:** the derivative at 0 is `E[G(1)] = θ λ μ_1`.
- **Adjustment coefficient:** `R` is the positive root of
  `E[e^{−R G(1)}] = 1`, the equivalent form (4.11) on p. 91 of
  `1 + (1 + θ) μ_1 R = m_X(R)` (Definition 4.3.1, eq. (4.10)). By convexity
  the function is below 1 between 0 and `R` and above 1 beyond it.
  Lundberg's bound `ψ(u) ≤ e^{−Ru}` (Theorem 4.3.4, p. 92) is then the job of
  the ruin functions of Topic 5.

For exponential claims with mean `1/β`, `R = θβ/(1 + θ)` (eq. (4.13), p. 92).

## Simulation

`simulate_surplus(process, severity_sampler=, n_periods=, n_paths=, rng=)`
returns `U(0), U(1), ..., U(n)` for many paths. It draws all the period
claims with the compound Poisson sampler of Topic 3 in one call. This is the
discrete-time view of Section 4.5 (eq. (4.33), p. 98) and gives a
simulation check for insolvency over `T` periods and for ruin estimates.

## How other Topic 5 functions can use it

| Contributor | Uses |
|---|---|
| Garcia (insolvency over T periods, `φ_v(u)`) | `simulate_surplus`, `surplus_moments(t)` |
| Monticelli (ruin probability, upper bound, Poisson and Pólya cases) | the process, `annual_result_generator` |
| Passanisi (stability, `λ`, loading as a function of `u`) | `safety_loading`, `premium_rate`, `annual_result_moments` |
| Pérez (safety equation, Lundberg's equation, adjustment coefficient) | `annual_result_generator(-R) = 1` |
| Valera (reinsurance effects) | build the retained process with new `c` and claim moments |

## Independent checks (`tests/test_collective_risk.py`)

- `c` from `θ` equals the expected value premium of Topic 1, and `θ` from `c`
  follows eq. (4.9).
- The closed form of the generating function for exponential claims.
- **Eqs. (4.10), (4.11) and (4.13):** with `R = θβ/(1 + θ)`, `1 + (1 + θ) μ_1 R
  = m_X(R)` and `E[e^{−R G(1)}] = 1`, for several `θ`; the function is
  below 1 at `R/2` and above 1 at `1.2 R`.
- The derivative at 0 equals `E[G(1)]`, and `E[e^{rG(3)}] = E[e^{rG(1)}]³`.
- The simulated `E[e^{rG}]` (400 000 periods) matches the formula within 5
  standard errors.
- The simulated period results match `E[G]`, `Var[G]` and the third central
  moment, and the simulated `U(4)` matches `E[U(4)]`.
- The hypotheses: a loading `θ ≤ 0` is flagged.
- Invalid inputs and overflow.
