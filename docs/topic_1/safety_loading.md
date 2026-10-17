# Topic 1: Safety Loading and Premium Principle Diagnostics

**Module:** `therisktheory.solvency_pricing.loadings`

**Status:** Experimental — pending independent human validation

**Reference:** Kaas, R., Goovaerts, M., Dhaene, J., and Denuit, M. (2008). *Modern Actuarial Risk Theory: Using R*, 2nd edition, Springer.

## 1. Scope

This module extends the premium foundations by providing monetary safety loading calculations, three classical loaded premium principles, and diagnostics of selected mathematical properties.

It builds on `expected_value_premium()` and the numerical validation utilities already present in TheRiskTheory.

The functions are:

- `safety_loading()`
- `loaded_premium()`
- `premium_principle_checks()`

The implementation supports NumPy arrays and broadcasting. Mathematical properties are documented separately from the numerical tests.

## 2. Collective risk model

Let `N` denote the number of claims and let `X_1, X_2, ...` represent claim amounts.

The aggregate loss is:

`S = X_1 + ... + X_N`, with `S = 0` when `N = 0`.

Assume that the claim amounts are independent and identically distributed and that `N` is independent of the claim amounts.

Under finite first moments:

`E[S] = E[N] E[X]`

This follows by conditioning on `N`, as in equation (3.3), page 42.

Under finite second moments:

`Var(S) = E[N] Var(X) + Var(N) (E[X])^2`

This is equation (3.4), page 43.

The functions operate on the supplied aggregate-loss moments. They do not estimate the claim frequency or severity distributions.

## 3. Monetary safety loading

For an aggregate loss `S` and a charged premium `P`, the monetary safety loading is:

`L = P - E[S]`

The implementation requires `P >= E[S]`, resulting in `L >= 0`.

When `E[S] > 0`, the relative loading coefficient is:

`theta = L / E[S]`

The monetary loading `L` and the dimensionless coefficient `theta` must not be confused.

For example, if the expected aggregate loss is 600 and the charged premium is 660:

`L = 660 - 600 = 60`

`theta = 60 / 600 = 0.10`

Thus, the monetary loading is 60 monetary units and the relative loading is 10%.

The function `safety_loading()` returns the monetary amount, not the relative coefficient.

## 4. Classical premium principles

Kaas et al. (2008), Section 5.3, page 119, describe the following principles.

| Principle | Premium formula | Coefficient |
|---|---|---|
| Expected value | `E[S] * (1 + theta)` | `theta >= 0` |
| Variance | `E[S] + alpha * Var(S)` | `alpha >= 0` |
| Standard deviation | `E[S] + alpha * sqrt(Var(S))` | `alpha >= 0` |

The referenced textbook uses strictly positive loading coefficients. This implementation also permits zero, representing the limiting net-premium case.

The expected value and standard deviation coefficients are dimensionless. For the variance principle, the coefficient has inverse monetary units when losses are expressed in currency.

All supplied moments and coefficients must be finite and non-negative.

The variance and standard deviation principles require a supplied aggregate-loss variance. The software assumes that this variance and the mean describe the same underlying risk.

## 5. Independent numerical example

Consider a compound Poisson model:

`N ~ Poisson(3)`

Suppose claim severities are exponentially distributed with mean 200, independently of `N`.

Then:

`E[N] = 3`

`Var(N) = 3`

`E[X] = 200`

`Var(X) = 200^2 = 40,000`

From the collective risk formulas:

`E[S] = 3 * 200 = 600`

`Var(S) = 3 * 40,000 + 3 * 200^2 = 240,000`

`sigma(S) = sqrt(240,000) = 489.897949...`

The resulting premiums are:

| Principle | Coefficient | Premium | Monetary loading |
|---|---:|---:|---:|
| Expected value | 0.10 | 660.00 | 60.00 |
| Variance | 0.0005 | 720.00 | 120.00 |
| Standard deviation | 0.10 | 648.99 | 48.99 |

These values provide independent reference calculations for automated tests.

The three coefficients are not interchangeable: each applies to a different risk measure.

## 6. Properties of premium principles

Section 5.3.1, pages 120–122, discusses five properties. The following results correspond to the classical principles with positive coefficients.

| Property | Expected value | Variance | Standard deviation |
|---|---|---|---|
| Non-negative loading | Yes | Yes | Yes |
| No rip-off, universally | No | No | No |
| Translation consistency | No | Yes | Yes |
| Additivity for independent risks | Yes | Yes | No |
| Iterativity | No | No | No |

The results are consistent with Table 5.1, page 122. A property marked "No" is not valid in general, although particular risks or coefficient values may satisfy it.

### 6.1 Non-negative loading

For all three principles, non-negative coefficients and non-negative variances imply:

`premium >= E[S]`

For the variance and standard deviation principles, zero variance also produces zero loading even when the coefficient is positive.

### 6.2 No rip-off

For a bounded risk, the no-rip-off property requires:

`premium <= ess sup(S)`

Here, `ess sup(S)` is the maximum possible loss in the probabilistic sense.

The property fails in general for all three loaded principles.

For example, if `S = 1` with certainty, the expected value premium with `theta = 0.10` is 1.10, although the maximum loss is 1.

For another example, suppose `S` equals 0 or 2 with probability 0.5 each. Then `E[S] = 1`, `Var(S) = 1`, and the maximum loss is 2.

With `alpha = 2`, both the variance and standard deviation premiums equal 3, exceeding the maximum loss of 2.

The `maximum_loss` input must represent the actual essential supremum if the diagnostic is to be interpreted as a no-rip-off test. An arbitrary upper bound is not sufficient for this interpretation.

The program cannot verify the distributional validity of the supplied maximum.

### 6.3 Translation consistency

Translation consistency requires:

`pi(S + c) = pi(S) + c`

For the expected value principle:

`pi(S + c) - pi(S) - c = theta * c`

Therefore, a positive coefficient and a positive shift imply failure of translation consistency.

For the variance and standard deviation principles:

`Var(S + c) = Var(S)`

`sigma(S + c) = sigma(S)`

Therefore, both principles are translation consistent.

### 6.4 Independent additivity

Let `S'` be an independent copy of `S`.

Then:

`E[S + S'] = 2 E[S]`

`Var(S + S') = 2 Var(S)`

The expected value principle is additive by linearity of expectation.

The variance principle is additive for independent risks because their variances add.

For the standard deviation principle:

`pi(S + S') = 2 E[S] + alpha * sqrt(2 Var(S))`

while:

`pi(S) + pi(S') = 2 E[S] + 2 alpha * sqrt(Var(S))`

These differ when `alpha > 0` and `Var(S) > 0`.

The standard deviation principle becomes additive in this independent-copy scenario if `alpha = 0` or `Var(S) = 0`.

### 6.5 Iterativity

Iterativity requires applying the premium principle to a conditional risk and then applying the same principle to the resulting random premium.

The three loaded principles do not satisfy iterativity in general, as shown in Table 5.1.

The current `premium_principle_checks()` interface does not receive conditional distributions, so it does not attempt an iterativity diagnostic.

## 7. Meaning of the diagnostics

`premium_principle_checks()` evaluates the selected principle for supplied risk moments.

Its output includes:

| Diagnostic | Meaning |
|---|---|
| `non_negative_loading` | Computed premium is at least the supplied mean |
| `no_rip_off` | Monetary loading does not exceed the margin between the maximum loss and the mean; `None` if the maximum is absent |
| `translation_consistency` | Algebraic consistency condition for a positive deterministic shift |
| `independent_additivity` | Algebraic additivity condition for an independent, identically distributed copy |

Translation consistency and independent-copy additivity are derived algebraically for the supported principles. They do not rely on approximate floating-point equality.

The no-rip-off diagnostic compares the monetary loading with the available margin up to the supplied essential supremum. A zero-margin safeguard avoids treating a strictly positive loading as admissible when the computed premium is rounded to the expected loss.

The non-negative loading diagnostic compares the computed premium with the supplied mean. Numerical checks remain subject to floating-point limitations.

All conclusions depend on the mathematical validity of the supplied moments and distributional assumptions.

## 8. Numerical implementation and limitations

Numeric inputs must be finite, real and non-negative. Invalid types, incompatible dimensions, unsupported principles and missing required moments are rejected.

Computed non-finite premium results raise `OverflowError`.

NumPy broadcasting is supported, and scalar inputs return zero-dimensional arrays.

The implementation uses floating-point arithmetic. Small mathematical loadings may be rounded away from a computed premium at extreme numerical scales. Consequently, computational equality must not automatically be interpreted as a universal mathematical identity.

The algebraic property checks avoid this ambiguity by evaluating the relevant coefficient and variance conditions.

The module does not validate claim independence, estimate moments from data, prove consistency of supplied distributional parameters, or compute conditional premium distributions.

## 9. Verification status

Automated verification performed on Windows with Python 3.13.3:

**114 tests passed, with 100% reported statement and branch coverage**, as of October 17, 2026.

This provides software testing evidence, not an independent actuarial proof.

**Human mathematical validation: PENDING.**

Before approval, the assigned student must independently reproduce the calculations, check the stated assumptions against the cited reference and confirm the counterexamples.

Final validation remains subject to peer review and the course maintainer's approval.

## 10. References

Kaas, R., Goovaerts, M., Dhaene, J., and Denuit, M. (2008). *Modern Actuarial Risk Theory: Using R*. Second edition. Springer.

Relevant material:

- Section 3.1, page 41: collective risk model and independence assumptions.
- Section 3.2, pages 42–43: aggregate-loss expectation and variance.
- Section 5.3, page 119: classical premium principles.
- Section 5.3.1, pages 120–122: properties of premium principles.
- Table 5.1, page 122: classification of premium-principle properties.

TheRiskTheory: `docs/topic_1/premium_foundations.md`, for the foundational functions contributed in the preceding Topic 1 work.