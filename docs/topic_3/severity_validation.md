# Topic 3 — Severity Mathematical Validation Record

Status: **experimental**
Date of local baseline verification: 2026-10-28

## Purpose

Document the assumptions, numerical evidence, limitations,
and review requirements for Garcia's four severity models.

This is not a claim of independent mathematical validation.

## Implemented distributions

| Distribution | Parameters | SciPy correspondence |
| --- | --- | --- |
| Lognormal | mu, sigma | lognorm(s=sigma, scale=exp(mu)) |
| Pareto Type I | alpha, minimum | pareto(b=alpha, scale=minimum) |
| Weibull | shape, scale | weibull_min(c=shape, scale=scale) |
| Scaled Beta | alpha, beta, maximum_loss | beta(a=alpha, b=beta, scale=maximum_loss) |

All SciPy correspondences use loc=0.

The four public operations per distribution are:

- pdf: probability density
- cdf: cumulative distribution function
- ppf: quantile or inverse CDF
- moments: mean, variance, second raw moment

Public functions are exported by
therisktheory.severity_aggregate.

See severity_contract.md for mathematical definitions.

## Mathematical conditions

Lognormal:
- sigma must be positive.
- All theoretical moments are finite for finite parameters.

Pareto Type I:
- alpha and minimum must be positive.
- The mean diverges for alpha <= 1.
- The variance and second raw moment diverge for alpha <= 2.

Weibull:
- shape and scale must be positive.
- The PDF at zero is singular for shape < 1.
- Shape=1 is the exponential special case.

Scaled Beta:
- alpha, beta and maximum_loss must be positive.
- Support is [0, maximum_loss].
- Boundary PDF values can be singular for shapes below 1.
- All theoretical moments are finite for valid parameters.

## Local automated verification

Environment:
- Windows
- Python 3.13.3
- NumPy 2.5.4
- SciPy 1.18.1

Verified before the new invariant test file:
- 183 severity tests passed.
- 188 total tests passed.
- 100% line and branch coverage of severity_aggregate.
- Ruff passed on the Topic 3 files.
- All four actuarial examples executed successfully.
- Git audit passed: 4 topic commits and 14 changed paths.
- Sixteen public functions checked.
- Scaling, CDF, quantile, moment and boundary identities passed.

Reproducible integration checks are stored in:
tests/test_severity_invariants.py

The additional invariant tests must pass locally before
the documentation changes are committed.

## Numerical interpretation

A PDF may equal positive infinity at a boundary without
implying a discrete probability mass.

A mathematically divergent Pareto moment is represented
by positive infinity.

OverflowError is used when a mathematically finite
quantity cannot be represented within float64.

The implementations use NumPy broadcasting and float64
arrays, including zero-dimensional scalar results.

No claim of accuracy for every extreme representable
floating-point input is made. Further edge-case review
may identify numerical improvements.

## Integration boundary

This contribution implements severity distributions,
not annual aggregate-loss calculations.

For independent iid severities and claim count:

E[S] = E[N] E[X]

Var(S) = E[N] Var(X) + Var(N) E[X]^2

These formulas require appropriate finite moments.
The downstream aggregate engine is outside this PR.

## Known integration and review constraints

- Mathematical peer review is pending.
- Final instructor approval is pending.
- GitHub Actions have not been confirmed passing for
  this Topic 3 contribution.
- Official repository CI targets Python 3.11 and 3.12;
  the recorded local run used Python 3.13.3.
- The upstream baseline has a pre-existing Ruff I001
  issue in examples/pure_premium.py. This contribution
  does not modify that file.
- On the Windows test environment, pytest-cov worked
  when coverage targeted the severity source directory
  and external pytest plugin autoload was disabled.
- Other students' modules are not modified.

## Human review checklist

- [ ] Verify all four mathematical definitions.
- [ ] Confirm PDF normalization and support.
- [ ] Verify moment derivations and existence conditions.
- [ ] Review parameter naming and SciPy correspondence.
- [ ] Check boundary and extreme numerical behavior.
- [ ] Confirm suitability for aggregate-loss integration.
- [ ] Verify bibliography and academic terminology.
- [ ] Review tests and reproducibility.
- [ ] Obtain independent contributor review.
- [ ] Obtain final approval from Prof. Luis Pitta.

## References

1. SciPy: scipy.stats.lognorm
   https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html

2. SciPy: scipy.stats.pareto
   https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pareto.html

3. SciPy: scipy.stats.weibull_min
   https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html

4. SciPy: scipy.stats.beta
   https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html

5. NIST/SEMATECH e-Handbook of Statistical Methods,
   section 8.1.6.4, Lognormal.
   https://www.itl.nist.gov/div898/handbook/apr/section1/apr164.htm

These are technical and mathematical reference materials.
Independent academic review remains required.

## AI-assisted development

AI assistance was used for implementation structure,
test generation, documentation and numerical checks.

AI output is not accepted as a mathematical authority.
Human validation is still pending.

Current validation state: experimental.
