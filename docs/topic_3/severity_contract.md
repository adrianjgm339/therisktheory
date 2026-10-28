# Topic 3: Claim Severity Distribution Contract

Status: experimental.
Peer mathematical review: pending.

## 1. Scope

Implement four claim severity distributions assigned
to Garcia in the semester development plan:

1. Lognormal
2. Pareto Type I
3. Weibull
4. Scaled Beta

The implementation must support downstream aggregate
loss models without duplicating other contributors' code.

## 2. Public API

Each distribution exposes four public functions:

- lognormal_severity_pdf/cdf/ppf/moments
- pareto_severity_pdf/cdf/ppf/moments
- weibull_severity_pdf/cdf/ppf/moments
- beta_severity_pdf/cdf/ppf/moments

PDF and CDF use an observed severity x.
PPF uses a probability q in [0, 1].
Moments return mean, variance and second_raw_moment.

All outputs use float64 NumPy arrays, including 0-D
arrays for scalar inputs.

## 3. Lognormal

Parameters:
- mu: finite real log-location.
- sigma: finite positive log-standard deviation.

Definition:
    log(X) ~ Normal(mu, sigma**2)

Support: x > 0.

Moments:
    mean = exp(mu + sigma**2/2)
    second_raw_moment = exp(2*mu + 2*sigma**2)
    variance = exp(2*mu + sigma**2)*expm1(sigma**2)

SciPy: lognorm(s=sigma, loc=0, scale=exp(mu)).

## 4. Pareto Type I

Parameters:
- alpha > 0: shape.
- minimum > 0: minimum claim severity.

Support: x >= minimum.

For x >= minimum:
    F(x) = 1 - (minimum/x)**alpha

Moments:
    mean = alpha*minimum/(alpha-1), alpha > 1
    second_raw_moment = alpha*minimum**2/(alpha-2),
                        alpha > 2
    variance = alpha*minimum**2 /
               ((alpha-1)**2*(alpha-2)), alpha > 2

Mean is infinite for alpha <= 1.
Variance and second raw moment are infinite
for alpha <= 2.

SciPy: pareto(b=alpha, loc=0, scale=minimum).

## 5. Weibull

Parameters:
- shape > 0
- scale > 0

Support: x >= 0.

CDF:
    F(x) = 1 - exp(-(x/scale)**shape)

Moments:
    mean = scale*Gamma(1+1/shape)
    second_raw_moment = scale**2*Gamma(1+2/shape)
    variance = second_raw_moment - mean**2

For shape=1, Weibull reduces to exponential.

SciPy: weibull_min(c=shape, loc=0, scale=scale).

## 6. Scaled Beta

Parameters:
- alpha > 0
- beta > 0
- maximum_loss > 0, default 1

Support: 0 <= x <= maximum_loss.

Let Y ~ Beta(alpha, beta).
Then X = maximum_loss*Y.

Moments:
    mean = M*alpha/(alpha+beta)
    second_raw_moment =
        M**2*alpha*(alpha+1) /
        ((alpha+beta)*(alpha+beta+1))
    variance =
        M**2*alpha*beta /
        ((alpha+beta)**2*(alpha+beta+1))

Here M = maximum_loss.

SciPy: beta(a=alpha, b=beta, loc=0, scale=M).

## 7. Validation policy

- Reject invalid types, including bool and complex.
- Reject invalid, NaN and infinite input parameters.
- Use keyword-only distribution parameters.
- Support NumPy broadcasting.
- Distinguish infinite mathematical moments from
  numerical overflow of finite moments.
- Respect endpoint behavior for PDF, CDF and PPF.
- Permit mathematically infinite endpoint densities
  and quantiles when appropriate.
- Test empty arrays and incompatible dimensions.
- Do not silently replace invalid results with zero.

## 8. Aggregate loss integration

For an independent claim count N and iid
severity variables X_i:

    S = sum(X_i, i=1..N)
    E[S] = E[N]*E[X]
    Var(S) = E[N]*Var(X) + Var(N)*E[X]**2

For Poisson frequency with parameter lambda:

    E[S] = lambda*E[X]
    Var(S) = lambda*E[X**2]

These formulas require the appropriate finite moments.
The aggregate-loss engine belongs to another contributor.

## 9. Verification requirements

- Analytical reference calculations.
- Independent comparisons with SciPy.
- PDF, CDF and PPF consistency.
- Mean, variance and second raw moment tests.
- Boundary and invalid-input tests.
- Vectorization and broadcasting.
- Ruff and pytest checks.
- Reproducible English-language examples.
- Bibliographic references.
- Independent mathematical peer review.

## 10. References

SciPy distribution references:

https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html

https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pareto.html

https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html

https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html

Course source:
TheRiskTheory Semester Development Plan, Topic 3.

Validation status: experimental.
