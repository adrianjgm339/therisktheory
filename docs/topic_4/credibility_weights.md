# Buhlmann Credibility Weights

**Topic:** Mathematical Risk Theory II, Topic IV.

**Scope:** Credibility weights and nonparametric component estimation.

**Validation status:** `experimental`.

## 1. Actuarial motivation

The Buhlmann credibility model balances the experience of an
individual risk against the collective experience of a portfolio.

The credibility weight controls the influence of individual
experience. This contribution supplies the weight and its variance
components. It does not implement the final credibility premium,
partial or full credibility standards, or Bayesian credibility.

## 2. Mathematical assumptions

Consider r independent risks observed for n equal-exposure periods.

For risk i, the observations are:

    X[i,1], X[i,2], ..., X[i,n].

The classical Buhlmann model assumes that:

- The risks arise from a common population of risk parameters.
- Conditional on the risk parameter, observations are identically
  distributed and independent across periods.
- Conditional means and variances exist and are finite.
- Every risk has the same number of comparable observation periods.
- The process is stationary within each risk.
- The classical model does not account for changing exposure volumes.

Observations are represented by a two-dimensional NumPy-compatible
matrix. Rows correspond to risks and columns to periods.

These assumptions are mathematical requirements. They cannot be
verified automatically from a matrix of observed values.

## 3. EPV and VHM

The expected process variance is:

\[
EPV = E_\Theta[\operatorname{Var}(X\mid\Theta)].
\]

The variance of hypothetical means is:

\[
VHM = \operatorname{Var}_\Theta[E(X\mid\Theta)].
\]

EPV measures within-risk variation, while VHM measures variation
between the conditional means of different risks.

Both population components are non-negative.

## 4. Credibility weight

For n observation periods, the Buhlmann credibility weight is:

\[
Z = \frac{n\,VHM}{n\,VHM+EPV}.
\]

When VHM is strictly positive, this is equivalent to:

\[
Z=\frac{n}{n+K},\qquad K=\frac{EPV}{VHM}.
\]

The implementation avoids explicitly forming K and uses
numerically scaled operations.

The basic properties for non-negative EPV and VHM are:

- 0 <= Z <= 1.
- Increasing the number of periods cannot decrease Z.
- Increasing VHM cannot decrease Z.
- Increasing EPV cannot increase Z.
- Multiplying both variance components by the same positive
  constant leaves Z unchanged.

The convention for zero observation periods is Z = 0, provided
the variance components do not both vanish.

For positive n:

- EPV = 0 and VHM > 0 imply Z = 1.
- VHM = 0 and EPV > 0 imply Z = 0.
- EPV = VHM = 0 is indeterminate and is rejected.

These conventions are explicit API decisions, not new estimates
derived from data.

## 5. Nonparametric estimation

For each risk i, define the sample mean:

\[
\bar X_i=\frac{1}{n}\sum_{j=1}^n X_{ij}.
\]

Its unbiased sample process variance is:

\[
s_i^2=\frac{1}{n-1}
\sum_{j=1}^n(X_{ij}-\bar X_i)^2.
\]

The EPV estimator is:

\[
\widehat{EPV}=\frac{1}{r}\sum_{i=1}^r s_i^2.
\]

The overall mean of risk means is:

\[
\bar X=\frac{1}{r}\sum_{i=1}^r\bar X_i.
\]

The untruncated method-of-moments VHM estimator is:

\[
\widehat{VHM}
=
\frac{1}{r-1}\sum_{i=1}^r
(\bar X_i-\bar X)^2
-\frac{\widehat{EPV}}{n}.
\]

Sample variances use the n-1 or r-1 denominators indicated
above, not population-variance denominators.

At least one risk and two periods are needed to estimate EPV.
At least two risks and two periods are required to estimate VHM.

## 6. Negative VHM estimates

Although population VHM is non-negative, its untruncated
finite-sample estimator may be negative.

A negative estimate indicates that within-risk variation is
large relative to measured between-risk variation.

The API retains this signed estimate rather than silently
clipping it to zero. A negative value cannot be passed to
credibility_weight(), which requires non-negative components.

The actuarial analyst must investigate or select a justified
alternative estimation approach.

## 7. Exact reference examples

### Example 12.5.1: Positive estimated credibility

Rows represent risks and columns represent years:

\[
X=
\begin{pmatrix}
0&1&0\\
2&1&2
\end{pmatrix}.
\]

The reference values are:

\[
\widehat{EPV}=\frac13,\qquad
\widehat{VHM}=\frac79,\qquad
Z=\frac78.
\]

The weight is common to both risks because each has three
observation periods and the same portfolio parameters.

### Example 12.5.2: Negative VHM estimate

\[
X=
\begin{pmatrix}
3&0&0\\
3&0&3
\end{pmatrix}.
\]

The reference gives:

\[
\widehat{EPV}=3,\qquad
\widehat{VHM}=-\frac12.
\]

No non-negative credibility weight is inferred from this
negative estimate.

## 8. Public API contract

### credibility_weight(n, epv, vhm)

Accepts scalar or broadcast-compatible array arguments.

- n: non-negative integers, excluding booleans.
- epv: non-negative finite real values.
- vhm: non-negative finite real values.
- Returns: NumPy float64 array, including a 0-D array for
  scalar arguments.

Invalid types or domains raise TypeError or ValueError.

### estimate_process_variance(observations)

Accepts a finite, two-dimensional real observation matrix.

- Requires at least one risk and two periods.
- Returns a 0-D NumPy float64 array.
- Signed observations are permitted.

### estimate_hypothetical_mean_variance(observations)

Accepts the same balanced-data convention.

- Requires at least two risks and two periods.
- Returns a signed, untruncated 0-D NumPy float64 array.
- Negative estimates are retained as diagnostics.

Both estimator functions reject nonfinite outputs resulting
from numerical overflow.

## 9. Numerical and modeling limitations

- Computation uses float64 and is subject to rounding.
- Large observations can produce numerical overflow.
- Small positive values may underflow.
- The API does not support missing values or ragged matrices.
- It does not estimate the collective mean as a public function.
- It does not implement Buhlmann-Straub variable exposure.
- An integer n supplied to credibility_weight() is not
  automatically checked against any observation matrix.
- A successful test suite does not verify the model assumptions
  or establish independent human mathematical validation.

## 10. Verification evidence

The implementation has been checked using:

- Exact rational examples from Sections 12.5.1 and 12.5.2.
- Independent pairwise-difference variance calculations.
- Invariance under risk and period permutations.
- Translation and quadratic scaling properties.
- Monotonicity and limiting behavior of credibility weights.
- A seeded hierarchical normal simulation with theoretical
  EPV = 9 and VHM = 4.
- Pytest line and branch coverage.

The initial local suite completed 43 passing tests with 100%
line and branch coverage for the credibility package.

Human mathematical peer review and maintainer approval remain
outstanding.

## 11. References

1. Loss Data Analytics, Second Edition. Chapter 12,
   "Experience Rating Using Credibility Theory."
   Section 12.2: Buhlmann Credibility, especially Eq. (12.2).
   Section 12.5.1: Nonparametric Estimation, especially
   Eqs. (12.5) and (12.6), Examples 12.5.1 and 12.5.2.
   https://openacttexts.github.io/LDAVer2/ChapCredibility.html

2. Buhlmann, H. and Gisler, A. (2005).
   A Course in Credibility Theory and Its Applications.
   Springer. Background reference; exact theorem or page
   correspondence remains subject to human verification.

## 12. Integration boundaries

Bracamonte owns the general credibility premium interface.
Monticelli and Passanisi own partial and full credibility
methods respectively.

This module exposes only its assigned parameters and weights.
Dependent methods should use this public API rather than
duplicate its estimator formulas.

The contribution is experimental until independent mathematical
review and final maintainer approval.
