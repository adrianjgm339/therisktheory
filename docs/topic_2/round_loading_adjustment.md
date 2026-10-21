# Topic 2: Round Loading and Unit-Step Adjustment Coefficient

**Package:** `therisktheory.finetti_ruin`

**Functions:**
- `round_safety_loading`
- `unit_step_adjustment_coefficient`

**Status:** experimental — pending independent mathematical review.

**Dependency:** Bladimir's Topic 2 gambler's ruin implementation,
based on commit `8ffde17`.

## 1. Scope

This contribution extends the fair-game and gambler's ruin
functions with two calculations for a favorable unit-step game:

1. Expected gain per round.
2. Adjustment coefficient for the associated discrete-time
   random walk.

These functions do not implement the complete De Finetti
theory or a general insurance safety index.

## 2. Mathematical model

At each independent round, the player's gain G is:

- G = +1 with probability p.
- G = -1 with probability q = 1 - p.

Each outcome is measured in units of the stake.

The implementation covers fair or favorable games:

    0.5 <= p <= 1

for expected gain, and:

    0.5 <= p < 1

for the finite adjustment-coefficient function.

The mathematical assumptions concern repeated independent
unit-step increments. The code cannot verify independence
from a supplied probability.

## 3. Expected gain per round

By the definition of mathematical expectation:

    E[G] = (+1)p + (-1)q
         = p - q
         = 2p - 1

When p = 0.5, the game is fair and E[G] = 0.

When p > 0.5, the game favors the player and E[G] > 0.

For p = 1, the gain is deterministically +1.

The function `round_safety_loading(p)` implements this
non-negative expected gain.

Terminology: this is the expected gain, or positive drift,
of the unit-step game. It is not the monetary premium loading
P - E[S] from Topic 1, nor a general actuarial safety index.

## 4. Adjustment coefficient

For a unit-step gain G, the equation for the adjustment
coefficient is:

    E[exp(-R G)] = 1

Therefore:

    p exp(-R) + q exp(R) = 1

Let y = exp(R). Multiplying by y gives:

    q y^2 - y + p = 0

Using p + q = 1, this factors as:

    (y - 1)(q y - p) = 0

The trivial solution is y = 1, or R = 0.

When 0.5 < p < 1, the positive nontrivial solution is:

    y = p / q

and hence:

    R = ln(p / q)
      = ln(p / (1 - p))

This expression is specific to the unit-step model.

### Boundary cases

- p = 0.5: R = 0. No strictly positive adjustment
  coefficient exists in the fair case.
- 0.5 < p < 1: R is strictly positive and finite.
- p = 1: the nontrivial formula tends to infinity.
  The public function excludes this endpoint.
- p < 0.5: the game has negative expected drift and is
  outside the supported favorable-game domain.

The implementation evaluates the formula using:

    log(p) - log1p(-p)

to avoid directly dividing by a very small q.

## 5. Relationship with the gambler's ruin

Bladimir's `gambler_ruin_probability` calculates the
probability of reaching zero before a specified target.

For an infinite target, initial capital z in units
of the stake, and 0.5 < p < 1:

    psi(z) = (q / p)^z
           = exp(-R z)

For p = 0.5 and finite z, eventual ruin is certain.

The unit-step model in Kaas et al. uses ruin defined
by U(t) < 0, rather than first reaching zero.
For its non-negative starting reserve u, the corresponding
unit-stake gambler starts at z = u + 1.

Thus the differing ruin boundaries must not be confused
when comparing formulas from different conventions.

## 6. Numerical example

Use initial gambler capital z = 3 and an infinite target.

| p | E[G] | R | Ruin probability |
|---|---:|---:|---:|
| 0.50 | 0.00 | 0.000000 | 1.000000 |
| 0.60 | 0.20 | 0.405465 | 0.296296 |
| 0.75 | 0.50 | 1.098612 | 0.037037 |

For p = 0.60:

    q = 0.40
    E[G] = 0.60 - 0.40 = 0.20
    R = ln(0.60 / 0.40) = ln(1.5)
    psi(3) = (0.40 / 0.60)^3 = 8/27

These values can be checked independently of Python.

## 7. Numerical implementation

The public functions use NumPy arrays and support
vectorized inputs.

Scalar inputs return zero-dimensional float64 arrays.

Inputs that are nonnumeric, NaN or infinite are rejected.
Probabilities outside the supported domain are rejected.

The adjustment coefficient is restricted to finite values.

Both functions reuse the validation utilities introduced
in the earlier premium-foundations contribution.

## 8. Verification

Automated tests include:

- Known analytical examples.
- Vectorized probabilities.
- Fair-game boundary.
- Probabilities close to 0.5 and 1.
- Invalid types and domains.
- The adjustment-coefficient root equation.
- Agreement with Bladimir's gambler's ruin probability.
- Agreement between expected drift and the logarithmic
  expression for the coefficient.
- Docstring verification.

As of October 20, 2026, the local test suite reported
179 passing tests and 100% line and branch coverage.

Passing automated tests is not equivalent to independent
mathematical validation.

**Mathematical review status: PENDING.**

## 9. Limitations

- Only independent unit-step games with gains +1 or -1.
- No arbitrary gain or loss distribution.
- No time-varying win probabilities.
- No general reserve-fund optimization.
- No dividend optimization or full De Finetti theorem.
- No general safety-index implementation.
- No insurance premium or reinsurance contract model.
- No guarantee that a supplied probability corresponds
  to a valid empirical or actuarial model.

These limitations must be preserved when integrating this
work with other course topics.

## 10. References

Kaas, R., Goovaerts, M., Dhaene, J., and Denuit, M. (2008).
*Modern Actuarial Risk Theory: Using R*. Second edition.
Springer. Section 4.5, particularly equation (4.34)
and the discrete-time ruin discussion.

TheRiskTheory:
`docs/topic_2/gambler_ruin.md`, for the underlying
gambler's ruin implementation and the comparison of
ruin-boundary conventions.
