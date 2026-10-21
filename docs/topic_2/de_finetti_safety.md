# Topic 2: De Finetti Foundations and Unit-Step Safety Measures

**Status:** experimental — pending independent mathematical review.

## 1. Scope and motivation

This contribution introduces two public safety-measure interfaces
for an independent unit-step reserve process.

The objective is to connect positive expected surplus, the adjustment
coefficient and eventual ruin probability in one analytically
verifiable case.

This is an introductory special case, not a complete implementation
of De Finetti's insurance or dividend-optimization theory.

## 2. Reserve model and assumptions

Let G be an independent per-round surplus increment:

- G = +1 with probability p.
- G = -1 with probability q = 1-p.

The reserve process is U(n) = u + sum(G_i), starting at u >= 0.

Under this model, positive expected surplus requires p > 0.5.

## 3. Safety loading in the unit-step model

The expected surplus increment is:

    E[G] = p-q = 2p-1

The function `de_finetti_safety_loading(p)` returns this value.

It reuses `round_safety_loading(p)` rather than duplicating
the formula.

This quantity is not an insurance premium loading coefficient.
The monetary premium loading of Topic 1 is a different object.

## 4. Safety index in the unit-step model

The exponential adjustment coefficient R satisfies:

    E[exp(-R G)] = 1

Therefore:

    p exp(-R) + q exp(R) = 1

For 0.5 < p < 1, its strictly positive solution is:

    R = ln(p/q)

The function `safety_index(p)` reuses the existing
`unit_step_adjustment_coefficient(p)` implementation.

At p = 0.5, the boundary value R = 0 is returned.
At p = 1, there is no finite positive nontrivial solution.

## 5. Relationship with ruin probability

For a unit-step process in which ruin means U(n) < 0,
starting from a non-negative integer reserve u:

    psi(u) = (q/p)^(u+1)
           = exp(-R(u+1))

when 0.5 < p < 1.

This implies:

    psi(u) <= exp(-R u)

The shift by one unit accounts for the strict negative-reserve
ruin convention. Bladimir's gambler model instead stops at zero.

This connection is tested using
`gambler_ruin_probability(u+1, np.inf, p)`.

## 6. Example

For p = 0.60:

    q = 0.40
    E[G] = 0.20
    R = ln(1.5) = 0.405465...

Starting at reserve u = 2, ruin requires reaching -1.
The equivalent gambler starts at z = 3.

    psi(2) = (2/3)^3 = 8/27 = 0.296296...

## 7. Public interfaces and limitations

- `safety_index(win_probability)`
- `de_finetti_safety_loading(win_probability)`

Both accept NumPy-compatible scalar and array inputs.
Both return float64 arrays and reuse prior validated functions.

These functions do not handle arbitrary claim distributions,
premium income processes, non-unit gains, dividend policies,
reserve optimization, or general reinsurance decisions.

The general interpretation and scope of De Finetti's safety
measures must be checked with the course professor before
claiming that the whole assigned theoretical topic is complete.

## 8. References and validation

Kaas, R., Goovaerts, M., Dhaene, J. and Denuit, M. (2008).
*Modern Actuarial Risk Theory: Using R*, second edition,
Springer, Section 4.5.

Moriconi, F. (1982). *A Pricing Model in a Sensitive Insurance
Market*. ASTIN Bulletin, 13(2), 135-149.
https://doi.org/10.1017/S0515036100004712

The latter source identifies the safety index with an adjustment
coefficient in an exponential ruin bound. Its complete pricing
model is not implemented here.

Peer mathematical review: PENDING.
Maintainer mathematical validation: PENDING.
