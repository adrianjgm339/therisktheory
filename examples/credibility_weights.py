"""Demonstrate classical Buhlmann credibility component estimation."""

import numpy as np

from therisktheory.credibility import (
    credibility_weight,
    estimate_hypothetical_mean_variance,
    estimate_process_variance,
)


def main() -> None:
    """Reproduce two published nonparametric credibility examples."""
    observations = np.array([[0.0, 1.0, 0.0], [2.0, 1.0, 2.0]])

    epv = estimate_process_variance(observations)
    vhm = estimate_hypothetical_mean_variance(observations)
    periods = observations.shape[1]

    weight = credibility_weight(periods, epv, vhm)

    np.testing.assert_allclose(epv, 1 / 3)
    np.testing.assert_allclose(vhm, 7 / 9)
    np.testing.assert_allclose(weight, 7 / 8)

    print("Buhlmann credibility components")
    print(f"Number of risks: {observations.shape[0]}")
    print(f"Number of periods: {periods}")
    print(f"Estimated EPV: {epv.item():.6f}")
    print(f"Estimated VHM: {vhm.item():.6f}")
    print(f"Credibility weight: {weight.item():.6f}")
    print("POSITIVE_REFERENCE=PASS")

    negative_example = np.array([[3.0, 0.0, 0.0], [3.0, 0.0, 3.0]])

    negative_epv = estimate_process_variance(negative_example)
    negative_vhm = estimate_hypothetical_mean_variance(negative_example)

    np.testing.assert_allclose(negative_epv, 3.0)
    np.testing.assert_allclose(negative_vhm, -0.5)

    try:
        credibility_weight(
            negative_example.shape[1],
            negative_epv,
            negative_vhm,
        )
    except ValueError as error:
        if "non-negative" not in str(error):
            raise
    else:
        raise AssertionError("Negative VHM must not yield a credibility weight")

    print(f"Diagnostic EPV: {negative_epv.item():.6f}")
    print(f"Diagnostic VHM: {negative_vhm.item():.6f}")
    print("NEGATIVE_REFERENCE=PASS")
    print("T4_A5_0_EXAMPLE=PASS")


if __name__ == "__main__":
    main()
