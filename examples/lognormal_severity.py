"""Reproducible lognormal claim severity example."""

import numpy as np

from therisktheory.severity_aggregate import (
    lognormal_severity_cdf,
    lognormal_severity_moments,
    lognormal_severity_pdf,
    lognormal_severity_ppf,
)


def main() -> None:
    """Display key quantities for lognormal claim severity."""
    mu = np.log(100.0)
    sigma = 0.5

    moments = lognormal_severity_moments(
        mu=mu,
        sigma=sigma,
    )

    print("Distribution: Lognormal")
    print("Median:", 100.0)
    print("Mean:", moments["mean"].item())
    print("Variance:", moments["variance"].item())
    print(
        "Second raw moment:",
        moments["second_raw_moment"].item(),
    )
    print(
        "PDF at 100:",
        lognormal_severity_pdf(100.0, mu=mu, sigma=sigma).item(),
    )
    print(
        "CDF at 100:",
        lognormal_severity_cdf(100.0, mu=mu, sigma=sigma).item(),
    )
    print(
        "90th percentile:",
        lognormal_severity_ppf(0.9, mu=mu, sigma=sigma).item(),
    )


if __name__ == "__main__":
    main()
