"""Reproducible Pareto Type I claim severity example."""

from therisktheory.severity_aggregate import (
    pareto_severity_cdf,
    pareto_severity_moments,
    pareto_severity_pdf,
    pareto_severity_ppf,
)


def main() -> None:
    """Display severity statistics for Pareto Type I."""
    alpha = 3.0
    minimum = 100.0

    moments = pareto_severity_moments(
        alpha=alpha,
        minimum=minimum,
    )

    print("Distribution: Pareto Type I")
    print("Shape:", alpha)
    print("Minimum:", minimum)
    print("Mean:", moments["mean"].item())
    print("Variance:", moments["variance"].item())
    print(
        "Second raw moment:",
        moments["second_raw_moment"].item(),
    )
    print(
        "PDF at 100:",
        pareto_severity_pdf(100, alpha=alpha, minimum=minimum).item(),
    )
    print(
        "CDF at 150:",
        pareto_severity_cdf(150, alpha=alpha, minimum=minimum).item(),
    )
    print(
        "90th percentile:",
        pareto_severity_ppf(0.9, alpha=alpha, minimum=minimum).item(),
    )

    heavy_tail = pareto_severity_moments(
        alpha=1.5,
        minimum=minimum,
    )

    print(
        "Variance at alpha=1.5:",
        heavy_tail["variance"].item(),
    )


if __name__ == "__main__":
    main()
