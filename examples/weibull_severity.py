"""Reproducible Weibull claim severity example."""

from therisktheory.severity_aggregate import (
    weibull_severity_cdf,
    weibull_severity_moments,
    weibull_severity_pdf,
    weibull_severity_ppf,
)


def main() -> None:
    """Display Weibull severity statistics."""
    shape = 2.0
    scale = 100.0

    result = weibull_severity_moments(
        shape=shape,
        scale=scale,
    )

    print("Distribution: Weibull")
    print("Shape:", shape)
    print("Scale:", scale)
    print("Mean:", result["mean"].item())
    print("Variance:", result["variance"].item())
    print(
        "Second raw moment:",
        result["second_raw_moment"].item(),
    )

    print(
        "PDF at 100:",
        weibull_severity_pdf(100, shape=shape, scale=scale).item(),
    )

    print(
        "CDF at 100:",
        weibull_severity_cdf(100, shape=shape, scale=scale).item(),
    )

    print(
        "90th percentile:",
        weibull_severity_ppf(0.9, shape=shape, scale=scale).item(),
    )

    exponential = weibull_severity_moments(
        shape=1,
        scale=100,
    )

    print(
        "Exponential mean at shape=1:",
        exponential["mean"].item(),
    )


if __name__ == "__main__":
    main()
