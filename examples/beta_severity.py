"""Reproducible scaled Beta claim severity example."""

from therisktheory.severity_aggregate import (
    beta_severity_cdf,
    beta_severity_moments,
    beta_severity_pdf,
    beta_severity_ppf,
)


def main() -> None:
    """Show moments and probabilities of bounded claim severity."""
    alpha = 2.0
    beta = 3.0
    maximum_loss = 1000.0

    result = beta_severity_moments(
        alpha=alpha,
        beta=beta,
        maximum_loss=maximum_loss,
    )

    print("Distribution: Scaled Beta")
    print("Alpha:", alpha)
    print("Beta:", beta)
    print("Maximum loss:", maximum_loss)
    print("Mean:", result["mean"].item())
    print("Variance:", result["variance"].item())

    print(
        "Second raw moment:",
        result["second_raw_moment"].item(),
    )

    print(
        "PDF at 400:",
        beta_severity_pdf(
            400,
            alpha=alpha,
            beta=beta,
            maximum_loss=maximum_loss,
        ).item(),
    )

    print(
        "CDF at 400:",
        beta_severity_cdf(
            400,
            alpha=alpha,
            beta=beta,
            maximum_loss=maximum_loss,
        ).item(),
    )

    print(
        "90th percentile:",
        beta_severity_ppf(
            0.9,
            alpha=alpha,
            beta=beta,
            maximum_loss=maximum_loss,
        ).item(),
    )

    print(
        "Singular density at zero:",
        beta_severity_pdf(
            0,
            alpha=0.5,
            beta=2,
            maximum_loss=maximum_loss,
        ).item(),
    )


if __name__ == "__main__":
    main()
