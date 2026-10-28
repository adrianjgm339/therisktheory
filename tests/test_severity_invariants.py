"""Cross-distribution mathematical invariants for claim severity.

These are automated checks, not independent human validation.
"""

import inspect

import numpy as np
import pytest
from scipy.stats import beta as scipy_beta
from scipy.stats import lognorm, pareto, weibull_min

from therisktheory import severity_aggregate as severity

CASES = (
    (
        "lognormal",
        {"mu": np.log(100.0), "sigma": 0.5},
        {"mu": np.log(300.0), "sigma": 0.5},
        lognorm(s=0.5, scale=100.0),
        120.0,
        500.0,
    ),
    (
        "pareto",
        {"alpha": 3.0, "minimum": 100.0},
        {"alpha": 3.0, "minimum": 300.0},
        pareto(b=3.0, scale=100.0),
        150.0,
        500.0,
    ),
    (
        "weibull",
        {"shape": 2.0, "scale": 100.0},
        {"shape": 2.0, "scale": 300.0},
        weibull_min(c=2.0, scale=100.0),
        120.0,
        500.0,
    ),
    (
        "beta",
        {
            "alpha": 2.0,
            "beta": 3.0,
            "maximum_loss": 1000.0,
        },
        {
            "alpha": 2.0,
            "beta": 3.0,
            "maximum_loss": 3000.0,
        },
        scipy_beta(a=2.0, b=3.0, scale=1000.0),
        400.0,
        1000.0,
    ),
)


def test_complete_public_api():
    """All sixteen required public functions must be exported."""
    kinds = ("lognormal", "pareto", "weibull", "beta")
    operations = ("pdf", "cdf", "ppf", "moments")

    expected = {f"{kind}_severity_{operation}" for kind in kinds for operation in operations}

    assert len(severity.__all__) == 16
    assert set(severity.__all__) == expected

    for name in expected:
        fn = getattr(severity, name)
        signature = inspect.signature(fn)

        assert callable(fn)
        assert fn.__doc__
        assert signature.return_annotation is not inspect.Signature.empty

        parameters = list(signature.parameters.values())

        if name.endswith("_moments"):
            assert all(parameter.kind == inspect.Parameter.KEYWORD_ONLY for parameter in parameters)
        else:
            assert parameters[0].name in ("x", "q")
            assert all(
                parameter.kind == inspect.Parameter.KEYWORD_ONLY for parameter in parameters[1:]
            )

        module = inspect.getmodule(fn)

        assert "experimental" in module.__doc__.lower()


@pytest.mark.parametrize(
    "kind,params,scaled,reference,x,grid_max",
    CASES,
)
def test_independent_scipy_reference(kind, params, scaled, reference, x, grid_max):
    """PDF, CDF, quantiles and moments agree with SciPy."""
    pdf = getattr(severity, f"{kind}_severity_pdf")
    cdf = getattr(severity, f"{kind}_severity_cdf")
    ppf = getattr(severity, f"{kind}_severity_ppf")
    moments = getattr(severity, f"{kind}_severity_moments")

    np.testing.assert_allclose(
        pdf(x, **params),
        reference.pdf(x),
        rtol=1e-11,
        atol=1e-14,
    )

    np.testing.assert_allclose(
        cdf(x, **params),
        reference.cdf(x),
        rtol=1e-11,
        atol=1e-14,
    )

    np.testing.assert_allclose(
        ppf(0.9, **params),
        reference.ppf(0.9),
        rtol=1e-11,
        atol=1e-12,
    )

    actual = moments(**params)

    np.testing.assert_allclose(
        actual["mean"],
        reference.mean(),
        rtol=1e-11,
    )

    np.testing.assert_allclose(
        actual["variance"],
        reference.var(),
        rtol=1e-11,
    )


@pytest.mark.parametrize(
    "kind,params,scaled,reference,x,grid_max",
    CASES,
)
def test_scale_invariants_and_second_moment(kind, params, scaled, reference, x, grid_max):
    """Scaling must preserve fundamental distribution identities."""
    pdf = getattr(severity, f"{kind}_severity_pdf")
    cdf = getattr(severity, f"{kind}_severity_cdf")
    ppf = getattr(severity, f"{kind}_severity_ppf")
    moments = getattr(severity, f"{kind}_severity_moments")

    factor = 3.0

    np.testing.assert_allclose(
        pdf(factor * x, **scaled),
        pdf(x, **params) / factor,
        rtol=1e-11,
        atol=1e-14,
    )

    np.testing.assert_allclose(
        cdf(factor * x, **scaled),
        cdf(x, **params),
        rtol=1e-11,
        atol=1e-14,
    )

    np.testing.assert_allclose(
        ppf(0.9, **scaled),
        factor * ppf(0.9, **params),
        rtol=1e-11,
    )

    original = moments(**params)
    enlarged = moments(**scaled)

    np.testing.assert_allclose(
        original["second_raw_moment"],
        original["variance"] + original["mean"] ** 2,
        rtol=1e-11,
    )

    for key, power in (
        ("mean", 1),
        ("variance", 2),
        ("second_raw_moment", 2),
    ):
        np.testing.assert_allclose(
            enlarged[key],
            factor**power * original[key],
            rtol=1e-11,
        )


@pytest.mark.parametrize(
    "kind,params,scaled,reference,x,grid_max",
    CASES,
)
def test_monotonicity_and_quantile_roundtrip(kind, params, scaled, reference, x, grid_max):
    """CDF and PPF must be monotone and internally consistent."""
    cdf = getattr(severity, f"{kind}_severity_cdf")
    ppf = getattr(severity, f"{kind}_severity_ppf")

    grid = np.linspace(0, grid_max, 51)
    cumulative = cdf(grid, **params)

    assert np.all(np.diff(cumulative) >= -1e-14)
    assert np.all((cumulative >= 0) & (cumulative <= 1))

    probabilities = np.array([0.001, 0.02, 0.1, 0.3, 0.7, 0.9, 0.99])

    quantiles = ppf(probabilities, **params)

    assert np.all(np.diff(quantiles) >= 0)

    np.testing.assert_allclose(
        cdf(quantiles, **params),
        probabilities,
        rtol=1e-10,
        atol=1e-12,
    )


def test_pareto_divergent_moments():
    """Pareto moment divergence must not be treated as overflow."""
    shapes = np.array([0.75, 1.0, 1.5, 2.0, 3.0])

    result = severity.pareto_severity_moments(
        alpha=shapes,
        minimum=100.0,
    )

    assert np.all(np.isposinf(result["mean"][:2]))
    assert np.all(np.isfinite(result["mean"][2:]))

    assert np.all(np.isposinf(result["variance"][:4]))
    assert np.isfinite(result["variance"][4])

    assert np.all(np.isposinf(result["second_raw_moment"][:4]))


def test_singular_density_is_not_point_mass():
    """Infinite boundary PDF does not imply positive boundary mass."""
    assert np.isposinf(severity.weibull_severity_pdf(0.0, shape=0.5, scale=1.0))

    assert severity.weibull_severity_cdf(0.0, shape=0.5, scale=1.0).item() == 0

    assert np.isposinf(severity.beta_severity_pdf(0.0, alpha=0.5, beta=2.0))

    assert severity.beta_severity_cdf(0.0, alpha=0.5, beta=2.0).item() == 0
