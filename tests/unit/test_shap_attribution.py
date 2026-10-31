"""Exact per-signal SHAP attribution for a fitted polynomial's terms."""

import numpy as np
import pandas as pd
import pytest

from forecasting_engine.models.base import ModelDescription
from forecasting_engine.reporting.polynomial_function import Origin, from_description
from forecasting_engine.reporting.shap_attribution import polynomial_shap

X, Y = "x", "y"


def _fn(terms, coefficients, *, intercept=0.0, standardisation=None):
    return from_description(
        ModelDescription(
            "DerivedPolynomial",
            tuple(terms),
            tuple(coefficients),
            intercept,
            standardisation=standardisation,
        ),
        origin=Origin.DERIVED,
        target="SPX",
        horizon=5,
        columns=[X, Y],
    )


def _frame(**columns) -> pd.DataFrame:
    return pd.DataFrame(columns)


def test_a_degree_1_shap_equals_coefficient_times_value_minus_training_mean():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    fn = _fn([X], [2.5])

    shap = polynomial_shap(fn, _frame(**{X: x}))

    expected = 2.5 * (x - x.mean())
    assert shap.terms == (X,)
    assert shap.coefficients[0] == pytest.approx(np.abs(expected).mean())


def test_a_square_term_attributes_wholly_to_its_one_signal():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    fn = _fn([f"{X}^2"], [1.5])

    shap = polynomial_shap(fn, _frame(**{X: x}))

    expected = 1.5 * (x**2 - (x**2).mean())
    assert shap.terms == (X,)
    assert shap.coefficients[0] == pytest.approx(np.abs(expected).mean())


def test_an_interaction_term_is_shared_between_its_two_signals():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([5.0, 1.0, -2.0, 0.5, 3.0])
    fn = _fn([f"{X} {Y}"], [2.0])

    shap = polynomial_shap(fn, _frame(**{X: x, Y: y}))

    assert set(shap.terms) == {X, Y}
    assert all(c > 0 for c in shap.coefficients)


def test_an_interaction_terms_two_shares_sum_to_the_terms_own_contribution():
    """Shapley values for a product are exact: per row, the two signals'
    shares always add back up to what the term itself contributed."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([5.0, 1.0, -2.0, 0.5, 3.0])
    coefficient = 2.0
    fn = _fn([f"{X} {Y}"], [coefficient])

    # Recover the per-row shares the same way polynomial_shap does internally,
    # via a term whose contribution is attributed to one signal at a time.
    from forecasting_engine.reporting.shap_attribution import _term_shapley

    shares = _term_shapley(fn.terms[0], None, None, _frame(**{X: x, Y: y}))

    # The conservation baseline is the term evaluated at both factors' own
    # means together (x̄·ȳ), not the mean of the row-wise product x·y — the
    # two differ whenever x and y are correlated in the sample.
    own_contribution = coefficient * (x * y - x.mean() * y.mean())
    assert shares[X] + shares[Y] == pytest.approx(own_contribution)


def test_terms_sharing_a_signal_are_summed_into_one_entry():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    fn = _fn([X, f"{X}^2"], [2.0, 0.5])

    shap = polynomial_shap(fn, _frame(**{X: x}))

    expected = 2.0 * (x - x.mean()) + 0.5 * (x**2 - (x**2).mean())
    assert shap.terms == (X,)
    assert shap.coefficients[0] == pytest.approx(np.abs(expected).mean())


def test_a_standardised_term_uses_z_not_the_raw_signal():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mean, sd = 2.0, 1.0
    fn = _fn([X], [3.0], standardisation={X: (mean, sd)})

    shap = polynomial_shap(fn, _frame(**{X: x}))

    z = (x - mean) / sd
    expected = 3.0 * (z - z.mean())
    assert shap.coefficients[0] == pytest.approx(np.abs(expected).mean())


def test_an_unexpandable_formula_has_no_shap():
    fn = from_description(
        ModelDescription("UserPolynomial", ("x / y",), (1.0,)),
        origin=Origin.USER_SUPPLIED,
        target="SPX",
        horizon=5,
    )

    assert polynomial_shap(fn, _frame(**{X: [1.0], Y: [2.0]})) is None


def test_no_surviving_terms_gives_an_empty_table():
    fn = _fn([], [])

    shap = polynomial_shap(fn, _frame(**{X: [1.0, 2.0]}))

    assert shap.terms == () and shap.coefficients == ()
