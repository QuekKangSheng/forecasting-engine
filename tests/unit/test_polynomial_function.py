"""The fitted polynomial as a readable function: parsing, expansion, rendering."""

import numpy as np
import pandas as pd
import pytest

from forecasting_engine.ingest.align import FeaturePanel
from forecasting_engine.models.base import ModelDescription
from forecasting_engine.models.polynomial import UserPolynomial
from forecasting_engine.models.sign_ruled import SignRuledPolynomial
from forecasting_engine.reporting.factor_labels import labeller
from forecasting_engine.reporting.polynomial_function import (
    MAX_EXPANDED_TERMS,
    Origin,
    PolynomialFunction,
    Term,
    dataset_fingerprint,
    from_description,
    shape_latex,
    significant,
    standardisation_lines,
    term_rows,
    to_latex,
)

VIX, IG = "VIX_Index_PX_LAST", "LUACOAS_Index_PX_LAST"


def derived(terms, coefficients, intercept=0.001, columns=None):
    return from_description(
        ModelDescription("DerivedPolynomial", tuple(terms), tuple(coefficients), intercept),
        origin=Origin.DERIVED,
        target="SPX_Index_PX_LAST",
        horizon=5,
        columns=columns,
    )


def user(formula):
    return from_description(
        ModelDescription("UserPolynomial", (formula,), (1.0,)),
        origin=Origin.USER_SUPPLIED,
        target="SPX_Index_PX_LAST",
        horizon=1,
    )


def as_dict(fn: PolynomialFunction) -> dict:
    return {term.factors: term.coefficient for term in fn.terms}


# --- derived fits ------------------------------------------------------------


def test_a_derived_power_term_is_parsed():
    fn = derived([f"{VIX}^2"], [0.5])
    assert as_dict(fn) == {((VIX, 2),): 0.5}


def test_a_derived_interaction_term_is_parsed():
    fn = derived([f"{VIX} {IG}"], [-0.2])
    assert as_dict(fn) == {((IG, 1), (VIX, 1)): -0.2}


def test_a_power_times_an_interaction_is_parsed():
    fn = derived([f"{VIX}^2 {IG}"], [0.3])
    assert as_dict(fn) == {((IG, 1), (VIX, 2)): 0.3}


def test_factors_are_sorted_by_column_whatever_order_the_model_used():
    assert derived([f"{VIX} {IG}"], [1.0]).terms[0].factors[0][0] == IG


def test_a_missing_intercept_becomes_zero():
    assert derived([VIX], [0.1], intercept=None).intercept == 0.0


def test_a_fit_with_no_surviving_terms_keeps_its_intercept():
    fn = derived([], [], intercept=0.0042)
    assert fn.terms == ()
    assert fn.intercept == 0.0042
    assert fn.formula is None


def test_an_unparseable_term_name_is_kept_as_one_raw_factor():
    fn = derived(["not^a^term"], [0.7])
    assert fn.terms == (Term(coefficient=0.7, factors=(("not^a^term", 1),)),)


def test_origin_target_and_horizon_are_recorded_as_given():
    fn = derived([VIX], [0.1])
    assert (fn.origin, fn.target, fn.horizon) == (Origin.DERIVED, "SPX_Index_PX_LAST", 5)


def test_a_column_name_containing_a_space_is_not_split_into_wrong_factors():
    # The field half of a column name comes straight from the export's header
    # and isn't sanitised, so "SPX_Index_Last Price" is possible. Splitting its
    # square on the space would give two plausible, wrong factors. Checked
    # against the known columns, it stays one raw factor instead.
    column = "SPX_Index_Last Price"
    fn = derived([f"{column}^2"], [0.4], columns=[column, VIX])
    assert fn.terms == (Term(coefficient=0.4, factors=((f"{column}^2", 1),)),)


def test_known_columns_still_parse_normally():
    fn = derived([f"{VIX}^2 {IG}"], [0.3], columns=[VIX, IG])
    assert as_dict(fn) == {((IG, 1), (VIX, 2)): 0.3}


# --- user-supplied formulas --------------------------------------------------


def test_a_plain_sum_expands_into_its_terms():
    assert as_dict(user("a + b")) == {(("a", 1),): 1.0, (("b", 1),): 1.0}


def test_a_squared_sum_expands_binomially():
    assert as_dict(user("(a + b)**2")) == {
        (("a", 2),): 1.0,
        (("a", 1), ("b", 1)): 2.0,
        (("b", 2),): 1.0,
    }


def test_a_constant_moves_into_the_intercept():
    fn = user("3 + 2*a")
    assert fn.intercept == 3.0
    assert as_dict(fn) == {(("a", 1),): 2.0}


def test_unary_minus_negates():
    assert as_dict(user("-a")) == {(("a", 1),): -1.0}


def test_division_by_a_constant_scales_the_coefficient():
    assert as_dict(user("a / 2")) == {(("a", 1),): 0.5}


def test_like_terms_that_cancel_are_dropped():
    fn = user("a - a + b")
    assert as_dict(fn) == {(("b", 1),): 1.0}


def test_a_power_of_zero_is_one():
    fn = user("a**0")
    assert fn.terms == ()
    assert fn.intercept == 1.0


def test_dividing_by_a_signal_falls_back_to_the_formula_as_entered():
    fn = user("a / b")
    assert (fn.terms, fn.intercept, fn.formula) == ((), 0.0, "a / b")


def test_an_expansion_past_the_cap_falls_back_to_the_formula():
    # (a+b+c+d+e)**4 has C(8, 4) = 70 distinct monomials.
    fn = user("(a + b + c + d + e)**4")
    assert MAX_EXPANDED_TERMS == 50
    assert fn.terms == ()
    assert fn.formula == "(a + b + c + d + e)**4"


def test_dividing_by_zero_falls_back_rather_than_raising():
    fn = user("a / (1 - 1)")
    assert fn.formula == "a / (1 - 1)"


def test_a_user_formula_has_no_unexpanded_formula_when_it_expands():
    assert user("a * b").formula is None


@pytest.mark.parametrize(
    "formula",
    [
        "2*a + 3*b**2 - a*b/4",
        "(a - b)**3",
        "-(a + 2)*(b - 1) + 5",
        "a**0 + c",
        "(a + b + c)**2 / 3",
        "+a - -b",
    ],
)
def test_the_expanded_terms_compute_exactly_what_the_fitted_function_does(formula):
    # Tests the algebra, not the formatting: the expanded terms, evaluated on
    # random data, must equal the fitted UserPolynomial's own prediction.
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-01", periods=40, freq="D")
    frame = pd.DataFrame(
        {name: rng.normal(size=40) for name in ("a", "b", "c", "target")}, index=idx
    )
    panel = FeaturePanel(frame=frame, signals=("a", "b", "c"), targets=("target",), lag_days=1)
    model = UserPolynomial(formula)
    model.fit(panel, idx)

    expected = model.predict(panel, idx)
    fn = from_description(
        model.describe(), origin=Origin.USER_SUPPLIED, target="SPX_Index_PX_LAST", horizon=1
    )
    assert fn.formula is None, "these formulas should all expand"
    actual = pd.Series(fn.intercept, index=idx)
    for term in fn.terms:
        product = pd.Series(1.0, index=idx)
        for column, power in term.factors:
            product = product * frame[column] ** power
        actual = actual + term.coefficient * product

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


# --- significant figures -----------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.000231, "0.0002310"),
        (0.0, "0"),
        (-0.000231, "−0.0002310"),
        (1234.5678, "1235"),
        (12345.678, "12350"),
        (1.0, "1.000"),
        (9.9996, "10.00"),
        (0.1, "0.1000"),
    ],
)
def test_significant_rounds_to_four_figures_and_keeps_trailing_zeros(value, expected):
    assert significant(value) == expected


def test_below_the_lower_boundary_is_scientific():
    assert significant(0.0000123456) == "1.235 × 10^-5"
    assert significant(-0.0000123456) == "−1.235 × 10^-5"


def test_exactly_the_lower_boundary_is_not_scientific():
    assert significant(1e-4) == "0.0001000"


def test_at_the_upper_boundary_is_scientific():
    assert significant(1e6) == "1.000 × 10^6"
    assert significant(999_900.0) == "999900"


def test_a_non_finite_value_does_not_raise():
    assert significant(float("nan")) == "nan"


# --- LaTeX -------------------------------------------------------------------


def label_for(*columns):
    return labeller(columns)


def test_the_equation_writes_signs_between_terms():
    fn = derived([f"{VIX}^2", f"{VIX} {IG}"], [0.0004521, -0.000231], intercept=0.001234)
    latex = to_latex(fn, label_for(VIX, IG))
    assert latex == (
        r"\hat{y} = 0.001234 + 0.0004521\,\text{VIX}^{2}"
        r" - 0.0002310\,\text{US IG credit spread}\,\text{VIX}"
    )


def test_a_zero_intercept_is_omitted_and_a_leading_minus_kept():
    fn = derived([VIX], [-0.5], intercept=0.0)
    assert to_latex(fn, label_for(VIX)) == r"\hat{y} = -0.5000\,\text{VIX}"


def test_a_coefficient_of_one_is_dropped_but_its_sign_kept():
    fn = derived([VIX, IG], [1.0, -1.0], intercept=0.0)
    assert to_latex(fn, label_for(VIX, IG)) == (
        r"\hat{y} = \text{VIX} - \text{US IG credit spread}"
    )


def test_power_one_has_no_exponent():
    assert r"\text{VIX}^" not in to_latex(derived([VIX], [0.2]), label_for(VIX))


def test_latex_special_characters_in_a_label_are_escaped():
    spx = "SPX_Index_PX_LAST"
    fn = derived([spx], [0.2], intercept=0.0)
    assert to_latex(fn, label_for(spx)) == r"\hat{y} = 0.2000\,\text{S\&P 500}"


@pytest.mark.parametrize(
    ("raw", "escaped"),
    [("a_b", r"a\_b"), ("50%", r"50\%"), ("$x", r"\$x"), ("#1", r"\#1"), ("{x}", r"\{x\}"),
     ("a~b", r"a\textasciitilde{}b"), ("a^b", r"a\textasciicircum{}b"),
     ("a\\b", r"a\textbackslash{}b")],
)
def test_every_special_character_is_escaped(raw, escaped):
    fn = derived([raw], [0.2], intercept=0.0)
    assert to_latex(fn, lambda column: column) == rf"\hat{{y}} = 0.2000\,\text{{{escaped}}}"


def test_an_intercept_only_function():
    assert to_latex(derived([], [], intercept=0.5), label_for()) == r"\hat{y} = 0.5000"


def test_a_zero_intercept_only_function_still_shows_zero():
    assert to_latex(derived([], [], intercept=0.0), label_for()) == r"\hat{y} = 0"


def test_a_scientific_coefficient_is_typeset():
    fn = derived([VIX], [0.0000123456], intercept=0.0)
    assert to_latex(fn, label_for(VIX)) == r"\hat{y} = 1.235 \times 10^{-5}\,\text{VIX}"


def test_a_fitted_user_function_is_written_as_its_scale_and_intercept():
    fn = from_description(
        ModelDescription("UserPolynomial", (f"{VIX} / {IG}",), (0.0004,), -0.0012),
        origin=Origin.USER_SUPPLIED,
        target="SPX_Index_PX_LAST",
        horizon=5,
    )
    expected = (
        r"\hat{y} = -0.001200 + 0.0004000 \cdot (\text{VIX} / \text{US IG credit spread})"
    )
    assert shape_latex(fn, label_for(VIX, IG)) == expected
    assert to_latex(fn, label_for(VIX, IG)) == expected


def test_an_expanded_user_function_keeps_its_fitted_form_too():
    fn = from_description(
        ModelDescription("UserPolynomial", ("2 * a + 1",), (-0.5,), 0.01),
        origin=Origin.USER_SUPPLIED,
        target="SPX_Index_PX_LAST",
        horizon=1,
    )
    assert as_dict(fn) == {(("a", 1),): -1.0}
    assert fn.intercept == pytest.approx(0.01 - 0.5)
    assert shape_latex(fn, lambda column: column) == (
        r"\hat{y} = 0.01000 - 0.5000 \cdot (2 \cdot \text{a} + 1)"
    )


def test_a_derived_equation_in_standardised_signals_reproduces_predict():
    rng = np.random.default_rng(5)
    idx = pd.date_range("2024-01-01", periods=200, freq="D")
    a, b = 20 + 5 * rng.normal(size=200), 1.5 + 0.3 * rng.normal(size=200)
    target = 0.0004 * (a - 20) ** 2 - 0.01 * (b - 1.5) + rng.normal(scale=0.002, size=200)
    frame = pd.DataFrame({"a": a, "b": b, "target": target}, index=idx)
    panel = FeaturePanel(frame=frame, signals=("a", "b"), targets=("target",), lag_days=1)
    model = SignRuledPolynomial({"a": 1, "b": -1})
    model.fit(panel, idx)
    description = model.describe()
    fn = derived(
        description.terms, description.coefficients, description.intercept, columns=("a", "b")
    )
    fn = PolynomialFunction(**{**fn.__dict__, "standardisation": description.standardisation})

    z = {
        column: (frame[column].clip(*description.input_bounds[column]) - mean) / sd
        for column, (mean, sd) in fn.standardisation.items()
    }
    actual = pd.Series(fn.intercept, index=idx)
    for term in fn.terms:
        product = pd.Series(1.0, index=idx)
        for column, power in term.factors:
            product = product * z[column] ** power
        actual = actual + term.coefficient * product

    assert fn.terms
    np.testing.assert_allclose(actual, model.predict(panel, idx), rtol=1e-9, atol=1e-12)


def test_a_standardised_equation_names_its_signals_z_and_says_how_each_is_made():
    fn = from_description(
        ModelDescription(
            "DerivedPolynomial",
            (f"{VIX}^2",),
            (-0.0002,),
            0.0012,
            standardisation={VIX: (18.2, 6.1), IG: (1.25, 0.3)},
        ),
        origin=Origin.DERIVED,
        target="SPX_Index_PX_LAST",
        horizon=5,
    )
    label = label_for(VIX, IG)

    assert to_latex(fn, label) == r"\hat{y} = 0.001200 - 0.0002000\,z_{\text{VIX}}^{2}"
    assert term_rows(fn, label)[1]["Factor"] == "z(VIX)"
    assert standardisation_lines(fn, label) == ["z(VIX) = (VIX − 18.20) / 6.100"]


def test_formula_only_mode_writes_the_formula_back_with_labels():
    fn = user(f"{VIX} / {IG}")
    assert to_latex(fn, label_for(VIX, IG)) == (
        r"\hat{y} = \text{VIX} / \text{US IG credit spread}"
    )


def test_formula_only_mode_brackets_only_where_precedence_needs_them():
    fn = user("(a + b) / c ** 2 - a * b / c")
    assert to_latex(fn, lambda column: column) == (
        r"\hat{y} = (\text{a} + \text{b}) / \text{c}^{2} - \text{a} \cdot \text{b} / \text{c}"
    )


def test_formula_only_mode_brackets_a_subtracted_sum_and_a_powered_sum():
    fn = user("a / (b - (c + a)) + (a + b)**2 / c")
    assert to_latex(fn, lambda column: column) == (
        r"\hat{y} = \text{a} / (\text{b} - (\text{c} + \text{a}))"
        r" + (\text{a} + \text{b})^{2} / \text{c}"
    )


# --- the term table ----------------------------------------------------------


def test_rows_follow_the_equation_order_with_the_intercept_first():
    fn = derived([f"{VIX}^2", f"{VIX} {IG}"], [0.0004521, -0.000231], intercept=0.001234)
    assert term_rows(fn, label_for(VIX, IG)) == [
        {"Factor": "(intercept)", "Exponent": "—", "Coefficient": "0.001234"},
        {"Factor": "VIX", "Exponent": "2", "Coefficient": "0.0004521"},
        {"Factor": "US IG credit spread × VIX", "Exponent": "1 × 1", "Coefficient": "−0.0002310"},
    ]


def test_every_value_is_a_string():
    fn = derived([VIX], [0.5], intercept=0.1)
    assert all(isinstance(v, str) for row in term_rows(fn, label_for(VIX)) for v in row.values())


def test_a_zero_intercept_has_no_row_when_terms_exist():
    fn = derived([VIX], [0.5], intercept=0.0)
    assert [row["Factor"] for row in term_rows(fn, label_for(VIX))] == ["VIX"]


def test_an_intercept_only_function_has_just_the_intercept_row():
    assert term_rows(derived([], [], intercept=0.5), label_for()) == [
        {"Factor": "(intercept)", "Exponent": "—", "Coefficient": "0.5000"}
    ]


def test_a_coefficient_of_one_still_shows_in_the_table():
    rows = term_rows(derived([VIX], [1.0], intercept=0.0), label_for(VIX))
    assert rows[0]["Coefficient"] == "1.000"


def test_formula_only_mode_has_no_rows():
    assert term_rows(user("a / b"), lambda column: column) == []


# --- dataset fingerprint -----------------------------------------------------


def _frame(n=10, columns=("Date", "VIX_Index_PX_LAST")):
    data = {"Date": pd.bdate_range("2024-01-01", periods=n), "VIX_Index_PX_LAST": range(n)}
    return pd.DataFrame({c: data[c] for c in columns})


def test_the_same_data_has_the_same_fingerprint():
    assert dataset_fingerprint(_frame()) == dataset_fingerprint(_frame())


def test_a_change_of_rows_dates_or_columns_changes_the_fingerprint():
    base = dataset_fingerprint(_frame())
    assert dataset_fingerprint(_frame(n=11)) != base
    assert dataset_fingerprint(_frame(columns=("Date",))) != base
    shifted = _frame()
    shifted["Date"] = pd.bdate_range("2025-01-01", periods=10)
    assert dataset_fingerprint(shifted) != base
