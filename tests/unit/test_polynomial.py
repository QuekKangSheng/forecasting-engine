import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import TimeSeriesSplit

from forecasting_engine.ingest.align import FeaturePanel
from forecasting_engine.models import polynomial
from forecasting_engine.models.polynomial import (
    MAX_DEGREE,
    MAX_FORMULA_LENGTH,
    MAX_FORMULA_NODES,
    DerivedPolynomial,
    PolynomialConfigError,
    UserPolynomial,
    evaluate_formula,
    placeholders,
    run_derived_polynomial,
)
from forecasting_engine.validation.harness import evaluate
from forecasting_engine.validation.splitters import PurgedWalkForward


def _formula_panel(n: int = 10) -> FeaturePanel:
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    frame = pd.DataFrame(
        {"vix": range(n), "credit_spread_hy": range(n), "target": range(n)}, index=idx
    )
    return FeaturePanel(
        frame=frame, signals=("vix", "credit_spread_hy"), targets=("target",), lag_days=1
    )


# --- UserPolynomial: FYP-118, FYP-125 ---------------------------------------


def _shape_panel(f, target) -> FeaturePanel:
    idx = pd.date_range("2024-01-01", periods=len(f), freq="D")
    frame = pd.DataFrame({"vix": f, "spread": np.arange(len(f)), "target": target}, index=idx)
    return FeaturePanel(frame=frame, signals=("vix", "spread"), targets=("target",), lag_days=1)


def test_user_polynomial_fits_a_scale_and_intercept_to_its_shape():
    f = np.random.default_rng(0).normal(20, 5, size=60)
    panel = _shape_panel(f, 0.01 + 0.5 * f)
    model = UserPolynomial("vix")

    model.fit(panel, panel.frame.index)
    description = model.describe()

    assert description.intercept == pytest.approx(0.01)
    assert description.coefficients == (pytest.approx(0.5),)
    assert description.terms == ("vix",)


def test_user_polynomial_fits_only_on_training_rows_with_both_values():
    f = np.random.default_rng(1).normal(size=40)
    target = 0.01 + 0.5 * f
    target[:5] = 99.0  # outside the training rows below
    f_with_gap = f.copy()
    f_with_gap[10] = np.nan
    panel = _shape_panel(f_with_gap, target)
    train = panel.frame.index[5:]

    model = UserPolynomial("vix")
    model.fit(panel, train)

    assert model.describe().coefficients == (pytest.approx(0.5),)
    assert np.isnan(model.predict(panel, train)[panel.frame.index[10]])


def test_a_constant_shape_forecasts_the_training_mean():
    target = np.random.default_rng(2).normal(0.001, 0.01, size=30)
    panel = _shape_panel(np.full(30, 7.0), target)
    model = UserPolynomial("vix")

    model.fit(panel, panel.frame.index)

    assert model.describe().coefficients == (0.0,)
    assert model.predict(panel, panel.frame.index).tolist() == pytest.approx(
        [target.mean()] * 30
    )


def test_a_formula_of_constants_alone_forecasts_the_training_mean():
    target = np.random.default_rng(3).normal(size=20)
    panel = _shape_panel(np.arange(20.0), target)
    model = UserPolynomial("2 + 3")

    model.fit(panel, panel.frame.index)

    assert model.predict(panel, panel.frame.index).tolist() == pytest.approx(
        [target.mean()] * 20
    )


def test_placeholders_stand_for_the_columns_they_are_bound_to():
    f = np.random.default_rng(4).normal(size=40)
    panel = _shape_panel(f, 0.002 - 0.3 * (f + np.arange(40.0) ** 2))
    bound = UserPolynomial("x + y ** 2", {"x": "vix", "y": "spread"})
    direct = UserPolynomial("vix + spread ** 2")

    for model in (bound, direct):
        model.fit(panel, panel.frame.index)

    assert bound.resolved_formula == "vix + spread ** 2"
    assert bound.describe() == direct.describe()
    pd.testing.assert_series_equal(
        bound.predict(panel, panel.frame.index), direct.predict(panel, panel.frame.index)
    )


@pytest.mark.parametrize(
    "bindings",
    [{"x": "vix", "vix": "spread"}, {"vix": "spread", "x": "vix"}],
    ids=["placeholder-first", "column-named-placeholder-first"],
)
def test_each_placeholder_reads_its_own_column_whatever_order_it_is_bound_in(bindings):
    # The name vix is itself a placeholder here, standing for spread; x stands for
    # the vix column, and must read it however the bindings are ordered.
    f = np.random.default_rng(8).normal(size=40)
    panel = _shape_panel(f, 0.01 + 0.5 * f)
    bound = UserPolynomial("x + 2 * vix", bindings)
    direct = UserPolynomial("vix + 2 * spread")

    for model in (bound, direct):
        model.fit(panel, panel.frame.index)

    assert bound.resolved_formula == "vix + 2 * spread"
    pd.testing.assert_series_equal(
        bound.predict(panel, panel.frame.index), direct.predict(panel, panel.frame.index)
    )


def test_a_placeholder_bound_to_an_unknown_column_is_refused():
    panel = _shape_panel(np.arange(10.0), np.arange(10.0))
    with pytest.raises(PolynomialConfigError, match="not a signal column"):
        UserPolynomial("x", {"x": "nope"}).fit(panel, panel.frame.index)


def test_placeholders_are_listed_once_in_the_order_they_appear():
    assert placeholders("b * a + a ** 2 - c") == ("b", "a", "c")


def test_predict_before_fit_is_an_error():
    panel = _shape_panel(np.arange(10.0), np.arange(10.0))
    with pytest.raises(RuntimeError):
        UserPolynomial("vix").predict(panel, panel.frame.index)


def test_user_polynomial_rejects_invalid_syntax():
    with pytest.raises(PolynomialConfigError):
        UserPolynomial("vix +")


def test_user_polynomial_rejects_function_calls_at_construction():
    # Structural rejection must happen eagerly, not deferred to predict() —
    # a portfolio manager's text box is untrusted input.
    with pytest.raises(PolynomialConfigError):
        UserPolynomial("abs(vix)")


def test_user_polynomial_rejects_non_integer_exponent_at_construction():
    with pytest.raises(PolynomialConfigError):
        UserPolynomial("vix ** 0.5")


@pytest.mark.parametrize("formula", ["vix ** 6", "vix ** 1e9", "vix * 7 ** 999999999", "vix ** -1"])
def test_an_exponent_must_be_a_whole_number_up_to_the_degree_cap(formula):
    with pytest.raises(PolynomialConfigError, match=f"a whole number from 0 to {MAX_DEGREE}"):
        UserPolynomial(formula)


@pytest.mark.parametrize(
    "formula",
    [
        "x ** 3 * y ** 3",
        "x * y * x * y * x * y",
        "(x ** 2) ** 3",
        "(x + y ** 2) ** 3",
        "x ** 5 / y",
    ],
)
def test_a_function_above_the_degree_cap_is_refused_at_construction(formula):
    refusal = f"degree is 6, above the highest allowed, {MAX_DEGREE}"
    with pytest.raises(PolynomialConfigError, match=refusal):
        UserPolynomial(formula)


@pytest.mark.parametrize(
    "formula", ["x ** 5", "x ** 3 * y ** 2", "(x + y) ** 5", "x ** 4 / y", "2 ** 5 * x ** 5 - 1"]
)
def test_a_function_at_the_degree_cap_is_accepted(formula):
    UserPolynomial(formula)


@pytest.mark.parametrize(
    "formula",
    [
        "vix + 1e400",
        "vix * (1e300 * 1e300)",
        f"vix + {'9' * 400}",
        f"vix + {'9' * 5000}",
        f"vix + 0x{'f' * 9000}",
        "vix * (((7 ** 5) ** 5) ** 5) ** 5",
    ],
    ids=[
        "infinite-literal",
        "infinite-product",
        "400-digit-literal",
        "5000-digit-literal",
        "9000-digit-hex-literal",
        "power-of-powers",
    ],
)
def test_a_number_too_large_to_use_is_refused_at_construction(formula):
    # Python itself won't read a 5,000-digit whole number, and its own refusal
    # talks about its settings; the user is told the number is too large.
    with pytest.raises(PolynomialConfigError, match="too large to work with; use a smaller"):
        UserPolynomial(formula)


@pytest.mark.parametrize("formula", ["vix / 0", "vix / (2 - 2)", "1 / 0.0 + vix"])
def test_dividing_by_zero_is_refused_at_construction(formula):
    with pytest.raises(PolynomialConfigError, match="divides by zero"):
        UserPolynomial(formula)


def test_a_function_with_too_many_terms_is_refused_at_construction():
    # 1,200 terms once nested past Python's recursion limit and broke the page.
    with pytest.raises(PolynomialConfigError, match=f"the most allowed is {MAX_FORMULA_NODES}"):
        UserPolynomial(" + ".join(["vix"] * 1200))


@pytest.mark.parametrize(
    "formula", ["-" * 3000 + "vix", "-" * 9000 + "vix"], ids=["3000-deep", "9000-deep"]
)
def test_a_deeply_nested_function_is_refused_rather_than_crashing(formula):
    with pytest.raises(PolynomialConfigError):
        UserPolynomial(formula)


def test_a_function_longer_than_the_length_cap_is_refused_at_construction():
    with pytest.raises(PolynomialConfigError, match="characters long"):
        UserPolynomial("vix" + " " * MAX_FORMULA_LENGTH)


#: Each of these once held the only model worker until the app was restarted:
#: the first built a number with hundreds of millions of digits, the second made
#: the shown equation multiply vix out a billion times, the third is a power of a
#: power of ... of 7.
_USED_TO_HANG = (
    "vix * 7 ** 999999999",
    "vix ** 1000000000",
    "(" * 20 + "7" + " ** 5)" * 20 + " * vix",
)

_REFUSE_EACH = """
import math
import sys
from forecasting_engine.models.base import ModelDescription
from forecasting_engine.models.polynomial import (
    PolynomialConfigError, UserPolynomial, evaluate_formula,
)
from forecasting_engine.reporting.polynomial_function import Origin, from_description, to_latex

def refused(check, formula):
    try:
        check(formula)
    except PolynomialConfigError:
        return True
    return False

def without_a_value(formula):
    # Refused, or NaN, as a power of vix beyond a float is: a scenario refuses both.
    try:
        return math.isnan(evaluate_formula(formula, {"vix": 1.5}))
    except PolynomialConfigError:
        return True

for formula in sys.argv[1:]:
    if not refused(UserPolynomial, formula):
        print("accepted as a new function:", formula)
    if not without_a_value(formula):
        print("evaluated when read back:", formula)
    saved = ModelDescription("UserPolynomial", (formula,), (1.0,), 0.0)
    to_latex(from_description(saved, origin=Origin.USER_SUPPLIED, target="", horizon=1), str)
"""


def _in_a_child(script: str, *args: str) -> str:
    """What ``script`` prints, run on ``args`` in a child process with a deadline,
    so a regression fails the test instead of hanging the suite: a big-number
    power can't be interrupted in-process."""
    src = Path(polynomial.__file__).resolve().parents[2]
    path = os.pathsep.join([str(src), os.environ.get("PYTHONPATH", "")])
    env = {**os.environ, "PYTHONPATH": path}
    done = subprocess.run(
        [sys.executable, "-c", script, *args],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


def test_functions_that_used_to_hang_are_refused_quickly_whether_new_or_saved():
    # A saved one is refused when evaluated, or has no value there, as 1.5 to the
    # power of a billion is beyond a float; it is shown as entered, never worked out.
    assert _in_a_child(_REFUSE_EACH, *_USED_TO_HANG) == ""


_EVALUATE_SAVED = """
import sys
from forecasting_engine.models.polynomial import PolynomialConfigError, evaluate_formula

for formula in sys.argv[1:]:
    try:
        print("evaluated:", evaluate_formula(formula, {"x": 3}))
    except PolynomialConfigError as exc:
        print("refused:", exc)
"""


def test_a_saved_function_raising_a_plain_number_past_a_float_is_refused_quickly_in_plain_words():
    # Read back, a signal may be raised to any whole power: its value is read as a
    # float, even a whole number such as this 3, so the power is quick and, past a
    # float, has no value. A plain number's power is worked out exactly, digit by
    # digit, so 10 ** 1000000000 would take hours: it is refused from its size first.
    refusal = "refused: a number in the function is too large to work with; use a smaller one."
    outcomes = {
        "10 ** 1000000000 * x": refusal,
        "x * 2 ** 100000": refusal,
        "x + 1.5 ** 3000": refusal,
        "x ** 1000000000": "evaluated: nan",
    }

    assert _in_a_child(_EVALUATE_SAVED, *outcomes).splitlines() == list(outcomes.values())


@pytest.mark.parametrize(
    ("formula", "value"),
    [("x ** 3 * y ** 3", 27.0), (" + ".join(["x", "y"] * 150), 525.0)],
    ids=["degree-6", "300-terms"],
)
def test_a_saved_function_past_the_caps_is_still_read_back(formula, value):
    # A function saved before the caps has run already: it is refused as a new
    # one, but reading it back still lists its names and evaluates it.
    with pytest.raises(PolynomialConfigError):
        UserPolynomial(formula)

    assert placeholders(formula) == ("x", "y")
    assert evaluate_formula(formula, {"x": 1.5, "y": 2.0}) == value


@pytest.mark.parametrize(
    ("formula", "values", "value"),
    [
        ("x ** 6 + y", {"x": 1.5, "y": 2.0}, 1.5**6 + 2.0),
        ("x ** 7 * y", {"x": 1.5, "y": 2.0}, 1.5**7 * 2.0),
        ("x ** 1000", {"x": 1.0001}, 1.0001**1000),
    ],
    ids=["power-6", "power-7", "power-1000"],
)
def test_a_saved_function_raising_a_signal_past_power_5_is_evaluated_though_a_new_one_is_refused(
    formula, values, value
):
    # Saved before the cap on a new function's exponents. Raising a signal's value
    # to a whole power is quick whatever the power, so it is read back as it ran.
    with pytest.raises(PolynomialConfigError, match=f"a whole number from 0 to {MAX_DEGREE}"):
        UserPolynomial(formula)

    assert placeholders(formula) == tuple(values)
    assert evaluate_formula(formula, values) == pytest.approx(value)


@pytest.mark.parametrize("formula", ["x ** -1", "x ** 0.5", "x ** 1e400", "x ** y"])
def test_reading_a_saved_function_back_still_refuses_a_negative_or_fractional_exponent(formula):
    with pytest.raises(PolynomialConfigError, match="a whole number of 0 or more"):
        evaluate_formula(formula, {"x": 1.5, "y": 2.0})


def test_reading_back_a_function_nested_too_deeply_to_evaluate_is_refused_not_crashed():
    # No size cap applies to a saved function, so it is the walk itself that can
    # run out of depth.
    with pytest.raises(PolynomialConfigError, match="nested too deeply"):
        evaluate_formula(" + ".join(["x"] * 1200), {"x": 1.0})


@pytest.mark.parametrize(
    "values",
    [
        pd.Series({"x": 1.5, "y": 0.0}),
        {"x": 1.5, "y": 0.0},
        pd.Series({"x": 1e70, "y": 1.0}),
        {"x": 1e70, "y": 1.0},
    ],
    ids=["zero-divisor-in-a-row", "zero-divisor", "too-large-in-a-row", "too-large"],
)
def test_a_saved_function_is_nan_on_values_it_cant_be_worked_out_on(values):
    # A panel's row holds numpy numbers, which give inf rather than raising as
    # plain floats do; either way there is no value to forecast with.
    assert np.isnan(evaluate_formula("x ** 5 / y", values))


def _dividing_panel(x, y) -> FeaturePanel:
    idx = pd.date_range("2024-01-01", periods=len(x), freq="D")
    target = np.arange(len(x), dtype=float)
    frame = pd.DataFrame({"vix": x, "spread": y, "target": target}, index=idx)
    return FeaturePanel(frame=frame, signals=("vix", "spread"), targets=("target",), lag_days=1)


def test_dividing_by_a_signal_on_days_it_is_zero_stops_the_fit_and_names_it():
    spread = np.arange(1.0, 41.0)
    spread[[0, 10, 20]] = 0.0
    panel = _dividing_panel(np.random.default_rng(5).normal(size=40), spread)
    model = UserPolynomial("x / y", {"x": "vix", "y": "spread"})

    with pytest.raises(
        PolynomialConfigError,
        match=r"divides by y \(spread\), which is zero on 3 days \(the first is 01 Jan 2024\)",
    ):
        model.fit(panel, panel.frame.index)


def test_a_zero_divisor_only_on_forecast_days_still_stops_the_run():
    # 0 / 0 is undefined rather than infinite, and is refused the same way.
    spread = np.arange(1.0, 41.0)
    spread[30] = 0.0
    vix = np.random.default_rng(6).normal(size=40)
    vix[30] = 0.0
    panel = _dividing_panel(vix, spread)
    model = UserPolynomial("vix / spread")
    model.fit(panel, panel.frame.index[:30])

    with pytest.raises(PolynomialConfigError, match=r"divides by spread, which is zero on 1 day "):
        model.predict(panel, panel.frame.index[30:])


def test_a_function_infinite_for_another_reason_still_stops_the_run():
    panel = _dividing_panel(np.full(20, 1e200), np.arange(1.0, 21.0))

    with pytest.raises(PolynomialConfigError, match="infinite or undefined on 20 days"):
        UserPolynomial("vix * vix * spread").fit(panel, panel.frame.index)


def test_a_missing_input_still_just_leaves_its_row_out():
    spread = np.arange(1.0, 41.0)
    spread[5] = np.nan
    panel = _dividing_panel(np.random.default_rng(7).normal(size=40), spread)
    model = UserPolynomial("vix / spread")

    model.fit(panel, panel.frame.index)

    assert np.isnan(model.predict(panel, panel.frame.index).iloc[5])


def test_user_polynomial_rejects_unknown_column_at_fit():
    panel = _formula_panel()
    model = UserPolynomial("unknown_signal * 2")
    with pytest.raises(PolynomialConfigError):
        model.fit(panel, panel.frame.index)


# --- DerivedPolynomial: FYP-119, FYP-120, FYP-124 ---------------------------


def _linear_panel(n: int = 60) -> FeaturePanel:
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    frame = pd.DataFrame({"x": x, "target": 2 * x}, index=idx)
    return FeaturePanel(frame=frame, signals=("x",), targets=("target",), lag_days=1)


def test_derived_polynomial_rejects_degree_above_five():
    with pytest.raises(PolynomialConfigError):
        DerivedPolynomial(degree=MAX_DEGREE + 1)


def test_derived_polynomial_rejects_unknown_regularizer():
    with pytest.raises(PolynomialConfigError):
        DerivedPolynomial(regularizer="ridge")


def test_derived_polynomial_recovers_a_clean_linear_relationship():
    # Regularization shrinks the coefficient, so this asserts structural
    # correctness (right term, right sign, fits well) rather than an exact
    # coefficient match — see the plan's note on why exact matching doesn't
    # hold under L1/L2 regularization.
    panel = _linear_panel()
    model = DerivedPolynomial(degree=1, regularizer="lasso")

    model.fit(panel, panel.frame.index)
    predicted = model.predict(panel, panel.frame.index)

    actual = panel.frame["target"]
    residual = ((predicted - actual) ** 2).sum()
    total = ((actual - actual.mean()) ** 2).sum()
    r_squared = 1 - residual / total
    assert r_squared > 0.9

    description = model.describe()
    assert "x" in description.terms
    coefficient = description.coefficients[description.terms.index("x")]
    assert coefficient > 0


def test_derived_polynomial_max_terms_caps_the_surviving_term_count():
    idx = pd.date_range("2024-01-01", periods=60, freq="D")
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({f"x{i}": rng.normal(size=60) for i in range(4)}, index=idx)
    frame["target"] = frame["x0"] * 2 - frame["x1"]
    panel = FeaturePanel(
        frame=frame, signals=tuple(f"x{i}" for i in range(4)), targets=("target",), lag_days=1
    )
    model = DerivedPolynomial(degree=2, regularizer="lasso", max_terms=3)

    model.fit(panel, panel.frame.index)

    assert len(model.describe().terms) <= 3


def test_derived_polynomial_rejects_too_few_training_rows():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    frame = pd.DataFrame({"x": range(5), "target": range(5)}, index=idx)
    panel = FeaturePanel(frame=frame, signals=("x",), targets=("target",), lag_days=1)

    with pytest.raises(PolynomialConfigError):
        DerivedPolynomial().fit(panel, panel.frame.index)


def test_derived_polynomial_predict_before_fit_raises():
    panel = _linear_panel()
    with pytest.raises(RuntimeError):
        DerivedPolynomial().predict(panel, panel.frame.index)


# --- the fit must not depend on the units its signals are quoted in --------


def _scaled_panel(factor: float, n: int = 400) -> FeaturePanel:
    """A real, quadratic relationship on a signal quoted in arbitrary units.

    The live dataset mixes an index level (4,000) with a volatility point (18)
    and a spread (1.5), and ``PolynomialFeatures`` squares and cubes all three,
    so its columns span several orders of magnitude. L1's penalty applies to
    coefficients, whose natural size depends on those units, so these pin down
    that the fit still behaves when the units change.
    """
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(7)
    x = rng.normal(size=n)
    target = 0.5 * x**2 + rng.normal(scale=0.05, size=n)
    frame = pd.DataFrame({"x": x * factor, "target": target}, index=idx)
    return FeaturePanel(frame=frame, signals=("x",), targets=("target",), lag_days=1)


@pytest.mark.parametrize("factor", [1.0, 1_000.0])
def test_derived_polynomial_finds_the_relationship_whatever_the_units(factor):
    panel = _scaled_panel(factor)
    model = DerivedPolynomial(degree=2, regularizer="lasso")

    model.fit(panel, panel.frame.index)

    assert model.describe().terms, "the quadratic term should survive regularization"
    predicted = model.predict(panel, panel.frame.index)
    actual = panel.frame["target"]
    r_squared = 1 - ((predicted - actual) ** 2).sum() / ((actual - actual.mean()) ** 2).sum()
    assert r_squared > 0.8


def test_derived_polynomial_scores_the_same_whatever_the_units():
    small, large = _scaled_panel(1.0), _scaled_panel(1_000.0)
    fits = []
    for panel in (small, large):
        model = DerivedPolynomial(degree=2, regularizer="lasso")
        model.fit(panel, panel.frame.index)
        fits.append(model.predict(panel, panel.frame.index))

    assert fits[0].corr(fits[1]) > 0.99


def _rebuilt(description, frame: pd.DataFrame) -> pd.Series:
    """The described equation evaluated by hand: each term in the standardised
    signal z = (clip(x) − mean) / sd."""
    z = {}
    for column, (mean, sd) in description.standardisation.items():
        low, high = description.input_bounds[column]
        z[column] = (frame[column].clip(low, high) - mean) / sd
    rebuilt = pd.Series(description.intercept, index=frame.index)
    for name, coefficient in zip(description.terms, description.coefficients, strict=True):
        product = pd.Series(1.0, index=frame.index)
        for part in name.split(" "):
            column, _, power = part.partition("^")
            product = product * z[column] ** (int(power) if power else 1)
        rebuilt = rebuilt + coefficient * product
    return rebuilt


def test_the_equation_in_standardised_signals_reproduces_predict():
    # The displayed equation is only meaningful if it is exactly what predict()
    # computes, so the description must undo the expanded terms' rescaling.
    panel = _scaled_panel(1_000.0)
    model = DerivedPolynomial(degree=2, regularizer="lasso")
    model.fit(panel, panel.frame.index)

    idx = panel.frame.index
    np.testing.assert_allclose(
        _rebuilt(model.describe(), panel.frame), model.predict(panel, idx), rtol=1e-9, atol=1e-12
    )


def test_each_signals_mean_and_sd_are_reported_from_the_training_window():
    panel = _scaled_panel(1_000.0)
    train = panel.frame.index[:200]
    model = DerivedPolynomial(degree=2, regularizer="lasso")
    model.fit(panel, train)

    x = panel.frame.loc[train, "x"]
    assert model.describe().standardisation == {
        "x": (pytest.approx(x.mean()), pytest.approx(x.std()))
    }


def _level_panel(n: int = 600, seed: int = 11) -> FeaturePanel:
    """A level signal (like VIX, around 20) whose effect is on its distance from
    the mean squared: raw, x and x² are near-collinear; centred, they are not."""
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    rng = np.random.default_rng(seed)
    x = 20 + 5 * rng.normal(size=n)
    target = 0.0004 * (x - 20) ** 2 + rng.normal(scale=0.002, size=n)
    frame = pd.DataFrame({"x": x, "target": target}, index=idx)
    return FeaturePanel(frame=frame, signals=("x",), targets=("target",), lag_days=1)


def test_a_squared_effect_keeps_its_squared_term_with_a_stable_sign_across_folds():
    folds = evaluate(
        lambda: DerivedPolynomial(degree=2, regularizer="lasso"),
        _level_panel(),
        PurgedWalkForward(train=120, test=20, embargo=5),
    )

    assert len(folds) > 10
    squared = [
        dict(zip(f.description.terms, f.description.coefficients, strict=True)) for f in folds
    ]
    assert all("x^2" in terms for terms in squared)
    assert all(terms["x^2"] > 0 for terms in squared)
    # Centred, the effect sits on the squared term. Expanded raw, it is split
    # between a large negative level term and a small square (about −0.016 x +
    # 0.0004 x²), which reads as the opposite of what the data does.
    assert all(abs(terms.get("x", 0.0)) < 0.5 * terms["x^2"] for terms in squared)


def test_deriving_from_one_candidate_reports_it_without_a_choice_to_make():
    panel = _linear_panel(n=120)
    splitter = PurgedWalkForward(train=40, test=10, embargo=2)

    result, description = run_derived_polynomial(
        panel, splitter, candidates=(DerivedPolynomial(degree=1),), shrinkage=False
    )

    assert description.name == "DerivedPolynomial"


def test_deriving_with_no_candidates_says_so_in_plain_words():
    panel = _linear_panel(n=120)
    splitter = PurgedWalkForward(train=40, test=10, embargo=2)

    with pytest.raises(PolynomialConfigError, match="at least one"):
        run_derived_polynomial(panel, splitter, candidates=(), shrinkage=False)


# --- scaling and validation stay inside the training window, in time order ----


def test_the_fit_ignores_rows_outside_the_training_window():
    panel = _scaled_panel(1.0)
    train = panel.frame.index[:200]
    shifted = panel.frame.copy()
    shifted.loc[panel.frame.index[200:], "x"] *= 1_000
    other = FeaturePanel(frame=shifted, signals=("x",), targets=("target",), lag_days=1)

    descriptions = []
    for p in (panel, other):
        model = DerivedPolynomial(degree=2, regularizer="lasso")
        model.fit(p, train)
        descriptions.append(model.describe())

    assert descriptions[0] == descriptions[1]


def test_the_inner_validation_is_time_ordered_with_a_horizon_gap(monkeypatch):
    seen = []
    real = polynomial._time_series_cv

    def recording(n_rows, gap):
        cv = real(n_rows, gap)
        seen.append(cv)
        return cv

    monkeypatch.setattr(polynomial, "_time_series_cv", recording)
    panel = replace(_scaled_panel(1.0), horizon=5)
    DerivedPolynomial(degree=1, regularizer="lasso").fit(panel, panel.frame.index)

    (cv,) = seen
    assert isinstance(cv, TimeSeriesSplit)
    assert cv.gap == 5
    assert cv.n_splits == 5


def test_a_short_window_validates_on_fewer_folds_and_too_short_says_so():
    assert polynomial._time_series_cv(20, gap=5).n_splits < 5
    with pytest.raises(PolynomialConfigError, match="too short"):
        polynomial._time_series_cv(10, gap=5)


# --- extreme inputs are clipped before they are raised to a power ---------------


def test_an_extreme_input_forecasts_no_further_than_one_at_the_clip_bound():
    panel = _scaled_panel(1.0)
    model = DerivedPolynomial(degree=2, regularizer="lasso")
    model.fit(panel, panel.frame.index)
    x = panel.frame["x"]
    bound = x.mean() + polynomial.CLIP_SD * x.std()

    probes = panel.frame.iloc[:3].copy()
    probes["x"] = [bound, bound * 10, bound * 1_000]
    probe_panel = FeaturePanel(frame=probes, signals=("x",), targets=("target",), lag_days=1)
    forecasts = model.predict(probe_panel, probes.index)

    assert forecasts.iloc[1] == pytest.approx(forecasts.iloc[0])
    assert forecasts.iloc[2] == pytest.approx(forecasts.iloc[0])


def test_the_clip_bounds_are_reported_in_raw_units():
    panel = _scaled_panel(1_000.0)
    model = DerivedPolynomial(degree=2, regularizer="lasso")
    model.fit(panel, panel.frame.index)

    low, high = model.describe().input_bounds["x"]
    x = panel.frame["x"]
    assert low == pytest.approx(x.mean() - 4 * x.std())
    assert high == pytest.approx(x.mean() + 4 * x.std())


# --- nothing from outside the training window reaches a forecast ---------------


def test_a_forecast_does_not_depend_on_the_rows_predicted_alongside_it():
    # Standardising (or clipping) with the rows being predicted, rather than the
    # training window's statistics, makes a day's forecast depend on which other
    # days are in the batch — and leaks the test window into every forecast.
    panel = _scaled_panel(1.0)
    train, test = panel.frame.index[:300], panel.frame.index[300:]
    model = DerivedPolynomial(degree=2, regularizer="lasso")
    model.fit(panel, train)

    batch = model.predict(panel, test)
    alone = pd.Series([model.predict(panel, test[i : i + 1]).iloc[0] for i in range(5)])

    np.testing.assert_allclose(alone, batch.iloc[:5], rtol=1e-12)


def test_the_fit_sees_each_signal_clipped_then_standardised(monkeypatch):
    panel = _scaled_panel(1.0)
    frame = panel.frame.copy()
    frame.iloc[10, frame.columns.get_loc("x")] = 50.0  # ~50 SD out
    panel = replace(panel, frame=frame)
    model = DerivedPolynomial(degree=1, regularizer="lasso")

    seen = []
    real = model._poly.fit_transform

    def recording(x, *args, **kwargs):
        seen.append(x.copy())
        return real(x, *args, **kwargs)

    monkeypatch.setattr(model._poly, "fit_transform", recording)
    model.fit(panel, panel.frame.index)

    (fitted_on,) = seen
    assert fitted_on["x"].max() == pytest.approx(polynomial.CLIP_SD)


def test_scaling_inside_validation_sees_only_earlier_rows(monkeypatch):
    # The penalty is chosen by validating on later rows of the training window.
    # Fitting the scaler on the whole window first would let those later rows
    # shape what they then validate.
    panel = _scaled_panel(1.0)
    model = DerivedPolynomial(degree=3, regularizer="lasso", max_terms=2)
    windows = []
    real = polynomial._prepare

    def recording(x):
        windows.append(x.index)
        return real(x)

    monkeypatch.setattr(polynomial, "_prepare", recording)
    model.fit(panel, panel.frame.index)

    everything = panel.frame.index
    inner = [w for w in windows if len(w) < len(everything)]
    assert inner, "validation must prepare each inner split itself"
    assert windows[-1].equals(everything)
    for window in inner:
        assert window.equals(everything[: len(window)]), "an inner split trains on a prefix"


# --- the term cap works through the penalty, never a separate ranking ----------


def _many_signals_panel(n: int = 200) -> FeaturePanel:
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(5)}, index=idx)
    frame["target"] = (
        frame["x0"] - 0.8 * frame["x1"] + 0.6 * frame["x2"] ** 2 + 0.4 * frame["x3"] * frame["x4"]
        + rng.normal(scale=0.3, size=n)
    )
    return FeaturePanel(
        frame=frame, signals=tuple(f"x{i}" for i in range(5)), targets=("target",), lag_days=1
    )


@pytest.mark.parametrize("cap", [1, 2, 3, 5])
def test_the_fit_never_keeps_more_terms_than_the_cap(cap):
    panel = _many_signals_panel()
    model = DerivedPolynomial(degree=2, regularizer="lasso", max_terms=cap)

    model.fit(panel, panel.frame.index)

    assert 1 <= len(model.describe().terms) <= cap


def test_a_cap_no_fit_reaches_changes_nothing():
    panel = _many_signals_panel()
    fits = []
    for cap in (None, 10_000):
        model = DerivedPolynomial(degree=2, regularizer="lasso", max_terms=cap)
        model.fit(panel, panel.frame.index)
        fits.append(model.describe())

    assert fits[0] == fits[1]


def _cap_inputs():
    panel = _many_signals_panel()
    x = pd.DataFrame(
        panel.frame[list(panel.signals)].to_numpy(), columns=panel.signals, index=panel.frame.index
    )
    return x, panel.frame["target"], polynomial._time_series_cv(len(x), gap=1)


def test_a_cap_can_only_raise_the_penalty_on_the_same_grid(monkeypatch):
    x, y, cv = _cap_inputs()
    grids = []
    real = polynomial._alpha_grid

    def recording(*args):
        grids.append(real(*args))
        return grids[-1]

    monkeypatch.setattr(polynomial, "_alpha_grid", recording)
    capped, _ = polynomial._choose_alpha(x, y, cv, 1, 1.0)
    free, _ = polynomial._choose_alpha(x, y, cv, None, 1.0)

    assert free in grids[1] and capped in grids[0]
    assert capped >= free  # a cap can only push the penalty up


def test_a_capped_path_never_fits_the_penalties_past_the_cap(monkeypatch):
    # The small penalties are the slowest to fit and a cap rules them out, so
    # the walk down the path stops soon after the cap is first exceeded.
    x, y, cv = _cap_inputs()
    grid = polynomial._alpha_grid(x, y, 1.0)
    fitted = []
    real = polynomial._path

    def recording(x_, y_, alphas, l1_ratio, coef_init=None):
        fitted.extend(alphas)
        return real(x_, y_, alphas, l1_ratio, coef_init=coef_init)

    monkeypatch.setattr(polynomial, "_path", recording)
    polynomial._choose_alpha(x, y, cv, 1, 1.0)

    assert min(fitted) > grid[-1]


def test_when_no_penalty_meets_the_cap_the_largest_is_used(monkeypatch):
    x, y, cv = _cap_inputs()
    # Penalties far too small to zero anything: every one keeps all five signals.
    monkeypatch.setattr(polynomial, "_alpha_grid", lambda *_: np.array([1e-6, 1e-7, 1e-8]))

    alpha, coef = polynomial._choose_alpha(x, y, cv, 1, 1.0)

    assert alpha == 1e-6
    assert (coef != 0).sum() == 5


def test_no_term_is_ranked_by_correlation_before_the_penalty():
    import inspect

    assert "corrwith" not in inspect.getsource(polynomial)
