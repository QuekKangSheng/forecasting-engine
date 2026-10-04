from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import TimeSeriesSplit

from forecasting_engine.ingest.align import FeaturePanel
from forecasting_engine.models import polynomial
from forecasting_engine.models.polynomial import (
    MAX_DEGREE,
    DerivedPolynomial,
    PolynomialConfigError,
    UserPolynomial,
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


def test_deriving_from_one_candidate_reports_no_pbo_rather_than_crashing():
    # PBO measures how often the best of several configurations was luck, so a
    # single configuration has nothing to compare against — the same "no
    # configuration search" case FF5 and a user formula report.
    panel = _linear_panel(n=120)
    splitter = PurgedWalkForward(train=40, test=10, embargo=2)

    result, description = run_derived_polynomial(
        panel, splitter, candidates=(DerivedPolynomial(degree=1),)
    )

    assert result.pbo is None
    assert description.name == "DerivedPolynomial"


def test_deriving_with_no_candidates_says_so_in_plain_words():
    panel = _linear_panel(n=120)
    splitter = PurgedWalkForward(train=40, test=10, embargo=2)

    with pytest.raises(PolynomialConfigError, match="at least one"):
        run_derived_polynomial(panel, splitter, candidates=())


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


def test_term_selection_and_scaling_inside_validation_see_only_earlier_rows(monkeypatch):
    # The penalty is chosen by validating on later rows of the training window.
    # Picking the top terms, or fitting the scaler, on the whole window first
    # would let those later rows shape what they then validate.
    panel = _scaled_panel(1.0)
    model = DerivedPolynomial(degree=3, regularizer="lasso", max_terms=2)
    windows = []
    real = polynomial._prepare

    def recording(x, y, max_terms):
        windows.append(x.index)
        return real(x, y, max_terms)

    monkeypatch.setattr(polynomial, "_prepare", recording)
    model.fit(panel, panel.frame.index)

    *inner, final = windows
    assert inner, "validation must prepare each inner split itself"
    everything = panel.frame.index
    assert final.equals(everything)
    for window in inner:
        assert len(window) < len(everything)
        assert window.equals(everything[: len(window)]), "an inner split trains on a prefix"
