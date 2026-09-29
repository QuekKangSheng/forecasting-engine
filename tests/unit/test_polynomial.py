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
    run_derived_polynomial,
)
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


def test_user_polynomial_applies_directly_with_no_fitting():
    panel = _formula_panel()
    model = UserPolynomial("2 * vix + credit_spread_hy ** 2")

    model.fit(panel, panel.frame.index)  # must be a no-op, not raise
    predicted = model.predict(panel, panel.frame.index)

    expected = 2 * panel.frame["vix"] + panel.frame["credit_spread_hy"] ** 2
    assert list(predicted) == list(expected.astype(float))


def test_user_polynomial_describe_reports_the_formula_as_its_own_term():
    description = UserPolynomial("vix + 1").describe()
    assert description.terms == ("vix + 1",)
    assert description.coefficients == (1.0,)


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


def test_user_polynomial_rejects_unknown_column_at_predict():
    panel = _formula_panel()
    model = UserPolynomial("unknown_signal * 2")
    with pytest.raises(PolynomialConfigError):
        model.predict(panel, panel.frame.index)


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


def test_derived_polynomial_reports_coefficients_in_the_signals_own_units():
    # The displayed equation is only meaningful if its coefficients apply to the
    # raw signal, so the description must undo any internal rescaling.
    panel = _scaled_panel(1_000.0)
    model = DerivedPolynomial(degree=2, regularizer="lasso")
    model.fit(panel, panel.frame.index)
    description = model.describe()

    idx = panel.frame.index
    rebuilt = pd.Series(description.intercept, index=idx)
    for name, coefficient in zip(description.terms, description.coefficients, strict=True):
        product = pd.Series(1.0, index=idx)
        for part in name.split(" "):
            column, _, power = part.partition("^")
            product = product * panel.frame[column] ** (int(power) if power else 1)
        rebuilt = rebuilt + coefficient * product

    np.testing.assert_allclose(rebuilt, model.predict(panel, idx), rtol=1e-9, atol=1e-12)


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
    real = polynomial._REGULARIZERS["lasso"]

    def recording(cv, **kwargs):
        seen.append(cv)
        return real(cv=cv, **kwargs)

    monkeypatch.setitem(polynomial._REGULARIZERS, "lasso", recording)
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
