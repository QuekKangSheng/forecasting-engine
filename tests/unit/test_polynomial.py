
import numpy as np
import pandas as pd
import pytest

from forecasting_engine.ingest.align import FeaturePanel
from forecasting_engine.models.polynomial import (
    PolynomialConfigError,
    UserPolynomial,
    placeholders,
)


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
