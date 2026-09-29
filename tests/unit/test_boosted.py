import numpy as np
import pandas as pd
import pytest

from forecasting_engine.ingest.align import FeaturePanel
from forecasting_engine.models.boosted import (
    BoostedConfigError,
    BoostedForecaster,
    tune_hyperparameters,
)

_N_TRIALS = 5  # kept tiny — tuning happens once per test, not per fold

#: Quiets each library's own stability/verbosity warnings on small test data,
#: without reaching into boosted.py's private defaults for them.
_QUIET_PARAMS = {
    "xgboost": {"verbosity": 0},
    "lightgbm": {"verbose": -1, "min_child_samples": 1, "min_data_in_leaf": 1},
}


def _panel(n: int = 80, seed: int = 0) -> FeaturePanel:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    sig_a = rng.normal(size=n)
    sig_b = rng.normal(size=n)
    target = 2 * sig_a - sig_b + rng.normal(scale=0.05, size=n)
    frame = pd.DataFrame({"sig_a": sig_a, "sig_b": sig_b, "target": target}, index=idx)
    return FeaturePanel(frame=frame, signals=("sig_a", "sig_b"), targets=("target",), lag_days=1)


# --- tune_hyperparameters: FYP-128, FYP-129 ---------------------------------


@pytest.mark.parametrize("library", ["xgboost", "lightgbm"])
def test_tune_hyperparameters_stays_within_the_declared_search_space(library):
    panel = _panel()
    params = tune_hyperparameters(panel, panel.frame.index, library, n_trials=_N_TRIALS)

    assert 20 <= params["n_estimators"] <= 100
    assert 2 <= params["max_depth"] <= 5
    assert 0.01 <= params["learning_rate"] <= 0.3


def test_tune_hyperparameters_rejects_an_unknown_library():
    panel = _panel()
    with pytest.raises(BoostedConfigError):
        tune_hyperparameters(panel, panel.frame.index, "not-a-library", n_trials=_N_TRIALS)


def test_tune_hyperparameters_rejects_too_few_training_rows():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    frame = pd.DataFrame({"sig_a": range(5), "sig_b": range(5), "target": range(5)}, index=idx)
    panel = FeaturePanel(frame=frame, signals=("sig_a", "sig_b"), targets=("target",), lag_days=1)

    with pytest.raises(BoostedConfigError):
        tune_hyperparameters(panel, panel.frame.index, "xgboost", n_trials=_N_TRIALS)


# --- BoostedForecaster: FYP-126, FYP-127, FYP-149 (SHAP) --------------------


def _params(library: str) -> dict:
    return {**_QUIET_PARAMS[library], "n_estimators": 30, "max_depth": 3, "learning_rate": 0.1}


@pytest.mark.parametrize("library", ["xgboost", "lightgbm"])
def test_boosted_forecaster_fits_and_predicts(library):
    panel = _panel()
    model = BoostedForecaster(library, _params(library))

    model.fit(panel, panel.frame.index)
    predicted = model.predict(panel, panel.frame.index)

    assert len(predicted) == len(panel.frame)
    assert predicted.notna().all()


@pytest.mark.parametrize("library", ["xgboost", "lightgbm"])
def test_boosted_forecaster_explain_reports_shap_importance_per_signal(library):
    panel = _panel()
    model = BoostedForecaster(library, _params(library))
    model.fit(panel, panel.frame.index)

    description = model.explain()

    assert description.terms == ("sig_a", "sig_b")
    assert len(description.coefficients) == 2
    # mean |SHAP value| is never negative
    assert all(c >= 0 for c in description.coefficients)
    # sig_a has twice sig_b's true coefficient magnitude in the synthetic
    # target — its attribution should come out higher.
    assert description.coefficients[0] > description.coefficients[1]


@pytest.mark.parametrize("library", ["xgboost", "lightgbm"])
def test_boosted_forecaster_describe_reports_native_importance_without_shap(library, monkeypatch):
    panel = _panel()
    model = BoostedForecaster(library, _params(library))
    model.fit(panel, panel.frame.index)
    monkeypatch.setattr(
        "forecasting_engine.models.boosted.shap.TreeExplainer",
        lambda *_: pytest.fail("describe() must not run SHAP"),
    )

    description = model.describe()

    assert description.terms == ("sig_a", "sig_b")
    assert description.coefficients[0] > description.coefficients[1]


def test_boosted_forecaster_rejects_an_unknown_library():
    with pytest.raises(BoostedConfigError):
        BoostedForecaster("not-a-library", {})


def test_boosted_forecaster_predict_masks_rows_with_a_missing_signal():
    panel = _panel()
    model = BoostedForecaster("xgboost", _params("xgboost"))
    model.fit(panel, panel.frame.index)
    panel.frame.loc[panel.frame.index[0], "sig_a"] = float("nan")

    predicted = model.predict(panel, panel.frame.index)

    assert pd.isna(predicted.iloc[0])
    assert not pd.isna(predicted.iloc[1])


def test_boosted_forecaster_rejects_too_few_training_rows():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    frame = pd.DataFrame({"sig_a": range(5), "sig_b": range(5), "target": range(5)}, index=idx)
    panel = FeaturePanel(frame=frame, signals=("sig_a", "sig_b"), targets=("target",), lag_days=1)
    model = BoostedForecaster("xgboost", _params("xgboost"))

    with pytest.raises(BoostedConfigError):
        model.fit(panel, panel.frame.index)


def test_boosted_forecaster_predict_before_fit_raises():
    model = BoostedForecaster("xgboost", _params("xgboost"))
    with pytest.raises(RuntimeError):
        model.predict(_panel(), _panel().frame.index)


@pytest.mark.parametrize("method", ["describe", "explain"])
def test_boosted_forecaster_describe_before_fit_raises(method):
    model = BoostedForecaster("xgboost", _params("xgboost"))
    with pytest.raises(RuntimeError):
        getattr(model, method)()
