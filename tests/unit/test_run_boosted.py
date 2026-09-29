import numpy as np
import pandas as pd
import pytest
import shap

from forecasting_engine.ingest.align import FeaturePanel
from forecasting_engine.models.boosted import BoostedConfigError, run_boosted
from forecasting_engine.reporting.model_metrics import build_metrics_rows
from forecasting_engine.validation.gates import evaluate_candidate
from forecasting_engine.validation.splitters import PurgedWalkForward

_N_TRIALS = 5  # kept tiny — Optuna tunes once per run, not once per fold


def _panel(n: int = 80) -> FeaturePanel:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    sig_a = rng.normal(size=n)
    sig_b = rng.normal(size=n)
    target = 2 * sig_a - sig_b + rng.normal(scale=0.05, size=n)
    frame = pd.DataFrame({"sig_a": sig_a, "sig_b": sig_b, "target": target}, index=idx)
    return FeaturePanel(frame=frame, signals=("sig_a", "sig_b"), targets=("target",), lag_days=1)


def _splitter() -> PurgedWalkForward:
    return PurgedWalkForward(train=30, test=8, embargo=2)


def test_run_boosted_computes_a_real_pbo():
    # Unlike FF5/UserPolynomial, there are always two candidates here (tuned
    # XGBoost vs. tuned LightGBM), so pbo is never None.
    result, description = run_boosted(_panel(), _splitter(), n_trials=_N_TRIALS, n_blocks=4)

    assert result.pbo is not None
    assert 0.0 <= result.pbo <= 1.0
    assert result.oos_rank_ic == result.oos_rank_ic  # a real number, not NaN
    assert description.terms == ("sig_a", "sig_b")


def test_run_boosted_raises_when_the_split_produces_no_folds():
    too_short = _panel(n=20)
    with pytest.raises(BoostedConfigError):
        run_boosted(too_short, _splitter(), n_trials=_N_TRIALS)


def test_run_boosted_result_matches_the_shared_comparison_contract():
    """Regression test against FYP-14's contract: an ML run must plug into
    evaluate_candidate() and build_metrics_rows() completely unmodified."""
    result, _description = run_boosted(_panel(), _splitter(), n_trials=_N_TRIALS, n_blocks=4)

    outcome = evaluate_candidate(result.oos_rank_ic, result.pbo)
    assert outcome.promoted in (True, False)

    rows = build_metrics_rows({"Machine Learning": result})
    assert rows[0]["Model"].text == "Machine Learning"


def test_run_boosted_runs_shap_once_not_once_per_fold(monkeypatch):
    calls = []
    real = shap.TreeExplainer

    def counting(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr("forecasting_engine.models.boosted.shap.TreeExplainer", counting)
    splitter = _splitter()
    assert len(list(splitter.split(_panel()))) > 1

    _result, description = run_boosted(_panel(), splitter, n_trials=_N_TRIALS, n_blocks=4)

    assert len(calls) == 1
    assert all(c >= 0 for c in description.coefficients)
