import numpy as np
import pandas as pd
import pytest

from forecasting_engine.ingest.align import FeaturePanel
from forecasting_engine.models.polynomial import (
    PolynomialConfigError,
    run_derived_polynomial,
    run_user_polynomial,
)
from forecasting_engine.reporting.model_metrics import build_metrics_rows
from forecasting_engine.validation.gates import evaluate_candidate
from forecasting_engine.validation.splitters import PurgedWalkForward


def _panel(n: int = 80) -> FeaturePanel:
    rng = np.random.default_rng(2)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    sig = rng.normal(size=n)
    target = 2 * sig + rng.normal(scale=0.1, size=n)
    frame = pd.DataFrame({"sig": sig, "target": target}, index=idx)
    return FeaturePanel(frame=frame, signals=("sig",), targets=("target",), lag_days=1)


def _splitter() -> PurgedWalkForward:
    return PurgedWalkForward(train=30, test=8, embargo=2)


def test_run_user_polynomial_is_scored_like_any_model():
    result, description = run_user_polynomial("2 * sig", _panel(), _splitter())
    assert result.oos_rank_ic == result.oos_rank_ic  # a real number, not NaN
    assert description.terms == ("2 * sig",)


def test_run_user_polynomial_result_matches_the_shared_comparison_contract():
    """Regression test: a single-configuration result must plug into
    evaluate_candidate() and build_metrics_rows() without raising."""
    result, _description = run_user_polynomial("2 * sig", _panel(), _splitter())

    outcome = evaluate_candidate(result.gated_rank_ic, result.signal_t)
    assert outcome.promoted in (True, False)

    rows = build_metrics_rows({"Polynomial": result})
    assert "t" in rows[0]


def test_run_user_polynomial_raises_when_the_split_produces_no_folds():
    too_short = _panel(n=20)
    with pytest.raises(PolynomialConfigError):
        run_user_polynomial("sig", too_short, _splitter())


def test_run_derived_polynomial_reports_the_winning_candidate():
    result, description = run_derived_polynomial(_panel(), _splitter())
    assert result.signal_rank_ic == result.signal_rank_ic  # a real number, not NaN
    assert description.terms  # the winning candidate's surviving terms


def test_run_derived_polynomial_result_matches_the_shared_comparison_contract():
    """Regression test against FYP-14's contract: a Polynomial run must plug
    into evaluate_candidate() and build_metrics_rows() completely unmodified."""
    result, _description = run_derived_polynomial(_panel(), _splitter())

    outcome = evaluate_candidate(result.gated_rank_ic, result.signal_t)
    assert outcome.promoted in (True, False)

    rows = build_metrics_rows({"Polynomial": result})
    assert rows[0]["Model"].text == "Polynomial"
    assert "t" in rows[0]
