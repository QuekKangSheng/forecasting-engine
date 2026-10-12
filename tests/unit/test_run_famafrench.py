import numpy as np
import pandas as pd
import pytest

from forecasting_engine.ingest.align import FeaturePanel
from forecasting_engine.models.famafrench import FACTOR_COLUMNS, FamaFrenchDataError, run_famafrench
from forecasting_engine.reporting.model_metrics import build_metrics_rows
from forecasting_engine.validation.gates import evaluate_candidate
from forecasting_engine.validation.splitters import PurgedWalkForward


def _panel(n: int = 80) -> FeaturePanel:
    rng = np.random.default_rng(4)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    factors = {c: rng.normal(size=n) for c in FACTOR_COLUMNS}
    target = 0.5 * factors["Mkt-RF"] - 0.2 * factors["SMB"] + rng.normal(scale=0.05, size=n)
    frame = pd.DataFrame({**factors, "target": target}, index=idx)
    return FeaturePanel(frame=frame, signals=FACTOR_COLUMNS, targets=("target",), lag_days=1)


def _splitter() -> PurgedWalkForward:
    return PurgedWalkForward(train=30, test=8, embargo=2)


def test_run_famafrench_fits_the_five_factors():
    result, description = run_famafrench(_panel(), _splitter())
    assert result.oos_rank_ic == result.oos_rank_ic  # a real number, not NaN
    assert description.terms == FACTOR_COLUMNS


def test_run_famafrench_raises_when_the_split_produces_no_folds():
    too_short = _panel(n=20)
    with pytest.raises(FamaFrenchDataError):
        run_famafrench(too_short, _splitter())


def test_run_famafrench_result_matches_the_shared_comparison_contract():
    """Regression test against FYP-14's contract: an FF5 run must plug into
    evaluate_candidate() and build_metrics_rows() completely unmodified."""
    result, _description = run_famafrench(_panel(), _splitter())

    outcome = evaluate_candidate(result.gated_rank_ic, result.signal_t)
    assert outcome.promoted in (True, False)

    rows = build_metrics_rows({"FF5 Benchmark": result})
    assert rows[0]["Model"].text == "FF5 Benchmark"
    assert "t" in rows[0]
