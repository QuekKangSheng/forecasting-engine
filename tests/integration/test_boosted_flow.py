"""End-to-end: raw frame -> lag-safe panel -> XGBoost/LightGBM (Optuna-tuned once,
SHAP-explained) -> the shared comparison contract (FYP-133)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting_engine.ingest.align import align_and_lag
from forecasting_engine.models.boosted import run_boosted
from forecasting_engine.reporting.model_metrics import build_metrics_rows
from forecasting_engine.validation.gates import evaluate_candidate
from forecasting_engine.validation.splitters import PurgedWalkForward

_N_TRIALS = 5  # kept tiny — Optuna tunes once per run, not once per fold


def _raw_frame(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(8)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    sig_a = rng.normal(size=n)
    sig_b = rng.normal(size=n)
    price = 100 + np.cumsum(0.3 * sig_a - 0.1 * sig_b + rng.normal(scale=0.2, size=n))
    return pd.DataFrame({"sig_a": sig_a, "sig_b": sig_b, "price": price}, index=idx)


def _splitter() -> PurgedWalkForward:
    return PurgedWalkForward(train=40, test=8, embargo=2)


def test_boosted_flows_from_a_raw_frame_through_the_shared_harness_to_the_comparison_table():
    frame = _raw_frame()
    panel = align_and_lag(frame, ["sig_a", "sig_b"], "price", horizon=1)

    result, description = run_boosted(panel, _splitter(), n_trials=_N_TRIALS, n_blocks=4)

    assert result.pbo is not None
    assert description.terms == ("sig_a", "sig_b")

    # The gate and the comparison table are FYP-45/FYP-14's already-built
    # contract — an ML result must satisfy both unmodified.
    outcome = evaluate_candidate(result.oos_rank_ic, result.pbo)
    assert outcome.promoted in (True, False)

    rows = build_metrics_rows({"Machine Learning": result})
    assert rows[2]["Model"].text == "Machine Learning"
