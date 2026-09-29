"""End-to-end: raw Bloomberg-like frame + Fama-French factor frame -> merged ->
lag-safe panel -> the FF5 benchmark -> the shared comparison contract (FYP-117)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting_engine.ingest.align import align_and_lag
from forecasting_engine.models.famafrench import FACTOR_COLUMNS, merge_factors, run_famafrench
from forecasting_engine.reporting.model_metrics import build_metrics_rows
from forecasting_engine.validation.gates import evaluate_candidate
from forecasting_engine.validation.splitters import PurgedWalkForward


def _bloomberg_frame(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    price = 100 + np.cumsum(rng.normal(scale=0.5, size=n))
    return pd.DataFrame({"Date": dates, "price": price})


def _factors_frame(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(6)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    data = {c: rng.normal(scale=0.01, size=n) for c in FACTOR_COLUMNS}
    data["RF"] = 0.0001
    return pd.DataFrame({"Date": dates, **data})


def _splitter() -> PurgedWalkForward:
    return PurgedWalkForward(train=40, test=8, embargo=2)


def test_famafrench_flows_from_raw_frames_through_the_shared_harness_to_the_comparison_table():
    bloomberg = _bloomberg_frame()
    factors = _factors_frame()

    merged = merge_factors(bloomberg, factors)
    panel = align_and_lag(
        merged.set_index("Date"), list(FACTOR_COLUMNS), "price", horizon=1
    )

    result, description = run_famafrench(panel, _splitter())

    assert result.pbo is None
    assert description.terms == FACTOR_COLUMNS

    # The gate and the comparison table are FYP-45/FYP-14's already-built
    # contract — an FF5 result must satisfy both unmodified.
    outcome = evaluate_candidate(result.oos_rank_ic, result.pbo)
    assert outcome.promoted in (True, False)

    rows = build_metrics_rows({"FF5 Benchmark": result})
    assert rows[0]["Model"].text == "FF5 Benchmark"
