"""End-to-end: raw frame -> lag-safe panel -> polynomial model -> the
shared comparison contract (FYP-125, and the Polynomial slice of FYP-141)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting_engine.ingest.align import align_and_lag
from forecasting_engine.models.polynomial import run_user_polynomial
from forecasting_engine.models.sign_ruled import run_sign_ruled_polynomial
from forecasting_engine.reporting.model_metrics import build_metrics_rows
from forecasting_engine.validation.gates import evaluate_candidate
from forecasting_engine.validation.splitters import PurgedWalkForward


def _raw_frame(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    price = 100 + np.cumsum(rng.normal(scale=0.5, size=n))
    strong_signal = np.roll(price, -1) - price
    return pd.DataFrame({"strong_signal": strong_signal, "price": price}, index=idx)


def _splitter() -> PurgedWalkForward:
    return PurgedWalkForward(train=40, test=8, embargo=2)


def test_user_supplied_formula_bypasses_fitting_and_applies_directly():
    frame = _raw_frame()
    panel = align_and_lag(frame, ["strong_signal"], "price", horizon=1)

    result, description = run_user_polynomial("strong_signal", panel, _splitter())

    # No configuration search happened, matching FF5's "no fitting" behaviour.
    assert result.pbo is None
    assert description.terms == ("strong_signal",)


def test_derived_fit_flows_through_to_the_promotion_gate_and_comparison_table():
    frame = _raw_frame()

    result, _description, _panel = run_sign_ruled_polynomial(
        frame, {"strong_signal": 1}, "price", 1, _splitter()
    )

    # The gate and the comparison table are FYP-45/FYP-14's already-built
    # contract — a Polynomial result must satisfy both unmodified.
    outcome = evaluate_candidate(result.oos_rank_ic, result.pbo)
    assert outcome.promoted in (True, False)
    assert set(outcome.failed_gates) <= {"signal_rank_ic", "pbo"}

    rows = build_metrics_rows({"Polynomial": result})
    assert rows[0]["Model"].text == "Polynomial"
