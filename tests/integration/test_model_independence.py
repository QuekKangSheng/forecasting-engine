"""Which models run together never changes any one model's result (FYP-161).

PBO is computed only within a family's own configurations, so ticking other
families on or off must leave every family's PBO and Rank IC exactly as it was.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting_engine.ingest.align import align_and_lag
from forecasting_engine.models.boosted import run_boosted
from forecasting_engine.models.naive import run_naive
from forecasting_engine.models.polynomial import run_derived_polynomial, run_user_polynomial
from forecasting_engine.validation.splitters import PurgedWalkForward

_N_TRIALS = 3


def _panel():
    rng = np.random.default_rng(8)
    n = 120
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    sig_a, sig_b = rng.normal(size=n), rng.normal(size=n)
    price = 100 + np.cumsum(0.3 * sig_a - 0.1 * sig_b + rng.normal(scale=0.2, size=n))
    frame = pd.DataFrame({"sig_a": sig_a, "sig_b": sig_b, "price": price}, index=idx)
    return align_and_lag(frame, ["sig_a", "sig_b"], "price", horizon=1)


def _splitter() -> PurgedWalkForward:
    return PurgedWalkForward(train=20, test=5, embargo=2, tuning_rows=45)


def _ml(panel):
    result, _description, _tuning = run_boosted(
        panel, _splitter(), n_trials=_N_TRIALS, retune_trials=_N_TRIALS, n_blocks=4
    )
    return result


def test_ml_alone_and_ml_with_every_other_model_score_identically():
    panel = _panel()
    alone = _ml(panel)

    run_naive(panel, _splitter())
    derived, _ = run_derived_polynomial(panel, _splitter(), n_blocks=4)
    run_user_polynomial("sig_a", panel, _splitter())
    together = _ml(panel)
    derived_after_ml, _ = run_derived_polynomial(panel, _splitter(), n_blocks=4)

    assert together.pbo == alone.pbo
    assert together.oos_rank_ic == alone.oos_rank_ic
    assert derived_after_ml.pbo == derived.pbo
    assert derived_after_ml.oos_rank_ic == derived.oos_rank_ic
