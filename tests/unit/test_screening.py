import numpy as np
import pandas as pd

from forecasting_engine.features.screening import screen_over_folds, screen_signals
from forecasting_engine.ingest.align import FeaturePanel
from forecasting_engine.validation.splitters import PurgedWalkForward


def _panel_with_known_signals(n: int = 60) -> FeaturePanel:
    rng = np.random.default_rng(16)
    target = rng.normal(size=n)
    strong = target * 2 + rng.normal(scale=0.01, size=n)
    weak = rng.normal(size=n)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    frame = pd.DataFrame(
        {"strong_signal": strong, "weak_signal": weak, "fwd_return_1d": target},
        index=idx,
    )
    return FeaturePanel(
        frame=frame,
        signals=("strong_signal", "weak_signal"),
        targets=("fwd_return_1d",),
        lag_days=1,
    )


def test_strong_signal_included_weak_signal_excluded():
    scores = {s.signal: s for s in screen_signals(_panel_with_known_signals())}
    assert scores["strong_signal"].included is True
    assert scores["weak_signal"].included is False


def test_every_signal_scored_even_when_excluded():
    scores = screen_signals(_panel_with_known_signals())
    assert {s.signal for s in scores} == {"strong_signal", "weak_signal"}


def test_screen_over_folds_uses_only_trailing_window():
    panel = _panel_with_known_signals()
    folds = list(PurgedWalkForward(train=30, test=5, embargo=2).split(panel))
    results = screen_over_folds(panel, folds)
    assert len(results) == len(folds)
    for fold_scores in results.values():
        assert {s.signal for s in fold_scores} == {"strong_signal", "weak_signal"}


def test_a_signal_exactly_at_the_threshold_is_excluded():
    panel = _panel_with_known_signals()
    strong = next(s for s in screen_signals(panel) if s.signal == "strong_signal")
    at_threshold = screen_signals(panel, threshold=abs(strong.ic))

    assert strong.included
    assert not next(s for s in at_threshold if s.signal == "strong_signal").included
