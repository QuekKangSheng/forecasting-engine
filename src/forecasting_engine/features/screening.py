"""Univariate IC screening: which signals carry enough standalone predictive
power to be worth modelling.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import pandas as pd

from forecasting_engine.ingest.align import FeaturePanel, align_and_lag
from forecasting_engine.validation.metrics import rank_ic

INCLUSION_THRESHOLD: float = 0.02
"""A signal is included in modelling when its absolute rank IC exceeds this."""


@dataclass(frozen=True)
class SignalScore:
    """One signal's screening result. Always produced, even when excluded."""

    signal: str
    ic: float
    included: bool


def screen_signals(
    panel: FeaturePanel, threshold: float = INCLUSION_THRESHOLD
) -> tuple[SignalScore, ...]:
    """Score every signal in ``panel`` against its target; never drop one."""
    target = panel.frame[panel.targets[0]]
    scores = []
    for name in panel.signals:
        ic = rank_ic(panel.frame[name], target)
        included = bool(not pd.isna(ic) and abs(ic) > threshold)
        scores.append(SignalScore(signal=name, ic=ic, included=included))
    return tuple(scores)


def screen_over_folds(
    panel: FeaturePanel,
    folds: Iterable[tuple[pd.DatetimeIndex, pd.DatetimeIndex]],
    threshold: float = INCLUSION_THRESHOLD,
) -> dict[int, tuple[SignalScore, ...]]:
    """Re-run screen_signals() per fold, using only that fold's train window."""
    results = {}
    for fold_index, (train_idx, _test_idx) in enumerate(folds):
        train_panel = FeaturePanel(
            frame=panel.frame.loc[train_idx],
            signals=panel.signals,
            targets=panel.targets,
            lag_days=panel.lag_days,
        )
        results[fold_index] = screen_signals(train_panel, threshold=threshold)
    return results


def run_screening(
    frame: pd.DataFrame,
    signal_cols: Sequence[str],
    price_col: str,
    threshold: float = INCLUSION_THRESHOLD,
) -> tuple[SignalScore, ...]:
    """align_and_lag() then screen_signals() — the call to make once Module 1's
    quality decisions are applied and a screening pass is wanted."""
    panel = align_and_lag(frame, signal_cols, price_col)
    return screen_signals(panel, threshold=threshold)
