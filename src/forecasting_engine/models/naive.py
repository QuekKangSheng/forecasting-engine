"""The naive baseline: forecast each fold's training-window mean return.

It answers "does a model beat doing nothing?", so it is scored like-for-like
with the signal models: it forecasts only on rows where every panel signal is
present, the rows a signal model can forecast on. It uses no signal itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from forecasting_engine.ingest.align import FeaturePanel
from forecasting_engine.models.base import ModelDescription
from forecasting_engine.reporting.model_metrics import ModelRunResult
from forecasting_engine.validation.harness import evaluate, summarize
from forecasting_engine.validation.splitters import PurgedWalkForward


class NaiveDataError(ValueError):
    """The message is written for a portfolio manager, like the other models' errors."""


@dataclass
class NaiveMean:
    name: str = field(default="NaiveMean", init=False)

    def __post_init__(self) -> None:
        self._mean: float | None = None

    def fit(self, panel: FeaturePanel, train: pd.DatetimeIndex) -> None:
        self._mean = float(panel.frame.loc[train, panel.targets[0]].mean())

    def predict(self, panel: FeaturePanel, idx: pd.DatetimeIndex) -> pd.Series:
        if self._mean is None:
            raise RuntimeError("predict() called before fit()")
        complete = panel.frame.loc[idx, list(panel.signals)].notna().all(axis=1)
        return pd.Series(np.where(complete, self._mean, np.nan), index=idx, dtype=float)

    def describe(self) -> ModelDescription:
        if self._mean is None:
            raise RuntimeError("describe() called before fit()")
        return ModelDescription(name=self.name, terms=(), coefficients=(), intercept=self._mean)


def run_naive(
    panel: FeaturePanel, splitter: PurgedWalkForward
) -> tuple[ModelRunResult, ModelDescription]:
    """One evaluate() pass, scored like any model; nothing is searched, so no PBO."""
    folds = evaluate(NaiveMean, panel, splitter)
    if not folds:
        raise NaiveDataError(splitter.too_short(panel))
    return summarize(folds, pbo=None)
