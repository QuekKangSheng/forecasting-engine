"""The Forecaster protocol: the one interface every model family implements.

FYP-42 (Fama-French), FYP-43 (polynomial) and FYP-44 (machine learning) each score
against the same walk-forward harness (``validation/harness.py``) and feed the same
comparison view (``reporting/model_metrics.py``). A shared protocol is what makes that
possible without the harness or the view knowing which model family produced a result.

Nothing here imports Streamlit or a concrete model library; the dashboard and the
model implementations are both callers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from forecasting_engine.ingest.align import FeaturePanel


@dataclass(frozen=True)
class ModelDescription:
    """What a fitted model is, in the sponsor's stated output format: terms and
    coefficients. ``UserPolynomial`` reports its formula as one term, with the
    fitted scale as its coefficient; a fitted model reports whatever survived
    fitting.
    """

    name: str
    terms: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float | None = None
    input_bounds: Mapping[str, tuple[float, float]] | None = None
    """Per input, the ``(low, high)`` range it is clipped to before the terms
    apply, for a model that clips; ``None`` if nothing is clipped."""
    standardisation: Mapping[str, tuple[float, float]] | None = None
    """Per input, the ``(mean, sd)`` its terms are written in: each term uses
    ``z = (x − mean) / sd`` of the clipped input. ``None`` if the terms use the
    inputs as they are."""

    def __post_init__(self) -> None:
        if len(self.terms) != len(self.coefficients):
            raise ValueError(
                f"terms and coefficients must be the same length, "
                f"got {len(self.terms)} and {len(self.coefficients)}"
            )


class Forecaster(Protocol):
    """A model family's fit/predict/describe contract.

    ``fit`` and ``predict`` both take the whole panel plus an index, rather than a
    pre-sliced frame, so a model can see lagged history right up to its own window
    without the caller re-deriving it — the same shape the architecture doc's
    ``Forecaster`` protocol specifies.
    """

    name: str

    def fit(self, panel: FeaturePanel, train: pd.DatetimeIndex) -> None: ...

    def predict(self, panel: FeaturePanel, idx: pd.DatetimeIndex) -> pd.Series: ...

    def describe(self) -> ModelDescription: ...
