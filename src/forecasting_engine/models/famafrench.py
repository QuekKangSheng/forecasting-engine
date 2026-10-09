"""FYP-42: the Fama-French five-factor benchmark Forecaster.

The factor *data* is already fully built (``ingest/fama_french.py``); what this module
adds is (1) joining that factor data into the same wide frame Bloomberg signals live in,
and (2) a ``Forecaster`` that runs an ordinary least-squares fit instead of a polynomial
one.

**Design decision:** the classic Fama-French regression explains *today's* return from
*today's* factor moves (attribution, not forecasting) — but this project needs every
model scored on the same out-of-sample, walk-forward Rank IC as Polynomial and ML. So
``FamaFrench5`` treats the five factors as lagged predictive signals, exactly like any
Bloomberg signal flowing through ``align_and_lag()``, not the textbook contemporaneous
use. This keeps "what OOS Rank IC measures" identical across every row of the comparison
table, at the cost of deviating from classical factor-attribution methodology.

**Documented simplification:** the target regressed on is the panel's own forward-return
column, not an "excess return" (target minus the risk-free rate) as classical Fama-French
monthly asset-pricing work would use. ``RF`` is still merged into the frame by
``merge_factors()`` if this needs revisiting; skipping the subtraction keeps
``FamaFrench5`` structurally identical to the polynomial models (same target column,
same shape) and keeps every model's IC/RankIC/RMSE comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm

from forecasting_engine.ingest.align import FeaturePanel
from forecasting_engine.models.base import ModelDescription
from forecasting_engine.reporting.model_metrics import ModelRunResult
from forecasting_engine.validation.harness import evaluate, summarize
from forecasting_engine.validation.splitters import PurgedWalkForward

FACTOR_COLUMNS: tuple[str, ...] = ("Mkt-RF", "SMB", "HML", "RMW", "CMA")

_MIN_TRAINING_ROWS: int = 15
"""Five factors plus an intercept is six parameters; below this a fit is more
noise than signal — reject with a clear message rather than let statsmodels
produce an unstable fit on a near-empty design matrix."""


class FamaFrenchDataError(ValueError):
    """The merged data or a fit request is invalid.

    The message is written for a portfolio manager and is safe to render
    directly in the dashboard — mirrors ``PolynomialConfigError``.
    """


def merge_factors(bloomberg_frame: pd.DataFrame, factors_frame: pd.DataFrame) -> pd.DataFrame:
    """Join the Fama-French factors onto the Bloomberg-merged frame by exact date.

    A left join: every Bloomberg row is kept, and a date the factor file doesn't
    cover yet (it lags the calendar by a month or two) has blank factors rather
    than being dropped, so ``factor_coverage`` can count it and the FF5 fit
    leaves it out like any other incomplete row.
    """
    from forecasting_engine.ingest.fama_french import DATE_COLUMN

    if bloomberg_frame.empty:
        raise FamaFrenchDataError("the Bloomberg data is empty — nothing to merge factors into.")
    return bloomberg_frame.merge(factors_frame, on=DATE_COLUMN, how="left")


@dataclass(frozen=True)
class FactorCoverage:
    """What the factor file contributed to an FF5 run."""

    first: pd.Timestamp
    last: pd.Timestamp
    rows_used: int
    """Target dates with every factor and a label, so usable for fitting and scoring."""
    missing_dates: int
    """Target dates the factor file has no row for."""


def factor_coverage(factors_frame: pd.DataFrame, panel: FeaturePanel) -> FactorCoverage:
    from forecasting_engine.ingest.fama_french import DATE_COLUMN

    dates = factors_frame[DATE_COLUMN]
    used = panel.frame[[*FACTOR_COLUMNS, panel.targets[0]]].dropna()
    return FactorCoverage(
        first=dates.min(),
        last=dates.max(),
        rows_used=len(used),
        missing_dates=int((~panel.frame.index.isin(dates)).sum()),
    )


@dataclass
class FamaFrench5:
    """An ordinary least-squares fit of the panel's target on the five (lagged)
    Fama-French factors. Reports every factor's coefficient — there is no
    regularization or shrinkage to zero any out."""

    name: str = field(default="FamaFrench5", init=False)

    def __post_init__(self) -> None:
        self._model = None
        self._intercept: float | None = None

    def fit(self, panel: FeaturePanel, train: pd.DatetimeIndex) -> None:
        frame = panel.frame.loc[train, [*FACTOR_COLUMNS, panel.targets[0]]].dropna()
        if len(frame) < _MIN_TRAINING_ROWS:
            raise FamaFrenchDataError(
                f"not enough complete training rows to fit the five-factor model "
                f"(need at least {_MIN_TRAINING_ROWS}, got {len(frame)})."
            )
        x = sm.add_constant(frame[list(FACTOR_COLUMNS)], has_constant="add")
        y = frame[panel.targets[0]]
        self._model = sm.OLS(y, x).fit()
        self._intercept = float(self._model.params["const"])

    def predict(self, panel: FeaturePanel, idx: pd.DatetimeIndex) -> pd.Series:
        if self._model is None:
            raise RuntimeError("predict() called before fit()")
        predicted = pd.Series(np.nan, index=idx, dtype=float)

        # statsmodels' OLS.predict() propagates NaN row-wise rather than
        # rejecting outright, but masking incomplete rows first keeps this
        # consistent with the polynomial models' predict() and avoids relying on
        # that propagation behaviour.
        raw = panel.frame.loc[idx, list(FACTOR_COLUMNS)].dropna()
        if raw.empty:
            return predicted

        x = sm.add_constant(raw, has_constant="add")
        predicted.loc[raw.index] = self._model.predict(x)
        return predicted

    def describe(self) -> ModelDescription:
        if self._model is None:
            raise RuntimeError("describe() called before fit()")
        params = self._model.params
        return ModelDescription(
            name=self.name,
            terms=FACTOR_COLUMNS,
            coefficients=tuple(float(params[c]) for c in FACTOR_COLUMNS),
            intercept=self._intercept,
        )


def run_famafrench(
    panel: FeaturePanel, splitter: PurgedWalkForward
) -> tuple[ModelRunResult, ModelDescription]:
    """Single evaluate() pass — no candidate grid, no PBO. Matches
    ``ModelRunResult``'s own docstring: ``pbo`` is ``None`` for FF5, "no
    configuration search" — there's nothing to tune, so there's nothing for
    PBO's CSCV to compare against, the same reason ``UserPolynomial`` reports
    ``pbo=None``."""
    folds = evaluate(FamaFrench5, panel, splitter)
    if not folds:
        raise FamaFrenchDataError(splitter.too_short(panel))
    return summarize(folds, pbo=None)
