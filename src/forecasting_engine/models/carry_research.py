"""The bond research model: a degree-1 polynomial in the levels of rates, spreads and
volatility gauges, fitted with each input's innovation over the label window taken out,
so that its forecast is an expected return.

Run on the Models page's Bond tab as its own row, "Polynomial – fixed carry inputs
(research)". Its inputs were chosen in the November 2026 model research
(``bond/nochange_levels``) by looking at results on the Sep 2026 upload (Aug 2016 - Sep
2026), which overlaps the scored period, so its scores there overstate it. See docs/methodology.md.

Why. A bond index's h-day return is, to first order, what it earns if nothing moves (its
yield, its spread and the roll down the curve) less its duration times the same-window
move in yields and spreads. The derived polynomial reads a rate as its 1-row
difference, so that carry is not in its panel. A regression on the levels alone does not
recover it either: a persistent level whose moves drive the return seems, in one sample,
to predict its own later change (Stambaugh bias), and the plain slope credits that to the
level.

What. Per fold, on the training rows only, with every input read as its level and lagged
one row like every signal:

1. each input ``x`` is clipped to its training mean ± ``CLIP_SD`` standard deviations and
   standardised with the same mean and SD, ``z = (x − mean) / sd``;
2. each input's innovation over the row's label window is formed (Amihud & Hurvich
   2004). A daily AR(1), ``x_s = θ + ρ·x_{s−1} + e``, is fitted to its training levels;
   ``ρ̂`` is bias-corrected, ``ρᶜ = ρ̂ + (1 + 3ρ̂)/n + 3(1 + 3ρ̂)/n²`` with ``n`` the
   number of daily pairs, and capped at ``RHO_CAP``; and compounded to the window,
   ``κᶜ = ρᶜ^lag · (ρᶜ^h − 1)``, the slope of the window's expected change on the
   level the forecast reads, which is ``lag`` rows older than the window's start. The
   innovation is ``û = Δx − (a + κᶜ·x)``, where ``Δx = x(t+h) − x(t)`` is the change over
   the label window and ``a`` makes ``û`` average zero over the training rows;
3. ``return(t, t+h) = α + Σ c_j·z_j(t) + Σ b_k·û_k(t) + e`` is fitted. ``α`` and the
   ``b`` are unpenalised; the ``c`` carry a lasso penalty chosen as the derived
   polynomial chooses its own (time-ordered splits with a gap of ``h``, the same
   penalty grid), by the validation error of the whole equation. By Frisch-Waugh the
   lasso is fitted to the target and the ``z`` once ``α`` and the ``û`` are regressed
   out of both;
4. the forecast is ``α + Σ c_j·z_j(t)``. Each ``û`` has mean zero given the level, so
   this is the return to expect, which the portfolio optimiser needs, not the return
   if nothing moves.

With plain-OLS ``κ`` the innovation is uncorrelated with the level in the training rows,
so ``c`` is the plain regression's slope and keeps the Stambaugh bias; the correction
works through ``ρᶜ``. With ``κ = 0`` the fit is the research's raw-move version, ``b``
soaking up the window's whole move. With several correlated inputs, correcting each
input's ``κ`` on its own is an approximation of the multi-predictor method (Amihud,
Hurvich & Wang 2009).

The innovation looks forward exactly as far as the label does. It is read on training
rows only, whose labels the splitter has already purged against the test window, and a
row whose label or innovation is not yet complete is not used. Every fold trains on all
earlier rows (an expanding window), as the research ran it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import enet_path

from forecasting_engine.extraction.bloomberg_csv import ColumnSource
from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.ingest.align import FeaturePanel, Transform, align_and_lag
from forecasting_engine.models.base import ModelDescription
from forecasting_engine.models.polynomial import (
    _ALPHA_EPS,
    _N_ALPHAS,
    CLIP_SD,
    PolynomialConfigError,
    _time_series_cv,
)
from forecasting_engine.models.sign_ruled import input_columns
from forecasting_engine.reporting.model_metrics import ModelRunResult
from forecasting_engine.validation.harness import evaluate, summarize
from forecasting_engine.validation.splitters import PurgedWalkForward, expanding

CARRY_INPUTS: Mapping[TargetRole, tuple[str, ...]] = {
    TargetRole.BOND: ("USGG10YR", "USGGBE10", "USYC2Y10", "LF98OAS", "LUACOAS", "VIX", "JPMVXYGL"),
}
"""Per target, the tickers of its inputs, read as levels. The research took every
signal the page reads as a level or a difference, never a price; on the Sep 2026 upload
those were these seven. Fixed here so an upload with other series doesn't change them."""

RHO_CAP: float = 0.9999
"""The bias-corrected daily AR(1) coefficient is capped here, as Amihud, Hurvich & Wang
(2009) cap theirs, so a near-unit-root level is never corrected into an explosive one."""

_MIN_TRAINING_ROWS: int = 30
"""Fewer complete training rows than this, and a fit with an intercept, one innovation
per input and a penalised slope per input is more noise than signal."""

VERSION: int = 1
"""Raise when a change above alters a fit, so results saved under the old one refit."""


class CarryResearchInputsError(PolynomialConfigError):
    """The committed data lacks an input the bond research model needs."""


class CarryResearchDataError(PolynomialConfigError):
    """The message is written for a portfolio manager, like the other models' errors."""


# ── The innovation ───────────────────────────────────────────────────────────


def label_window_moves(panel: FeaturePanel) -> pd.DataFrame:
    """Each signal's change over each row's label window, ``level(t+h) − level(t)``.

    A panel's signals are lagged one row, so the level dated ``t`` sits on the next
    row and this reads ``h + 1`` rows ahead: as far as the row's label reaches and no
    further. That is a look ahead, so it is for training rows only."""
    levels = panel.frame[list(panel.signals)].shift(-panel.lag_days)
    return levels.shift(-panel.horizon) - levels


def corrected_rho(levels: pd.Series) -> float:
    """``ρᶜ``: the daily AR(1) coefficient of ``levels``, bias-corrected (Amihud &
    Hurvich 2004) and capped at ``RHO_CAP``."""
    pairs = pd.concat([levels.shift(1), levels], axis=1).dropna().to_numpy(float)
    n = len(pairs)
    lagged = pairs[:, 0] - pairs[:, 0].mean()
    if not lagged @ lagged > 0:  # a level that never moves has nothing to predict
        return 0.0
    rho = float(lagged @ (pairs[:, 1] - pairs[:, 1].mean()) / (lagged @ lagged))
    return min(rho + (1 + 3 * rho) / n + 3 * (1 + 3 * rho) / n**2, RHO_CAP)


def corrected_kappa(
    daily: pd.DataFrame, x: np.ndarray, moves: np.ndarray, horizon: int, lag: int
) -> np.ndarray:
    """Per input, ``κᶜ = ρᶜ^lag · (ρᶜ^h − 1)`` from its training levels ``daily``: the
    slope of the expected change over a label window on the level the forecast reads."""
    rho = np.array([corrected_rho(daily[c]) for c in daily.columns])
    return rho**lag * (rho**horizon - 1)


Kappa = Callable[[pd.DataFrame, np.ndarray, np.ndarray, int, int], np.ndarray]
"""``(daily training levels, levels on the complete rows, their label-window moves,
horizon, lag) -> κ per input``."""


def innovations(x: np.ndarray, moves: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    """``û = Δx − (a + κ·x)`` per input, with ``a`` making each average zero."""
    unexpected = moves - x * kappa
    return unexpected - unexpected.mean(axis=0)


# ── The penalised fit, with the innovations unpenalised ──────────────────────


def _lstsq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _residualised(
    z: np.ndarray, controls: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``z`` and ``y`` with their least-squares fit on ``controls`` taken out, and each
    residual ``z`` column's SD (1 where it has none), which the penalty is scaled by."""
    ry = y - controls @ _lstsq(controls, y)
    rz = z - controls @ _lstsq(controls, z)
    scale = rz.std(axis=0)
    return rz, ry, np.where(scale == 0, 1.0, scale)


def _alpha_grid(z: np.ndarray, controls: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``_N_ALPHAS`` penalties, largest first, from the smallest that zeroes every
    residual ``z`` down by ``_ALPHA_EPS``: the derived polynomial's grid, on what is
    left once the controls are taken out."""
    rz, ry, scale = _residualised(z, controls, y)
    largest = float(np.abs((rz / scale).T @ ry).max()) / len(y)
    if not largest > 0:  # nothing left to explain: every penalty gives the same fit
        return np.array([1.0])
    return np.geomspace(largest, largest * _ALPHA_EPS, _N_ALPHAS)


def lasso_given_controls(
    z: np.ndarray, controls: np.ndarray, y: np.ndarray, alphas: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The lasso of ``y`` on ``z`` with ``controls`` unpenalised, at every penalty: by
    Frisch-Waugh, the lasso of the residualised ``y`` on the residualised ``z``. Returns
    the coefficients of ``z`` and of ``controls``, one column per penalty."""
    rz, ry, scale = _residualised(z, controls, y)
    _, coefs, _ = enet_path(rz / scale, ry, l1_ratio=1.0, alphas=alphas, max_iter=10_000)
    slopes = coefs / scale[:, None]
    return slopes, _lstsq(controls, y[:, None] - z @ slopes)


def _fit(z: np.ndarray, controls: np.ndarray, y: np.ndarray, gap: int) -> tuple[np.ndarray, float]:
    """The slopes of ``z`` and the intercept at the penalty with the lowest validation
    error over time-ordered splits, the error of the whole equation, controls
    included."""
    alphas = _alpha_grid(z, controls, y)
    slopes, fitted = lasso_given_controls(z, controls, y, alphas)
    errors = np.zeros(len(alphas))
    for train, validate in _time_series_cv(len(y), gap=gap).split(z):
        s, c = lasso_given_controls(z[train], controls[train], y[train], alphas)
        forecast = controls[validate] @ c + z[validate] @ s
        errors += ((forecast - y[validate][:, None]) ** 2).mean(axis=0)
    best = int(np.argmin(errors))
    return slopes[:, best], float(fitted[0, best])


# ── The Forecaster ───────────────────────────────────────────────────────────


@dataclass
class CarryResearchPolynomial:
    """``forecast = α + Σ c_j · z_j``, one term per input the penalty keeps, ``z_j`` its
    clipped, standardised level: the return the training rows say to expect."""

    kappa: Kappa = corrected_kappa
    """How each input's ``κ`` is found. The bias-corrected one; tests swap in others."""
    name: str = field(default="CarryResearchPolynomial", init=False)

    def __post_init__(self) -> None:
        self._coef: np.ndarray | None = None

    def fit(self, panel: FeaturePanel, train: pd.DatetimeIndex) -> None:
        signals, target = list(panel.signals), panel.targets[0]
        rows = panel.frame.loc[train, [*signals, target]]
        moves = label_window_moves(panel).loc[train]
        complete = rows.notna().all(axis=1) & moves.notna().all(axis=1)
        if complete.sum() < _MIN_TRAINING_ROWS:
            raise CarryResearchDataError(
                f"not enough complete training rows for the bond research model (need at "
                f"least {_MIN_TRAINING_ROWS}, got {int(complete.sum())})."
            )
        x = rows.loc[complete, signals]
        levels, window = x.to_numpy(float), moves.loc[complete].to_numpy(float)
        kappa = self.kappa(rows[signals], levels, window, panel.horizon, panel.lag_days)
        surprise = innovations(levels, window, kappa)
        spread = surprise.std(axis=0)
        controls = np.column_stack(
            [np.ones(len(surprise)), surprise / np.where(spread == 0, 1.0, spread)]
        )
        self._signals = signals
        self._mean, sd = x.mean(), x.std()
        self._sd = sd.where(sd > 0, 1.0)
        self._low, self._high = self._mean - CLIP_SD * self._sd, self._mean + CLIP_SD * self._sd
        self._coef, self._intercept = _fit(
            self._z(x), controls, rows.loc[complete, target].to_numpy(float), panel.horizon
        )

    def predict(self, panel: FeaturePanel, idx: pd.DatetimeIndex) -> pd.Series:
        if self._coef is None:
            raise RuntimeError("predict() called before fit()")
        predicted = pd.Series(np.nan, index=idx, dtype=float)
        x = panel.frame.loc[idx, self._signals].dropna()
        if not x.empty:
            predicted.loc[x.index] = self._intercept + self._z(x) @ self._coef
        return predicted

    def describe(self) -> ModelDescription:
        """Only the terms the penalty kept, in the standardised levels, with every
        input's clip bounds and standardisation."""
        if self._coef is None:
            raise RuntimeError("describe() called before fit()")
        kept = [j for j, c in enumerate(self._coef) if c != 0]
        return ModelDescription(
            name=self.name,
            terms=tuple(self._signals[j] for j in kept),
            coefficients=tuple(float(self._coef[j]) for j in kept),
            intercept=self._intercept,
            input_bounds={s: (float(self._low[s]), float(self._high[s])) for s in self._signals},
            standardisation={s: (float(self._mean[s]), float(self._sd[s])) for s in self._signals},
        )

    def _z(self, x: pd.DataFrame) -> np.ndarray:
        clipped = x[self._signals].clip(lower=self._low, upper=self._high, axis=1)
        return ((clipped - self._mean) / self._sd).to_numpy(float)


# ── Inputs and the run ───────────────────────────────────────────────────────


def carry_inputs(
    columns: Sequence[str], sources: Mapping[str, ColumnSource], role: TargetRole
) -> tuple[str, ...]:
    """The signal columns of ``role``'s inputs, found by each column's ticker."""
    wanted = CARRY_INPUTS.get(role)
    if wanted is None:
        raise CarryResearchInputsError("the bond research model is for the bond target only.")
    return tuple(
        input_columns(columns, sources, wanted, "the bond research model", CarryResearchInputsError)
    )


def run_carry_research_polynomial(
    frame: pd.DataFrame,
    inputs: Sequence[str],
    price_col: str,
    horizon: int,
    splitter: PurgedWalkForward,
) -> tuple[ModelRunResult, ModelDescription, FeaturePanel]:
    """Walk the bond research model forward on ``splitter``'s test windows.

    Its inputs are read as levels whatever the page's transform map says (rates are
    otherwise differenced), and every fold trains on all earlier rows. One
    configuration. Returns the panel too, for the equation's columns."""
    panel = align_and_lag(
        frame,
        list(inputs),
        price_col,
        horizon=horizon,
        transforms=dict.fromkeys(inputs, Transform.LEVEL),
    )
    folds = evaluate(CarryResearchPolynomial, panel, expanding(splitter))
    if not folds:
        raise PolynomialConfigError(expanding(splitter).too_short(panel))
    result, description = summarize(folds)
    return result, description, panel
