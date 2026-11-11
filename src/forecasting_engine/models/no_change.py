"""The no-change polynomial: a degree-1 polynomial in the levels of the rates, spreads
and gauges, fitted with each training row's own news taken out, that forecasts the
return expected if none of them moves.

Why. An index's h-day return is, to first order, what it earns if nothing changes (for
a bond index its yield, its spread and the roll down the curve) less its sensitivity to
how its rates and spreads move over those same h days. Over Aug 2016 - Sep 2024 the
same-window changes of the rates and gauges explained about 97% of a 20-day bond return.
The page reads a rate as its 1-row difference, never its level, so the part a bond
forecast can hope for, its carry, is not in the derived polynomial's panel: in about
half the bond folds its lasso kept no term at all. Handing it the levels alone is not
enough either. A persistent level whose changes move the return looks, in one sample,
as if it predicted its own later change (Stambaugh bias), and a plain regression on it
credits that to the level at many times the size of the carry.

What. Per fold, on the training rows only:

1. the inputs are every signal the page reads as a level or a difference (a rate,
   spread or gauge: a "state" series), never one it reads as a log return (a price),
   chosen by kind, not by name. Each enters as its level, read as of each target date
   and lagged one row like every signal (``level_panel``);
2. each input is clipped to its training mean ± ``CLIP_SD`` standard deviations and
   standardised with the same mean and SD, ``z = (x − mean) / sd``;
3. ``return(t, t+h) = a + Σ c_j z_j(t) + Σ b_k news_k(t) + e`` is fitted, where
   ``news_k(t)`` is the change of input k's level over that row's own label window,
   ``level_k(t+h) − level_k(t)`` (``label_window_news``). The ``b`` and ``a`` are
   unpenalised; the ``c`` carry a lasso penalty chosen as the derived polynomial
   chooses its own (the same time-ordered splits, penalty grid and term cap), by the
   validation error of the whole equation, news included. By Frisch-Waugh the lasso
   is fitted to the target and the ``z`` once the intercept and news are regressed out
   of both;
4. the forecast is ``a + Σ c_j z_j(t)``: the fitted return if no input moves over the
   horizon. The ``b`` soak up the news, so the ``c`` are estimated against what is left
   of the return (about a sixth of it on bond), and a level's drift through one sample
   is no longer credited to it.

The news looks forward exactly as far as the label does. It is read on training rows
only, whose labels the splitter has already purged against the test window, and a
row whose label or news is not yet complete is not used.

Its level is its own intercept, not the training-mean target. The harness judges
every model's forecasts against the model's own mean in-sample forecast
(``validation.harness.fold_baseline``), so that gap earns it nothing, and a fold that
keeps no term adds exactly nothing to the Signal Rank IC.

From the November 2026 model research (``bond/nochange_levels``), where on bond it
was the one derivation that gave a usable, sign-stable polynomial: about 0.25% per 20
days on average against a carry of about 0.20%, the 10-year yield's level kept with a
positive sign in every refit that kept it. It is a candidate in the derived
polynomial's grid (``polynomial.run_derived_polynomial``) for every target, on the
page's rolling window and on an expanding one, chosen only when it scores best.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import enet_path

from forecasting_engine.ingest.align import FeaturePanel, Transform, align_and_lag
from forecasting_engine.models.base import ModelDescription
from forecasting_engine.models.polynomial import (
    _ALPHA_EPS,
    _N_ALPHAS,
    CLIP_SD,
    PolynomialConfigError,
    _time_series_cv,
)

_MIN_TRAINING_ROWS: int = 30
"""Fewer complete training rows than this, and a fit with an intercept, one news term
per input and a penalised slope per input is more noise than signal."""


class NoChangeDataError(PolynomialConfigError):
    """The message is written for a portfolio manager, like the other models' errors."""


# ── Inputs: the state series, as levels ──────────────────────────────────────


def state_columns(
    signal_cols: Sequence[str], transforms: Mapping[str, Transform]
) -> tuple[str, ...]:
    """The signals the page reads as a level or a difference: its rates, spreads and
    gauges. A signal it reads as a log return is a price, and is left out."""
    return tuple(
        c for c in signal_cols if transforms.get(c, Transform.LEVEL) is not Transform.LOG_RETURN
    )


def level_panel(
    frame: pd.DataFrame,
    signal_cols: Sequence[str],
    transforms: Mapping[str, Transform],
    price_col: str,
    horizon: int,
) -> FeaturePanel | None:
    """The page's panel with every state series read as its level, whatever the page's
    transform map says, or ``None`` when the data has no state series. Aligned, lagged
    and labelled by ``align_and_lag`` exactly as the page's own panel is."""
    states = state_columns(signal_cols, transforms)
    if not states:
        return None
    return align_and_lag(
        frame,
        list(states),
        price_col,
        horizon=horizon,
        transforms=dict.fromkeys(states, Transform.LEVEL),
    )


def label_window_news(panel: FeaturePanel) -> pd.DataFrame:
    """Each signal's change over each row's label window, ``level(t+h) − level(t)``.

    A panel's signals are lagged one row, so the level dated ``t`` sits on the next
    row and this reads ``h + 1`` rows ahead: as far as the row's label reaches and no
    further. That is a look ahead, so it is for training rows only."""
    levels = panel.frame[list(panel.signals)].shift(-panel.lag_days)
    return levels.shift(-panel.horizon) - levels


# ── The penalised fit, with the news unpenalised ─────────────────────────────


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


def _lasso_given_controls(
    z: np.ndarray, controls: np.ndarray, y: np.ndarray, alphas: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The lasso of ``y`` on ``z`` with ``controls`` unpenalised, at every penalty: by
    Frisch-Waugh, the lasso of the residualised ``y`` on the residualised ``z``. Returns
    the coefficients of ``z`` and of ``controls``, one column per penalty."""
    rz, ry, scale = _residualised(z, controls, y)
    _, coefs, _ = enet_path(rz / scale, ry, l1_ratio=1.0, alphas=alphas, max_iter=10_000)
    slopes = coefs / scale[:, None]
    return slopes, _lstsq(controls, y[:, None] - z @ slopes)


def _eligible(
    alphas: np.ndarray, slopes: np.ndarray, controls: np.ndarray, max_terms: int | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The penalties a term cap leaves eligible, as the derived polynomial's cap does:
    those before the first whose fit keeps more than ``max_terms`` terms, or the
    largest alone if even it keeps more."""
    if max_terms is None or slopes.shape[0] <= max_terms:  # the cap can't bind
        return alphas, slopes, controls
    over = np.flatnonzero((slopes != 0).sum(axis=0) > max_terms)
    stop = max(int(over[0]) if len(over) else len(alphas), 1)
    return alphas[:stop], slopes[:, :stop], controls[:, :stop]


def _fit(
    z: np.ndarray, controls: np.ndarray, y: np.ndarray, gap: int, max_terms: int | None
) -> tuple[np.ndarray, float]:
    """The slopes of ``z`` and the intercept at the penalty with the lowest validation
    error over time-ordered splits, the error of the whole equation, controls
    included."""
    alphas = _alpha_grid(z, controls, y)
    slopes, fitted = _lasso_given_controls(z, controls, y, alphas)
    alphas, slopes, fitted = _eligible(alphas, slopes, fitted, max_terms)
    errors = np.zeros(len(alphas))
    for train, validate in _time_series_cv(len(y), gap=gap).split(z):
        s, c = _lasso_given_controls(z[train], controls[train], y[train], alphas)
        forecast = controls[validate] @ c + z[validate] @ s
        errors += ((forecast - y[validate][:, None]) ** 2).mean(axis=0)
    best = int(np.argmin(errors))
    return slopes[:, best], float(fitted[0, best])


# ── The Forecaster ───────────────────────────────────────────────────────────


@dataclass
class NoChangePolynomial:
    """``forecast = a + Σ c_j · z_j``, one term per state series the penalty keeps,
    ``z_j`` its clipped, standardised level: the return the training rows say to
    expect if no state series moves over the horizon."""

    max_terms: int | None = None
    name: str = field(default="NoChangePolynomial", init=False)

    def __post_init__(self) -> None:
        self._coef: np.ndarray | None = None

    def fit(self, panel: FeaturePanel, train: pd.DatetimeIndex) -> None:
        signals, target = list(panel.signals), panel.targets[0]
        rows = panel.frame.loc[train, [*signals, target]]
        news = label_window_news(panel).loc[train]
        complete = rows.notna().all(axis=1) & news.notna().all(axis=1)
        if complete.sum() < _MIN_TRAINING_ROWS:
            raise NoChangeDataError(
                f"not enough complete training rows for the no-change polynomial (need at "
                f"least {_MIN_TRAINING_ROWS}, got {int(complete.sum())})."
            )
        rows, news = rows[complete], news[complete]
        x = rows[signals]
        self._signals = signals
        self._mean, sd = x.mean(), x.std()
        self._sd = sd.where(sd > 0, 1.0)
        self._low, self._high = self._mean - CLIP_SD * self._sd, self._mean + CLIP_SD * self._sd
        moves = news.to_numpy(float)
        spread = moves.std(axis=0)
        controls = np.column_stack(
            [np.ones(len(moves)), moves / np.where(spread == 0, 1.0, spread)]
        )
        self._coef, self._intercept = _fit(
            self._z(x), controls, rows[target].to_numpy(float), panel.horizon, self.max_terms
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
