"""Monte Carlo VaR, CVaR and maximum drawdown over the next forecast horizon (FYP-36).

Where ``risk.tail`` reads the worst days the backtest actually had, this asks what
the next ``horizon`` days could bring from here: it simulates many paths of daily
equity and bond returns, holds a portfolio through each, and reads the tail off the
simulated outcomes. The figures are tagged ``tail.MONTE_CARLO`` so a display can
tell them apart from the historical ones, which answer a different question.

Each path is built from the base case the active models give:

- **Centre.** Each asset's daily returns are shifted so that, over ``horizon`` days,
  they add up to that asset's forecast (``forecast / horizon`` a day). Compounding
  adds a little on top, about half the horizon's variance.
- **Spread.** The two indices' daily returns over the ``window`` trading days up to
  the as-of date, demeaned, so the history sets the shape and size of the moves but
  not their direction. Equity and bond are drawn together, keeping their
  correlation.
- **Sampling.** ``PARAMETRIC`` fits a two-asset normal distribution to that history
  and draws from it; ``BOOTSTRAP`` resamples whole historical days, so fat tails and
  the equity/bond relationship on stress days are kept as they were.

A portfolio's weights are set at the as-of date and left to drift through the path,
as the backtest drifts them between rebalances; nothing is traded, so no costs.

Definitions match ``risk.tail``, so every figure is a simulated path rather than an
interpolation between two: with ``n`` paths and confidence ``c`` the tail is the
``⌈n·(1−c)⌉`` worst paths, **VaR** is the loss on the last of them and **CVaR** the
mean loss across them. **Max drawdown** at ``c`` is the drawdown only those paths
exceed, the path's starting value counting as its first peak.

Each path's first day is also read on its own, as a one-day VaR and CVaR
(``as_tail_risk``), so the simulation can sit in the tail-risk table beside the
historical one-day figures on the same footing.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.risk.tail import (
    MONTE_CARLO,
    VAR_CONFIDENCES,
    TailLevel,
    TailRisk,
    _var_cvar,
)

PARAMETRIC: str = "parametric"
BOOTSTRAP: str = "bootstrap"
SAMPLING_METHODS: tuple[str, ...] = (BOOTSTRAP, PARAMETRIC)
"""Bootstrap first: it is the default, since a fitted normal thins the tails that
VaR at 99% is about."""

DEFAULT_PATHS: int = 100_000
MIN_PATHS: int = 1_000
MAX_PATHS: int = 1_000_000

SEED: int = 0
"""Fixed so the same inputs always give the same figures: a page re-running the
simulation should not move the numbers a risk manager has just read."""

_CHUNK_PATHS: int = 50_000
"""Paths simulated at once. Bounds memory at a million paths to one chunk's draws
(50,000 × horizon × 2 floats) without changing the result: the draws come off one
random stream in the same order whatever the chunking."""


class MonteCarloDataError(ValueError):
    """The inputs can't support a simulation. The message is written for a risk
    manager, like the models' errors."""


@dataclass(frozen=True)
class SimulatedLevel:
    confidence: float
    var: float
    """Loss over the horizon exceeded on at most ``1 − c`` of paths, positive."""
    cvar: float
    """Mean loss over the horizon across those paths, never below VaR."""
    max_drawdown: float
    """The drawdown only those paths exceed: negative, like ``performance.max_drawdown``."""


@dataclass(frozen=True)
class SimulatedTailRisk:
    """One portfolio's simulated tail, at each confidence."""

    levels: tuple[SimulatedLevel, ...]
    mean_return: float
    """Mean simulated return over the horizon: the base case the paths centre on."""
    one_day: tuple[SimulatedLevel, ...]
    """VaR and CVaR of each path's first day, comparable with the historical
    one-day figures. Its ``max_drawdown`` is the first day's loss, if any."""


@dataclass(frozen=True)
class MonteCarloResult:
    risk: Mapping[str, SimulatedTailRisk]
    """Per portfolio, all simulated on the same paths so they can be compared."""
    forecasts: pd.Series
    """The base-case return over the horizon each asset's paths centre on."""
    method: str
    paths: int
    horizon: int
    as_of: pd.Timestamp
    history_start: pd.Timestamp
    """The first day of the history the moves are drawn from."""
    history_days: int
    seed: int
    label: str = MONTE_CARLO


def history_window(
    prices: Mapping[TargetRole, pd.Series], as_of: pd.Timestamp, *, window: int
) -> pd.DataFrame:
    """The ``window`` daily returns up to and including ``as_of``, on the days both
    indices traded: a column per asset, in ``prices``' order.

    Up to and including, not before: no forecast is fitted here, so the as-of day's
    own close is history like any other, and nothing after it is used.
    """
    calendar = None
    for series in prices.values():
        dates = series.dropna().index
        calendar = dates if calendar is None else calendar.intersection(dates)
    calendar = calendar.sort_values()
    calendar = calendar[calendar <= as_of]
    returns = pd.DataFrame(
        {role: prices[role].reindex(calendar).pct_change() for role in prices}
    ).iloc[1:]
    if len(returns) < window:
        raise MonteCarloDataError(
            f"A simulation draws its moves from the {window} trading days up to "
            f"{as_of:%d %b %Y}, but only {len(returns)} are available."
        )
    return returns.iloc[-window:]


def simulate(
    history: pd.DataFrame,
    forecasts: pd.Series,
    portfolios: Mapping[str, pd.Series],
    *,
    horizon: int,
    method: str = BOOTSTRAP,
    paths: int = DEFAULT_PATHS,
    confidences: Sequence[float] = VAR_CONFIDENCES,
    seed: int = SEED,
    chunk_paths: int = _CHUNK_PATHS,
) -> MonteCarloResult:
    """Simulate ``paths`` paths of ``horizon`` daily returns around ``forecasts``
    and read each portfolio's tail off them.

    ``history`` is daily returns with a column per asset (``history_window``),
    ``forecasts`` each asset's base-case return over ``horizon`` days, and each of
    ``portfolios`` a weight per asset, summing to 1.
    """
    if method not in SAMPLING_METHODS:
        raise ValueError(f"method must be one of {SAMPLING_METHODS}, got {method!r}")
    if not MIN_PATHS <= paths <= MAX_PATHS:
        raise MonteCarloDataError(
            f"The number of paths must be between {MIN_PATHS:,} and {MAX_PATHS:,}, "
            f"got {paths:,}."
        )
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1 day, got {horizon}")
    assets = list(history.columns)
    base = forecasts.reindex(assets)
    if base.isna().any():
        missing = ", ".join(str(a) for a in base.index[base.isna()])
        raise MonteCarloDataError(f"There is no base-case forecast for {missing}.")
    moves = history.dropna().to_numpy(dtype=float)
    if len(moves) < 2:
        raise MonteCarloDataError(
            "A simulation needs at least 2 days of history with both indices' returns."
        )
    weights = {}
    for name, w in portfolios.items():
        vector = w.reindex(assets).to_numpy(dtype=float)
        if np.isnan(vector).any() or not math.isclose(vector.sum(), 1.0, abs_tol=1e-9):
            raise ValueError(f"{name}'s weights must cover {assets} and sum to 1, got {w}")
        weights[name] = vector

    drift = base.to_numpy(dtype=float) / horizon
    draw = _sampler(moves, method)
    rng = np.random.default_rng(seed)
    outcomes = {name: np.empty(paths) for name in weights}
    first_days = {name: np.empty(paths) for name in weights}
    drawdowns = {name: np.empty(paths) for name in weights}
    for start in range(0, paths, chunk_paths):
        size = min(chunk_paths, paths - start)
        growth = np.cumprod(1 + drift + draw(rng, size, horizon), axis=1)
        for name, w in weights.items():
            value = growth @ w
            outcomes[name][start : start + size] = value[:, -1] - 1
            first_days[name][start : start + size] = value[:, 0] - 1
            drawdowns[name][start : start + size] = path_drawdowns(value)

    risk = {
        name: SimulatedTailRisk(
            levels=_levels(outcomes[name], drawdowns[name], confidences),
            mean_return=float(outcomes[name].mean()),
            one_day=_levels(
                first_days[name], np.minimum(first_days[name], 0.0), confidences
            ),
        )
        for name in weights
    }
    return MonteCarloResult(
        risk=risk,
        forecasts=base,
        method=method,
        paths=paths,
        horizon=horizon,
        as_of=history.index[-1],
        history_start=history.index[0],
        history_days=len(moves),
        seed=seed,
    )


def as_tail_risk(result: MonteCarloResult, portfolio: str) -> TailRisk:
    """``portfolio``'s one-day simulated VaR and CVaR in the shape the tail-risk
    table takes for every method, tagged ``MONTE_CARLO``.

    A simulation has no realised days to test a VaR against or to draw down
    over, so there are no breaches (``days_tested`` is 0, the breach rate
    undefined) and the drawdown is ``NaN``; the horizon's drawdown is in
    ``result.risk`` instead."""
    return TailRisk(
        levels=tuple(
            TailLevel(level.confidence, level.var, level.cvar, breaches=0, days_tested=0)
            for level in result.risk[portfolio].one_day
        ),
        max_drawdown=float("nan"),
        days=result.paths,
        window=0,
        method=MONTE_CARLO,
    )


def _levels(
    outcomes: np.ndarray, drawdowns: np.ndarray, confidences: Sequence[float]
) -> tuple[SimulatedLevel, ...]:
    levels = []
    for confidence in confidences:
        var, cvar = _var_cvar(outcomes, confidence)
        # The k-th worst (most negative) drawdown, as VaR is the k-th worst loss.
        drawdown, _ = _var_cvar(drawdowns, confidence)
        levels.append(SimulatedLevel(confidence, var, cvar, -drawdown))
    return tuple(levels)


def path_drawdowns(value: np.ndarray) -> np.ndarray:
    """Each path's maximum drawdown, ``value`` being its value at each day's close
    (a row per path) from a start of 1, which counts as the first peak."""
    peak = np.maximum(np.maximum.accumulate(value, axis=1), 1.0)
    return np.minimum((value / peak - 1).min(axis=1), 0.0)


def _sampler(moves: np.ndarray, method: str):
    """A function drawing ``(size, horizon, assets)`` demeaned daily moves."""
    centred = moves - moves.mean(axis=0)
    if method == BOOTSTRAP:

        def draw(rng: np.random.Generator, size: int, horizon: int) -> np.ndarray:
            return centred[rng.integers(0, len(centred), size=(size, horizon))]

        return draw

    # The covariance's square root by eigendecomposition rather than Cholesky, so a
    # degenerate history (an index that never moved) draws zeros instead of failing.
    covariance = np.atleast_2d(np.cov(centred, rowvar=False, ddof=1))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    root = eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))

    def draw(rng: np.random.Generator, size: int, horizon: int) -> np.ndarray:
        return rng.standard_normal((size, horizon, centred.shape[1])) @ root.T

    return draw
