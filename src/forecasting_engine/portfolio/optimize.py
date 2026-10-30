"""Turn the active equity and bond models' signals into a weight schedule — the
input ``portfolio.backtest.run_backtest`` chains through history.

The allocation starts from the risk level's own starting weight and
tilts away from it by what the signals say: maximise the expected signal
return of the tilt less ``k/2`` times its variance, where ``k``
(``TILT_STRENGTH``) is one fixed tilt strength shared by every level — the
level sets where the allocation starts, not how hard a signal pulls. With no
signal the portfolio sits at the starting weight. The 50/50 benchmark used for
comparison (``backtest.BASELINE_WEIGHTS``) is separate and never moves.

A model's signal is its forecast less its fold's training-mean return
(``validation.harness.signal_forecasts``). The training means are left out on
purpose: equity's trailing mean is far above bond's in most windows and swings
with recent performance, so fed to the optimiser it pins the allocation to a
bound whatever the signals say.

A rebalance happens once every ``horizon`` trading days over the out-of-sample
period, so each allocation is held for exactly as long as the forecast it was
set from looks ahead.

Between rebalances the portfolio drifts with the two indices' returns, and a
rebalance trades only when the new target is worth its cost: within a
no-trade band of ``c / (k·D)`` around the drifted weight, where ``c`` is the cost
of moving the equity weight by 1 (equity's plus bond's cost) and ``D`` the variance
of the difference between the two indices' returns, the portfolio holds; outside
it, it trades only as far as the band's near edge. A portfolio that has drifted
outside the weight bounds is traded back to the nearest bound, whatever the band.
The first rebalance buys the target from cash.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.ingest.align import FeaturePanel
from forecasting_engine.portfolio.backtest import (
    ASSETS,
    DEFAULT_COSTS_BPS,
    asset_returns,
)
from forecasting_engine.validation.splitters import TUNING_ROWS, PurgedWalkForward


class OptimizeDataError(ValueError):
    """The message is written for a portfolio manager, like the other models' errors."""


RISK_LEVELS: tuple[int, ...] = (1, 2, 3, 4, 5)
"""The sponsor's 1-5 risk scale. ``starting_weight_for`` maps a level to its
starting equity weight, scaled across whatever bounds are configured —
reproduces the sponsor's `{1: 80%, 2: 65%, 3: 50%, 4: 35%, 5: 20%}` at the
default bounds. Replaces ``RISK_AVERSION_SCALE``, which varied tilt strength
instead of the starting point."""

TILT_STRENGTH: float = 40.0
"""``k``: how strongly a signal tilts the allocation, fixed across every level.
Raised from 10 after testing showed level 3 moving too far from 50/50. Not
yet PO-confirmed."""

DEFAULT_RISK_LEVEL: int = 3
"""Working default, not sponsor-confirmed: the middle of the scale."""

DEFAULT_WEIGHT_BOUNDS: tuple[float, float] = (0.2, 0.8)
"""Working default, not sponsor-confirmed: stops either asset being pushed to a
corner by a small, noisy difference between the two forecasts. Revisit once
Alpha Norm states how far the allocation may tilt from 50/50."""


def _common_price_calendar(prices: Mapping[TargetRole, pd.Series]) -> pd.DatetimeIndex:
    calendars = [prices[role].dropna().index for role in ASSETS]
    common = calendars[0]
    for other in calendars[1:]:
        common = common.intersection(other)
    return common.sort_values()


def common_rebalance_dates(
    prices: Mapping[TargetRole, pd.Series],
    *,
    horizon: int,
    train_window: int,
    test_window: int,
    embargo: int,
) -> pd.DatetimeIndex:
    """Every ``horizon``-th out-of-sample date on the calendar both targets share,
    from the first walk-forward test window's start.

    An allocation is held until the next rebalance, so holding it for the
    forecast's own horizon is what the forecast speaks to: a 5-day forecast held
    20 days says nothing about the last 15 of them.

    Equity's and bond's own active models were each walk-forward split over
    their *own* calendar, so splitting one shared calendar keeps both targets'
    out-of-sample period, and so the rebalance dates, in step.
    """
    calendar = _common_price_calendar(prices)
    panel = FeaturePanel(
        frame=pd.DataFrame(index=calendar), signals=(), targets=(), horizon=horizon
    )
    splitter = PurgedWalkForward(
        train=train_window, test=test_window, embargo=embargo, tuning_rows=TUNING_ROWS
    )
    tested = [test for _, test in splitter.split(panel) if len(test)]
    if not tested:
        return pd.DatetimeIndex([])
    return tested[0].append(tested[1:])[::horizon]


def expected_returns(
    rebalance_dates: pd.DatetimeIndex, forecasts: Mapping[TargetRole, pd.Series]
) -> pd.DataFrame:
    """Each target's own forecast as of each rebalance date — the mean-variance
    solver's expected-return input, one row per rebalance, a column per asset.

    Reads the most recent forecast at or before the rebalance date rather than
    requiring an exact match: equity's and bond's own saved forecasts can fall
    on slightly different calendars, and a model's most recent view as of that
    date is what it would actually be using if asked that day — never a date
    after it, so this stays as leakage-free as an exact match would be.
    """
    return pd.DataFrame(
        {
            role: forecasts[role].sort_index().reindex(rebalance_dates, method="ffill")
            for role in ASSETS
        }
    )


def training_window(
    rebalance_date: pd.Timestamp,
    calendar: pd.DatetimeIndex,
    *,
    horizon: int,
    train_window: int,
    embargo: int,
) -> pd.DatetimeIndex:
    """The ``train_window`` dates on ``calendar`` ending ``embargo`` days before
    ``rebalance_date`` — the same purge gap a forecasting model itself trains
    under, so covariance never reaches into days the model couldn't see either.
    """
    if rebalance_date not in calendar:
        raise OptimizeDataError(
            f"{rebalance_date:%d %b %Y} is not a trading day on the shared equity/bond calendar."
        )
    panel = FeaturePanel(
        frame=pd.DataFrame(index=calendar), signals=(), targets=(), horizon=horizon
    )
    splitter = PurgedWalkForward(train=train_window, test=0, embargo=embargo)
    position = calendar.get_loc(rebalance_date)
    return splitter.window_before(panel, position, train_window)


def covariance_at_rebalance(
    rebalance_date: pd.Timestamp,
    prices: Mapping[TargetRole, pd.Series],
    *,
    horizon: int,
    train_window: int,
    embargo: int,
) -> pd.DataFrame:
    """Equity/bond daily-return covariance over the training window ending before
    ``rebalance_date``, scaled from daily to ``horizon``-day terms."""
    calendar = _common_price_calendar(prices)
    train_dates = training_window(
        rebalance_date, calendar, horizon=horizon, train_window=train_window, embargo=embargo
    )
    returns = pd.DataFrame(
        {role: prices[role].reindex(train_dates).pct_change() for role in ASSETS}
    ).dropna()
    if len(returns) < 2:
        raise OptimizeDataError(
            f"Not enough trading days before {rebalance_date:%d %b %Y} to estimate a "
            f"covariance (need at least 2 daily returns, got {len(returns)})."
        )
    return returns.cov() * horizon


@dataclass(frozen=True)
class WeightBreakdown:
    """How one rebalance's equity weight was reached, for a reader to follow."""

    anchor: float
    """Equity's starting weight for this risk level: where the allocation sits
    with no signal, before any tilt."""
    forecast_tilt: float
    """What the difference between the signals adds to equity at the fixed tilt
    strength, before any bound."""
    equity: float
    """Equity's final weight, after the bounds and any no-trade band."""
    target: float | None = None
    """Equity's weight after the bounds, before any no-trade band. ``None`` where
    no band applied, so it is ``equity``."""
    band: float = 0.0
    """The no-trade band's half-width around ``drifted``: 0 where none applied."""
    drifted: float | None = None
    """Equity's weight as the last allocation drifted to by this rebalance, or
    ``None`` at the first, which buys from cash."""
    held: bool = False
    """Whether this rebalance traded nothing: the target was within the band."""

    @property
    def unconstrained(self) -> float:
        return self.anchor + self.forecast_tilt

    @property
    def bounded(self) -> float:
        return self.equity if self.target is None else self.target

    @property
    def capped(self) -> bool:
        return self.bounded != self.unconstrained


def starting_weight_for(
    level: int, bounds: tuple[float, float] = DEFAULT_WEIGHT_BOUNDS
) -> float:
    """Equity's starting weight for a level, spaced evenly across ``bounds``:
    1 at the upper bound, 5 at the lower, 3 always at the midpoint."""
    if level not in RISK_LEVELS:
        raise ValueError(f"risk level must be 1 to 5, got {level}")
    lower, upper = bounds
    return lower + (5 - level) / 4 * (upper - lower)


def bounds_for_level(
    level: int, bounds: tuple[float, float] = DEFAULT_WEIGHT_BOUNDS
) -> tuple[float, float]:
    """``bounds``, narrowed so the signal tilt can never cross 50% the wrong
    way: levels 1-2 always end equity-heavy, levels 4-5 bond-heavy. Level 3
    keeps ``bounds`` as given."""
    if level not in RISK_LEVELS:
        raise ValueError(f"risk level must be 1 to 5, got {level}")
    lower, upper = bounds
    if level in (1, 2):
        lower = max(lower, 0.5)
        upper = max(upper, lower)
    elif level in (4, 5):
        upper = min(upper, 0.5)
        lower = min(lower, upper)
    return lower, upper


def weight_breakdown(
    expected_signal: pd.Series,
    covariance: pd.DataFrame,
    *,
    starting_weight: float = 0.5,
    tilt_strength: float = TILT_STRENGTH,
    bounds: tuple[float, float] = DEFAULT_WEIGHT_BOUNDS,
) -> WeightBreakdown:
    """``solve_weights``'s equity weight, split into the starting weight and the
    signals' tilt from it: ``w = start + (s_e − s_b)/(k·D)``, with
    ``D = σ_e² − 2σ_eb + σ_b²``, then clipped to ``bounds``."""
    equity, bond = ASSETS
    anchor = float(starting_weight)
    signal_diff = expected_signal[equity] - expected_signal[bond]
    var_diff = _var_diff(covariance)
    tilt = float(signal_diff / (tilt_strength * var_diff)) if var_diff > 0 else 0.0
    w_equity = min(max(anchor + tilt, bounds[0]), bounds[1])
    return WeightBreakdown(anchor=anchor, forecast_tilt=tilt, equity=w_equity)


def solve_weights(
    expected_signal: pd.Series,
    covariance: pd.DataFrame,
    *,
    starting_weight: float = 0.5,
    tilt_strength: float = TILT_STRENGTH,
    bounds: tuple[float, float] = DEFAULT_WEIGHT_BOUNDS,
) -> pd.Series:
    """The long-only two-asset weights maximising the starting-weight-relative
    mean-variance objective ``a @ expected_signal - tilt_strength / 2 * a @
    covariance @ a``, where ``a`` is the tilt from ``starting_weight``, subject
    to the two weights summing to 1 and each lying within ``bounds``.

    Two assets collapse that to one free variable — equity's tilt, bond's being
    its negative — with a closed form, then clipped to ``bounds``. When the two
    assets' returns are so alike that tilting between them changes no risk at
    all (equal variance, perfect correlation), the covariance can't size the
    tilt, and the starting weight is kept rather than dividing by zero.
    """
    equity, bond = ASSETS
    w_equity = weight_breakdown(
        expected_signal,
        covariance,
        starting_weight=starting_weight,
        tilt_strength=tilt_strength,
        bounds=bounds,
    ).equity
    return pd.Series({equity: w_equity, bond: 1 - w_equity})


def weight_schedule(
    signals: Mapping[TargetRole, pd.Series],
    prices: Mapping[TargetRole, pd.Series],
    *,
    horizon: int,
    train_window: int,
    test_window: int,
    embargo: int,
    starting_weight: float = 0.5,
    tilt_strength: float = TILT_STRENGTH,
    bounds: tuple[float, float] = DEFAULT_WEIGHT_BOUNDS,
    costs_bps: Mapping[TargetRole, float] = DEFAULT_COSTS_BPS,
) -> pd.DataFrame:
    """The full weight schedule ``portfolio.backtest.run_backtest`` chains
    through history: one row of equity/bond weights per rebalance date, from
    ``schedule_breakdowns``."""
    return schedule_weights(
        schedule_breakdowns(
            signals,
            prices,
            horizon=horizon,
            train_window=train_window,
            test_window=test_window,
            embargo=embargo,
            starting_weight=starting_weight,
            tilt_strength=tilt_strength,
            bounds=bounds,
            costs_bps=costs_bps,
        )
    )


def schedule_weights(breakdowns: Mapping[pd.Timestamp, WeightBreakdown]) -> pd.DataFrame:
    """One row of equity/bond weights per rebalance in ``breakdowns``. A held
    rebalance's row is the drifted weights, so the backtest trades nothing there."""
    equity, bond = ASSETS
    weights = np.array([b.equity for b in breakdowns.values()], dtype=float)
    return pd.DataFrame(
        {equity: weights, bond: 1 - weights}, index=pd.DatetimeIndex(list(breakdowns))
    )


def schedule_breakdowns(
    signals: Mapping[TargetRole, pd.Series],
    prices: Mapping[TargetRole, pd.Series],
    *,
    horizon: int,
    train_window: int,
    test_window: int,
    embargo: int,
    starting_weight: float = 0.5,
    tilt_strength: float = TILT_STRENGTH,
    bounds: tuple[float, float] = DEFAULT_WEIGHT_BOUNDS,
    costs_bps: Mapping[TargetRole, float] = DEFAULT_COSTS_BPS,
) -> dict[pd.Timestamp, WeightBreakdown]:
    """How each rebalance's equity weight was reached, in date order: the target
    solved from that date's signals and the covariance over its own training
    window, then the no-trade band against the weight the last allocation drifted
    to. Zero costs leave no band, so every rebalance moves to its target.

    A rebalance whose signal is missing (a model had no prediction that day) or
    whose training window can't support a covariance estimate is left out rather
    than guessed at; the portfolio drifts on through it.
    """
    rebalance_dates = common_rebalance_dates(
        prices, horizon=horizon, train_window=train_window, test_window=test_window, embargo=embargo
    )
    returns = expected_returns(rebalance_dates, signals)
    targets: dict[pd.Timestamp, tuple[WeightBreakdown, float]] = {}
    for date in rebalance_dates:
        row = returns.loc[date]
        if row.isna().any():
            continue
        try:
            covariance = covariance_at_rebalance(
                date, prices, horizon=horizon, train_window=train_window, embargo=embargo
            )
        except OptimizeDataError:
            continue
        breakdown = weight_breakdown(
            row,
            covariance,
            starting_weight=starting_weight,
            tilt_strength=tilt_strength,
            bounds=bounds,
        )
        targets[date] = (breakdown, _var_diff(covariance))
    if not targets:
        return {}

    cost = sum(costs_bps[role] for role in ASSETS) / 1e4
    daily_returns = asset_returns(prices, pd.DataFrame(index=pd.DatetimeIndex(list(targets))))
    held: np.ndarray | None = None
    breakdowns: dict[pd.Timestamp, WeightBreakdown] = {}
    for i, (day, row) in enumerate(daily_returns[list(ASSETS)].iterrows()):
        if i > 0 and held is not None:
            # Drifted as backtest.chain drifts it, so a held rebalance trades nothing.
            daily = row.to_numpy(dtype=float)
            held = held * (1 + daily) / (1 + float(held @ daily))
        if day not in targets:
            continue
        breakdown, var_diff = targets[day]
        if held is not None and cost > 0:
            band = cost / (tilt_strength * var_diff) if var_diff > 0 else math.inf
            breakdown = _banded(breakdown, float(held[0]), band, bounds)
        breakdowns[day] = breakdown
        held = np.array([breakdown.equity, 1 - breakdown.equity])
    return breakdowns


def _var_diff(covariance: pd.DataFrame) -> float:
    """``D = σ_e² − 2σ_eb + σ_b²``: the variance of the difference between the
    two indices' returns, which a tilt from the benchmark is sized against."""
    equity, bond = ASSETS
    return float(
        covariance.loc[equity, equity]
        - 2 * covariance.loc[equity, bond]
        + covariance.loc[bond, bond]
    )


def _banded(
    target: WeightBreakdown, drifted: float, band: float, bounds: tuple[float, float]
) -> WeightBreakdown:
    """Trade to the nearest bound if the portfolio has drifted outside ``bounds``.
    Otherwise hold at ``drifted`` if the target is within ``band`` of it (an
    infinite band, where D ≤ 0, always holds), or move to the band's near edge."""
    lower, upper = bounds
    if not lower <= drifted <= upper:
        held, equity = False, min(max(drifted, lower), upper)
    else:
        gap = target.equity - drifted
        held = abs(gap) <= band
        edge = target.equity - math.copysign(band, gap)
        equity = drifted if held else min(max(edge, lower), upper)
    return WeightBreakdown(
        anchor=target.anchor,
        forecast_tilt=target.forecast_tilt,
        equity=equity,
        target=target.equity,
        band=band,
        drifted=drifted,
        held=held,
    )
