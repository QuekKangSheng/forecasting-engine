"""Turn the active equity and bond models' signals into a weight schedule — the
input ``portfolio.backtest.run_backtest`` chains through history.

The allocation starts from the 50/50 benchmark and tilts away from it by what the
signals say, in a benchmark-relative mean-variance problem: maximise the
expected signal return of the tilt less ``λ/2`` times its variance. Risk here is
how far the portfolio strays from the benchmark, so risk aversion sets how much
a signal is acted on, and with no signal the portfolio is the benchmark.

A model's signal is its forecast less its fold's training-mean return
(``validation.harness.signal_forecasts``). The training means are left out on
purpose: equity's trailing mean is far above bond's in most windows and swings
with recent performance, so fed to the optimiser it pins the allocation to a
bound whatever the signals say.

A rebalance happens once every ``horizon`` trading days over the out-of-sample
period, so each allocation is held for exactly as long as the forecast it was
set from looks ahead.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.ingest.align import FeaturePanel
from forecasting_engine.portfolio.backtest import ASSETS, BASELINE_WEIGHTS
from forecasting_engine.validation.splitters import TUNING_ROWS, PurgedWalkForward


class OptimizeDataError(ValueError):
    """The message is written for a portfolio manager, like the other models' errors."""


RISK_AVERSION_SCALE: Mapping[int, float] = {1: 1.0, 2: 3.0, 3: 10.0, 4: 30.0, 5: 100.0}
"""The sponsor's 1-5 risk-aversion scale (1 most risk-loving, 5 most risk-averse)
mapped to the optimiser's λ. Working calibration, not sponsor-confirmed: each step
multiplies λ by about 3. On the Sep 2026 Bloomberg data, at a 10-day horizon and a
252-day train window, level 1 sits at a weight bound on most rebalances where a
signal is present, and level 5 keeps nine rebalances in ten within about 12
percentage points of 50/50."""

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
    """Equity's benchmark weight: where the allocation sits when the two
    signals agree."""
    forecast_tilt: float
    """What the difference between the signals adds to equity at this risk
    aversion, before any bound."""
    equity: float
    """Equity's final weight, after the bounds."""

    @property
    def unconstrained(self) -> float:
        return self.anchor + self.forecast_tilt

    @property
    def capped(self) -> bool:
        return self.equity != self.unconstrained


def risk_aversion_for(level: int) -> float:
    """λ for a level on the sponsor's 1-5 scale (``RISK_AVERSION_SCALE``)."""
    if level not in RISK_AVERSION_SCALE:
        raise ValueError(f"risk aversion level must be 1 to 5, got {level}")
    return RISK_AVERSION_SCALE[level]


def weight_breakdown(
    expected_signal: pd.Series,
    covariance: pd.DataFrame,
    *,
    risk_aversion: float = RISK_AVERSION_SCALE[DEFAULT_RISK_LEVEL],
    bounds: tuple[float, float] = DEFAULT_WEIGHT_BOUNDS,
) -> WeightBreakdown:
    """``solve_weights``'s equity weight, split into the benchmark weight and the
    signals' tilt from it: ``w = w_bench + (s_e − s_b)/(λ·D)``, with
    ``D = σ_e² − 2σ_eb + σ_b²``, then clipped to ``bounds``."""
    equity, bond = ASSETS
    anchor = float(BASELINE_WEIGHTS[equity])
    signal_diff = expected_signal[equity] - expected_signal[bond]
    var_diff = (
        covariance.loc[equity, equity]
        - 2 * covariance.loc[equity, bond]
        + covariance.loc[bond, bond]
    )
    tilt = float(signal_diff / (risk_aversion * var_diff)) if var_diff > 0 else 0.0
    w_equity = min(max(anchor + tilt, bounds[0]), bounds[1])
    return WeightBreakdown(anchor=anchor, forecast_tilt=tilt, equity=w_equity)


def solve_weights(
    expected_signal: pd.Series,
    covariance: pd.DataFrame,
    *,
    risk_aversion: float = RISK_AVERSION_SCALE[DEFAULT_RISK_LEVEL],
    bounds: tuple[float, float] = DEFAULT_WEIGHT_BOUNDS,
) -> pd.Series:
    """The long-only two-asset weights maximising the benchmark-relative
    mean-variance objective ``a @ expected_signal - risk_aversion / 2 * a @
    covariance @ a``, where ``a`` is the tilt from ``BASELINE_WEIGHTS``, subject
    to the two weights summing to 1 and each lying within ``bounds``.

    Two assets collapse that to one free variable — equity's tilt, bond's being
    its negative — with a closed form, then clipped to ``bounds``. When the two
    assets' returns are so alike that tilting between them changes no risk at
    all (equal variance, perfect correlation), the covariance can't size the
    tilt, and the benchmark is kept rather than dividing by zero.
    """
    equity, bond = ASSETS
    w_equity = weight_breakdown(
        expected_signal, covariance, risk_aversion=risk_aversion, bounds=bounds
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
    risk_aversion: float = RISK_AVERSION_SCALE[DEFAULT_RISK_LEVEL],
    bounds: tuple[float, float] = DEFAULT_WEIGHT_BOUNDS,
) -> pd.DataFrame:
    """The full weight schedule ``portfolio.backtest.run_backtest`` chains
    through history: one row of equity/bond weights per rebalance date, solved
    from each active model's signal and the equity/bond covariance over that
    rebalance's own training window.

    A rebalance whose signal is missing (a model had no prediction that day) or
    whose training window can't support a covariance estimate is left out rather
    than guessed at.
    """
    rebalance_dates = common_rebalance_dates(
        prices, horizon=horizon, train_window=train_window, test_window=test_window, embargo=embargo
    )
    returns = expected_returns(rebalance_dates, signals)
    rows: dict[pd.Timestamp, pd.Series] = {}
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
        rows[date] = solve_weights(row, covariance, risk_aversion=risk_aversion, bounds=bounds)
    return pd.DataFrame(rows).T.sort_index()
