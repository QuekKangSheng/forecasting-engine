"""Chain an optimised allocation and the 50/50 baseline through the same days, gross
and net of trading costs, and score both.

Takes the allocation as a weight schedule: a row of equity and bond weights for
each rebalance date, decided at that day's close. Whatever produces the weights
(the optimiser) is not this module's concern, so a schedule can come from a test
as easily as from a model.

Between rebalances a portfolio is left alone, so its weights drift with returns:
holding the target weights fixed every day would quietly be a daily rebalance,
earning a different return and trading every day for free. Turnover is measured
against those drifted weights.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.ingest.align import MAX_STALENESS, _as_of
from forecasting_engine.portfolio.performance import (
    RISK_FREE_RATE,
    PerformanceMetrics,
    RelativeMetrics,
    performance,
    relative,
)
from forecasting_engine.risk.tail import TailRisk, historical_tail_risk

ASSETS: tuple[TargetRole, ...] = (TargetRole.EQUITY, TargetRole.BOND)

BASELINE_WEIGHTS: Mapping[TargetRole, float] = {TargetRole.EQUITY: 0.5, TargetRole.BOND: 0.5}
"""The reference portfolio: half in each index, put back to half at every rebalance."""

DEFAULT_COSTS_BPS: Mapping[TargetRole, float] = {TargetRole.EQUITY: 3.0, TargetRole.BOND: 5.0}
"""Cost of trading each index, in basis points of the value traded."""

REBALANCE_FREQUENCY: str = "monthly"
"""The baseline is rebalanced on the last trading day of each month."""

_WEIGHT_TOLERANCE: float = 1e-9

PORTFOLIOS: tuple[str, ...] = ("optimised", "baseline")
BASES: tuple[str, ...] = ("gross", "net")


class BacktestDataError(ValueError):
    """The prices or the weight schedule can't support a backtest. The message is
    written for a portfolio manager, like the models' errors."""


@dataclass(frozen=True)
class PortfolioPath:
    """One portfolio's run through the backtest."""

    gross: pd.Series
    """Daily return before costs, from the day after the first allocation."""
    net: pd.Series
    """Daily return after costs, compounded in on the day each trade is made."""
    weights: pd.DataFrame
    """The target weights set at each rebalance."""
    turnover: pd.Series
    """Per rebalance, the sum of absolute weight changes from the drifted weights.
    The first allocation is bought from cash, so it is a full turnover of 1."""
    costs: pd.Series
    """Per rebalance, the fraction of the portfolio's value paid to trade."""


@dataclass(frozen=True)
class BacktestResult:
    """Everything a display or a risk or significance check needs from a backtest."""

    start: pd.Timestamp
    """The first allocation's date. Returns begin the next trading day."""
    end: pd.Timestamp
    """The last day with a return, the same for both portfolios."""
    optimised: PortfolioPath
    baseline: PortfolioPath
    metrics: Mapping[tuple[str, str], PerformanceMetrics]
    """Keyed by ``(portfolio, basis)``: ``("optimised", "net")``, and so on."""
    relative: Mapping[str, RelativeMetrics]
    """The optimised portfolio against the baseline, keyed by basis."""
    costs_bps: Mapping[TargetRole, float]
    rebalance_frequency: str = REBALANCE_FREQUENCY
    risk_free_rate: float = RISK_FREE_RATE
    active_models: tuple[str, ...] = ()
    """The forecasting models whose output the optimised weights came from."""
    tail_risk: Mapping[tuple[str, str], TailRisk] = field(default_factory=dict)
    """Historical VaR and CVaR of each path, keyed like ``metrics``; each carries
    the same maximum drawdown as ``metrics`` so the two are read together."""


def run_backtest(
    prices: Mapping[TargetRole, pd.Series],
    weights: pd.DataFrame,
    *,
    costs_bps: Mapping[TargetRole, float] = DEFAULT_COSTS_BPS,
    risk_free_rate: float = RISK_FREE_RATE,
    active_models: Sequence[str] = (),
) -> BacktestResult:
    """Backtest ``weights`` against the 50/50 baseline over the same days.

    ``prices`` are each index's total-return levels. ``weights`` has a row per
    rebalance date and a column per index. The backtest starts at the first
    rebalance and ends on the last day both indices have a price.
    """
    schedule = _checked_schedule(weights)
    returns = asset_returns(prices, schedule)
    start, end = returns.index[0], returns.index[-1]

    optimised = chain(returns, schedule, costs_bps)
    month_ends = month_end_rebalance_dates(returns.index)
    baseline_dates = pd.DatetimeIndex([start]).append(
        month_ends[(month_ends > start) & (month_ends < end)]
    )
    baseline_schedule = pd.DataFrame(
        [dict(BASELINE_WEIGHTS)] * len(baseline_dates), index=baseline_dates
    )
    baseline = chain(returns, baseline_schedule, costs_bps)

    paths = {"optimised": optimised, "baseline": baseline}
    return BacktestResult(
        start=start,
        end=end,
        optimised=optimised,
        baseline=baseline,
        metrics={
            (name, basis): performance(getattr(path, basis), risk_free_rate=risk_free_rate)
            for name, path in paths.items()
            for basis in BASES
        },
        relative={
            basis: relative(getattr(optimised, basis), getattr(baseline, basis)) for basis in BASES
        },
        costs_bps=dict(costs_bps),
        risk_free_rate=risk_free_rate,
        active_models=tuple(active_models),
        tail_risk={
            (name, basis): historical_tail_risk(getattr(path, basis))
            for name, path in paths.items()
            for basis in BASES
        },
    )


def asset_returns(prices: Mapping[TargetRole, pd.Series], schedule: pd.DataFrame) -> pd.DataFrame:
    """Each index's daily return on the joint calendar from the first rebalance to
    the last day both have a price. The first row is that rebalance, with no return.

    The two markets close on different days. On a day one is shut, its last price
    carries forward, so its return is 0 and the move lands the next day it trades;
    nothing is lost or invented. A price older than ``MAX_STALENESS`` rows is not a
    holiday but missing data, and is refused rather than treated as a flat market.
    """
    missing = [role.value for role in ASSETS if role not in prices]
    if missing:
        raise BacktestDataError(f"No price series for the {', '.join(missing)} index.")
    observed = {role: prices[role].dropna().sort_index() for role in ASSETS}
    for role, series in observed.items():
        if series.empty:
            raise BacktestDataError(f"The {role.value} index has no prices.")

    first = max(series.index[0] for series in observed.values())
    last = min(series.index[-1] for series in observed.values())
    start = schedule.index[0]
    if start < first:
        raise BacktestDataError(
            f"The first rebalance, {start:%d %b %Y}, is before both indices have prices "
            f"(from {first:%d %b %Y})."
        )
    calendar = observed[ASSETS[0]].index.union(observed[ASSETS[1]].index)
    off_calendar = schedule.index.difference(calendar)
    if len(off_calendar):
        raise BacktestDataError(
            f"Rebalance date {off_calendar[0]:%d %b %Y} is not a trading day for either index."
        )
    late = schedule.index[schedule.index > last]
    if len(late):
        raise BacktestDataError(
            f"Rebalance date {late[0]:%d %b %Y} is after the last day both indices have "
            f"prices ({last:%d %b %Y})."
        )
    calendar = calendar[(calendar >= start) & (calendar <= last)]
    if len(calendar) < 2:
        raise BacktestDataError("There is no trading day after the first rebalance to backtest.")

    levels = {}
    for role, series in observed.items():
        level, _ = _as_of(series, calendar)
        stale = level.index[level.isna()]
        if len(stale):
            raise BacktestDataError(
                f"The {role.value} index has no price for more than {MAX_STALENESS} trading "
                f"days around {stale[0]:%d %b %Y} — a data gap, not a holiday. Fill the "
                "export, or start the backtest after it."
            )
        levels[role] = level
    return pd.DataFrame(levels).pct_change().fillna(0.0)


def chain(
    returns: pd.DataFrame,
    schedule: pd.DataFrame,
    costs_bps: Mapping[TargetRole, float],
) -> PortfolioPath:
    """Run one weight schedule through ``returns``.

    ``returns``' first row is the first rebalance, which earns nothing; each later
    row is a day's return per index. A rebalance trades at that day's close, after
    the day's return: its cost is compounded into that day's net return, or, for
    the first allocation, into the first day's.
    """
    targets = schedule.reindex(columns=list(ASSETS)).astype(float)
    rates = np.array([costs_bps[role] for role in ASSETS], dtype=float) / 1e4
    held = np.zeros(len(ASSETS))
    gross, net, turnover, costs = [], [], {}, {}
    opening_cost = 0.0

    for i, (day, row) in enumerate(returns[list(ASSETS)].iterrows()):
        if i > 0:
            daily = row.to_numpy(dtype=float)
            portfolio = float(held @ daily)
            held = held * (1 + daily) / (1 + portfolio)
        cost = 0.0
        if day in targets.index:
            target = targets.loc[day].to_numpy()
            traded = np.abs(target - held)
            turnover[day] = float(traded.sum())
            cost = float(traded @ rates)
            costs[day] = cost
            held = target.copy()
        if i == 0:
            opening_cost = cost
            continue
        gross.append(portfolio)
        net.append((1 + portfolio) * (1 - cost) * (1 - opening_cost) - 1)
        opening_cost = 0.0

    days = returns.index[1:]
    return PortfolioPath(
        gross=pd.Series(gross, index=days, dtype=float),
        net=pd.Series(net, index=days, dtype=float),
        weights=targets,
        turnover=pd.Series(turnover, dtype=float),
        costs=pd.Series(costs, dtype=float),
    )


def month_end_rebalance_dates(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The last trading day of each month in ``calendar``."""
    days = pd.Series(calendar.sort_values(), index=calendar.sort_values())
    return pd.DatetimeIndex(days.groupby(days.index.to_period("M")).max().to_numpy())


def _checked_schedule(weights: pd.DataFrame) -> pd.DataFrame:
    if weights.empty:
        raise BacktestDataError("The weight schedule needs at least one rebalance.")
    missing = [role.value for role in ASSETS if role not in weights.columns]
    if missing:
        raise BacktestDataError(f"The weight schedule has no {', '.join(missing)} column.")
    schedule = weights.reindex(columns=list(ASSETS)).astype(float).sort_index()
    if schedule.index.has_duplicates:
        raise BacktestDataError("The weight schedule lists a rebalance date twice.")
    totals = schedule.sum(axis=1, skipna=False)
    bad = totals.index[~np.isclose(totals, 1.0, rtol=0, atol=_WEIGHT_TOLERANCE)]
    if len(bad):
        raise BacktestDataError(
            f"The weights on {bad[0]:%d %b %Y} must sum to 1 (got {totals[bad[0]]:g})."
        )
    return schedule
