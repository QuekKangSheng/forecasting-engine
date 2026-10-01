"""Risk-adjusted performance of a daily return series, and of one series against another.

Every figure is annualised from daily returns. The conventions below are where a
backtest's numbers most often go quietly wrong, so each is a deliberate choice:

- the risk-free rate is an annual rate, compounded down to a day before it is
  subtracted, not divided by the number of days;
- the annual return is compounded (geometric), not the daily mean times a year;
- downside deviation is the root mean squared shortfall below the risk-free rate
  over *every* day, gains counting as zero, not over the losing days alone;
- drawdown is measured from the starting value, so a loss on the first day counts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR: int = 252

RISK_FREE_RATE: float = 0.0
"""Annual. A working default: the app holds no cash-rate series to take it from."""


@dataclass(frozen=True)
class PerformanceMetrics:
    annual_return: float
    """Compounded annual growth rate."""
    annual_volatility: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    """The worst fall from a running peak, as a negative fraction (-0.2 is -20%)."""


@dataclass(frozen=True)
class RelativeMetrics:
    """A portfolio judged against a baseline over the same days."""

    tracking_error: float
    """Annualised standard deviation of the daily difference in returns."""
    information_ratio: float
    """Annualised mean daily difference over the tracking error."""


def performance(
    returns: pd.Series,
    *,
    risk_free_rate: float = RISK_FREE_RATE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> PerformanceMetrics:
    """Every metric for one daily return series. A ratio whose denominator is zero
    (no variation, no losing day, no drawdown) is NaN rather than infinite."""
    r = returns.dropna().to_numpy(dtype=float)
    daily_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess = r - daily_rf
    root_year = math.sqrt(periods_per_year)

    # A series that never varies has a standard deviation of rounding error, not
    # zero, which would make a meaningless Sharpe of 1e15; it has no Sharpe.
    varies = len(r) > 1 and np.ptp(r) > 0
    sd = float(np.std(excess, ddof=1)) if varies else float("nan")
    downside = math.sqrt(float(np.mean(np.minimum(excess, 0.0) ** 2)))
    growth = float(np.prod(1 + r))
    annual_return = growth ** (periods_per_year / len(r)) - 1 if len(r) else float("nan")
    drawdown = max_drawdown(returns)

    return PerformanceMetrics(
        annual_return=annual_return,
        annual_volatility=float(np.std(r, ddof=1)) * root_year if varies else 0.0,
        sharpe=_ratio(float(np.mean(excess)) * root_year, sd),
        sortino=_ratio(float(np.mean(excess)) * root_year, downside),
        calmar=_ratio(annual_return, abs(drawdown)),
        max_drawdown=drawdown,
    )


def max_drawdown(returns: pd.Series) -> float:
    """The worst fall from a running peak of the compounded path, the starting
    value of 1 included as the first peak."""
    path = np.concatenate([[1.0], np.cumprod(1 + returns.dropna().to_numpy(dtype=float))])
    return float(np.min(path / np.maximum.accumulate(path) - 1))


def relative(
    portfolio: pd.Series,
    baseline: pd.Series,
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> RelativeMetrics:
    """Tracking error and information ratio of ``portfolio`` against ``baseline``.

    Both must cover the same days: a difference taken over mismatched dates would
    compare one portfolio's good weeks with the other's bad ones."""
    if not portfolio.index.equals(baseline.index):
        raise ValueError("a portfolio and its baseline must cover the same days")
    active = (portfolio - baseline).to_numpy(dtype=float)
    tracking_error = float(np.std(active, ddof=1)) * math.sqrt(periods_per_year)
    return RelativeMetrics(
        tracking_error=tracking_error,
        information_ratio=_ratio(float(np.mean(active)) * periods_per_year, tracking_error),
    )


def _ratio(numerator: float, denominator: float) -> float:
    if not denominator > 0:  # zero or NaN
        return float("nan")
    return numerator / denominator
