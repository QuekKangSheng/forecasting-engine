"""Risk-adjusted performance of a daily return series, and of one series against another.

Every figure is annualised from daily returns. The conventions below are where a
backtest's numbers most often go quietly wrong, so each is a deliberate choice:

- the risk-free rate is an annual rate, compounded down to a day before it is
  subtracted, not divided by the number of days;
- the annual return is compounded (geometric), not the daily mean times a year;
- downside deviation is the root mean squared shortfall below the risk-free rate
  over *every* day, gains counting as zero, not over the losing days alone;
- drawdown is measured from the starting value, so a loss on the first day counts.

Whether one portfolio's outperformance of another is more than luck (FYP-52) is a
paired test on their daily difference, with a Newey-West standard error: a day's
difference is often related to the days around it, and treating each day as an
independent draw would overstate how sure the test can be.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd
import statsmodels.api as sm

from forecasting_engine.validation.gates import SIGNIFICANCE_T_GATE

TRADING_DAYS_PER_YEAR: int = 252

RISK_FREE_RATE: float = 0.0
"""Annual. A working default: the app holds no cash-rate series to take it from."""

SIGNIFICANCE_CONFIDENCE: float = 0.95
"""The confidence a significant verdict states. The cut-off itself is the sponsor's
t-statistic bar, ``validation.gates.SIGNIFICANCE_T_GATE`` (2), the one every model
is gated on: beyond ±2 is about 95.4% two-sided, which the verdict rounds to 95%."""


class Significance(StrEnum):
    """The spread t-statistic's verdict, read against ``SIGNIFICANCE_T_GATE``."""

    OUTPERFORMS = "outperforms"
    """Significantly better than the baseline."""
    UNDERPERFORMS = "underperforms"
    """Significantly worse than the baseline."""
    NOT_SIGNIFICANT = "not significant"
    """The difference could be noise."""
    UNDEFINED = "undefined"
    """Too few days, or a difference that never varies, so no test is possible."""


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
    spread_t: float = float("nan")
    """Paired t-statistic of the mean daily difference, on a Newey-West standard
    error. NaN when it can't be computed."""
    spread_t_lags: int = 0
    """Days of neighbouring correlation the Newey-West standard error allows for."""
    days: int = 0
    """Daily differences the test was run on."""

    @property
    def significance(self) -> Significance:
        return significance(self.spread_t)


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
    lags = newey_west_lags(len(active))
    return RelativeMetrics(
        tracking_error=tracking_error,
        information_ratio=_ratio(float(np.mean(active)) * periods_per_year, tracking_error),
        spread_t=spread_t_stat(active, lags=lags),
        spread_t_lags=lags,
        days=len(active),
    )


def newey_west_lags(days: int) -> int:
    """Newey and West's (1994) rule of thumb, ``⌊4 · (days / 100)^(2/9)⌋``: more lags
    for a longer series, growing slowly. 1,000 days gives 6."""
    if days < 1:
        return 0
    return math.floor(4 * (days / 100) ** (2 / 9))


def spread_t_stat(active: np.ndarray, *, lags: int) -> float:
    """t-statistic of the mean of ``active``, the daily difference between two
    portfolios, against zero: the mean over its Newey-West standard error with
    ``lags`` lags. Regressing the difference on a constant alone gives exactly
    that, so the test is statsmodels' HAC estimate of the constant.

    NaN with fewer than three days, or when the difference never varies (two
    identical portfolios), where a standard error of zero would make t meaningless.
    """
    d = np.asarray(active, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 3 or np.ptp(d) == 0:
        return float("nan")
    fit = sm.OLS(d, np.ones((len(d), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(fit.tvalues[0])


def significance(t: float, threshold: float = SIGNIFICANCE_T_GATE) -> Significance:
    """Whether ``t`` is strictly beyond ``±threshold``, and which way — the same
    strict "t > 2" the model gate applies."""
    if t != t:  # NaN
        return Significance.UNDEFINED
    if t > threshold:
        return Significance.OUTPERFORMS
    if t < -threshold:
        return Significance.UNDERPERFORMS
    return Significance.NOT_SIGNIFICANT


def _ratio(numerator: float, denominator: float) -> float:
    if not denominator > 0:  # zero or NaN
        return float("nan")
    return numerator / denominator
