"""Historical Value at Risk and Conditional VaR, read straight off realised returns.

No volatility model and no simulation: the figures are the strategy's own worst
days. They are tagged ``HISTORICAL`` so a display can tell them apart from any
simulated (Monte Carlo) figures, which answer a different question.

Definitions, chosen so every figure is a day that actually happened rather than
an interpolation between two:

- with ``n`` daily returns and confidence ``c``, the tail is the ``k = ⌈n·(1−c)⌉``
  worst days;
- **VaR** is the loss on the ``k``-th worst day: the loss exceeded on at most
  ``1 − c`` of days;
- **CVaR** is the mean loss over those ``k`` days, so it is never below VaR.

Both are one-day figures from daily returns, reported as positive losses (0.02 is
a 2% loss); a tail of gains gives a negative loss rather than being hidden as 0.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecasting_engine.portfolio.performance import max_drawdown

HISTORICAL: str = "Historical"

VAR_CONFIDENCES: tuple[float, ...] = (0.95, 0.99)

VAR_WINDOW: int = 252
"""Trading days of history behind each day's VaR in the breach count: a year,
the minimum the Basel backtesting standard asks for."""


@dataclass(frozen=True)
class TailLevel:
    confidence: float
    var: float
    """One-day loss, over the whole period."""
    cvar: float
    """Mean one-day loss beyond VaR, over the whole period."""
    breaches: int
    """Days whose loss exceeded a VaR built from the ``VAR_WINDOW`` days before."""
    days_tested: int

    @property
    def expected_breach_rate(self) -> float:
        return 1 - self.confidence

    @property
    def breach_rate(self) -> float:
        return self.breaches / self.days_tested if self.days_tested else float("nan")


@dataclass(frozen=True)
class TailRisk:
    """A return series' tail measures, beside its maximum drawdown."""

    levels: tuple[TailLevel, ...]
    max_drawdown: float
    days: int
    window: int
    method: str = HISTORICAL


def historical_tail_risk(
    returns: pd.Series,
    *,
    confidences: Sequence[float] = VAR_CONFIDENCES,
    window: int = VAR_WINDOW,
) -> TailRisk:
    levels = []
    for confidence in confidences:
        var, cvar = historical_var(returns, confidence)
        count, tested = breaches(returns, confidence, window=window)
        levels.append(TailLevel(confidence, var, cvar, count, tested))
    return TailRisk(
        levels=tuple(levels),
        max_drawdown=max_drawdown(returns),
        days=int(returns.notna().sum()),
        window=window,
    )


def historical_var(returns: pd.Series, confidence: float) -> tuple[float, float]:
    """``(VaR, CVaR)`` of ``returns`` at ``confidence``, as positive one-day losses."""
    return _var_cvar(returns.dropna().to_numpy(dtype=float), confidence)


def breaches(returns: pd.Series, confidence: float, *, window: int) -> tuple[int, int]:
    """``(breaches, days tested)``: how often a day's loss exceeded the VaR of the
    ``window`` days before it.

    Each day is tested against a VaR it played no part in. Counting breaches of
    the whole period's own VaR instead would be circular: that VaR is defined as
    the loss exceeded on ``1 − c`` of those same days, so the rate would come out
    at about ``1 − c`` whatever the strategy did. Only a strict excess counts.
    """
    r = returns.dropna().to_numpy(dtype=float)
    count = 0
    for day in range(window, len(r)):
        var, _ = _var_cvar(r[day - window : day], confidence)
        count += int(-r[day] > var)
    return count, max(len(r) - window, 0)


def _var_cvar(r: np.ndarray, confidence: float) -> tuple[float, float]:
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be between 0 and 1, got {confidence}")
    if len(r) == 0:
        return float("nan"), float("nan")
    k = math.ceil(len(r) * (1 - confidence) - 1e-9)  # guard 100 × 0.05 = 5.000000000000001
    worst = np.sort(r)[: max(k, 1)]
    return float(-worst[-1]), float(-worst.mean())
