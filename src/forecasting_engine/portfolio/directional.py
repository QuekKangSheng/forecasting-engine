"""FYP-162: what holding one index only on days its forecast is positive would have
earned, against simply holding it throughout.

A tangible check of whether a forecast's direction is useful. The strategy is fully
invested on a step whose forecast is above zero and fully in cash otherwise; its
return comes from the index's realised return alone, so the forecast's size never
enters. Cash earns nothing and no trading cost is charged: every figure is gross.

Forecasts and realised returns are the run's out-of-sample series, indexed by the
date the forecast was made, with each realised value the ``horizon``-day forward
return from that date. At ``horizon`` above 1 neighbouring rows' returns overlap,
so the window is stepped through ``horizon`` rows at a time: each step's return
ends where the next one starts, and compounding them never counts a day twice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_WINDOW: int = 60
"""Trading days shown by default: about a quarter, the most recent ones."""


class DirectionalDataError(ValueError):
    """The message is written for a portfolio manager, like the models' errors."""


@dataclass(frozen=True)
class DirectionalResult:
    """One index's long/cash strategy against buy-and-hold over one window."""

    horizon: int
    window: int
    """Trading days the window spans; may be fewer than asked for if the run
    produced fewer out-of-sample days."""
    strategy: pd.Series
    """Per step, the strategy's return: the realised return when invested, else 0."""
    buy_and_hold: pd.Series
    """Per step, the index's realised return."""
    invested: pd.Series
    """Per step, whether the strategy held the index (the forecast was above zero)."""
    hit_rate: float
    """Share of steps with a forecast whose direction matched the realised return's.
    NaN if no step had both."""
    no_forecast: int
    """Steps the model made no forecast for (a signal was missing). They are held
    in cash and left out of the hit rate."""

    @property
    def calls(self) -> int:
        """Steps in the window: the number of in-or-out decisions made."""
        return len(self.strategy)

    @property
    def share_invested(self) -> float:
        return float(self.invested.mean()) if self.calls else float("nan")

    @property
    def strategy_cumulative(self) -> pd.Series:
        return cumulative(self.strategy)

    @property
    def buy_and_hold_cumulative(self) -> pd.Series:
        return cumulative(self.buy_and_hold)

    @property
    def start(self) -> pd.Timestamp:
        return self.strategy.index[0]

    @property
    def end(self) -> pd.Timestamp:
        return self.strategy.index[-1]


def directional_pnl(
    forecast: pd.Series,
    realised: pd.Series,
    *,
    horizon: int,
    window: int = DEFAULT_WINDOW,
) -> DirectionalResult:
    """The long/cash strategy and buy-and-hold over the last ``window`` trading days
    of ``forecast``'s out-of-sample dates, stepped every ``horizon`` days.

    Only dates with a realised return count: the last ``horizon`` dates of a run
    have none yet. A date with a realised return but no forecast is held in cash.
    """
    if horizon < 1:
        raise DirectionalDataError(f"The horizon must be at least 1 day, got {horizon}.")
    if window < 1:
        raise DirectionalDataError(f"The window must be at least 1 day, got {window}.")
    frame = pd.DataFrame(
        {"forecast": forecast, "realised": realised.reindex(forecast.index)}
    ).sort_index()
    frame = frame[frame["realised"].notna()]
    if frame.empty:
        raise DirectionalDataError(
            "This run has no out-of-sample day with a realised return to compare against."
        )

    recent = frame.iloc[-window:]
    steps = recent.iloc[::horizon]
    invested = steps["forecast"] > 0
    strategy = steps["realised"].where(invested, 0.0)

    has_forecast = steps["forecast"].notna()
    judged = steps[has_forecast]
    hits = (judged["forecast"] > 0) == (judged["realised"] > 0)

    return DirectionalResult(
        horizon=horizon,
        window=len(recent),
        strategy=strategy.astype(float),
        buy_and_hold=steps["realised"].astype(float),
        invested=invested,
        hit_rate=float(hits.mean()) if len(judged) else float("nan"),
        no_forecast=int((~has_forecast).sum()),
    )


def cumulative(returns: pd.Series) -> pd.Series:
    """Compounded growth after each step, as a fraction (0.03 is +3%)."""
    return pd.Series(np.cumprod(1 + returns.to_numpy(dtype=float)) - 1, index=returns.index)
