"""FYP-162: what holding one index only on days its forecast is positive would have
earned, against simply holding it throughout.

A long/cash investor's trade: cash earns nothing, so the index is worth holding
whenever its expected return is above zero. The forecast used is the model's raw
out-of-sample forecast, level included. Whether a model's forecasts rank the good
days above the bad ones is a separate question, answered by the Signal Rank IC on
the forecast less the model's own level; this asks only whether acting on the
forecast's sign would have paid.

The strategy is fully invested on a step whose forecast is above zero and fully
in cash otherwise; its return comes from the index's realised return alone, so the
forecast's size never enters. No trading cost is charged: every figure is gross.

Forecasts and realised returns are the run's out-of-sample series, indexed by the
date the forecast was made, with each realised value the ``horizon``-day forward
return from that date. Every out-of-sample date the run produced is replayed. At
``horizon`` above 1 neighbouring rows' returns overlap, so the dates are stepped
through ``horizon`` rows at a time: each step's return
ends where the next one starts, and compounding them never counts a day twice.
The cumulative returns are dated by when each step's return was earned, the end
of its holding period, starting from zero on the first step's date.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class DirectionalDataError(ValueError):
    """The message is written for a portfolio manager, like the models' errors."""


@dataclass(frozen=True)
class DirectionalResult:
    """One index's long/cash strategy against buy-and-hold over a run's
    out-of-sample period."""

    horizon: int
    days: int
    """Out-of-sample trading days replayed: every one with a realised return."""
    strategy: pd.Series
    """Per step, dated when the call was made, the strategy's return: the realised
    return when invested, else 0."""
    buy_and_hold: pd.Series
    """Per step, the index's realised return."""
    invested: pd.Series
    """Per step, whether the strategy held the index (the forecast was above zero)."""
    hit_rate: float
    """Share of steps with a forecast whose sign matched the realised return's.
    NaN if no step had both."""
    no_forecast: int
    """Steps the model made no forecast for (an input was missing). They are held
    in cash and left out of the hit rate."""
    held_until: pd.Series
    """Per step, dated when the call was made, the date its ``horizon``-day return
    ends."""

    @property
    def calls(self) -> int:
        """Steps replayed: the number of in-or-out decisions made."""
        return len(self.strategy)

    @property
    def share_invested(self) -> float:
        return float(self.invested.mean()) if self.calls else float("nan")

    @property
    def strategy_cumulative(self) -> pd.Series:
        """Compounded growth, dated by the end of each step's holding period, from
        zero on the first step's date."""
        return self._dated_by_end(cumulative(self.strategy))

    @property
    def buy_and_hold_cumulative(self) -> pd.Series:
        return self._dated_by_end(cumulative(self.buy_and_hold))

    def _dated_by_end(self, growth: pd.Series) -> pd.Series:
        dated = pd.Series(growth.to_numpy(), index=pd.DatetimeIndex(self.held_until.to_numpy()))
        return pd.concat([pd.Series([0.0], index=pd.DatetimeIndex([self.start])), dated])

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
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> DirectionalResult:
    """The long/cash strategy and buy-and-hold over ``forecast``'s out-of-sample
    dates, stepped every ``horizon`` days from the first.

    Every date is replayed unless ``start`` and ``end`` pick a window of them
    (both inclusive); the steps then begin at the window's first date, so the
    window reads as if it were the whole period.

    Only dates with a realised return count: the last ``horizon`` dates of a run
    have none yet. A date with a realised return but no forecast is held in cash.

    Each step's return ends ``horizon`` of ``forecast``'s dates later. Where the
    forecast's dates run out first, it ends ``horizon`` business days later.
    """
    if horizon < 1:
        raise DirectionalDataError(f"The horizon must be at least 1 day, got {horizon}.")
    frame = pd.DataFrame(
        {"forecast": forecast, "realised": realised.reindex(forecast.index)}
    ).sort_index()
    dates = frame.index
    frame = frame[frame["realised"].notna()]
    if start is not None or end is not None:
        frame = frame.loc[start:end]
    if frame.empty:
        raise DirectionalDataError(
            "This run has no out-of-sample day with a realised return to compare against"
            + (" in the chosen window." if start is not None or end is not None else ".")
        )

    steps = frame.iloc[::horizon]
    invested = steps["forecast"] > 0
    strategy = steps["realised"].where(invested, 0.0)

    has_forecast = steps["forecast"].notna()
    judged = steps[has_forecast]
    hits = (judged["forecast"] > 0) == (judged["realised"] > 0)

    ends = dates.get_indexer(steps.index) + horizon
    held_until = [
        dates[end] if end < len(dates) else day + pd.offsets.BDay(horizon)
        for day, end in zip(steps.index, ends, strict=True)
    ]

    return DirectionalResult(
        horizon=horizon,
        days=len(frame),
        strategy=strategy.astype(float),
        buy_and_hold=steps["realised"].astype(float),
        invested=invested,
        hit_rate=float(hits.mean()) if len(judged) else float("nan"),
        no_forecast=int((~has_forecast).sum()),
        held_until=pd.Series(held_until, index=steps.index),
    )


def cumulative(returns: pd.Series) -> pd.Series:
    """Compounded growth after each step, as a fraction (0.03 is +3%)."""
    return pd.Series(np.cumprod(1 + returns.to_numpy(dtype=float)) - 1, index=returns.index)
