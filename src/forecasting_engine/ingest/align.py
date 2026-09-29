"""FeaturePanel: the lag-safe dataset screening and modelling consume.

Built by align_and_lag() from the committed Bloomberg merge. No screening or
fitting function accepts a bare DataFrame, so there is no type-legal way to run
on unlagged data.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

#: Signal lag, in rows. A longer lag throws away usable data, a shorter one uses
#: data not yet published, so it is not a setting.
PRODUCTION_LAG_DAYS: int = 1


@dataclass(frozen=True)
class FeaturePanel:
    """A dataset where every signal is dated to when it was observable."""

    frame: pd.DataFrame
    signals: tuple[str, ...]
    targets: tuple[str, ...]
    lag_days: int = PRODUCTION_LAG_DAYS
    horizon: int = 1
    """Trading days a target label looks forward from its own date. Lets a
    splitter purge training rows whose label window would reach past a
    rebalance date, regardless of how large an embargo the caller chose."""
    label_end: pd.Series | None = None
    """For each row, the date of the price its target label is computed from,
    or NaT where there is no label. ``None`` on a panel built by hand, in which
    case a splitter can only assume a label reaches ``horizon`` rows ahead.

    The two differ whenever the target's market was shut on a day another
    series traded: that day is a row of the merged frame but not a trading day
    of the target, so a label crossing it reaches further than ``horizon`` rows."""

    def __post_init__(self) -> None:
        if self.lag_days != PRODUCTION_LAG_DAYS:
            raise ValueError(f"lag_days must be {PRODUCTION_LAG_DAYS}, got {self.lag_days}")
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        overlap = set(self.signals) & set(self.targets)
        if overlap:
            raise ValueError(f"target column(s) {overlap} also listed as signals")


def align_and_lag(
    frame: pd.DataFrame,
    signal_cols: Sequence[str],
    price_col: str,
    horizon: int = 5,
) -> FeaturePanel:
    """Lag every signal, derive a forward-return target, freeze both into a FeaturePanel.

    ``price_col`` is a price/level column already present in ``frame``; the
    target is the ``horizon``-day forward return computed from it. Every
    signal in ``signal_cols`` is shifted forward ``PRODUCTION_LAG_DAYS`` so its value on
    a given date is what was actually observable on that date.

    The horizon is counted in the target's own trading days — the rows where
    ``price_col`` has a value — not in rows of ``frame``. A merged frame has a
    row for every date any series traded, so on a day the target's market was
    shut (Columbus Day for a US bond index, say) there is a row but no price.
    Counting rows across it made a "5-day" label a 4-day return and dropped the
    real move across the closed day entirely. That day now gets no label, and
    ``label_end`` records how far each label really reaches so the splitter can
    purge on it.

    This assumes ``price_col`` is not forward-filled. A filled cell on a closed
    day looks like a trading day and would reintroduce the error.
    """
    target_col = f"fwd_return_{horizon}d"
    out = frame.copy()

    prices = out[price_col].dropna()
    out[target_col] = prices.pct_change(horizon).shift(-horizon).reindex(out.index)
    label_end = pd.Series(prices.index, index=prices.index).shift(-horizon).reindex(out.index)

    out[list(signal_cols)] = out[list(signal_cols)].shift(PRODUCTION_LAG_DAYS)
    return FeaturePanel(
        frame=out,
        signals=tuple(signal_cols),
        targets=(target_col,),
        horizon=horizon,
        label_end=label_end,
    )
