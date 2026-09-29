"""FeaturePanel: the lag-safe dataset screening and modelling consume.

Built by align_and_lag() from the committed Bloomberg merge. No screening or
fitting function accepts a bare DataFrame, so there is no type-legal way to run
on unlagged data.

A panel lives on its target's own calendar: the dates the target has a price.
Every other row of the merged frame is dropped, never filled, and each signal
is read as of each of those dates, then made stationary on that calendar, so a
change spans any dropped date instead of losing the move across it.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import pandas as pd

from forecasting_engine.extraction.bloomberg_csv import ColumnSource

#: Signal lag, in rows. A longer lag throws away usable data, a shorter one uses
#: data not yet published, so it is not a setting.
PRODUCTION_LAG_DAYS: int = 1

#: A signal's last value is carried onto a calendar date for at most this many
#: rows; older than that it is treated as missing.
MAX_STALENESS: int = 3


class Transform(StrEnum):
    LOG_RETURN = "log return"
    DIFFERENCE = "difference"
    NONE = "none"


#: Bloomberg ticker -> how its level is made stationary. A ticker not listed
#: here is differenced; a total-return index field is always a log return.
TICKER_TRANSFORMS: Mapping[str, Transform] = {
    "LF98TRUU": Transform.LOG_RETURN,
    "LEGATRUU": Transform.LOG_RETURN,
    "SPX": Transform.LOG_RETURN,
    "LUACOAS": Transform.DIFFERENCE,
    "USGGBE10": Transform.DIFFERENCE,
    "VIX": Transform.DIFFERENCE,
    "JPMVXYGL": Transform.DIFFERENCE,
}

_TOTAL_RETURN_FIELD = "TOT_RETURN_INDEX"
_PRICE_FIELD = "PX_LAST"
_QUOTE_FIELDS = frozenset({"PX_BID", "PX_ASK"})


def transform_for(source: ColumnSource | None) -> Transform:
    if source is None:
        return Transform.DIFFERENCE
    if source.field.startswith(_TOTAL_RETURN_FIELD):
        return Transform.LOG_RETURN
    return TICKER_TRANSFORMS.get(source.ticker, Transform.DIFFERENCE)


def select_signals(
    columns: Sequence[str],
    target_columns: Collection[str],
    sources: Mapping[str, ColumnSource],
) -> list[str]:
    """The signal columns to model, one per security.

    Every field of a target's security is left out, not only the target column
    itself: SPX's bid is SPX. Of a security's remaining fields, the total-return
    index is kept if there is one, otherwise ``PX_LAST``, otherwise its first
    other field; bid and ask quotes never are. A column with no known source
    stands alone.
    """
    target_securities = {sources[c].security for c in target_columns if c in sources}
    by_security: dict[str, list[str]] = {}
    for column in columns:
        if column in target_columns:
            continue
        source = sources.get(column)
        if source is None:
            by_security[column] = [column]
        elif source.security not in target_securities and source.field not in _QUOTE_FIELDS:
            by_security.setdefault(source.security, []).append(column)

    def rank(column: str) -> int:
        source = sources.get(column)
        if source is None:
            return 0
        if source.field.startswith(_TOTAL_RETURN_FIELD):
            return 0
        return 1 if source.field == _PRICE_FIELD else 2

    kept = {min(group, key=rank) for group in by_security.values()}
    return [c for c in columns if c in kept]


@dataclass(frozen=True)
class SignalAlignment:
    """How one signal was put on the target's calendar."""

    transform: Transform
    carried_forward: int
    """Calendar rows given a value observed on an earlier date, within the limit."""
    excluded: int
    """Calendar rows where the signal, once transformed and lagged, has no value."""


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
    case a splitter can only assume a label reaches ``horizon`` rows ahead."""
    alignment: Mapping[str, SignalAlignment] = field(default_factory=dict)
    """Per signal, how it was aligned. Empty on a panel built by hand."""

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
    *,
    transforms: Mapping[str, Transform] | None = None,
    exact: Collection[str] = (),
) -> FeaturePanel:
    """Put every signal on ``price_col``'s calendar, make it stationary, lag it,
    and derive a forward-return target.

    a. The calendar is the dates ``price_col`` has a price; every other row is
       dropped and the target is never filled.
    b. Each signal takes its last value on or before each calendar date, unless
       that value is more than ``MAX_STALENESS`` calendar rows old. A column in
       ``exact`` is matched by date only, never carried forward.
    c. Each signal is transformed per ``transforms`` (``Transform.NONE`` where
       unlisted), so a change spans any dropped date.
    d. Signals are lagged ``PRODUCTION_LAG_DAYS`` row.
    e. The target is the ``horizon``-row forward return on the same calendar,
       and ``label_end`` records the date of the price each label reaches.

    A row missing any signal a model uses is left in the panel; each model
    drops its own incomplete rows when fitting and predicting.
    """
    target_col = f"fwd_return_{horizon}d"
    transforms = transforms or {}
    frame = frame.sort_index()
    prices = frame[price_col].dropna()
    calendar = prices.index

    out = pd.DataFrame(index=calendar)
    alignment = {}
    for signal in signal_cols:
        if signal in exact:
            level, carried = frame[signal].reindex(calendar), 0
        else:
            level, carried = _as_of(frame[signal], calendar)
        transform = transforms.get(signal, Transform.NONE)
        lagged = _transform(level, transform).shift(PRODUCTION_LAG_DAYS)
        out[signal] = lagged
        alignment[signal] = SignalAlignment(
            transform=transform, carried_forward=carried, excluded=int(lagged.isna().sum())
        )

    out[price_col] = prices
    out[target_col] = prices.pct_change(horizon).shift(-horizon)
    label_end = pd.Series(calendar, index=calendar).shift(-horizon)
    return FeaturePanel(
        frame=out,
        signals=tuple(signal_cols),
        targets=(target_col,),
        horizon=horizon,
        label_end=label_end,
        alignment=alignment,
    )


def _as_of(series: pd.Series, calendar: pd.DatetimeIndex) -> tuple[pd.Series, int]:
    """``series``' last value on or before each calendar date, and how many
    calendar dates took a value observed earlier. A value more than
    ``MAX_STALENESS`` calendar rows old is NaN."""
    observed = series.dropna()
    if observed.empty:
        return pd.Series(np.nan, index=calendar), 0
    position = observed.index.searchsorted(calendar, side="right") - 1
    has_value = position >= 0
    safe = np.where(has_value, position, 0)
    observed_on = observed.index[safe]
    # Calendar rows since the observation: 0 when it was made on the date itself.
    age = np.arange(1, len(calendar) + 1) - calendar.searchsorted(observed_on, side="right")
    fresh = has_value & (age <= MAX_STALENESS)
    values = np.where(fresh, observed.to_numpy(dtype=float)[safe], np.nan)
    return pd.Series(values, index=calendar), int((fresh & (age > 0)).sum())


def _transform(level: pd.Series, transform: Transform) -> pd.Series:
    if transform is Transform.LOG_RETURN:
        return np.log(level.where(level > 0)).diff()
    if transform is Transform.DIFFERENCE:
        return level.diff()
    return level
