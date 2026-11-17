"""The S&P 500 as it trades, for the Home page: candles and volume from Yahoo Finance's
chart service, or, when that does not answer, candles from the committed data's own
closes.

A candle is a day, a week, a month or a quarter (``INTERVALS``), and each comes with
as much history as is worth sending to a page: ten years of days, twenty of weeks,
and every month and quarter the service has (from 1985). The page then pans through
that history like a trading chart.

The service needs no key and was checked against the team's Bloomberg export
(16 Nov 2026, over Nov 2016 to Sep 2026): 99.7% of 2,480 shared daily closes of
``^GSPC`` and ``SPX Index PX_LAST`` agree to the cent (median gap 0.0001; the worst,
11 Aug 2021, by 5.29 points), and all 118 monthly closes agree to within 0.0002. It
is an unofficial endpoint, so nothing here is relied on: any failure gives an
``IndexFeedError`` and the page draws the committed data instead. Asked for "max",
the service quietly sends quarters whatever the interval, so history is asked for by
its start and end dates.

Nothing here feeds a model or the portfolio; it is the market, shown.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd

#: The S&P 500 price index, the figure quoted in the news.
SYMBOL: str = "^GSPC"

#: Candle shown -> (days of history asked for, or ``None`` for all there is; the
#: service's interval). History is what a drag can pan back through.
INTERVALS: Mapping[str, tuple[int | None, str]] = {
    "1D": (3_653, "1d"),
    "1W": (7_305, "1wk"),
    "1M": (None, "1mo"),
    "3M": (None, "3mo"),
}

#: The service's interval -> the pandas period one candle covers.
_PERIODS: Mapping[str, str] = {"1d": "D", "1wk": "W", "1mo": "M", "3mo": "Q"}

_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    "?period1={start}&period2={end}&interval={interval}"
)

#: Short: the page waits on this while it draws.
_TIMEOUT_SECONDS: float = 8.0

#: A generic browser string; the service refuses requests with none. It says nothing
#: about who is asking.
_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

_PRICES = ("open", "high", "low", "close")


class IndexFeedError(Exception):
    """The live feed did not answer, or answered with something unreadable."""


@dataclass(frozen=True)
class Candles:
    """One candle per period: when it opened, its highest and lowest, where it closed,
    and how much traded."""

    frame: pd.DataFrame
    """Indexed by each period's first date, in the exchange's own time; columns
    ``open``, ``high``, ``low``, ``close`` and ``volume`` (NaN where none is known)."""
    period: str
    """The pandas period one candle covers: ``D``, ``W``, ``M`` or ``Q``."""
    source: str
    live: bool
    """True when the candles came from the live feed, False when from committed data."""
    last: float
    as_of: pd.Timestamp
    """When ``last`` was struck: the latest trade, or the committed data's last day."""


def fetch(
    interval: str,
    *,
    symbol: str = SYMBOL,
    opener: Callable[..., object] = urlopen,
    now: float | None = None,
) -> Candles:
    """``symbol``'s candles of ``interval`` (a key of ``INTERVALS``), over the history
    ``INTERVALS`` gives it, from the live feed.

    Raises ``ValueError`` for an unknown interval and ``IndexFeedError`` when the feed
    does not answer in time or answers with something unreadable."""
    if interval not in INTERVALS:
        raise ValueError(f"unknown interval {interval!r}; one of {', '.join(INTERVALS)}")
    days, service_interval = INTERVALS[interval]
    end = int(time.time() if now is None else now)
    start = 0 if days is None else end - days * 86_400
    url = _URL.format(
        symbol=quote(symbol, safe=""), start=start, end=end, interval=service_interval
    )
    try:
        with opener(Request(url, headers=_HEADERS), timeout=_TIMEOUT_SECONDS) as response:
            payload = json.load(response)
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        raise IndexFeedError(f"the live feed did not answer ({exc})") from exc
    return parse_chart(payload, service_interval)


def parse_chart(payload: Mapping, interval: str) -> Candles:
    """Candles from the chart service's answer, at the service's ``interval``.

    A period with no trade (the service sends nulls) is left out. Above a day, the
    service dates the period still running at its latest trade rather than at its
    start, so each candle is filed under its period and the latest kept."""
    try:
        chart = payload["chart"]
        if chart.get("error"):
            raise IndexFeedError(f"the live feed refused the request: {chart['error']}")
        result = chart["result"][0]
        meta = result["meta"]
        quote_ = result["indicators"]["quote"][0]
        stamps = pd.to_datetime(result["timestamp"], unit="s", utc=True)
        zone = meta.get("exchangeTimezoneName", "America/New_York")
        columns = {c: quote_[c] for c in _PRICES}
        columns["volume"] = quote_.get("volume") or [None] * len(stamps)
        frame = pd.DataFrame(columns, index=stamps, dtype=float)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise IndexFeedError(f"the live feed's answer could not be read ({exc})") from exc
    frame.index = frame.index.tz_convert(zone).tz_localize(None)
    frame = frame.dropna(subset=list(_PRICES))
    if frame.empty:
        raise IndexFeedError("the live feed sent no prices")
    period = _PERIODS.get(interval, "D")
    frame = frame.groupby(frame.index.to_period(period)).last()
    frame.index = frame.index.start_time
    price = meta.get("regularMarketPrice")
    stamp = meta.get("regularMarketTime")
    last = float(price) if price is not None else float(frame["close"].iloc[-1])
    as_of = (
        pd.Timestamp(stamp, unit="s", tz="UTC").tz_convert(zone).tz_localize(None)
        if stamp is not None
        else frame.index[-1]
    )
    return Candles(frame, period, "Yahoo Finance", True, last, as_of)


def from_closes(closes: pd.Series, interval: str, source: str = "committed data") -> Candles:
    """Candles of ``interval`` built from every one of the daily ``closes``, for when
    the live feed is not there. Each candle opens at the previous one's close; its
    high and low are of the closes alone, since a file of closes holds no range within
    a day, and no volume is known."""
    if interval not in INTERVALS:
        raise ValueError(f"unknown interval {interval!r}; one of {', '.join(INTERVALS)}")
    closes = closes.dropna().sort_index()
    if closes.empty:
        raise ValueError("no closes to draw")
    period = _PERIODS[INTERVALS[interval][1]]
    groups = closes.groupby(closes.index.to_period(period))
    summary = groups.agg(["first", "max", "min", "last"])
    opens = summary["last"].shift(1).fillna(summary["first"])
    frame = pd.DataFrame(
        {
            "open": opens,
            "high": summary["max"].combine(opens, max),
            "low": summary["min"].combine(opens, min),
            "close": summary["last"],
            "volume": float("nan"),
        }
    )
    frame.index = frame.index.start_time
    return Candles(frame, period, source, False, float(closes.iloc[-1]), closes.index[-1])
