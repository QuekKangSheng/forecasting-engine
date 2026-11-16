"""The S&P 500 as it trades, for the Home page: candles from Yahoo Finance's chart
service, or, when that does not answer, from the committed data's own closes.

The service needs no key and was checked against the team's Bloomberg export
(16 Nov 2026, over Nov 2016 to Sep 2026): 99.7% of 2,480 shared daily closes of
``^GSPC`` and ``SPX Index PX_LAST`` agree to the cent (median gap 0.0001; the worst,
11 Aug 2021, by 5.29 points), and all 118 monthly closes agree to within 0.0002.
It is an unofficial endpoint, so nothing here is relied on: any failure gives a
``IndexFeedError`` and the page draws the committed data instead.

Nothing here feeds a model or the portfolio; it is the market, shown.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd

#: The S&P 500 price index, the figure quoted in the news.
SYMBOL: str = "^GSPC"

#: Span shown -> (the service's range, its candle interval). Each gives a few dozen
#: to a few hundred candles: few enough to tell apart, enough to see a trend.
SPANS: Mapping[str, tuple[str, str]] = {
    "3M": ("3mo", "1d"),
    "1Y": ("1y", "1wk"),
    "5Y": ("5y", "1mo"),
    "10Y": ("10y", "3mo"),
}

#: The candle interval -> the pandas period a candle covers.
_PERIODS: Mapping[str, str] = {"1d": "D", "1wk": "W", "1mo": "M", "3mo": "Q"}

#: Span shown -> how far back the committed data's own candles reach, in rows.
_FALLBACK_ROWS: Mapping[str, int] = {"3M": 63, "1Y": 252, "5Y": 1_260, "10Y": 2_520}

_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={range}&interval={interval}"

#: Short: the page waits on this while it draws.
_TIMEOUT_SECONDS: float = 6.0

#: A generic browser string; the service refuses requests with none. It says nothing
#: about who is asking.
_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

_COLUMNS = ("open", "high", "low", "close")


class IndexFeedError(Exception):
    """The live feed did not answer, or answered with something unreadable."""


@dataclass(frozen=True)
class Candles:
    """One candle per period: when it opened, its highest and lowest, where it closed."""

    frame: pd.DataFrame
    """Indexed by each period's first date, in the exchange's own time; columns
    ``open``, ``high``, ``low``, ``close``."""
    period: str
    """The pandas period one candle covers: ``D``, ``W``, ``M`` or ``Q``."""
    source: str
    live: bool
    """True when the candles came from the live feed, False when from committed data."""
    last: float
    as_of: pd.Timestamp
    """When ``last`` was struck: the latest trade, or the committed data's last day."""


def fetch(
    span: str,
    *,
    symbol: str = SYMBOL,
    opener: Callable[..., object] = urlopen,
) -> Candles:
    """``symbol``'s candles over ``span`` (a key of ``SPANS``) from the live feed.

    Raises ``ValueError`` for an unknown span and ``IndexFeedError`` when the feed
    does not answer in time or answers with something unreadable."""
    if span not in SPANS:
        raise ValueError(f"unknown span {span!r}; one of {', '.join(SPANS)}")
    range_, interval = SPANS[span]
    url = _URL.format(symbol=quote(symbol, safe=""), range=range_, interval=interval)
    try:
        with opener(Request(url, headers=_HEADERS), timeout=_TIMEOUT_SECONDS) as response:
            payload = json.load(response)
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        raise IndexFeedError(f"the live feed did not answer ({exc})") from exc
    return parse_chart(payload, interval)


def parse_chart(payload: Mapping, interval: str) -> Candles:
    """Candles from the chart service's answer, at ``interval``.

    A day with no trade (the service sends nulls) is left out. Above a day, the
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
        frame = pd.DataFrame({c: quote_[c] for c in _COLUMNS}, index=stamps, dtype=float)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise IndexFeedError(f"the live feed's answer could not be read ({exc})") from exc
    frame.index = frame.index.tz_convert(zone).tz_localize(None)
    frame = frame.dropna()
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


def from_closes(closes: pd.Series, span: str, source: str = "committed data") -> Candles:
    """Candles over ``span`` built from daily ``closes``, for when the live feed is
    not there. Each candle opens at the previous one's close; its high and low are of
    the closes alone, since a file of closes holds no range within a day."""
    if span not in SPANS:
        raise ValueError(f"unknown span {span!r}; one of {', '.join(SPANS)}")
    closes = closes.dropna().sort_index().iloc[-_FALLBACK_ROWS[span] :]
    if closes.empty:
        raise ValueError("no closes to draw")
    period = _PERIODS[SPANS[span][1]]
    groups = closes.groupby(closes.index.to_period(period))
    summary = groups.agg(["first", "max", "min", "last"])
    opens = summary["last"].shift(1).fillna(summary["first"])
    frame = pd.DataFrame(
        {
            "open": opens,
            "high": summary["max"].combine(opens, max),
            "low": summary["min"].combine(opens, min),
            "close": summary["last"],
        }
    )
    frame.index = frame.index.start_time
    return Candles(frame, period, source, False, float(closes.iloc[-1]), closes.index[-1])
