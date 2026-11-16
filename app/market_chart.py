"""The market on the Home page: the S&P 500 as candles, live where the feed answers.

A candle per day, week, month or quarter, by the span picked above the chart: green
and outlined where it closed above its open, red and filled where it closed below,
so the two are told apart without the colours. A hover shows that candle's date,
open, high, low, close and move; a drag pans and Shift with the wheel zooms, in the
browser alone (the wheel alone still scrolls the page past it), and a double click
puts it back. A dotted line marks the latest price.

The live feed (``ingest.index_feed``) is asked at most once every few minutes per
span, and a failure is remembered as long, so a page with no network draws the
committed data at once instead of waiting on the feed at every click.
"""

from __future__ import annotations

import html
from typing import Any

import pandas as pd
import streamlit as st

import charts
import tokens
from forecasting_engine.ingest import index_feed
from forecasting_engine.ingest.index_feed import Candles, IndexFeedError

#: The span drawn until another is picked: five years of monthly candles.
DEFAULT_SPAN = "5Y"

#: Seconds an answer of the live feed, or its failure, is kept before asking again.
REFRESH_SECONDS = 300

#: The chart's height on the page.
HEIGHT = 330

#: The share of its period a candle's body fills, leaving a gap between candles.
BODY_SHARE = 0.68

#: The time axis by the period a candle covers: where its ticks fall, and their text.
_AXES = {
    "D": {"tickCount": "week", "format": "%d %b"},
    "W": {"tickCount": "month", "format": "%b %Y"},
    "M": {"tickCount": "year", "format": "%Y"},
    "Q": {"tickCount": "year", "format": "%Y"},
}

#: What a candle's date is shown as, by the period it covers.
_DATE_FORMATS = {"D": "%d %b %Y", "W": "w/c %d %b %Y", "M": "%b %Y", "Q": "Q%q %Y"}


@st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
def live(span: str) -> tuple[Candles | None, str | None]:
    """The live feed's candles over ``span``, or why there are none. Kept for
    ``REFRESH_SECONDS`` either way: a failure is not retried on every rerun."""
    try:
        return index_feed.fetch(span), None
    except IndexFeedError as exc:
        return None, str(exc)


def candles_table(candles: Candles) -> pd.DataFrame:
    """One row per candle, with where its body starts and ends on the time axis
    (``BODY_SHARE`` of its period, centred), its move, and its date as text."""
    frame = candles.frame
    starts = frame.index
    if len(starts) > 1:
        step = pd.Series(starts).diff().median()
    else:
        step = pd.Timedelta(days=1)
    half = step * BODY_SHARE / 2
    mids = starts + step / 2 if candles.period != "D" else starts
    table = pd.DataFrame(
        {
            "mid": mids,
            "start": mids - half,
            "end": mids + half,
            "open": frame["open"].to_numpy(),
            "high": frame["high"].to_numpy(),
            "low": frame["low"].to_numpy(),
            "close": frame["close"].to_numpy(),
        }
    )
    table["move"] = table["close"] / table["open"] - 1
    table["up"] = table["close"] >= table["open"]
    table["when"] = [_when(stamp, candles.period) for stamp in starts]
    return table


def _when(stamp: pd.Timestamp, period: str) -> str:
    if period == "Q":
        return f"Q{stamp.quarter} {stamp.year}"
    return stamp.strftime(_DATE_FORMATS[period])


def candle_chart(candles: Candles) -> dict[str, Any]:
    """The Vega-Lite specification of ``candles``, its figures as named tables."""
    up = {"test": "datum.up"}
    wick = {
        "data": {"name": "candles"},
        "mark": {"type": "rule", "strokeWidth": 1.2},
        "encoding": {
            "x": {"field": "mid", "type": "temporal"},
            "y": {"field": "low", "type": "quantitative"},
            "y2": {"field": "high"},
            "color": {"condition": {**up, "value": tokens.RISE}, "value": tokens.FALL},
        },
    }
    # Closed up: outlined, a faint fill. Closed down: filled. Never colour alone.
    body = {
        "data": {"name": "candles"},
        "mark": {"type": "bar", "strokeWidth": 1.2, "cornerRadius": 0.5},
        "encoding": {
            "x": {"field": "start", "type": "temporal"},
            "x2": {"field": "end"},
            "y": {"field": "open", "type": "quantitative"},
            "y2": {"field": "close"},
            "stroke": {"condition": {**up, "value": tokens.RISE}, "value": tokens.FALL},
            "fill": {"condition": {**up, "value": tokens.BG}, "value": tokens.FALL},
        },
    }
    crosshair = {
        "data": {"name": "candles"},
        "mark": {"type": "rule", "strokeDash": [3, 3], "strokeWidth": 1},
        "encoding": {
            "x": {"field": "mid", "type": "temporal"},
            "color": {"value": tokens.MUTED},
            "opacity": {"condition": {"param": "nearest", "empty": False, "value": 1}, "value": 0},
        },
    }
    latest = {
        "data": {"values": [{"last": candles.last}]},
        "mark": {"type": "rule", "strokeDash": [2, 3], "strokeWidth": 1},
        "encoding": {
            "y": {"field": "last", "type": "quantitative"},
            "color": {"value": tokens.ACCENT_BRIGHT},
        },
    }
    targets = {
        "data": {"name": "candles"},
        "name": "candles_hit",
        "mark": {"type": "point", "opacity": 0, "size": 400},
        # All on one height, so the nearest candle is the one under the pointer's
        # left-right position, wherever up or down the pointer is.
        "encoding": {
            "x": {"field": "mid", "type": "temporal"},
            "y": {"value": HEIGHT / 2},
            "tooltip": [
                {"field": "when", "type": "nominal", "title": "Period"},
                {"field": "open", "type": "quantitative", "title": "Open", "format": ",.2f"},
                {"field": "high", "type": "quantitative", "title": "High", "format": ",.2f"},
                {"field": "low", "type": "quantitative", "title": "Low", "format": ",.2f"},
                {"field": "close", "type": "quantitative", "title": "Close", "format": ",.2f"},
                {"field": "move", "type": "quantitative", "title": "Move", "format": "+.2%"},
            ],
        },
    }
    return {
        "config": {
            **charts._CONFIG,
            "axisY": {**charts._CONFIG["axisY"], "orient": "right", "format": ",.0f"},
        },
        "height": HEIGHT,
        "encoding": {
            "x": {
                "type": "temporal",
                "title": None,
                "axis": {**_AXES[candles.period], "labelOverlap": "greedy"},
            },
            "y": {"type": "quantitative", "title": None, "scale": {"zero": False, "nice": True}},
        },
        "layer": [wick, body, latest, crosshair, targets],
        "params": [
            {
                "name": "nearest",
                "select": {
                    "type": "point",
                    "nearest": True,
                    "on": "pointerover",
                    "clear": "pointerout",
                    "fields": ["mid"],
                },
                "views": ["candles_hit"],
            },
            {
                "name": "zoom",
                # Shift and the wheel: the wheel alone scrolls the page past the chart.
                "select": {
                    "type": "interval",
                    "encodings": ["x"],
                    "zoom": "wheel![event.shiftKey]",
                },
                "bind": "scales",
                "views": ["candles_hit"],
            },
        ],
        "usermeta": charts._CANVAS,
        "datasets": {"candles": candles_table(candles)},
    }


def header(candles: Candles, name: str) -> str:
    """The line above the chart: the index, its latest price and the span's move."""
    first = float(candles.frame["open"].iloc[0])
    change = candles.last / first - 1 if first else float("nan")
    tone = "fe-tick-up" if change >= 0 else "fe-tick-down"
    arrow = "▲" if change >= 0 else "▼"
    return (
        '<div class="fe-market-head">'
        f'<span class="fe-market-name">{html.escape(name)}</span>'
        f'<span class="fe-market-last">{candles.last:,.2f}</span>'
        f'<span class="{tone}">{arrow} {abs(change):.1%}</span></div>'
    )


def source(candles: Candles) -> str:
    """The line under the chart: where the figures come from and when, a green dot
    when live and amber when drawn from the committed data, and how to move it."""
    if candles.live:
        where = f"Live · {candles.source} · {candles.as_of:%d %b %Y %H:%M} New York"
        dot = "var(--fe-success-fg)"
    else:
        where = f"From the {candles.source} · to {candles.as_of:%d %b %Y}"
        dot = "var(--fe-warning-fg)"
    return (
        f'<div class="fe-market-where"><i style="background:{dot}"></i>{html.escape(where)}'
        '<span class="fe-market-hint">Drag to pan · Shift + scroll to zoom · '
        "double-click to reset</span></div>"
    )
