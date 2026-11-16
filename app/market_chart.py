"""The market on the Home page: the S&P 500 as a trading chart, live where the feed
answers.

Drawn by TradingView's Lightweight Charts (Apache 2.0; the app serves its own copy,
``app/static/lightweight-charts.standalone.production.js``) in a frame of its own,
because a chart that pans and zooms needs a script and ``st.markdown`` runs none.

It works as a trading chart does. The buttons pick the size of a candle (a day, a
week, a month, a quarter), and each comes with its whole history (``index_feed``):
the chart opens on the latest stretch (``VISIBLE`` candles), a drag pans back
through the past and forward to now, the wheel zooms, and "Now" jumps back to the
latest candle. Volume stands under the candles. The line above gives the latest
price and the move over what is in view, kept up to date as the view moves; the
candle under the pointer is written out in a box beside it.

Every candle size is fetched at once, in parallel, and sent together, so a change of
size happens in the browser: picking a span used to rerun the whole page, about
300 ms a click; in the browser it takes about 40 ms.

At rest the chart stands tilted back on a dotted floor, for depth; the moment the
pointer is over it, it lies flat, so it is always read flat and exact. The tilt is a
transform the browser's compositor does, so it costs nothing to move. A 3D switch
turns it off, and it starts off for a viewer who asked for less motion.
"""

from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import streamlit as st

import tokens
from forecasting_engine.ingest import index_feed
from forecasting_engine.ingest.index_feed import Candles, IndexFeedError

#: The candle the chart opens on: a day.
DEFAULT_INTERVAL = "1D"

#: Candles in the first view of each size: about six months of days, two years of
#: weeks, ten of months, twenty-five of quarters. The rest is a drag away.
VISIBLE = {"1D": 126, "1W": 104, "1M": 120, "3M": 100}

#: Seconds the feed's answer, or its failure, is kept before asking again.
REFRESH_SECONDS = 300

#: The frame's height on the page, in pixels: the line of figures, the controls, the
#: chart and the line under it.
HEIGHT = 470

#: Where the browser loads the chart library from: the app's own static files.
LIBRARY = "/app/static/lightweight-charts.standalone.production.js"

#: How a candle's period is written, by the period it covers.
_PERIOD_TEXT = {"D": "day", "W": "week", "M": "month", "Q": "quarter"}


@st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
def live() -> tuple[dict[str, Candles], str | None]:
    """Every candle size from the live feed, asked in parallel, and the first reason
    any had none. A size the feed did not answer is left out. Kept for
    ``REFRESH_SECONDS``, failures included, so no rerun waits on the feed."""
    intervals = list(index_feed.INTERVALS)
    with ThreadPoolExecutor(len(intervals)) as pool:
        futures = {i: pool.submit(index_feed.fetch, i) for i in intervals}
    answered, reasons = {}, []
    for interval, future in futures.items():
        try:
            answered[interval] = future.result()
        except IndexFeedError as exc:
            reasons.append(str(exc))
    return answered, (reasons[0] if reasons else None)


def rows(candles: Candles) -> list[list[Any]]:
    """The candles as compact rows, ``[day, open, high, low, close, volume]``, prices to
    the cent and volume ``None`` where none is known: a fifth the size of objects, for
    a page sent whole on every visit."""
    out = []
    for stamp, row in candles.frame.iterrows():
        volume = row.get("volume", float("nan"))
        out.append(
            [
                f"{stamp:%Y-%m-%d}",
                round(float(row.open), 2),
                round(float(row.high), 2),
                round(float(row.low), 2),
                round(float(row.close), 2),
                None if volume is None or math.isnan(volume) else int(volume),
            ]
        )
    return out


def payload(bundles: dict[str, Candles]) -> dict[str, Any]:
    """Each candle size's rows, how many to show first, and the figures above it."""
    out = {}
    for interval, candles in bundles.items():
        if candles.live:
            where = f"Live · {candles.source} · {candles.as_of:%d %b %Y %H:%M} New York"
        else:
            where = f"From the {candles.source} · to {candles.as_of:%d %b %Y}"
        out[interval] = {
            "rows": rows(candles),
            "visible": VISIBLE.get(interval, 120),
            "period": candles.period,
            "unit": _PERIOD_TEXT[candles.period],
            "last": round(candles.last, 2),
            "live": candles.live,
            "where": where,
        }
    return out


def _script_safe(value: Any) -> str:
    """``value`` as JSON that can sit inside a script element: no text in it can end
    the element early."""
    return json.dumps(value, separators=(",", ":")).replace("</", "<\\/").replace("<!--", "<\\!--")


def chart_html(bundles: dict[str, Candles], name: str, default: str = DEFAULT_INTERVAL) -> str:
    """The frame's whole page: the line of figures, the candle-size buttons, the
    readout, the chart on its stage, and where the figures come from."""
    intervals = [i for i in index_feed.INTERVALS if i in bundles]
    first = default if default in bundles else intervals[0]
    buttons = "".join(
        f'<button type="button" data-span="{i}" aria-pressed="{str(i == first).lower()}">'
        f"{i}</button>"
        for i in intervals
    )
    colours = {
        "muted": tokens.MUTED,
        "border": tokens.BORDER,
        "raised": tokens.RAISED,
        "rise": tokens.RISE,
        "fall": tokens.FALL,
        "accentBright": tokens.ACCENT_BRIGHT,
        "up": tokens.STATE["success"][0],
        "down": tokens.STATE["danger"][0],
        "warn": tokens.STATE["warning"][0],
    }
    return (
        _PAGE.replace("__NAME__", html_text(name))
        .replace("__BUTTONS__", buttons)
        .replace("__LIBRARY__", LIBRARY)
        .replace("__DATA__", _script_safe(payload(bundles)))
        .replace("__FIRST__", _script_safe(first))
        .replace("__COLOURS__", _script_safe(colours))
        .replace("__HEIGHT__", str(HEIGHT))
    )


def html_text(text: str) -> str:
    """``text`` escaped for the page."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


#: The frame's page, its placeholders (__NAME__ and the like) filled by chart_html.
_PAGE = Path(__file__).with_name("market_chart.html").read_text(encoding="utf-8")
