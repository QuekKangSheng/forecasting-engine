"""The market field on the Home page: the equity index as the committed data has it.

The index's last months stand as candles on a dotted floor that runs back to a
horizon, the oldest month furthest away and smallest. A faint wall from the floor to
each month's close holds them up, and the largest falls are arcs from the month the
index peaked to the month it bottomed. Every mark is read from the committed data;
with nothing committed the floor is empty and says so.

``field`` returns an SVG as text for ``st.markdown``. The drawing is plain geometry
worked out here, so it needs no JavaScript, which ``st.markdown`` would not run. Its
motion is in ``hero.CSS``.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

import tokens

#: Months of candles on the field: five years, enough to hold a fall and its recovery
#: and few enough that the nearest are wide enough to read.
MONTHS: int = 60

#: The fewest months worth drawing as a field; fewer and the floor stays empty.
MIN_MONTHS: int = 6

#: A fall between month-end closes smaller than this is not drawn as an arc.
LEAST_FALL: float = 0.05

#: How many of the largest falls are drawn.
FALLS_SHOWN: int = 3

# The field's geometry, in the drawing's own units. The candles stand on a line that
# runs from ``_NEAR``, where the newest month stands, towards ``_VANISH`` on the
# horizon; a month at the back is drawn at ``_FAR_SCALE`` of the size of one at the
# front. The horizon is low, so the months rise into open space above it.
_VIEW_W, _VIEW_H = 760, 470
_VANISH = (-210.0, 268.0)
_NEAR = (672.0, 418.0)
_FAR_SCALE = 0.3
_BODY = 21.0
_RISE = 318.0
_DOT_GAP = 26.0
_FLOOR_LEFT, _FLOOR_RIGHT = -30, 5
_MIN_ROW_GAP = 2.6


@dataclass(frozen=True)
class Candle:
    """One month of daily closes: where it started, how far it ranged, where it ended."""

    month: pd.Timestamp
    open: float
    high: float
    low: float
    close: float

    @property
    def up(self) -> bool:
        return self.close >= self.open


@dataclass(frozen=True)
class Fall:
    """A fall between two month-end closes."""

    peak: int
    trough: int
    depth: float
    """Negative: ``trough close / peak close - 1``."""


def monthly_candles(prices: pd.Series, months: int = MONTHS) -> tuple[Candle, ...]:
    """The last ``months`` calendar months of ``prices`` as candles.

    A month opens at the previous month's last close, so its body is the month's
    whole move and not the move after its first day; the first month opens at its
    own first close. The high and low are of the daily closes, the open included:
    an export of closes has no intraday range.
    """
    closes = prices.dropna().sort_index()
    if closes.empty:
        return ()
    by_month = closes.groupby(closes.index.to_period("M"))
    summary = by_month.agg(["first", "max", "min", "last"])
    opens = summary["last"].shift(1).fillna(summary["first"])
    starts = by_month.apply(lambda month: month.index[0])
    candles = [
        Candle(
            month=starts[period],
            open=float(opens[period]),
            high=float(max(row["max"], opens[period])),
            low=float(min(row["min"], opens[period])),
            close=float(row["last"]),
        )
        for period, row in summary.iterrows()
    ]
    return tuple(candles[-months:])


def largest_falls(
    closes: Sequence[float], count: int = FALLS_SHOWN, least: float = LEAST_FALL
) -> tuple[Fall, ...]:
    """The ``count`` deepest falls in ``closes`` that share no month, deepest first.

    The deepest fall in a stretch is found, then the stretches before its peak and
    after its trough are searched the same way, so no two falls overlap.
    """
    found: list[Fall] = []
    stretches = [(0, len(closes) - 1)]
    while stretches:
        start, end = stretches.pop()
        fall = _deepest_fall(closes, start, end)
        if fall is None or fall.depth > -least:
            continue
        found.append(fall)
        stretches += [(start, fall.peak - 1), (fall.trough + 1, end)]
    return tuple(sorted(found, key=lambda fall: fall.depth)[:count])


def _deepest_fall(closes: Sequence[float], start: int, end: int) -> Fall | None:
    best: Fall | None = None
    peak = start
    for i in range(start, end + 1):
        if closes[i] > closes[peak]:
            peak = i
        depth = closes[i] / closes[peak] - 1
        if depth < 0 and (best is None or depth < best.depth):
            best = Fall(peak=peak, trough=i, depth=depth)
    return best


# ── The field ────────────────────────────────────────────────────────────────


def _scale(i: int, n: int) -> float:
    """How large month ``i`` of ``n`` is drawn: 1 at the front, ``_FAR_SCALE`` at the back."""
    back = (n - 1 - i) / max(n - 1, 1)
    return 1 / (1 + (1 / _FAR_SCALE - 1) * back)


def _floor(scale: float, lateral: float = 0.0) -> tuple[float, float]:
    """Where a point on the floor is drawn: ``lateral`` units to the side of the line
    the candles stand on, at the depth drawn at ``scale``."""
    x = _VANISH[0] + (_NEAR[0] + lateral - _VANISH[0]) * scale
    y = _VANISH[1] + (_NEAR[1] - _VANISH[1]) * scale
    return x, y


def _fade(scale: float) -> float:
    """0 at the back of the field, 1 at the front."""
    return (scale - _FAR_SCALE) / (1 - _FAR_SCALE)


def _n(value: float) -> str:
    return f"{value:.1f}"


@dataclass(frozen=True)
class _Layout:
    """Where each month stands and how a price becomes a height."""

    candles: tuple[Candle, ...]
    low: float
    high: float

    @classmethod
    def of(cls, candles: tuple[Candle, ...]) -> _Layout:
        return cls(candles, min(c.low for c in candles), max(c.high for c in candles))

    def scale(self, i: int) -> float:
        return _scale(i, len(self.candles))

    def base(self, i: int) -> tuple[float, float]:
        return _floor(self.scale(i))

    def lift(self, i: int, price: float) -> float:
        """How far above the floor ``price`` is drawn at month ``i``."""
        spread = self.high - self.low
        share = 0.5 if spread <= 0 else (price - self.low) / spread
        return (0.07 + 0.93 * share) * _RISE * self.scale(i)

    def y(self, i: int, price: float) -> float:
        return self.base(i)[1] - self.lift(i, price)


def _defs() -> str:
    return (
        "<defs>"
        '<linearGradient id="fe-field-fade" x1="0" x2="1" y1="0" y2="0">'
        '<stop offset="0" stop-color="#000"/><stop offset="0.2" stop-color="#fff"/>'
        '<stop offset="1" stop-color="#fff"/></linearGradient>'
        '<mask id="fe-field-mask"><rect width="100%" height="100%" fill="url(#fe-field-fade)"/>'
        "</mask>"
        '<radialGradient id="fe-field-glow" cx="0.5" cy="0.5" r="0.5">'
        f'<stop offset="0" stop-color="{tokens.ACCENT}" stop-opacity="0.22"/>'
        f'<stop offset="1" stop-color="{tokens.ACCENT}" stop-opacity="0"/></radialGradient>'
        '<linearGradient id="fe-field-wall" x1="0" x2="0" y1="0" y2="1">'
        f'<stop offset="0" stop-color="{tokens.ACCENT}" stop-opacity="0.2"/>'
        f'<stop offset="1" stop-color="{tokens.ACCENT}" stop-opacity="0.01"/></linearGradient>'
        '<linearGradient id="fe-field-limb" x1="0" x2="1" y1="0" y2="0">'
        f'<stop offset="0" stop-color="{tokens.TEXT}" stop-opacity="0"/>'
        f'<stop offset="0.45" stop-color="{tokens.TEXT}" stop-opacity="0.22"/>'
        f'<stop offset="1" stop-color="{tokens.TEXT}" stop-opacity="0.04"/></linearGradient>'
        "</defs>"
    )


def _horizon() -> str:
    """The far edge of the floor, as the limb of something large, with a glow behind it."""
    top = _VANISH[1] - 6
    return (
        f'<ellipse cx="420" cy="{_n(top + 4)}" rx="440" ry="58" fill="url(#fe-field-glow)"/>'
        f'<circle cx="430" cy="{_n(top + 2100)}" r="2100" fill="none" '
        'stroke="url(#fe-field-limb)" stroke-width="1"/>'
    )


def _wall(layout: _Layout) -> str:
    """What holds the months up: a faint wall from the floor to each month's close,
    its top edge the line of closes, and a thread from each month down to the floor."""
    count = len(layout.candles)
    floor = [layout.base(i) for i in range(count)]
    closes = [(floor[i][0], layout.y(i, layout.candles[i].close)) for i in range(count)]
    outline = " ".join(f"{_n(x)},{_n(y)}" for x, y in [*floor, *reversed(closes)])
    ridge = " ".join(f"{_n(x)},{_n(y)}" for x, y in closes)
    threads = "".join(
        f'<line x1="{_n(x)}" x2="{_n(x)}" y1="{_n(layout.y(i, layout.candles[i].low))}" '
        f'y2="{_n(y)}" stroke-opacity="{0.05 + 0.13 * _fade(layout.scale(i)):.2f}"/>'
        for i, (x, y) in enumerate(floor)
    )
    return (
        f'<g class="fe-field-floor"><polygon points="{outline}" fill="url(#fe-field-wall)"/>'
        f'<polyline points="{ridge}" fill="none" stroke="{tokens.ACCENT_BRIGHT}" '
        f'stroke-opacity="0.28" stroke-width="1"/>'
        f'<g stroke="{tokens.TEXT}" stroke-width="1">{threads}</g></g>'
    )


def _floor_rows(layout: _Layout | None) -> str:
    """The floor: a row of dots at each month's depth, brighter under a month that
    ranged widely. With no data, evenly spaced rows of one brightness."""
    count = len(layout.candles) if layout else MONTHS
    widest = max((c.high - c.low) / c.open for c in layout.candles) if layout else 0.0
    rows, last_y = [], None
    for i in range(count - 1, -1, -1):
        scale = _scale(i, count)
        x1, y = _floor(scale, _FLOOR_LEFT * _DOT_GAP)
        x2, _ = _floor(scale, _FLOOR_RIGHT * _DOT_GAP)
        if last_y is not None and last_y - y < _MIN_ROW_GAP:
            continue
        last_y = y
        ranged = 0.0
        if layout and widest > 0:
            candle = layout.candles[i]
            ranged = (candle.high - candle.low) / candle.open / widest
        opacity = (0.2 + 0.4 * ranged) * (0.3 + 0.7 * _fade(scale))
        rows.append(
            f'<line x1="{_n(x1)}" y1="{_n(y)}" x2="{_n(x2)}" y2="{_n(y)}" '
            f'stroke-width="{0.9 + 1.9 * scale:.2f}" '
            f'stroke-dasharray="0 {_DOT_GAP * scale:.2f}" stroke-opacity="{opacity:.3f}"/>'
        )
    return (
        f'<g class="fe-field-floor" stroke="{tokens.TEXT}" stroke-linecap="round" '
        f'mask="url(#fe-field-mask)">{"".join(rows)}</g>'
    )


def _candle(layout: _Layout, i: int) -> str:
    candle, scale = layout.candles[i], layout.scale(i)
    x, _ = layout.base(i)
    colour = tokens.RISE if candle.up else tokens.FALL
    width = max(1.2, _BODY * scale * scale)
    top = layout.y(i, max(candle.open, candle.close))
    height = max(layout.y(i, min(candle.open, candle.close)) - top, 0.9)
    # A month that closed up is outlined, one that closed down is filled, so the two
    # are told apart without the colours.
    fill = f'fill="{colour}" fill-opacity="0.2"' if candle.up else f'fill="{colour}"'
    return (
        f'<g class="fe-candle" style="--i:{i}" opacity="{0.3 + 0.7 * _fade(scale):.2f}" '
        f'stroke="{colour}" stroke-width="{max(0.7, 1.15 * scale):.2f}">'
        f'<line x1="{_n(x)}" x2="{_n(x)}" y1="{_n(layout.y(i, candle.high))}" '
        f'y2="{_n(layout.y(i, candle.low))}"/>'
        f'<rect x="{_n(x - width / 2)}" y="{_n(top)}" width="{_n(width)}" '
        f'height="{_n(height)}" {fill}/></g>'
    )


def _years(layout: _Layout) -> str:
    """The year under each January far enough forward to be read."""
    labels, last_x = [], None
    for i, candle in enumerate(layout.candles):
        if candle.month.month != 1:
            continue
        scale = layout.scale(i)
        x, y = layout.base(i)
        if last_x is not None and x - last_x < 34:
            continue
        last_x = x
        labels.append(
            f'<line x1="{_n(x)}" x2="{_n(x)}" y1="{_n(y + 3 * scale)}" y2="{_n(y + 9 * scale)}" '
            f'stroke="{tokens.BORDER_STRONG}" stroke-width="1"/>'
            f'<text x="{_n(x)}" y="{_n(y + 12 * scale + 9)}" text-anchor="middle" '
            f'font-size="{max(8.5, 11 * scale):.1f}" '
            f'fill-opacity="{0.6 + 0.4 * _fade(scale):.2f}">{candle.month.year}</text>'
        )
    return f'<g class="fe-field-label" fill="{tokens.TEXT_SECONDARY}">{"".join(labels)}</g>'


def _arcs(layout: _Layout, falls: Sequence[Fall]) -> str:
    """Each fall as an arc from the top of the month it peaked to the foot of the month
    it bottomed, its size written at the top of the arc. The deepest is brightest."""
    parts = []
    for rank, fall in enumerate(falls):
        x1, _ = layout.base(fall.peak)
        x2, _ = layout.base(fall.trough)
        y1 = layout.y(fall.peak, layout.candles[fall.peak].close)
        y2 = layout.y(fall.trough, layout.candles[fall.trough].close)
        reach = 30 + 0.3 * (x2 - x1)
        cx, cy = (x1 + x2) / 2, min(y1, y2) - reach
        opacity = (0.95, 0.6, 0.4)[min(rank, 2)]
        parts.append(
            f'<path class="fe-field-arc" style="--i:{rank}" pathLength="1" '
            f'd="M{_n(x1)} {_n(y1)} Q{_n(cx)} {_n(cy)} {_n(x2)} {_n(y2)}" '
            f'stroke-opacity="{opacity}"/>'
            f'<circle cx="{_n(x2)}" cy="{_n(y2)}" r="2.2" fill="{tokens.ACCENT_BRIGHT}" '
            f'fill-opacity="{opacity}" class="fe-field-late"/>'
            f'<text class="fe-field-late" x="{_n(cx)}" y="{_n((cy + min(y1, y2)) / 2 - 5)}" '
            f'text-anchor="middle" font-size="10" fill="{tokens.ACCENT_BRIGHT}" '
            f'fill-opacity="{opacity}">{fall.depth:+.1%}</text>'
        )
    return (
        f'<g class="fe-field-label" fill="none" stroke="{tokens.ACCENT_BRIGHT}" '
        f'stroke-width="1.1">{"".join(parts)}</g>'
    )


def _latest(layout: _Layout, name: str) -> str:
    """The newest month: two rings where it stands, and its date, close and move
    written above it."""
    i = len(layout.candles) - 1
    candle = layout.candles[i]
    x, y = layout.base(i)
    move = candle.close / candle.open - 1 if candle.open else 0.0
    colour = tokens.STATE["success"][0] if move >= 0 else tokens.STATE["danger"][0]
    top = max(layout.y(i, candle.high) - 58, 14)
    edge = _VIEW_W - 6
    return (
        f'<g fill="none" stroke="{tokens.ACCENT_BRIGHT}" stroke-width="1">'
        f'<circle class="fe-field-ring" cx="{_n(x)}" cy="{_n(y)}" r="9"/>'
        f'<circle class="fe-field-ring fe-field-ring-2" cx="{_n(x)}" cy="{_n(y)}" r="9"/></g>'
        f'<g class="fe-field-label fe-field-late" text-anchor="end">'
        f'<text x="{edge}" y="{_n(top)}" font-size="10" fill="{tokens.MUTED}">'
        f"{html.escape(name)} · {candle.month:%b %Y}</text>"
        f'<text class="fe-field-figure" x="{edge}" y="{_n(top + 24)}" font-size="22" '
        f'fill="{tokens.TEXT}">{candle.close:,.2f}</text>'
        f'<text x="{edge}" y="{_n(top + 40)}" font-size="10" fill="{colour}">'
        f"{move:+.1%} this month</text></g>"
    )


def _key(months: int, falls: int) -> str:
    """What the marks are, on one line in the field's bottom-left corner."""
    y = _VIEW_H - 13
    arcs = (
        f'<path d="M244 {y - 3} q9 -10 18 0" fill="none" stroke="{tokens.ACCENT_BRIGHT}"/>'
        f'<text x="268" y="{y}">largest falls</text>'
        if falls
        else ""
    )
    return (
        f'<g class="fe-field-label" font-size="9.5" fill="{tokens.MUTED}">'
        f'<text x="6" y="{y}">{months} monthly candles</text>'
        f'<rect x="130" y="{y - 9}" width="6" height="10" fill="{tokens.RISE}" '
        f'fill-opacity="0.2" stroke="{tokens.RISE}"/><text x="141" y="{y}">up</text>'
        f'<rect x="170" y="{y - 9}" width="6" height="10" fill="{tokens.FALL}" '
        f'stroke="{tokens.FALL}"/><text x="181" y="{y}">down</text>{arcs}</g>'
    )


def _described(candles: Sequence[Candle], falls: Sequence[Fall], name: str) -> str:
    first, last = candles[0], candles[-1]
    text = (
        f"{name}, one candle a month from {first.month:%B %Y} to {last.month:%B %Y}. "
        f"It last closed at {last.close:,.2f}."
    )
    if falls:
        deepest = falls[0]
        text += (
            f" Its largest fall between month-end closes was {deepest.depth:.1%}, from "
            f"{candles[deepest.peak].month:%B %Y} to {candles[deepest.trough].month:%B %Y}."
        )
    return text


def field(candles: Sequence[Candle], name: str) -> str:
    """The market field for ``candles`` of the index called ``name``, as an SVG."""
    candles = tuple(candles)
    if len(candles) < MIN_MONTHS:
        return empty_field(
            f"Fewer than {MIN_MONTHS} months of {name}", "Too short a history to draw"
        )
    layout = _Layout.of(candles)
    falls = largest_falls([c.close for c in candles])
    body = "".join(_candle(layout, i) for i in range(len(candles)))
    return (
        f'<svg class="fe-field" viewBox="0 0 {_VIEW_W} {_VIEW_H}" role="img" '
        f'aria-label="{html.escape(_described(candles, falls, name), quote=True)}">'
        f"{_defs()}{_horizon()}{_floor_rows(layout)}{_wall(layout)}{_years(layout)}{body}"
        f"{_arcs(layout, falls)}{_latest(layout, name)}{_key(len(candles), len(falls))}</svg>"
    )


def empty_field(
    reason: str = "No data committed", detail: str = "The index is drawn here once it is"
) -> str:
    """The floor with nothing standing on it, and why: by default, that no data has
    been committed."""
    reason, detail = html.escape(reason, quote=True), html.escape(detail, quote=True)
    return (
        f'<svg class="fe-field" viewBox="0 0 {_VIEW_W} {_VIEW_H}" role="img" '
        f'aria-label="An empty field. {reason}. {detail}.">'
        f"{_defs()}{_horizon()}{_floor_rows(None)}"
        f'<g class="fe-field-label" text-anchor="end" fill="{tokens.MUTED}">'
        f'<text x="{_VIEW_W - 6}" y="212" font-size="10">{reason}</text>'
        f'<text x="{_VIEW_W - 6}" y="228" font-size="10">{detail}</text>'
        "</g></svg>"
    )
