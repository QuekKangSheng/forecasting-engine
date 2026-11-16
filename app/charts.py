"""Charts both the Portfolio page and an opened run draw, so a saved run looks the same
as the page that first showed it.

A chart is a Vega-Lite specification written out here as a plain dict, with its
figures beside it as named tables. Two things follow, and both are why a click no
longer waits on the charts:

- Nothing is built or validated on the server. The charts used to be assembled
  through Altair, which checks every chart against the whole Vega-Lite schema each
  time it is drawn: a third of a second for a page of five, on every rerun.
- A chart's specification never holds its figures, so it is the same from one rerun
  to the next and only the tables change. The browser then keeps the chart it has
  drawn and swaps the figures in, where a changed specification makes it throw the
  chart away and build it again.
"""

from __future__ import annotations

import hashlib
import html
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

import tokens

# One colour per series, the same on every chart, so a line means the same thing
# wherever it is drawn: the optimised portfolio is always blue and the 50/50
# benchmark always grey, the equity index orange and the bond index green.
OPTIMISED = tokens.SERIES_OURS
BENCHMARK = tokens.SERIES_REFERENCE
EQUITY_INDEX = tokens.SERIES_EQUITY
BOND_INDEX = tokens.SERIES_BOND
BASELINE = tokens.SERIES_BASELINE
_OTHERS = tokens.SERIES_OTHERS

_FIXED = {
    "Optimised": OPTIMISED,
    "Equal-weight benchmark": BENCHMARK,
    # The Models page's direction test: the strategy that acts on the forecast is
    # drawn like the optimised portfolio, the passive one like the benchmark.
    "Long/cash strategy": OPTIMISED,
    "Buy and hold": BENCHMARK,
    # A recorded fit's forecasts (Run history): the model's own line is the blue one,
    # what happened is the grey it is read against.
    "Forecast": OPTIMISED,
    "Realised": BENCHMARK,
    "Baseline": BASELINE,
}

#: The most days a line chart lets a hover land on: about one to the pixel across the
#: widest chart the app draws.
MAX_HOVER_TARGETS = 600

#: The most rows a line chart is drawn from: more than that and it is thinned (``_drawn``).
MAX_DRAWN_ROWS = 1_500

#: A line chart's height unless told otherwise, and the weights chart's.
LINE_HEIGHT = 300
WEIGHTS_HEIGHT = 280

#: Canvas, not Streamlit's default SVG: SVG makes every day's point a page element,
#: thousands per chart, which stalls scrolling and every redraw.
_CANVAS = {"embedOptions": {"renderer": "canvas", "actions": False}}

#: How every chart is dressed. Stated in full and drawn with ``theme=None``, so a chart
#: looks the same whatever Streamlit's own chart theme does: quiet horizontal
#: gridlines, no box, monospace figures on the axes, the legend underneath.
_CONFIG: dict[str, Any] = {
    "background": "transparent",
    "font": tokens.SANS,
    "padding": {"left": 2, "right": 8, "top": 6, "bottom": 2},
    "view": {"stroke": None},
    "axis": {
        # The text face, whose figures have a plain zero: at this size the monospace
        # face's dotted zero reads as an eight.
        "labelFont": tokens.SANS,
        "labelFontSize": 11.5,
        "labelColor": tokens.MUTED,
        "labelPadding": 8,
        "titleFont": tokens.MONO,
        "titleFontSize": 11,
        "titleFontWeight": "normal",
        "titleColor": tokens.MUTED,
        "gridColor": tokens.BORDER,
        "domainColor": tokens.BORDER_STRONG,
        "tickColor": tokens.BORDER_STRONG,
        "tickSize": 4,
    },
    "axisX": {"grid": False, "labelFlush": False},
    "axisY": {"domain": False, "ticks": False},
    "legend": {
        "labelFont": tokens.SANS,
        "labelFontSize": 12,
        "labelColor": tokens.TEXT_SECONDARY,
        "symbolStrokeWidth": 2,
        "symbolSize": 140,
        "columnPadding": 20,
        "padding": 0,
        "offset": 14,
    },
    "line": {"strokeJoin": "round", "strokeCap": "round"},
}

#: What every index this app is given is called, and so says nothing where space is short.
_SUFFIX = " (total return)"


def series_colour(name: str, position: int = 0) -> str:
    """The colour ``name`` is drawn in on every chart. Index names come from the
    data, so they are matched by what they contain rather than listed."""
    if name in _FIXED:
        return _FIXED[name]
    lowered = name.lower()
    if "s&p" in lowered or "spx" in lowered or "equity" in lowered:
        return EQUITY_INDEX
    if "bond" in lowered or "agg" in lowered or "legatruu" in lowered or "lbustruu" in lowered:
        return BOND_INDEX
    return _OTHERS[position % len(_OTHERS)]


def short_name(name: str) -> str:
    """An index's name without "(total return)", for a legend, a tooltip or a narrow label."""
    return name.replace(_SUFFIX, "")


def _key(
    names: list[str],
    latest: Mapping[str, float],
    figure: str,
    as_of: pd.Timestamp | None,
    *,
    bands: bool = False,
) -> str:
    """A chart's key, as HTML to stand above it: each series' colour and name, and
    its latest figure, with the day those figures are for.

    It replaces the legend Vega would draw under the chart. That named the lines and
    nothing else; where each one stands now, the first thing asked of a chart of
    returns, had to be read off the axis or found with a hover. The names and figures
    are in the text colours and only the swatch carries the series colour, so the key
    reads the same to someone who can't tell the colours apart."""
    swatch = "fe-readout-swatch fe-readout-band" if bands else "fe-readout-swatch"
    items = []
    for position, name in enumerate(names):
        value = latest.get(name, float("nan"))
        shown = "—" if value != value else format(value, figure)
        items.append(
            '<span class="fe-readout-item">'
            f'<span class="{swatch}" style="background:{series_colour(name, position)}"></span>'
            f"{html.escape(short_name(name))}"
            f'<span class="fe-readout-value">{shown}</span></span>'
        )
    when = "" if as_of is None else f'<span class="fe-readout-asof">{as_of:%d %b %Y}</span>'
    return f'<div class="fe-readout">{"".join(items)}{when}</div>'


def colour_scale(names: list[str]) -> dict[str, Any]:
    """An explicit domain and range, so Vega cannot reassign colours by order."""
    return {"domain": names, "range": [series_colour(n, i) for i, n in enumerate(names)]}


def _colour(field: str, names: list[str]) -> dict[str, Any]:
    # No legend of Vega's: the chart's key stands above it (``_key``).
    return {
        "field": field,
        "type": "nominal",
        "scale": colour_scale(names),
        "sort": names,
        "legend": None,
    }


def _years() -> dict[str, Any]:
    return {
        "field": "date",
        "type": "temporal",
        "title": None,
        "axis": {"tickCount": "year", "format": "%Y"},
    }


def draw(chart: dict[str, Any]) -> None:
    """Show ``chart`` across the width of wherever it is called.

    ``theme=None``: the chart states its own dress (``_CONFIG``), and Streamlit's chart
    theme would otherwise restyle the axes and legend over it. The chart's key
    (``_key``) is written above it, and a ``key_below`` under it, for a chart whose
    lower part has a key of its own. Neither is part of what the chart itself is sent."""
    spec = {name: part for name, part in chart.items() if name not in ("key", "key_below")}
    if chart.get("key"):
        st.markdown(chart["key"], unsafe_allow_html=True)
    st.vega_lite_chart(_named_by_content(spec), width="stretch", theme=None)
    if chart.get("key_below"):
        st.markdown(chart["key_below"], unsafe_allow_html=True)


def _named_by_content(spec: dict[str, Any]) -> dict[str, Any]:
    """``spec`` with each named table renamed to its name plus a hash of its contents.

    Streamlit keeps a drawn chart and swaps new figures in only when it can tell the
    figures changed, and it tells by the tables' names: it expects them to carry a
    hash of the data, as Altair's names do. With fixed names ("drawn", "weights") a
    chart whose figures changed but whose specification didn't kept showing the old
    figures: another model on the direction check, or another risk level's weights,
    updated the numbers above the chart and left the lines as they were. Named by
    content, a change of figures is a change of name, so the chart is drawn afresh,
    and figures that didn't change keep their name and the chart is kept."""
    datasets = spec.get("datasets")
    if not datasets:
        return spec
    names = {name: f"{name}-{_content_hash(table)}" for name, table in datasets.items()}
    renamed = _rename_data(spec, names)
    renamed["datasets"] = {names[name]: table for name, table in datasets.items()}
    return renamed


def _content_hash(table: pd.DataFrame) -> str:
    digest = hashlib.blake2b(digest_size=8)
    digest.update(repr(list(table.columns)).encode())
    digest.update(pd.util.hash_pandas_object(table, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def _rename_data(part: Any, names: Mapping[str, str]) -> Any:
    """``part`` with every ``"data": {"name": ...}`` reference renamed by ``names``.
    Other ``name`` keys (a layer's, a parameter's) are left as they are."""
    if isinstance(part, list):
        return [_rename_data(item, names) for item in part]
    if not isinstance(part, dict):
        return part
    out = {}
    for key, value in part.items():
        if key == "datasets":
            out[key] = value
        elif key == "data" and isinstance(value, dict) and value.get("name") in names:
            out[key] = {**value, "name": names[value["name"]]}
        else:
            out[key] = _rename_data(value, names)
    return out


def weights_chart(
    renamed: pd.DataFrame,
    *,
    limits: tuple[float, float] | None = None,
    until: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Each named asset's share of the portfolio over time, stepping at each
    rebalance: the weights hold between rebalances, so a step is what actually
    happened, and a time axis shows the years rather than 99 date labels.

    The two shares are tinted bands with the first asset's weight drawn as a line
    where they meet. Solid blocks of colour hid the gridlines, so no weight could be
    read off the axis, and put the page's two heaviest marks on its least eventful
    chart. ``limits`` are the lowest and highest weight the first asset may take,
    drawn as dashed lines so a weight pinned at one is seen to be. ``until`` is the
    last day the final allocation was held to, so its step is drawn too.
    """
    names = [str(c) for c in renamed.columns]
    held = renamed.copy()
    held.columns = names
    if until is not None and len(held) and until > held.index[-1]:
        held.loc[until] = held.iloc[-1]
    wide = held.rename_axis("date").reset_index()
    long = wide.melt("date", var_name="Asset", value_name="Weight")
    # Stacked in the order given, first asset at the bottom, so the line below sits
    # on the boundary between the bands whatever the assets are called.
    long["position"] = long["Asset"].map({name: i for i, name in enumerate(names)})

    def y(field: str, **extra: Any) -> dict[str, Any]:
        # Every layer states the same axis: layers that left it out fell back to
        # Vega's own ticks, and the merged axis dropped half the labels.
        return {
            "field": field,
            "type": "quantitative",
            "title": None,
            "axis": {"format": "%", "values": [0, 0.2, 0.4, 0.6, 0.8, 1]},
            "scale": {"domain": [0, 1]},
            **extra,
        }

    first = names[0]
    bands = {
        "data": {"name": "shares"},
        "mark": {"type": "area", "interpolate": "step-after", "opacity": 0.16},
        "encoding": {
            "x": _years(),
            "y": y("Weight", stack="zero"),
            "color": _colour("Asset", names),
            "order": {"field": "position", "type": "quantitative", "sort": "ascending"},
        },
    }
    boundary = {
        "data": {"name": "weights"},
        "mark": {
            "type": "line",
            "interpolate": "step-after",
            "strokeWidth": 2,
            "color": series_colour(first),
        },
        "encoding": {"x": _years(), "y": y(first)},
    }
    # A step line only answers a hover within a pixel of one of its corners, so an
    # unseen point per rebalance takes it: the nearest rebalance to the pointer is
    # marked and shows its date and every asset's weight.
    points = {
        "data": {"name": "weights"},
        "name": "rebalances",
        "mark": {"type": "point", "filled": True, "size": 60, "color": series_colour(first)},
        "encoding": {
            "x": _years(),
            "y": y(first),
            "opacity": {"condition": {"param": "nearest", "value": 1, "empty": False}, "value": 0},
            "tooltip": [
                {"field": "date", "type": "temporal", "title": "From", "format": "%d %b %Y"},
                *(
                    {
                        "field": name,
                        "type": "quantitative",
                        "title": short_name(name),
                        "format": ".0%",
                    }
                    for name in names
                ),
            ],
        },
    }
    layers = [bands, boundary, points]
    datasets = {"shares": long, "weights": wide}
    if limits is not None:
        datasets["limits"] = pd.DataFrame({"Limit": sorted(set(limits))})
        layers.insert(
            2,
            {
                "data": {"name": "limits"},
                "mark": {
                    "type": "rule",
                    "strokeDash": [4, 4],
                    "strokeWidth": 1,
                    "color": tokens.MUTED,
                },
                "encoding": {"y": y("Limit")},
            },
        )
    return {
        "config": _CONFIG,
        "height": WEIGHTS_HEIGHT,
        "layer": layers,
        "params": [
            {
                "name": "nearest",
                "select": {
                    "type": "point",
                    "nearest": True,
                    "on": "pointerover",
                    "clear": "pointerout",
                    "encodings": ["x"],
                },
                "views": ["rebalances"],
            }
        ],
        "usermeta": _CANVAS,
        "datasets": datasets,
        "key": _key(
            names,
            {} if held.empty else held.iloc[-1].to_dict(),
            ".0%",
            None if held.empty else held.index[-1],
            bands=True,
        ),
    }


def _extremes(values: np.ndarray, starts: np.ndarray, stretch: np.ndarray) -> np.ndarray:
    """The row of the highest figure in each stretch of ``values`` (``starts`` are the
    stretches' first rows, ``stretch`` each row's stretch), leaving out a stretch with
    no figure at all. Pass ``-values`` for the lowest."""
    filled = np.where(np.isnan(values), -np.inf, values)
    highest = np.maximum.reduceat(filled, starts)
    at_highest = np.flatnonzero((filled == highest[stretch]) & np.isfinite(filled))
    # The first row at its stretch's highest: ``at_highest`` is in row order.
    _, first = np.unique(stretch[at_highest], return_index=True)
    return at_highest[first]


def _drawn(wide: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """The rows of ``wide`` its lines are drawn from: every day, until there are more
    days than the chart has room to show apart.

    Past ``MAX_DRAWN_ROWS`` the days are cut into equal stretches, each a few pixels
    wide, and every line keeps its highest and its lowest day in each stretch (and the
    first and last day of all). So every peak and trough is still drawn, on its own
    date and at its own value, and the line reads as it did, from a third of the
    points: twenty years of days is four to the pixel, and drawing them all was most
    of what the browser spent on each chart."""
    stretches = MAX_DRAWN_ROWS // (2 * max(len(names), 1))
    if len(wide) <= MAX_DRAWN_ROWS or stretches < 1:
        return wide
    edges = np.linspace(0, len(wide), stretches + 1).astype(int)
    starts = edges[:-1]
    stretch = np.repeat(np.arange(stretches), np.diff(edges))
    keep = [np.array([0, len(wide) - 1])]
    for name in names:
        values = wide[name].to_numpy(dtype=float)
        keep.append(_extremes(values, starts, stretch))
        keep.append(_extremes(-values, starts, stretch))
    return wide.iloc[np.unique(np.concatenate(keep))]


def line_chart(
    paths: pd.DataFrame,
    value: str,
    *,
    percent: bool = True,
    title: str | None = None,
    height: int = LINE_HEIGHT,
) -> dict[str, Any]:
    """A line per column of ``paths`` on one time axis, in percent unless told it is a
    ratio. Hovering or tapping anywhere marks that day on every line and shows its date
    and each line's value. ``title`` labels the vertical axis; without one the axis is
    left to the heading the page puts above the chart.

    The chart is sent one row per day and unfolded into its lines in the browser, and
    a hover lands on one unseen target per day. A long chart is thinned to what can be
    seen (``_drawn``) and to as many hover targets as can be told apart
    (``MAX_HOVER_TARGETS``). It used to be sent a row per day per line with an unseen
    point on each. Over twenty years that is twelve thousand points a chart for the
    browser to lay out before it could draw: about three seconds of a frozen page for
    four charts, every time their figures changed."""
    names = [str(c) for c in paths.columns]
    wide = paths.set_axis(names, axis=1).rename_axis("date").reset_index()
    ours = [name for name in names if series_colour(name) == OPTIMISED]
    y = {
        "field": value,
        "type": "quantitative",
        "title": title,
        "axis": {
            "format": "%" if percent else ".1f",
            "tickCount": 6,
            # The zero line a little stronger than the rest: a return, a drawdown and
            # a Sharpe ratio are all read against it.
            "gridColor": {
                "condition": {"test": "datum.value === 0", "value": tokens.BORDER_STRONG},
                "value": tokens.BORDER,
            },
        },
    }
    colour = _colour("Series", names)
    figure = ".2%" if percent else ".2f"
    unfold = {"fold": names, "as": ["Series", value]}
    hovering = {"filter": {"param": "nearest", "empty": False}}
    lines = {
        "data": {"name": "drawn"},
        "transform": [unfold],
        "mark": {"type": "line"},
        "encoding": {
            "x": _years(),
            "y": y,
            "color": colour,
            # The app's own line a little heavier than the ones it is read against.
            "strokeWidth": {
                "condition": {
                    "test": {"field": "Series", "oneOf": ours},
                    "value": 2,
                },
                "value": 1.5,
            },
        },
    }
    # Vega only shows a line's tooltip within a pixel or two of one of its days, too
    # fine to tap. So the day nearest the pointer, wherever in the chart that is, is
    # the one marked: a rule down it, a dot on each line, every line's value shown.
    crosshair = {
        "data": {"name": "hover"},
        "transform": [hovering],
        "mark": {"type": "rule", "color": tokens.MUTED, "strokeWidth": 1},
        "encoding": {"x": _years()},
    }
    dots = {
        "data": {"name": "hover"},
        "transform": [unfold, hovering],
        "mark": {
            "type": "point",
            "filled": True,
            "size": 56,
            "opacity": 1,
            # A ring in the page's colour, so a dot stays whole where lines cross.
            "stroke": tokens.BG,
            "strokeWidth": 2,
        },
        "encoding": {"x": _years(), "y": y, "color": colour},
    }
    targets = {
        "data": {"name": "hover"},
        "name": "days",
        "mark": {"type": "point", "opacity": 0},
        "encoding": {
            "x": _years(),
            "tooltip": [
                {"field": "date", "type": "temporal", "title": "Date", "format": "%d %b %Y"},
                *(
                    {
                        "field": name,
                        "type": "quantitative",
                        "title": short_name(name),
                        "format": figure,
                    }
                    for name in names
                ),
            ],
        },
    }
    # No more targets than a chart has room to tell apart: finding which of four
    # thousand days is nearest the pointer was the rest of what the browser did with
    # each chart. A hover lands on the nearest day kept and shows that day's own
    # figures, read from the data and not off the drawn line.
    every = -(-len(wide) // MAX_HOVER_TARGETS) or 1
    kept = wide.iloc[::every]
    if len(wide) and kept.index[-1] != wide.index[-1]:
        kept = pd.concat([kept, wide.iloc[[-1]]])  # the latest day can always be read
    return {
        "config": _CONFIG,
        "height": height,
        "layer": [lines, crosshair, dots, targets],
        "params": [
            {
                "name": "nearest",
                "select": {
                    "type": "point",
                    "nearest": True,
                    "on": "pointerover",
                    "clear": "pointerout",
                    "fields": ["date"],
                },
                "views": ["days"],
            }
        ],
        "usermeta": _CANVAS,
        "datasets": {"drawn": _drawn(wide, names), "hover": kept},
        "key": _key(
            names,
            # Each line's own latest figure: one may end before another does.
            {name: series.iloc[-1] for name in names if len(series := wide[name].dropna())},
            figure,
            None if wide.empty else wide["date"].iloc[-1],
        ),
    }


#: Each panel's height in the weights-and-volatility chart, and the gap between them:
#: two panels in a little more than the room the two charts took side by side.
PANEL_HEIGHT = 200
PANEL_GAP = 34


def weights_and_volatility_chart(
    renamed: pd.DataFrame,
    volatility: pd.DataFrame,
    *,
    limits: tuple[float, float] | None = None,
    until: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """The weights per rebalance above each line's rolling volatility, on one time axis.

    They used to stand side by side, each on its own axis covering slightly different
    years, so matching a move in the weights to the volatility around it meant reading
    across two charts. Stacked on a shared axis, a date sits directly above itself, and
    a hover anywhere draws that day through both panels and shows its weights and
    volatility together.

    It is one layered chart, not two stacked ones: Vega-Lite fits only a single or
    layered chart to its container's width, and a stacked pair ran past the right
    edge. The weights' layers and the volatility's each keep their own vertical scale,
    placed in the top and bottom band of the one plot. The layers are
    ``weights_chart``'s and ``line_chart``'s, less their own hovers, which are
    replaced by one that marks both: a day's weights are the allocation held that
    day, the last rebalance's."""
    top = weights_chart(renamed, limits=limits, until=until)
    bottom = line_chart(volatility, "Rolling volatility", height=PANEL_HEIGHT)
    weight_names = [str(c) for c in renamed.columns]
    vol_names = [str(c) for c in volatility.columns]
    first = weight_names[0]
    total = 2 * PANEL_HEIGHT + PANEL_GAP
    # Pixel bands, top of the plot first: Vega measures y downwards.
    top_band = [PANEL_HEIGHT, 0]
    bottom_band = [total, PANEL_HEIGHT + PANEL_GAP]

    # One table of hover days for both panels: the volatility chart's days, each with
    # the weights held that day. Weight columns are prefixed, since the indices share
    # their names with the volatility lines.
    days = bottom["datasets"]["hover"].copy()
    held = top["datasets"]["weights"].set_index("date")[weight_names].sort_index()
    on_day = held.reindex(held.index.union(days["date"])).ffill().reindex(days["date"])
    for name in weight_names:
        days[f"Weight: {name}"] = on_day[name].to_numpy()

    def in_band(layer: dict[str, Any], band: list[int], title: str) -> dict[str, Any]:
        """``layer`` with its vertical scale placed in ``band`` and its axis titled."""
        encoding = dict(layer["encoding"])
        if "y" in encoding:
            y = dict(encoding["y"])
            y["scale"] = {**y.get("scale", {}), "range": band}
            y["axis"] = {**y.get("axis", {}), "title": title, "orient": "left"}
            encoding["y"] = y
        return {**layer, "encoding": encoding}

    hovering = {"filter": {"param": "nearest", "empty": False}}
    weight_layers = [
        in_band(layer, top_band, "Weight")
        for layer in top["layer"]
        if layer.get("name") != "rebalances"
    ]
    weight_y = next(
        layer["encoding"]["y"] for layer in weight_layers if layer["data"]["name"] == "weights"
    )
    weight_layers.append(
        {
            "data": {"name": "hover"},
            "transform": [hovering],
            "mark": {
                "type": "point",
                "filled": True,
                "size": 56,
                "opacity": 1,
                "color": series_colour(first),
                "stroke": tokens.BG,
                "strokeWidth": 2,
            },
            "encoding": {"x": _years(), "y": {**weight_y, "field": f"Weight: {first}"}},
        }
    )
    vol_layers = [
        in_band(layer, bottom_band, "Volatility")
        for layer in bottom["layer"]
        if layer.get("name") != "days" and layer["mark"]["type"] != "rule"
    ]
    crosshair = {
        "data": {"name": "hover"},
        "transform": [hovering],
        "mark": {"type": "rule", "color": tokens.MUTED, "strokeWidth": 1},
        "encoding": {"x": _years()},
    }
    targets = {
        "data": {"name": "hover"},
        "name": "days",
        "mark": {"type": "point", "opacity": 0},
        "encoding": {
            "x": _years(),
            "tooltip": [
                {"field": "date", "type": "temporal", "title": "Date", "format": "%d %b %Y"},
                *(
                    {
                        "field": f"Weight: {name}",
                        "type": "quantitative",
                        "title": f"{short_name(name)} weight",
                        "format": ".0%",
                    }
                    for name in weight_names
                ),
                *(
                    {
                        "field": name,
                        "type": "quantitative",
                        "title": f"{short_name(name)} volatility",
                        "format": ".2%",
                    }
                    for name in vol_names
                ),
            ],
        },
    }
    return {
        "config": _CONFIG,
        "height": total,
        # Fit the width only. Fitting the height too squeezed the plot to make room for
        # the year labels, and the panels, placed in fixed bands, ran into them.
        "autosize": {"type": "fit-x", "contains": "padding"},
        "layer": [
            {"layer": weight_layers},
            {"layer": vol_layers},
            crosshair,
            targets,
        ],
        # Colour too: shared, the weights' two-colour range was used for the volatility
        # lines as well, and the third line (the optimised portfolio) wrapped round to
        # the first colour instead of its own.
        "resolve": {
            "scale": {"y": "independent", "color": "independent"},
            "axis": {"y": "independent"},
        },
        "params": [
            {
                "name": "nearest",
                "select": {
                    "type": "point",
                    "nearest": True,
                    "on": "pointerover",
                    "clear": "pointerout",
                    "fields": ["date"],
                },
                "views": ["days"],
            }
        ],
        "usermeta": _CANVAS,
        "datasets": {
            **top["datasets"],
            "drawn": bottom["datasets"]["drawn"],
            "hover": days,
        },
        # Each panel's key beside it: the weights' above the chart, the volatility's
        # under it.
        "key": _labelled(top["key"], "Weight"),
        "key_below": _labelled(bottom["key"], "Volatility"),
    }


def _labelled(key: str, label: str) -> str:
    """A chart key with a short label leading its row, so two keys over one chart say
    which panel each belongs to."""
    opening = '<div class="fe-readout">'
    return key.replace(
        opening, f'{opening}<span class="fe-readout-label">{html.escape(label)}</span>', 1
    )
