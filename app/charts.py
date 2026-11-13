"""Charts both the Portfolio page and an opened run draw, so a saved run looks the same
as the page that first showed it."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

# One colour per series, the same on every chart, so a line means the same thing
# wherever it is drawn: the optimised portfolio is always blue and the 50/50
# benchmark always grey, the equity index orange and the bond index green.
#
# The three hues are Atlassian's blue, orange and green, and were chosen by
# measurement rather than by eye: every pair stays apart under simulated red-green
# colour blindness and for full colour vision, and each clears 3:1 against both the
# light and the dark page (so one set serves both themes). The earlier purple for
# the equity index did not: beside the optimised portfolio's blue, on the three-line
# risk charts, the two were hard to tell apart even with full colour vision. Grey is
# kept for whatever a chart is read against: a benchmark, or what really happened.
OPTIMISED = "#0C66E4"
BENCHMARK = "#8590A2"
EQUITY_INDEX = "#E56910"
BOND_INDEX = "#22A06B"
BASELINE = "#CD519D"
_OTHERS = ("#AE2E24", "#946F00", "#943D73", "#216E4E")

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

#: Canvas, not Streamlit's default SVG: SVG makes every day's point a page element,
#: thousands per chart, which stalls scrolling and every redraw.
_CANVAS = {"embedOptions": {"renderer": "canvas"}}


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


#: What every index this app is given is called, and so says nothing where space is short.
_SUFFIX = " (total return)"


def short_name(name: str) -> str:
    """An index's name without "(total return)", for a legend, a tooltip or a narrow label."""
    return name.replace(_SUFFIX, "")


def _legend(**overrides) -> alt.Legend:
    """A legend under the chart, its labels shortened so three fit across a
    half-width chart: "(total return)" says nothing the reader needs here."""
    return alt.Legend(
        title=None,
        orient="bottom",
        labelLimit=0,
        labelExpr=f"replace(datum.label, '{_SUFFIX}', '')",
        **overrides,
    )


def colour_scale(names: list[str]) -> alt.Scale:
    """An explicit domain and range, so Vega cannot reassign colours by order."""
    return alt.Scale(domain=names, range=[series_colour(n, i) for i, n in enumerate(names)])


def weights_chart(
    renamed: pd.DataFrame,
    *,
    limits: tuple[float, float] | None = None,
    until: pd.Timestamp | None = None,
) -> alt.LayerChart:
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

    x = alt.X("date:T", title=None, axis=alt.Axis(tickCount="year", format="%Y"))

    def y(field: str, **kwargs) -> alt.Y:
        # Every layer states the same axis: layers that left it out fell back to
        # Vega's own ticks, and the merged axis dropped half the labels.
        return alt.Y(
            field,
            title=None,
            axis=alt.Axis(format="%", values=[0, 0.2, 0.4, 0.6, 0.8, 1]),
            scale=alt.Scale(domain=[0, 1]),
            **kwargs,
        )

    # The legend's swatches at full strength, not the bands' tint.
    legend = _legend(symbolOpacity=1)
    bands = (
        alt.Chart(long)
        .mark_area(interpolate="step-after", opacity=0.28)
        .encode(
            x=x,
            y=y("Weight:Q", stack="zero"),
            color=alt.Color("Asset:N", scale=colour_scale(names), sort=names, legend=legend),
            order=alt.Order("position:Q", sort="ascending"),
        )
    )
    first = names[0]
    boundary = (
        alt.Chart(wide)
        .mark_line(interpolate="step-after", strokeWidth=2, color=series_colour(first))
        .encode(x=x, y=y(f"{first}:Q"))
    )
    # A step line only answers a hover within a pixel of one of its corners, so an
    # unseen point per rebalance takes it: the nearest rebalance to the pointer is
    # marked and shows its date and every asset's weight.
    nearest = alt.selection_point(
        nearest=True, on="pointerover", clear="pointerout", empty=False, encodings=["x"]
    )
    points = (
        alt.Chart(wide)
        .mark_point(filled=True, size=50, color=series_colour(first))
        .encode(
            x=x,
            y=y(f"{first}:Q"),
            opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
            tooltip=[
                alt.Tooltip("date:T", title="From", format="%d %b %Y"),
                *(
                    alt.Tooltip(f"{name}:Q", title=short_name(name), format=".0%")
                    for name in names
                ),
            ],
        )
        .add_params(nearest)
    )
    layers = [bands, boundary, points]
    if limits is not None:
        bounds = pd.DataFrame({"Limit": sorted(set(limits))})
        layers.insert(
            2,
            alt.Chart(bounds)
            .mark_rule(strokeDash=[4, 4], strokeWidth=1, color=BENCHMARK)
            .encode(y=y("Limit:Q")),
        )
    return alt.layer(*layers).properties(height=280, usermeta=_CANVAS)


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
    keep = {0, len(wide) - 1}
    for name in names:
        values = wide[name].to_numpy(dtype=float)
        for start, end in zip(edges[:-1], edges[1:], strict=True):
            stretch = values[start:end]
            if np.isnan(stretch).all():
                continue  # the line has no figure anywhere in this stretch
            keep.add(start + int(np.nanargmin(stretch)))
            keep.add(start + int(np.nanargmax(stretch)))
    return wide.iloc[sorted(keep)]


def line_chart(
    paths: pd.DataFrame, value: str, *, percent: bool = True, title: str | None = None
) -> alt.LayerChart:
    """A line per column of ``paths`` on one time axis, in percent unless told it is a
    ratio. Hovering or tapping anywhere marks that day on every line and shows its date
    and each line's value.

    The chart is sent one row per day and unfolded into its lines in the browser, and
    a hover lands on one unseen target per day. A long chart is thinned to what can be
    seen (``_drawn``) and to as many hover targets as can be told apart
    (``MAX_HOVER_TARGETS``). It used to be sent a row per day per line with an unseen
    point on each. Over twenty years that is twelve thousand points a chart for the
    browser to lay out before it could draw: about three seconds of a frozen page for
    four charts, every time their figures changed."""
    names = [str(c) for c in paths.columns]
    wide = paths.set_axis(names, axis=1).rename_axis("date").reset_index()
    x = alt.X("date:T", title=None, axis=alt.Axis(tickCount="year", format="%Y"))
    y = alt.Y(
        f"{value}:Q", title=title or value, axis=alt.Axis(format="%" if percent else ".1f")
    )
    colour = alt.Color("Series:N", scale=colour_scale(names), sort=names, legend=_legend())
    figure = ".2%" if percent else ".2f"
    unfold = {"fold": names, "as_": ["Series", value]}
    lines = (
        alt.Chart(_drawn(wide, names))
        .transform_fold(**unfold)
        .mark_line()
        .encode(x=x, y=y, color=colour)
    )
    # Vega only shows a line's tooltip within a pixel or two of one of its days, too
    # fine to tap. So the day nearest the pointer, wherever in the chart that is, is
    # the one marked: a rule down it, a dot on each line, every line's value shown.
    nearest = alt.selection_point(
        nearest=True, on="pointerover", clear="pointerout", fields=["date"], empty=False
    )
    # No more targets than a chart has room to tell apart: finding which of four
    # thousand days is nearest the pointer was the rest of what the browser did with
    # each chart. A hover lands on the nearest day kept and shows that day's own
    # figures, read from the data and not off the drawn line.
    every = -(-len(wide) // MAX_HOVER_TARGETS) or 1
    kept = wide.iloc[::every]
    if len(wide) and kept.index[-1] != wide.index[-1]:
        kept = pd.concat([kept, wide.iloc[[-1]]])  # the latest day can always be read
    hovered = alt.Chart(kept)
    targets = (
        hovered.mark_point(opacity=0)
        .encode(
            x=x,
            tooltip=[
                alt.Tooltip("date:T", title="Date", format="%d %b %Y"),
                *(
                    alt.Tooltip(f"{name}:Q", title=short_name(name), format=figure)
                    for name in names
                ),
            ],
        )
        .add_params(nearest)
    )
    crosshair = (
        hovered.mark_rule(color=BENCHMARK, strokeWidth=1).encode(x=x).transform_filter(nearest)
    )
    dots = (
        hovered.transform_fold(**unfold)
        .transform_filter(nearest)
        .mark_point(filled=True, size=40, opacity=1)
        .encode(x=x, y=y, color=colour)
    )
    return alt.layer(lines, crosshair, dots, targets).properties(usermeta=_CANVAS)
