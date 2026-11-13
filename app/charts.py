"""Charts both the Portfolio page and an opened run draw, so a saved run looks the same
as the page that first showed it."""

from __future__ import annotations

import altair as alt
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


def line_chart(
    paths: pd.DataFrame, value: str, *, percent: bool = True, title: str | None = None
) -> alt.LayerChart:
    """A line per column of ``paths`` on one time axis, in percent unless told it is a
    ratio. Hovering or tapping anywhere marks the nearest day on a line and shows its
    date and value."""
    long = paths.rename_axis("date").reset_index().melt("date", var_name="Series", value_name=value)
    nearest = alt.selection_point(nearest=True, on="pointerover", clear="pointerout", empty=False)
    lines = alt.Chart(long).encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(tickCount="year", format="%Y")),
        y=alt.Y(
            f"{value}:Q", title=title or value, axis=alt.Axis(format="%" if percent else ".1f")
        ),
        color=alt.Color(
            "Series:N",
            scale=colour_scale([str(c) for c in paths.columns]),
            legend=_legend(),
        ),
    )
    # Vega only shows a line's tooltip within a pixel or two of one of its days,
    # too fine to tap; invisible points with a nearest-day selection catch any hover.
    points = (
        lines.mark_point(filled=True)
        .encode(
            opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
            tooltip=[
                alt.Tooltip("date:T", title="Date", format="%d %b %Y"),
                "Series:N",
                alt.Tooltip(f"{value}:Q", format=".2%" if percent else ".2f"),
            ],
        )
        .add_params(nearest)
    )
    return (lines.mark_line() + points).properties(usermeta=_CANVAS)
