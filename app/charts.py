"""Charts both the Portfolio page and an opened run draw, so a saved run looks the same
as the page that first showed it."""

from __future__ import annotations

import altair as alt
import pandas as pd

# One colour per series, the same on every chart, so a line means the same thing
# wherever it is drawn: the optimised portfolio is always blue and the 50/50
# benchmark always grey, the equity index purple and the bond index teal. The
# values are Atlassian's chart colours, readable on both the light and dark theme.
OPTIMISED = "#0C66E4"
BENCHMARK = "#8590A2"
EQUITY_INDEX = "#6E5DC6"
BOND_INDEX = "#1D7F8C"
NO_FORECASTS = "#C25100"
_OTHERS = ("#AE2E24", "#946F00", "#943D73", "#216E4E")

_FIXED = {
    "Optimised": OPTIMISED,
    "Equal-weight benchmark": BENCHMARK,
    # The Models page's direction test: the strategy that acts on the forecast is
    # drawn like the optimised portfolio, the passive one like the benchmark.
    "Long/cash strategy": OPTIMISED,
    "Buy and hold": BENCHMARK,
}


def series_colour(name: str, position: int = 0) -> str:
    """The colour ``name`` is drawn in on every chart. Index names come from the
    data, so they are matched by what they contain rather than listed."""
    if name in _FIXED:
        return _FIXED[name]
    lowered = name.lower()
    if "no forecast" in lowered:
        return NO_FORECASTS
    if "s&p" in lowered or "spx" in lowered or "equity" in lowered:
        return EQUITY_INDEX
    if "bond" in lowered or "agg" in lowered or "legatruu" in lowered or "lbustruu" in lowered:
        return BOND_INDEX
    return _OTHERS[position % len(_OTHERS)]


def _legend() -> alt.Legend:
    """A legend under the chart, its labels shortened so three fit across a
    half-width chart: "(total return)" says nothing the reader needs here."""
    return alt.Legend(
        title=None,
        orient="bottom",
        labelLimit=0,
        labelExpr="replace(datum.label, ' (total return)', '')",
    )


def colour_scale(names: list[str]) -> alt.Scale:
    """An explicit domain and range, so Vega cannot reassign colours by order."""
    return alt.Scale(domain=names, range=[series_colour(n, i) for i, n in enumerate(names)])


def weights_chart(renamed: pd.DataFrame) -> alt.Chart:
    """Each named asset's share of the portfolio over time, as stacked areas that
    step at each rebalance: the weights hold between rebalances, so a step is what
    actually happened, and a time axis shows the years rather than 99 date labels."""
    names = [str(c) for c in renamed.columns]
    long = renamed.rename_axis("date").reset_index().melt(
        "date", var_name="Asset", value_name="Weight"
    )
    return (
        alt.Chart(long)
        .mark_area(interpolate="step-after", opacity=0.9)
        .encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(tickCount="year", format="%Y")),
            y=alt.Y(
                "Weight:Q",
                stack="zero",
                title=None,
                axis=alt.Axis(format="%", values=[0, 0.2, 0.4, 0.6, 0.8, 1]),
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color(
                "Asset:N",
                scale=colour_scale(names),
                sort=names,
                legend=_legend(),
            ),
            order=alt.Order("Asset:N", sort="ascending"),
            tooltip=[
                alt.Tooltip("date:T", title="Rebalance", format="%d %b %Y"),
                "Asset:N",
                alt.Tooltip("Weight:Q", format=".0%"),
            ],
        )
        .properties(height=320)
    )


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
    # Canvas, not Streamlit's default SVG: SVG makes every day's point a page element,
    # thousands per chart, which stalls scrolling and every redraw.
    canvas = {"embedOptions": {"renderer": "canvas"}}
    return (lines.mark_line() + points).properties(usermeta=canvas)
