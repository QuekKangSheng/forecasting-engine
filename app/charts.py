"""Charts both the Portfolio page and an opened run draw, so a saved run looks the same
as the page that first showed it."""

from __future__ import annotations

import altair as alt
import pandas as pd


def weights_chart(renamed: pd.DataFrame) -> alt.Chart:
    """A stacked bar per rebalance of ``renamed``, one column per named asset."""
    long = renamed.rename_axis("date").reset_index().melt(
        "date", var_name="Asset", value_name="Weight"
    )
    return (
        alt.Chart(long)
        .mark_bar()
        .encode(
            # Ordinal, not temporal: rebalances are sparse relative to the full date
            # range, so a time axis squeezes every bar into a sliver with long empty
            # gaps between them. Each rebalance instead gets an equal-width slot.
            x=alt.X("yearmonthdate(date):O", title="Rebalance date", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("Weight:Q", stack="zero", axis=alt.Axis(format="%")),
            color=alt.Color("Asset:N", legend=alt.Legend(title=None)),
            tooltip=["date:T", "Asset:N", alt.Tooltip("Weight:Q", format=".0%")],
        )
        .properties(height=600)
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
        x=alt.X("date:T", title="Date", axis=alt.Axis(tickCount="year")),
        y=alt.Y(
            f"{value}:Q", title=title or value, axis=alt.Axis(format="%" if percent else ".1f")
        ),
        color=alt.Color("Series:N", legend=alt.Legend(title=None, orient="bottom", labelLimit=0)),
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
