"""The charts' colours and the weights chart's shape, read from the specs they build."""

import numpy as np
import pandas as pd

import charts

EQUITY, BOND = "S&P 500 (total return)", "US Aggregate Bond (total return)"


def _weights(names=(EQUITY, BOND)) -> pd.DataFrame:
    dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-28"])
    return pd.DataFrame({names[0]: [0.2, 0.5, 0.8], names[1]: [0.8, 0.5, 0.2]}, index=dates)


def _spec(chart) -> dict:
    return chart.to_dict()


def _rows(spec: dict, column: str) -> list[dict]:
    """The dataset that has ``column``: a layered chart keeps each layer's data by name."""
    return next(rows for rows in spec["datasets"].values() if column in rows[0])


# --- colours ---------------------------------------------------------------------------


def test_a_series_is_one_colour_wherever_it_is_drawn():
    assert charts.series_colour("Optimised") == charts.OPTIMISED
    assert charts.series_colour("Long/cash strategy") == charts.OPTIMISED
    assert charts.series_colour("Equal-weight benchmark") == charts.BENCHMARK
    assert charts.series_colour(EQUITY) == charts.series_colour("SPX Index") == charts.EQUITY_INDEX
    assert charts.series_colour(BOND) == charts.series_colour("LBUSTRUU Index") == charts.BOND_INDEX


def test_the_lines_a_chart_draws_together_never_share_a_colour():
    together = (
        (EQUITY, BOND, "Optimised"),  # volatility, drawdown, rolling Sharpe
        ("Optimised", "Equal-weight benchmark"),  # cumulative return
        ("Long/cash strategy", "Buy and hold"),  # the Models page's direction check
        ("Realised", "Forecast", "Baseline"),  # a recorded fit's forecasts
    )

    for names in together:
        colours = [charts.series_colour(name) for name in names]
        assert len(set(colours)) == len(names), names


def test_an_index_with_an_unknown_name_still_gets_a_colour_of_its_own():
    known = {charts.OPTIMISED, charts.BENCHMARK, charts.EQUITY_INDEX, charts.BOND_INDEX}

    first, second = charts.series_colour("Gold", 0), charts.series_colour("Copper", 1)

    assert first != second and not {first, second} & known


def test_an_indexs_name_is_shortened_where_space_is_tight():
    assert charts.short_name(EQUITY) == "S&P 500"
    assert charts.short_name("VIX") == "VIX"


# --- the weights chart -----------------------------------------------------------------


def test_weights_are_tinted_bands_with_the_first_assets_weight_as_a_line():
    spec = _spec(charts.weights_chart(_weights()))

    bands, boundary, points = spec["layer"]
    assert (bands["mark"]["type"], boundary["mark"]["type"]) == ("area", "line")
    assert points["mark"]["type"] == "point"
    # Tinted, so the gridlines behind show through and a weight can be read off.
    assert bands["mark"]["opacity"] < 0.5
    assert boundary["encoding"]["y"]["field"] == EQUITY
    assert boundary["mark"]["color"] == charts.EQUITY_INDEX
    scale = bands["encoding"]["color"]["scale"]
    assert dict(zip(scale["domain"], scale["range"], strict=True)) == {
        EQUITY: charts.EQUITY_INDEX,
        BOND: charts.BOND_INDEX,
    }


def test_every_layer_draws_the_same_axis_so_no_label_is_dropped():
    spec = _spec(charts.weights_chart(_weights(), limits=(0.2, 0.8)))

    axes = [layer["encoding"]["y"]["axis"] for layer in spec["layer"]]

    assert all(axis == axes[0] for axis in axes)
    assert axes[0]["values"] == [0, 0.2, 0.4, 0.6, 0.8, 1]


def test_the_first_asset_is_stacked_at_the_bottom_whatever_the_assets_are_called():
    # Alphabetically the bond would come first; the line is the equity weight, so
    # equity has to be the band beneath it.
    names = ("Zeta equity", "Alpha bond")

    spec = _spec(charts.weights_chart(_weights(names)))

    rows = _rows(spec, "position")
    assert {row["Asset"]: row["position"] for row in rows} == {"Zeta equity": 0, "Alpha bond": 1}
    assert spec["layer"][0]["encoding"]["order"]["field"] == "position"


def test_weight_limits_are_drawn_as_dashed_lines_only_when_given():
    without = _spec(charts.weights_chart(_weights()))
    with_limits = _spec(charts.weights_chart(_weights(), limits=(0.2, 0.8)))

    assert "rule" not in [layer["mark"]["type"] for layer in without["layer"]]
    (rule,) = [layer for layer in with_limits["layer"] if layer["mark"]["type"] == "rule"]
    assert rule["mark"]["strokeDash"]
    assert [row["Limit"] for row in _rows(with_limits, "Limit")] == [0.2, 0.8]


def test_the_last_allocation_is_drawn_held_to_the_day_given():
    weights = _weights()
    until = pd.Timestamp("2024-04-30")

    spec = _spec(charts.weights_chart(weights, until=until))

    rows = [row for row in _rows(spec, "position") if row["Asset"] == EQUITY]
    assert len(rows) == len(weights) + 1
    assert pd.Timestamp(rows[-1]["date"]) == until
    assert rows[-1]["Weight"] == weights[EQUITY].iloc[-1]
    assert len(weights) == 3, "the caller's frame is left as it was"


def test_a_day_no_later_than_the_last_rebalance_adds_nothing():
    weights = _weights()

    spec = _spec(charts.weights_chart(weights, until=weights.index[-1]))

    assert len([r for r in _rows(spec, "position") if r["Asset"] == EQUITY]) == len(weights)


def test_hovering_a_rebalance_gives_its_date_and_every_assets_weight():
    spec = _spec(charts.weights_chart(_weights()))

    tooltip = spec["layer"][-1]["encoding"]["tooltip"]

    assert [t["field"] for t in tooltip] == ["date", EQUITY, BOND]
    assert [t.get("title") for t in tooltip][1:] == ["S&P 500", "US Aggregate Bond"]


# --- line charts -----------------------------------------------------------------------


def _paths(days: int) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=days)
    steps = pd.Series(range(days), index=dates, dtype=float)
    return pd.DataFrame({"Optimised": steps / 100, "Equal-weight benchmark": steps / 200})


def test_a_line_chart_is_sent_a_row_per_day_and_unfolded_into_its_lines():
    paths = _paths(30)

    spec = _spec(charts.line_chart(paths, "Cumulative return"))

    lines = spec["layer"][0]
    assert lines["mark"]["type"] == "line"
    assert lines["transform"] == [
        {"fold": ["Optimised", "Equal-weight benchmark"], "as": ["Series", "Cumulative return"]}
    ]
    (rows,) = spec["datasets"].values()
    assert len(rows) == len(paths) and set(rows[0]) == {"date", *paths.columns}


def test_a_hover_marks_the_day_on_every_line_and_shows_each_lines_value():
    spec = _spec(charts.line_chart(_paths(30), "Cumulative return"))

    marks = [layer["mark"]["type"] for layer in spec["layer"]]
    assert marks == ["line", "rule", "point", "point"]
    tooltip = spec["layer"][-1]["encoding"]["tooltip"]
    assert [t["field"] for t in tooltip] == ["date", "Optimised", "Equal-weight benchmark"]
    assert {t["format"] for t in tooltip[1:]} == {".2%"}
    # The targets a hover lands on are there but never seen.
    assert spec["layer"][-1]["mark"]["opacity"] == 0


def test_a_ratio_is_shown_as_a_number_not_a_percentage():
    spec = _spec(charts.line_chart(_paths(30), "Rolling Sharpe", percent=False))

    assert {t["format"] for t in spec["layer"][-1]["encoding"]["tooltip"][1:]} == {".2f"}
    assert spec["layer"][0]["encoding"]["y"]["axis"]["format"] == ".1f"


def _layer_rows(spec: dict, layer: int) -> list[dict]:
    return spec["datasets"][spec["layer"][layer]["data"]["name"]]


def _wandering(days: int) -> pd.DataFrame:
    """Two lines that wander like returns do, so their peaks fall on different days."""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2006-01-02", periods=days)
    steps = rng.normal(0, 0.01, (days, 2))
    return pd.DataFrame(steps.cumsum(axis=0), index=dates, columns=["Optimised", "Buy and hold"])


def test_a_long_chart_takes_a_hover_on_no_more_days_than_it_can_show():
    paths = _wandering(charts.MAX_HOVER_TARGETS * 7 + 3)

    spec = _spec(charts.line_chart(paths, "Cumulative return"))

    hovered = _layer_rows(spec, -1)
    assert len(hovered) <= charts.MAX_HOVER_TARGETS + 1
    # The first and the latest day can always be read, whatever was thinned between,
    # and a hover shows a day's own figures, not ones read off the drawn line.
    assert pd.Timestamp(hovered[0]["date"]) == paths.index[0]
    assert pd.Timestamp(hovered[-1]["date"]) == paths.index[-1]
    assert hovered[-1]["Optimised"] == paths["Optimised"].iloc[-1]
    # The rule and the dots a hover draws come from those same days.
    assert spec["layer"][1]["data"] == spec["layer"][2]["data"] == spec["layer"][-1]["data"]


def test_a_chart_short_enough_to_show_is_drawn_from_every_day():
    paths = _wandering(charts.MAX_DRAWN_ROWS)

    spec = _spec(charts.line_chart(paths, "Cumulative return"))

    assert len(_layer_rows(spec, 0)) == len(paths)


def test_a_long_chart_is_drawn_from_fewer_days_that_keep_every_peak_and_trough():
    paths = _wandering(5_000)
    paths.iloc[2_345, 0] = 9.0  # one day's spike, far outside everything around it
    stretches = charts.MAX_DRAWN_ROWS // 4

    spec = _spec(charts.line_chart(paths, "Cumulative return"))

    drawn = pd.DataFrame(_layer_rows(spec, 0))
    drawn["date"] = pd.to_datetime(drawn["date"])
    drawn = drawn.set_index("date")
    assert len(drawn) <= charts.MAX_DRAWN_ROWS
    assert drawn.index[0] == paths.index[0] and drawn.index[-1] == paths.index[-1]
    assert drawn.index.is_monotonic_increasing and drawn.index.is_unique
    # Every row drawn is a real day at its real values.
    pd.testing.assert_frame_equal(drawn, paths.loc[drawn.index], check_freq=False)
    assert drawn.loc[paths.index[2_345], "Optimised"] == 9.0
    # In each stretch of days, each line's highest and lowest day is among those drawn.
    edges = np.linspace(0, len(paths), stretches + 1).astype(int)
    for start, end in zip(edges[:-1], edges[1:], strict=True):
        stretch = paths.iloc[start:end]
        kept = drawn.loc[stretch.index[0] : stretch.index[-1]]
        assert (kept.max() == stretch.max()).all() and (kept.min() == stretch.min()).all()


def test_a_line_with_no_figure_yet_is_thinned_like_any_other():
    """A rolling Sharpe has nothing to show until its window has passed."""
    paths = _wandering(4_000)
    paths.iloc[:300, 1] = float("nan")

    spec = _spec(charts.line_chart(paths, "Rolling Sharpe", percent=False))

    drawn = pd.DataFrame(_layer_rows(spec, 0))
    assert len(drawn) <= charts.MAX_DRAWN_ROWS
    assert drawn["Buy and hold"].notna().any()
    assert drawn["Optimised"].max() == paths["Optimised"].max()
    assert drawn["Buy and hold"].min() == paths["Buy and hold"].min()
