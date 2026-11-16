"""The market chart on Home: candles as a Vega-Lite chart, and the line above it."""

import pandas as pd

import market_chart
import tokens
from forecasting_engine.ingest.index_feed import Candles


def _candles(live: bool = True, period: str = "M") -> Candles:
    starts = pd.date_range("2026-08-01", periods=3, freq="MS")
    frame = pd.DataFrame(
        {
            "open": [7500.0, 7690.0, 7650.0],
            "high": [7820.0, 7790.0, 7700.0],
            "low": [7500.0, 7510.0, 7350.0],
            "close": [7686.0, 7650.0, 7418.2],
        },
        index=starts,
    )
    source = "Yahoo Finance" if live else "committed data"
    return Candles(frame, period, source, live, 7418.2, pd.Timestamp("2026-11-16 16:00"))


def test_each_candle_has_its_figures_its_move_and_whether_it_closed_up():
    table = market_chart.candles_table(_candles())

    assert table["up"].tolist() == [True, False, False]
    assert round(table["move"].iloc[0], 4) == round(7686 / 7500 - 1, 4)
    assert table["when"].tolist() == ["Aug 2026", "Sep 2026", "Oct 2026"]
    # The body fills part of its month, centred, leaving a gap to the next.
    assert (table["start"] < table["mid"]).all() and (table["mid"] < table["end"]).all()
    assert (table["end"] < table["start"].shift(-1)).iloc[:-1].all()


def test_a_quarter_and_a_week_are_named_as_such():
    quarters = market_chart.candles_table(_candles(period="Q"))
    weeks = market_chart.candles_table(_candles(period="W"))

    assert quarters["when"].iloc[0] == "Q3 2026"
    assert weeks["when"].iloc[0].startswith("w/c 01 Aug 2026")


def test_the_chart_shows_every_candles_figures_on_hover_and_zooms_in_the_browser():
    spec = market_chart.candle_chart(_candles())

    assert len(spec["datasets"]["candles"]) == 3
    hit = next(layer for layer in spec["layer"] if layer.get("name") == "candles_hit")
    titles = [field["title"] for field in hit["encoding"]["tooltip"]]
    assert titles == ["Period", "Open", "High", "Low", "Close", "Move"]
    zoom = next(p for p in spec["params"] if p["name"] == "zoom")
    assert zoom["bind"] == "scales" and zoom["select"]["encodings"] == ["x"]


def test_a_candle_that_closed_up_is_outlined_and_one_that_closed_down_filled():
    spec = market_chart.candle_chart(_candles())
    body = next(layer for layer in spec["layer"] if layer["mark"]["type"] == "bar")

    fill = body["encoding"]["fill"]
    assert fill["condition"]["value"] == tokens.BG and fill["value"] == tokens.FALL
    assert body["encoding"]["stroke"]["condition"]["value"] == tokens.RISE


def test_the_header_gives_the_latest_price_and_the_spans_move():
    header = market_chart.header(_candles(), "S&P 500 · ^GSPC")

    assert ">7,418.20<" in header
    assert "▼ 1.1%" in header  # 7,500.00 at the span's first open to 7,418.20
    assert "<b>" not in market_chart.header(_candles(), "S&P 500 <b>")


def test_the_line_under_the_chart_says_the_feed_is_live_and_when():
    line = market_chart.source(_candles())

    assert "Live · Yahoo Finance · 16 Nov 2026 16:00 New York" in line


def test_the_line_under_the_chart_says_when_the_committed_data_is_drawn_instead():
    line = market_chart.source(_candles(live=False))

    assert "From the committed data · to 16 Nov 2026" in line


def test_the_wheel_alone_scrolls_the_page_and_shift_with_it_zooms():
    spec = market_chart.candle_chart(_candles())
    zoom = next(p for p in spec["params"] if p["name"] == "zoom")

    assert zoom["select"]["zoom"] == "wheel![event.shiftKey]"


def test_the_time_axis_is_written_for_the_candles_period():
    monthly = market_chart.candle_chart(_candles())["encoding"]["x"]["axis"]
    daily = market_chart.candle_chart(_candles(period="D"))["encoding"]["x"]["axis"]

    assert (monthly["tickCount"], monthly["format"]) == ("year", "%Y")
    assert (daily["tickCount"], daily["format"]) == ("week", "%d %b")
