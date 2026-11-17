"""The market chart on Home: every candle size fetched at once, and the page that draws it."""

import json
import re

import pandas as pd

import market_chart
from forecasting_engine.ingest import index_feed
from forecasting_engine.ingest.index_feed import Candles, IndexFeedError


def _candles(live: bool = True, period: str = "M", volume=(4.1e9, None, 3.9e9)) -> Candles:
    starts = pd.date_range("2026-08-01", periods=3, freq="MS")
    frame = pd.DataFrame(
        {
            "open": [7500.0, 7690.0, 7650.0],
            "high": [7820.0, 7790.0, 7700.0],
            "low": [7500.0, 7510.0, 7350.0],
            "close": [7686.0, 7650.0, 7418.2],
            "volume": [float("nan") if v is None else v for v in volume],
        },
        index=starts,
    )
    source = "Yahoo Finance" if live else "committed data"
    return Candles(frame, period, source, live, 7418.2, pd.Timestamp("2026-11-16 16:00"))


def _data(page: str) -> dict:
    return json.loads(re.search(r"const DATA = (.*?);\n", page).group(1))


def test_every_candle_size_is_asked_of_the_feed_and_one_it_refused_is_left_out(monkeypatch):
    asked = []

    def fetch(interval, **kwargs):
        asked.append(interval)
        if interval == "1D":
            raise IndexFeedError("the live feed did not answer (timed out)")
        return _candles()

    monkeypatch.setattr(index_feed, "fetch", fetch)

    answered, reason = market_chart.live()

    assert sorted(asked) == sorted(index_feed.INTERVALS)
    assert sorted(answered) == ["1M", "1W", "3M"]
    assert reason == "the live feed did not answer (timed out)"


def test_each_candle_goes_to_the_page_as_a_compact_row_with_its_volume():
    first, second, _ = market_chart.rows(_candles())

    assert first == ["2026-08-01", 7500.0, 7820.0, 7500.0, 7686.0, 4100000000]
    assert second[-1] is None  # no volume known


def test_each_size_carries_its_history_its_first_view_and_where_it_is_from():
    data = market_chart.payload({"1M": _candles(), "1W": _candles(live=False, period="W")})

    assert len(data["1M"]["rows"]) == 3
    assert data["1M"]["visible"] == market_chart.VISIBLE["1M"]
    assert data["1M"]["where"] == "Live · Yahoo Finance · 16 Nov 2026 16:00 New York"
    assert data["1W"]["where"] == "From the committed data · to 16 Nov 2026"
    assert (data["1W"]["period"], data["1W"]["unit"], data["1W"]["live"]) == ("W", "week", False)


def test_the_page_holds_every_size_so_changing_it_needs_no_rerun():
    page = market_chart.chart_html({"1D": _candles(period="D"), "1W": _candles()}, "S&P 500")

    assert sorted(_data(page)) == ["1D", "1W"]
    assert 'data-span="1D" aria-pressed="true"' in page
    assert 'data-span="1W" aria-pressed="false"' in page
    assert market_chart.LIBRARY in page


def test_the_first_size_falls_back_when_the_default_is_missing():
    page = market_chart.chart_html({"1W": _candles()}, "S&P 500")

    assert 'data-span="1W" aria-pressed="true"' in page
    assert 'let span = "1W";' in page


def test_it_opens_on_the_latest_stretch_and_pans_back_through_the_rest():
    page = market_chart.chart_html({"1M": _candles()}, "S&P 500")

    assert "setVisibleLogicalRange({ from: Math.max(0, bars.length - n)" in page
    assert "handleScroll: { mouseWheel: true, pressedMouseMove: true" in page
    assert "handleScale: { mouseWheel: true" in page
    assert '$("now").addEventListener("click", latest)' in page


def test_the_move_above_is_over_what_is_in_view_and_follows_the_view():
    page = market_chart.chart_html({"1M": _candles()}, "S&P 500")

    assert "subscribeVisibleLogicalRangeChange(inView)" in page
    assert "% in view" in page


def test_volume_stands_under_the_candles():
    page = market_chart.chart_html({"1M": _candles()}, "S&P 500")

    assert 'addHistogramSeries({ priceScaleId: "volume"' in page
    assert "scaleMargins: { top: 0.8, bottom: 0 }" in page


def test_a_candle_that_closed_up_is_outlined_and_one_that_closed_down_filled():
    page = market_chart.chart_html({"1M": _candles()}, "S&P 500")

    assert 'upColor: "rgba(0,0,0,0)", downColor: C.fall' in page
    assert "borderUpColor: C.rise" in page


def test_a_hover_shows_the_candles_figures_in_a_box_beside_the_pointer():
    page = market_chart.chart_html({"1M": _candles()}, "S&P 500")

    assert '<div class="tip" id="tip" role="tooltip"></div>' in page
    for label in ('"Open"', '"High"', '"Low"', '"Close"', '"Open → close"', '"Volume"'):
        assert f"row({label}" in page
    assert "row(`vs last ${d.unit}`" in page
    assert "showTip(param, d);" in page


def test_nothing_in_the_data_or_the_name_can_break_out_of_the_page():
    page = market_chart.chart_html({"1M": _candles()}, "</script><script>alert(1)</script>")

    assert "<script>alert(1)" not in page
    assert market_chart._script_safe({"x": "</script>"}) == r'{"x":"<\/script>"}'


def test_the_chart_rests_in_3d_and_lies_flat_under_the_pointer():
    page = market_chart.chart_html({"1M": _candles()}, "S&P 500")

    assert ".tilted .plane { transform: rotateX(" in page
    flat = ".tilted .stage:hover .plane, .tilted .stage:focus-within .plane { transform: none;"
    assert flat in page
    assert 'id="depth" aria-pressed="true"' in page
