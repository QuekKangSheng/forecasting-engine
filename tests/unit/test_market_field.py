"""The market field on the Home page: months as candles, falls as arcs."""

import re

import pandas as pd
import pytest

import market_field
import tokens
from market_field import Candle, Fall


def _daily(values_by_month: dict[str, list[float]]) -> pd.Series:
    """Daily closes, each month's values on its first business days."""
    parts = [
        pd.Series(values, index=pd.bdate_range(f"{month}-01", periods=len(values)))
        for month, values in values_by_month.items()
    ]
    return pd.concat(parts)


def _months(closes: list[float], start: str = "2024-01-01") -> tuple[Candle, ...]:
    """A candle a month whose close is each of ``closes``, opening at the last close."""
    starts = pd.date_range(start, periods=len(closes), freq="MS")
    opens = [closes[0], *closes[:-1]]
    return tuple(
        Candle(month=m, open=o, high=max(o, c), low=min(o, c), close=c)
        for m, o, c in zip(starts, opens, closes, strict=True)
    )


def test_a_month_opens_at_the_previous_months_last_close():
    prices = _daily({"2024-01": [100, 104], "2024-02": [110, 108]})

    first, second = market_field.monthly_candles(prices)

    assert (second.open, second.close) == (104, 108)
    assert first.open == 100  # nothing before it: its own first close


def test_a_months_high_and_low_take_in_its_open():
    # February never closes as low as January's last close, where it opened.
    prices = _daily({"2024-01": [100, 90], "2024-02": [95, 99]})

    february = market_field.monthly_candles(prices)[1]

    assert (february.low, february.high) == (90, 99)


def test_a_month_that_ends_where_it_opened_counts_as_up():
    assert Candle(pd.Timestamp("2024-01-01"), 5, 6, 4, 5).up
    assert not Candle(pd.Timestamp("2024-01-01"), 5, 6, 4, 4.9).up


def test_only_the_latest_months_are_kept_and_missing_prices_are_left_out():
    prices = _daily({f"2024-{m:02d}": [float(m), float("nan")] for m in range(1, 13)})

    candles = market_field.monthly_candles(prices, months=3)

    assert [c.month.month for c in candles] == [10, 11, 12]
    assert [c.close for c in candles] == [10, 11, 12]


def test_no_prices_make_no_candles():
    assert market_field.monthly_candles(pd.Series(dtype=float)) == ()


def test_the_deepest_fall_runs_from_its_peak_to_its_trough():
    (fall,) = market_field.largest_falls([100, 120, 90, 96, 130], count=1)

    assert fall == Fall(peak=1, trough=2, depth=pytest.approx(-0.25))


def test_falls_share_no_month_and_come_deepest_first():
    closes = [100, 80, 100, 120, 60, 130, 117]

    falls = market_field.largest_falls(closes, count=3)

    assert [(f.peak, f.trough) for f in falls] == [(3, 4), (0, 1), (5, 6)]
    assert [round(f.depth, 2) for f in falls] == [-0.5, -0.2, -0.1]


def test_a_fall_smaller_than_the_least_worth_drawing_is_left_out():
    assert market_field.largest_falls([100, 97, 110], least=0.05) == ()


def test_a_series_that_only_rises_has_no_fall():
    assert market_field.largest_falls([1, 2, 3, 4]) == ()


def test_the_field_draws_a_candle_for_every_month():
    field = market_field.field(_months([100, 110, 105, 120, 118, 125, 140, 133]), "S&P 500")

    assert field.count('class="fe-candle"') == 8


def test_a_month_that_closed_down_is_filled_and_one_that_closed_up_is_outlined():
    field = market_field.field(_months([100, 110, 105, 120, 118, 125, 140, 133]), "S&P 500")

    candles = re.findall(r'<g class="fe-candle".*?</g>', field)
    up, down = candles[1], candles[2]  # 100 -> 110, then 110 -> 105
    assert f'stroke="{tokens.RISE}"' in up and 'fill-opacity="0.2"' in up
    assert f'fill="{tokens.FALL}"/>' in down and "fill-opacity" not in down.split("<rect")[1]


def test_an_older_month_is_drawn_smaller_than_a_newer_one():
    field = market_field.field(_months([100.0] * 12), "S&P 500")

    candles = re.findall(r'<g class="fe-candle".*?</g>', field)
    widths = [float(re.search(r'<rect [^>]*width="([^"]+)"', c).group(1)) for c in candles]
    assert widths == sorted(widths) and widths[0] < widths[-1]


def test_the_field_says_in_words_what_it_draws():
    field = market_field.field(_months([100, 120, 90, 96, 99, 104, 130, 133]), "S&P 500")

    label = re.search(r'aria-label="([^"]+)"', field).group(1)
    assert "S&amp;P 500, one candle a month from January 2024 to August 2024" in label
    assert "last closed at 133.00" in label
    assert "largest fall between month-end closes was -25.0%, from February 2024" in label


def test_each_fall_is_an_arc_with_its_size_written_on_it():
    field = market_field.field(_months([100, 120, 90, 96, 99, 104, 130, 133]), "S&P 500")

    assert field.count('class="fe-field-arc"') == 1
    assert "-25.0%" in field


def test_the_newest_months_close_and_move_are_written_out():
    field = market_field.field(_months([100, 110, 105, 120, 118, 125, 140, 133]), "S&P 500")

    assert "S&amp;P 500 · Aug 2024" in field
    assert ">133.00<" in field
    assert "-5.0% this month" in field


def test_the_index_name_is_escaped():
    field = market_field.field(_months([100.0] * 8), "<script>alert(1)</script>")

    assert "<script>" not in field


def test_the_field_is_one_line_so_markdown_cannot_break_it_up():
    field = market_field.field(_months([100, 110, 105, 120, 118, 125, 140, 133]), "S&P 500")

    assert "\n" not in field


def test_a_flat_index_is_still_drawn():
    field = market_field.field(_months([100.0] * 8), "S&P 500")

    assert field.count('class="fe-candle"') == 8 and "nan" not in field


def test_too_few_months_leave_the_floor_empty_and_say_why():
    field = market_field.field(_months([100, 110, 105]), "S&P 500")

    assert "fe-candle" not in field
    assert f"Fewer than {market_field.MIN_MONTHS} months of S&amp;P 500" in field


def test_with_no_data_the_floor_is_empty_and_says_so():
    field = market_field.empty_field()

    assert "fe-candle" not in field and "No data committed" in field
    assert 'role="img"' in field
