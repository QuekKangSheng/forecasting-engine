"""The S&P 500 as it trades: the live feed's answer read, and candles from closes."""

import io
import json
from urllib.error import URLError

import pandas as pd
import pytest

from forecasting_engine.ingest import index_feed
from forecasting_engine.ingest.index_feed import IndexFeedError

NEW_YORK = "America/New_York"


def _payload(stamps, opens, highs, lows, closes, *, price=None, at=None) -> dict:
    """The chart service's answer, as far as it is read."""
    meta = {"symbol": "^GSPC", "exchangeTimezoneName": NEW_YORK}
    if price is not None:
        meta |= {"regularMarketPrice": price, "regularMarketTime": at}
    return {
        "chart": {
            "result": [
                {
                    "meta": meta,
                    "timestamp": stamps,
                    "indicators": {
                        "quote": [{"open": opens, "high": highs, "low": lows, "close": closes}]
                    },
                }
            ],
            "error": None,
        }
    }


def _at(text: str) -> int:
    """Seconds since 1970 for a New York wall-clock time."""
    return int(pd.Timestamp(text, tz=NEW_YORK).timestamp())


def test_a_days_candle_is_read_in_new_york_time():
    payload = _payload(
        [_at("2026-11-13 09:30"), _at("2026-11-16 09:30")],
        [7400.0, 7390.0],
        [7450.0, 7430.0],
        [7380.0, 7370.0],
        [7410.0, 7418.2],
        price=7418.2,
        at=_at("2026-11-16 16:00"),
    )

    candles = index_feed.parse_chart(payload, "1d")

    assert list(candles.frame.index) == [pd.Timestamp("2026-11-13"), pd.Timestamp("2026-11-16")]
    prices = candles.frame.loc["2026-11-16", ["open", "high", "low", "close"]]
    assert prices.tolist() == [7390.0, 7430.0, 7370.0, 7418.2]
    assert candles.live and candles.source == "Yahoo Finance"
    assert (candles.last, candles.as_of) == (7418.2, pd.Timestamp("2026-11-16 16:00"))


def test_a_day_with_no_trade_is_left_out():
    payload = _payload(
        [_at("2026-11-13 09:30"), _at("2026-11-16 09:30")],
        [7400.0, None],
        [7450.0, None],
        [7380.0, None],
        [7410.0, None],
    )

    candles = index_feed.parse_chart(payload, "1d")

    assert len(candles.frame) == 1
    assert candles.last == 7410.0  # no live price sent: the last close


def test_the_month_still_running_is_filed_under_its_month_and_kept_once():
    # The service dates the running month at its latest trade, beside its start.
    payload = _payload(
        [_at("2026-10-01 09:30"), _at("2026-11-02 09:30"), _at("2026-11-16 16:00")],
        [7600.0, 7700.0, 7700.0],
        [7800.0, 7750.0, 7760.0],
        [7550.0, 7350.0, 7350.0],
        [7700.0, 7500.0, 7418.2],
    )

    candles = index_feed.parse_chart(payload, "1mo")

    assert list(candles.frame.index) == [pd.Timestamp("2026-10-01"), pd.Timestamp("2026-11-01")]
    assert candles.frame["close"].tolist() == [7700.0, 7418.2]


@pytest.mark.parametrize(
    "payload",
    [
        {"chart": {"result": None, "error": {"code": "Not Found"}}},
        {"chart": {"result": []}},
        {"unexpected": True},
        _payload([_at("2026-11-16 09:30")], [None], [None], [None], [None]),
    ],
)
def test_an_answer_that_cannot_be_drawn_is_a_feed_error(payload):
    with pytest.raises(IndexFeedError):
        index_feed.parse_chart(payload, "1d")


def _opener(body: bytes | Exception, seen: list):
    def opener(request, timeout):
        seen.append((request.full_url, request.get_header("User-agent"), timeout))
        if isinstance(body, Exception):
            raise body
        return io.BytesIO(body)

    return opener


def test_fetch_asks_for_the_candles_history_by_its_dates_and_reads_the_answer():
    seen = []
    body = json.dumps(
        _payload([_at("2026-11-02 09:30")], [7700.0], [7760.0], [7350.0], [7418.2])
    ).encode()
    now = _at("2026-11-16 16:00")

    candles = index_feed.fetch("1M", opener=_opener(body, seen), now=now)

    url, agent, timeout = seen[0]
    # Every month there is: from 1970, not "max", which the service turns into quarters.
    assert f"chart/%5EGSPC?period1=0&period2={now}&interval=1mo" in url
    assert agent and timeout <= 10
    assert candles.period == "M" and candles.frame["close"].iloc[-1] == 7418.2


def test_daily_candles_reach_back_ten_years():
    seen = []
    body = json.dumps(
        _payload([_at("2026-11-16 09:30")], [7400.0], [7450.0], [7380.0], [7418.2])
    ).encode()
    now = _at("2026-11-16 16:00")

    index_feed.fetch("1D", opener=_opener(body, seen), now=now)

    assert f"period1={now - 3_653 * 86_400}&period2={now}&interval=1d" in seen[0][0]


def test_volume_is_read_where_the_service_sends_it():
    payload = _payload(
        [_at("2026-11-13 09:30"), _at("2026-11-16 09:30")],
        [7400.0, 7390.0],
        [7450.0, 7430.0],
        [7380.0, 7370.0],
        [7410.0, 7418.2],
    )
    payload["chart"]["result"][0]["indicators"]["quote"][0]["volume"] = [4.1e9, None]

    candles = index_feed.parse_chart(payload, "1d")

    assert candles.frame["volume"].iloc[0] == 4.1e9
    assert pd.isna(candles.frame["volume"].iloc[1])


@pytest.mark.parametrize("failure", [URLError("offline"), TimeoutError("slow"), OSError("reset")])
def test_a_feed_that_does_not_answer_is_a_feed_error(failure):
    with pytest.raises(IndexFeedError, match="did not answer"):
        index_feed.fetch("1D", opener=_opener(failure, []))


def test_an_answer_that_is_not_json_is_a_feed_error():
    with pytest.raises(IndexFeedError):
        index_feed.fetch("1D", opener=_opener(b"<html>verify you are human</html>", []))


def test_an_unknown_candle_size_is_refused_before_anything_is_asked():
    seen = []
    with pytest.raises(ValueError, match="unknown interval"):
        index_feed.fetch("2W", opener=_opener(b"{}", seen))
    assert not seen


def test_candles_from_closes_open_at_the_last_close_before_them():
    days = pd.bdate_range("2024-01-01", "2024-03-29")
    closes = pd.Series(range(100, 100 + len(days)), index=days, dtype=float)

    candles = index_feed.from_closes(closes, "1M")

    january, february = candles.frame.iloc[0], candles.frame.iloc[1]
    assert january["open"] == 100  # nothing before it: its own first close
    assert february["open"] == january["close"]
    assert february["high"] >= max(february["open"], february["close"])
    assert not candles.live and candles.source == "committed data"
    assert (candles.last, candles.as_of) == (closes.iloc[-1], days[-1])


def test_candles_from_closes_cover_every_close_at_the_candles_size():
    days = pd.bdate_range("2020-01-01", periods=600)
    closes = pd.Series(1.0, index=days)

    daily, quarterly = index_feed.from_closes(closes, "1D"), index_feed.from_closes(closes, "3M")

    assert daily.period == "D" and len(daily.frame) == 600
    assert quarterly.period == "Q" and len(quarterly.frame) == days.to_period("Q").nunique()
    assert quarterly.frame["volume"].isna().all()
