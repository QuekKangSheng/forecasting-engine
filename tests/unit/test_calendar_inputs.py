"""Calendar counts over a label window (``models/calendar_inputs``): what kind of days the
next ``h`` trading days are, on the target's own calendar and past its last date."""

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import pytest

from forecasting_engine.models.calendar_inputs import (
    CALENDAR_INPUTS,
    CALENDAR_NAMES,
    HOLIDAY_CALENDAR,
    WEEKDAY_INPUTS,
    coming_sessions,
    inputs_for,
    label_window_calendar,
)
from forecasting_engine.reporting.factor_labels import BUILT_LABELS


def _sessions(start: str, end: str) -> pd.DatetimeIndex:
    """The US bond market's trading days from ``start`` to ``end``."""
    days = mcal.get_calendar(HOLIDAY_CALENDAR).valid_days(start, end)
    return pd.DatetimeIndex(days.tz_localize(None))


def _row(columns: pd.DataFrame, date: str) -> dict[str, float]:
    return columns.loc[pd.Timestamp(date)].to_dict()


#: The second half of 2018 on the bond market's calendar: Labor Day (3 Sep), Columbus
#: Day (8 Oct), Veterans Day (12 Nov) and Thanksgiving (22 Nov) are not trading days.
AUTUMN_2018 = _sessions("2018-07-02", "2018-12-14")


def test_the_columns_are_the_three_counts_and_at_one_day_the_weekday():
    assert inputs_for(1) == CALENDAR_INPUTS + WEEKDAY_INPUTS
    assert CALENDAR_NAMES == set(CALENDAR_INPUTS + WEEKDAY_INPUTS)
    for horizon in (5, 10, 20):
        assert inputs_for(horizon) == CALENDAR_INPUTS
        assert list(label_window_calendar(AUTUMN_2018, horizon).columns) == list(CALENDAR_INPUTS)


def test_every_calendar_column_has_a_readable_label():
    assert set(BUILT_LABELS) == CALENDAR_NAMES


def test_a_rows_one_day_window_is_the_next_trading_day():
    columns = label_window_calendar(AUTUMN_2018, 1)

    # 30 Aug 2018's label window is 31 Aug, the last trading day of August.
    assert _row(columns, "2018-08-30")["CAL_MONTH_END"] == 1
    assert _row(columns, "2018-08-29")["CAL_MONTH_END"] == 0
    # 31 Aug's is Tuesday 4 Sep, the first of September once Labor Day is skipped.
    assert _row(columns, "2018-08-31") == {
        "CAL_MONTH_END": 0,
        "CAL_MONTH_START": 1,
        "CAL_JOBS_FRIDAY": 0,
        "CAL_MON": 0,
        "CAL_TUE": 1,
        "CAL_WED": 0,
        "CAL_THU": 0,
    }
    # 7 Sep, 5 Oct and 2 Nov 2018 are the first Friday of their month; 14 Sep is not.
    for before in ("2018-09-06", "2018-10-04", "2018-11-01"):
        assert _row(columns, before)["CAL_JOBS_FRIDAY"] == 1
    assert _row(columns, "2018-09-13")["CAL_JOBS_FRIDAY"] == 0


@pytest.mark.parametrize("horizon", [5, 10, 20])
def test_an_h_day_count_is_the_sum_of_the_next_h_days_flags(horizon):
    one_day = label_window_calendar(AUTUMN_2018, 1)[list(CALENDAR_INPUTS)]
    window = label_window_calendar(AUTUMN_2018, horizon)

    expected = one_day.rolling(horizon).sum().shift(-(horizon - 1))
    pd.testing.assert_frame_equal(window.iloc[:-horizon], expected.iloc[:-horizon])


def test_the_counts_match_the_research_wherever_the_data_runs():
    # The November 2026 research's own function (s14-bond-short/grid14.py), as it ran:
    # the data's dates only, so it has no value for the last h rows.
    def research(index, horizon):
        month = index.year * 12 + index.month
        flags = {
            "CAL_MONTH_END": (month != np.r_[month[1:], -1]).astype(float),
            "CAL_MONTH_START": (month != np.r_[-1, month[:-1]]).astype(float),
            "CAL_JOBS_FRIDAY": ((index.dayofweek == 4) & (index.day <= 7)).astype(float),
        }
        if horizon == 1:
            for d, name in enumerate(WEEKDAY_INPUTS):
                flags[name] = (index.dayofweek == d).astype(float)
        return pd.DataFrame(flags, index=index).rolling(horizon).sum().shift(-horizon)

    for horizon in (1, 5, 20):
        ours, theirs = label_window_calendar(AUTUMN_2018, horizon), research(AUTUMN_2018, horizon)
        # Its last row counts as a month-end whatever its date (nothing follows it), so the
        # last label reaching it is where the two part.
        keep = len(AUTUMN_2018) - horizon - 1
        pd.testing.assert_frame_equal(ours.iloc[:keep], theirs.iloc[:keep])
        assert theirs.iloc[keep + 1 :].isna().all().all()


def test_the_data_calendar_is_used_where_it_runs_even_where_it_holds_a_holiday():
    # Good Friday 29 Mar 2024 is a bond-market holiday but a row of the data here, and
    # the last of March, so the data's calendar puts the month-end there.
    index = _sessions("2024-03-01", "2024-04-30").union([pd.Timestamp("2024-03-29")])

    columns = label_window_calendar(index, 1)

    assert _row(columns, "2024-03-28")["CAL_MONTH_END"] == 1
    assert _row(columns, "2024-03-27")["CAL_MONTH_END"] == 0


def test_the_coming_trading_days_are_the_bond_markets():
    # Veterans Day (12 Nov 2018) and Thanksgiving (22 Nov) are skipped; the NYSE trades on
    # the first, the bond market does not.
    assert list(coming_sessions(pd.Timestamp("2018-11-09"), 3)) == list(
        pd.to_datetime(["2018-11-13", "2018-11-14", "2018-11-15"])
    )
    assert list(coming_sessions(pd.Timestamp("2018-11-21"), 2)) == list(
        pd.to_datetime(["2018-11-23", "2018-11-26"])
    )


@pytest.mark.parametrize("horizon", [1, 5, 10, 20])
def test_the_last_rows_get_a_value_from_the_holiday_calendar(horizon):
    columns = label_window_calendar(AUTUMN_2018, horizon)

    assert columns.notna().all().all()
    # And they are what a longer history would have given them.
    longer = label_window_calendar(_sessions("2018-07-02", "2019-03-29"), horizon)
    pd.testing.assert_frame_equal(columns, longer.loc[columns.index])


def test_the_last_rows_window_skips_the_holidays_ahead():
    # The data stops on Friday 21 Dec 2018. Its week is 24, 26, 27, 28 and 31 Dec, with
    # Christmas skipped, so it holds the year's last trading day; five weekdays would not.
    december = label_window_calendar(_sessions("2018-10-01", "2018-12-21"), 5)

    assert _row(december, "2018-12-21") == {
        "CAL_MONTH_END": 1,
        "CAL_MONTH_START": 0,
        "CAL_JOBS_FRIDAY": 0,
    }


def test_no_dates_give_no_rows():
    assert label_window_calendar(pd.DatetimeIndex([]), 5).empty
