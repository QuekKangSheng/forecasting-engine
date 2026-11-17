"""The bond calendar model: what kind of days each label window holds, screened and fitted
per fold by the shrinkage fit, on the page's own walk-forward."""

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import pytest

from forecasting_engine.models.calendar_research import (
    CALENDAR_INPUTS,
    HOLIDAY_CALENDAR,
    WEEKDAY_INPUTS,
    calendar_panel,
    coming_sessions,
    inputs_for,
    label_window_calendar,
    run_calendar_research_polynomial,
)
from forecasting_engine.models.polynomial import PolynomialConfigError
from forecasting_engine.models.shrinkage import ShrinkagePolynomial
from forecasting_engine.reporting.factor_labels import BUILT_LABELS
from forecasting_engine.validation.harness import evaluate, summarize
from forecasting_engine.validation.splitters import PurgedWalkForward


def _sessions(start: str, end: str) -> pd.DatetimeIndex:
    """The US bond market's trading days from ``start`` to ``end``."""
    days = mcal.get_calendar(HOLIDAY_CALENDAR).valid_days(start, end)
    return pd.DatetimeIndex(days.tz_localize(None))


def _row(columns: pd.DataFrame, date: str) -> dict[str, float]:
    return columns.loc[pd.Timestamp(date)].to_dict()


#: The second half of 2018 on the bond market's calendar: Labor Day (3 Sep), Columbus
#: Day (8 Oct), Veterans Day (12 Nov) and Thanksgiving (22 Nov) are not trading days.
AUTUMN_2018 = _sessions("2018-07-02", "2018-12-14")


# --- the columns -------------------------------------------------------------------


def test_the_columns_are_the_three_counts_and_at_one_day_the_weekday():
    assert inputs_for(1) == CALENDAR_INPUTS + WEEKDAY_INPUTS
    for horizon in (5, 10, 20):
        assert inputs_for(horizon) == CALENDAR_INPUTS
        assert list(label_window_calendar(AUTUMN_2018, horizon).columns) == list(CALENDAR_INPUTS)


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
    # A Friday is the left-out weekday.
    assert sum(_row(columns, "2018-09-13")[d] for d in WEEKDAY_INPUTS) == 0


def test_a_longer_window_counts_its_days():
    columns = label_window_calendar(AUTUMN_2018, 5)

    # 27 Aug's window: 28, 29, 30, 31 Aug and 4 Sep.
    assert _row(columns, "2018-08-27") == {
        "CAL_MONTH_END": 1,
        "CAL_MONTH_START": 1,
        "CAL_JOBS_FRIDAY": 0,
    }
    # 30 Aug's: 31 Aug, 4, 5, 6 and 7 Sep, the jobs Friday.
    assert _row(columns, "2018-08-30")["CAL_JOBS_FRIDAY"] == 1


@pytest.mark.parametrize("horizon", [5, 10, 20])
def test_an_h_day_count_is_the_sum_of_the_next_h_days_flags(horizon):
    one_day = label_window_calendar(AUTUMN_2018, 1)[list(CALENDAR_INPUTS)]
    window = label_window_calendar(AUTUMN_2018, horizon)

    expected = one_day.rolling(horizon).sum().shift(-(horizon - 1))
    pd.testing.assert_frame_equal(window.iloc[:-horizon], expected.iloc[:-horizon])


def test_almost_every_20_day_window_holds_one_month_end():
    # Why the model carries nothing at h = 20: the month-end count barely varies, where
    # at h = 5 it is a one in about four windows.
    twenty = label_window_calendar(AUTUMN_2018, 20)["CAL_MONTH_END"]
    five = label_window_calendar(AUTUMN_2018, 5)["CAL_MONTH_END"]

    assert set(twenty.unique()) <= {0.0, 1.0, 2.0}
    assert (twenty == 1).mean() > 0.9
    assert 0.15 < (five == 1).mean() < 0.3


def test_the_columns_match_the_research_wherever_the_data_runs():
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
    # Since Nov 2023 the exports carry a row on bond-market holidays. Good Friday 29 Mar
    # 2024 is one, and the last row of March, so the data's calendar puts the month-end
    # there, as the research's did.
    index = _sessions("2024-03-01", "2024-04-30").union([pd.Timestamp("2024-03-29")])

    columns = label_window_calendar(index, 1)

    assert _row(columns, "2024-03-28")["CAL_MONTH_END"] == 1
    assert _row(columns, "2024-03-27")["CAL_MONTH_END"] == 0


# --- past the data's last date ---------------------------------------------------------


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


def test_a_month_end_just_past_the_data_is_found_across_a_holiday():
    # The data stops on Thursday 29 Mar 2018; Good Friday is not a bond trading day, so
    # 29 Mar is the last of March and the next trading day is Monday 2 Apr.
    index = _sessions("2018-01-02", "2018-03-29")

    one_day = label_window_calendar(index, 1)
    week = label_window_calendar(index, 5)

    assert _row(one_day, "2018-03-28")["CAL_MONTH_END"] == 1
    assert _row(one_day, "2018-03-29")["CAL_MONTH_START"] == 1
    assert _row(one_day, "2018-03-29")["CAL_MON"] == 1
    assert _row(week, "2018-03-23")["CAL_MONTH_END"] == 1


def test_the_last_rows_window_skips_the_holidays_ahead():
    # The data stops on Friday 21 Dec 2018. Its week is 24, 26, 27, 28 and 31 Dec, with
    # Christmas skipped, so it holds the year's last trading day; five weekdays would not.
    december = label_window_calendar(_sessions("2018-10-01", "2018-12-21"), 5)

    assert _row(december, "2018-12-21") == {
        "CAL_MONTH_END": 1,
        "CAL_MONTH_START": 0,
        "CAL_JOBS_FRIDAY": 0,
    }


# --- the panel and the run -------------------------------------------------------------


H = 5


def _bond_world(n_days: int = 900, seed: int = 4, month_end: float = 0.0015) -> pd.DataFrame:
    """A bond index that earns ``month_end`` more on each month's last trading day, on the
    bond market's own calendar."""
    index = _sessions("2014-01-02", "2022-12-30")[:n_days]
    rng = np.random.default_rng(seed)
    last = label_window_calendar(index[:-1], 1)["CAL_MONTH_END"].to_numpy()
    daily = np.r_[0.0, 0.0001 + month_end * last + rng.normal(scale=0.002, size=len(index) - 1)]
    return pd.DataFrame({"price": 100 * np.exp(np.cumsum(daily))}, index=index)


def _splitter() -> PurgedWalkForward:
    return PurgedWalkForward(train=252, test=20, embargo=20, tuning_rows=252)


def test_the_panel_is_the_targets_calendar_with_the_columns_unlagged():
    frame = _bond_world()
    panel = calendar_panel(frame, "price", H)

    assert panel.signals == CALENDAR_INPUTS
    assert list(panel.frame.columns) == [*CALENDAR_INPUTS, "price", "fwd_return_5d"]
    pd.testing.assert_index_equal(panel.frame.index, frame.index)
    pd.testing.assert_frame_equal(
        panel.frame[list(CALENDAR_INPUTS)], label_window_calendar(frame.index, H)
    )
    pd.testing.assert_series_equal(
        panel.frame["fwd_return_5d"], frame["price"].pct_change(H).shift(-H), check_names=False
    )


def test_the_run_is_the_shrinkage_fit_screened_per_fold_on_the_splitters_own_window():
    frame = _bond_world()

    result, description, panel = run_calendar_research_polynomial(frame, "price", H, _splitter())

    folds = evaluate(ShrinkagePolynomial, panel, _splitter(), screen=True)
    expected, latest = summarize(folds)
    pd.testing.assert_series_equal(result.forecast, expected.forecast)
    assert description == latest
    assert {len(f.train) for f in folds} == {252}  # a rolling window, not every earlier row
    assert len({f.train[0] for f in folds}) == len(folds)


def test_on_a_world_where_month_ends_earn_more_the_fit_keeps_the_month_end_positive():
    result, description, _ = run_calendar_research_polynomial(
        _bond_world(), "price", 1, _splitter()
    )

    slopes = dict(zip(description.terms, description.coefficients, strict=True))
    assert slopes["CAL_MONTH_END"] > 0
    assert result.gated_rank_ic > 0
    assert result.forecast.notna().all()


def test_its_result_carries_its_pre_2016_score():
    result, _, _ = run_calendar_research_polynomial(_bond_world(), "price", H, _splitter())

    assert result.holdout is not None
    assert result.holdout.before == pd.Timestamp("2016-08-01")
    assert result.holdout.last_label_end < result.holdout.before
    assert result.holdout.windows > 0


def test_data_too_short_for_one_fold_is_refused_with_the_splitters_reason():
    with pytest.raises(PolynomialConfigError, match="too few for one walk-forward fold"):
        run_calendar_research_polynomial(_bond_world(280), "price", H, _splitter())


# --- no look-ahead -------------------------------------------------------------------------


@pytest.mark.parametrize("horizon", [1, 5, 20])
def test_no_price_on_or_after_a_folds_first_test_date_changes_its_fit(horizon):
    frame = _bond_world()
    panel = calendar_panel(frame, "price", horizon)
    folds = list(_splitter().split(panel))
    number = len(folds) // 2
    _, test = folds[number]
    later = frame.index >= test[0]
    rewritten = frame.copy()
    rng = np.random.default_rng(0)
    rewritten.loc[later, "price"] *= rng.uniform(0.5, 1.5, size=int(later.sum()))

    before = evaluate(ShrinkagePolynomial, panel, _splitter(), screen=True)
    after = evaluate(
        ShrinkagePolynomial, calendar_panel(rewritten, "price", horizon), _splitter(), screen=True
    )

    assert after[number].description == before[number].description
    assert after[number].screening == before[number].screening
    # Every earlier fold is untouched too; a later one is not, or the check is vacuous.
    assert all(a.description == b.description for a, b in zip(after[:number], before, strict=False))
    assert after[number + 1].description != before[number + 1].description


def test_the_last_price_a_training_label_reaches_does_change_the_fit():
    frame = _bond_world()
    panel = calendar_panel(frame, "price", H)
    train, _ = list(_splitter().split(panel))[10]
    last_reached = panel.label_end.loc[train].max()
    rewritten = frame.copy()
    rewritten.loc[last_reached, "price"] *= 1.05

    model, again = ShrinkagePolynomial(), ShrinkagePolynomial()
    model.fit(panel, train)
    again.fit(calendar_panel(rewritten, "price", H), train)

    assert again.describe() != model.describe()


def test_the_columns_read_no_price():
    frame = _bond_world()
    shuffled = frame.assign(price=frame["price"].sample(frac=1, random_state=0).to_numpy())

    pd.testing.assert_frame_equal(
        calendar_panel(frame, "price", H).frame[list(CALENDAR_INPUTS)],
        calendar_panel(shuffled, "price", H).frame[list(CALENDAR_INPUTS)],
    )


def test_every_column_has_a_plain_label():
    assert set(inputs_for(1)) <= set(BUILT_LABELS)


def test_a_panel_built_from_a_shuffled_frame_is_the_same():
    # The calendar is the target's sorted dates, whatever order the frame arrives in.
    frame = _bond_world()

    shuffled = calendar_panel(frame.sample(frac=1, random_state=1), "price", H)

    pd.testing.assert_frame_equal(shuffled.frame, calendar_panel(frame, "price", H).frame)
