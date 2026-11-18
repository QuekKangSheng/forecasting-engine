"""The extraction package's pandera schema and plain-pandas report."""

import numpy as np
import pandas as pd

from forecasting_engine.extraction.bloomberg_csv import BloombergCsvExport
from forecasting_engine.extraction.validation import (
    _format_failure,
    _robust_z,
    drop_exact_repeats,
    repeated_date_error,
    schema_errors,
    validate,
)


def frame(dates, **columns):
    return pd.DataFrame({"Date": pd.to_datetime(dates), **columns})


def test_a_clean_file_reports_nothing():
    data = frame(["2024-01-02", "2024-01-03"], A_Index_PX_LAST=[100.0, 101.0])
    report = validate(data)
    assert report.is_clean


def test_duplicate_dates_are_counted_not_dropped():
    data = frame(["2024-01-02", "2024-01-02"], A_Index_PX_LAST=[100.0, 101.0])
    report = validate(data)
    assert report.duplicate_dates == 1
    assert len(data) == 2  # nothing removed


def test_duplicate_dates_alone_are_not_a_schema_error():
    # A repeated date is a revision, not a fault — it must not block the file
    # the way a genuinely bad column does.
    data = frame(["2024-01-02", "2024-01-02"], A_Index_PX_LAST=[100.0, 101.0])
    report = validate(data)
    assert not report.schema_errors


def _export(data: pd.DataFrame, security: str = "A Index") -> BloombergCsvExport:
    return BloombergCsvExport(filename="a.csv", security=security, frame=data)


def test_rows_repeating_an_earlier_row_exactly_are_dropped_and_counted():
    data = frame(["2024-01-02", "2024-01-02", "2024-01-03"], A_Index_PX_LAST=[1.0, 1.0, 2.0])

    export = drop_exact_repeats(_export(data))

    assert export.frame["A_Index_PX_LAST"].tolist() == [1.0, 2.0]
    (note,) = export.notes
    assert "a.csv" in note and "1 row" in note and "dropped" in note


def test_a_file_with_no_repeated_rows_is_returned_as_it_was():
    export = _export(frame(["2024-01-02", "2024-01-03"], A_Index_PX_LAST=[1.0, 2.0]))

    assert drop_exact_repeats(export) is export


def test_a_date_repeated_with_different_values_is_named_with_its_security():
    data = frame(["2024-01-02", "2024-01-02", "2024-01-03"], A_Index_PX_LAST=[1.0, 9.0, 2.0])

    message = repeated_date_error(_export(data))

    assert "A Index" in message
    assert "02/01/2024" in message
    assert "03/01/2024" not in message


def test_a_repeated_date_error_names_the_first_few_dates_and_counts_the_rest():
    dates = pd.bdate_range("2024-01-01", periods=12)
    data = frame([*dates, *dates], A_Index_PX_LAST=[1.0] * 12 + [2.0] * 12)

    message = repeated_date_error(_export(data))

    assert message.count("/2024") == 10
    assert "and 2 more" in message


def test_no_repeated_date_gives_no_error():
    data = frame(["2024-01-02", "2024-01-03"], A_Index_PX_LAST=[1.0, 2.0])

    assert repeated_date_error(_export(data)) is None


def test_weekend_rows_are_counted():
    # 2024-01-06 and 2024-01-07 are a Saturday and Sunday.
    data = frame(["2024-01-05", "2024-01-06", "2024-01-07"], A_Index_PX_LAST=[1.0, 2.0, 3.0])
    report = validate(data)
    assert report.weekend_rows == 2
    assert report.weekend_date_values == ("2024-01-06", "2024-01-07")


def test_duplicate_dates_are_named_not_just_counted():
    data = frame(["2024-01-02", "2024-01-02"], A_Index_PX_LAST=[100.0, 101.0])
    report = validate(data)
    assert report.duplicate_date_values == ("2024-01-02",)


def test_missing_pct_is_reported_per_column_and_nulls_are_not_an_error():
    data = frame(["2024-01-01", "2024-01-02"], A_Index_PX_LAST=[100.0, None])
    report = validate(data)
    assert report.missing_pct["A_Index_PX_LAST"] == 50.0
    assert not report.schema_errors


def test_an_isolated_move_is_flagged_by_robust_mad_and_its_rebound_is_collapsed():
    dates = pd.date_range("2024-01-01", periods=60)
    values = [100.0 + i for i in range(60)]
    values[30] = 200.0
    data = frame(dates, A_Index_PX_LAST=values)

    report = validate(data)

    assert len(report.big_moves) == 1
    assert report.big_moves.iloc[0]["column"] == "A_Index_PX_LAST"
    assert report.big_moves.iloc[0]["robust_score"] > 8


def test_large_percentage_moves_are_not_outliers_when_they_are_typical_for_the_series():
    data = frame(
        pd.date_range("2024-01-01", periods=8),
        A_Index_SPREAD=[1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8],
    )

    report = validate(data)

    assert report.big_moves.empty


def test_a_negative_total_return_index_is_a_schema_error():
    data = frame(
        ["2024-01-01", "2024-01-02"], SPX_Index_TOT_RETURN_INDEX_GROSS_DVDS=[100.0, -5.0]
    )
    report = validate(data)
    assert any("TOT_RETURN_INDEX_GROSS_DVDS" in e for e in report.schema_errors)


def test_a_negative_px_last_is_accepted_since_not_every_series_is_a_price():
    # The 2s10s curve was inverted through 2022-24.
    data = frame(["2022-07-05", "2022-07-06"], USYC2Y10_Index_PX_LAST=[-3.1, -12.4])
    report = validate(data)
    assert not report.schema_errors


def test_px_last_is_not_held_to_the_generic_range():
    data = frame(["2024-01-01", "2024-01-02"], NDX_Index_PX_LAST=[16_800.0, 16_900.0])
    assert schema_errors(data) == []


def test_text_in_a_px_last_column_is_still_a_schema_error():
    data = frame(["2024-01-01", "2024-01-02"], A_Index_PX_LAST=["1.0", "abc"])
    assert any("PX_LAST" in e for e in schema_errors(data))


def test_a_non_price_column_uses_the_generic_sane_range_not_positivity():
    # A spread-like column can legitimately be negative (e.g. a breakeven rate).
    data = frame(["2024-01-01", "2024-01-02"], A_Index_SPREAD=[-1.5, 2.0])
    report = validate(data)
    assert not report.schema_errors


def test_a_known_non_price_ticker_can_go_negative_despite_its_px_field():
    # USYC2Y10 (the 2s10s Treasury spread) genuinely inverts; PX_LAST here
    # names a spread, not a price, so it must not be forced positive.
    data = frame(
        ["2024-01-01", "2024-01-02"], USYC2Y10_Index_PX_LAST=[-108.714, -97.309]
    )
    report = validate(data)
    assert not report.schema_errors


def test_an_out_of_range_non_price_value_is_a_schema_error():
    data = frame(["2024-01-01", "2024-01-02"], A_Index_SPREAD=[1.0, 50_000.0])
    report = validate(data)
    assert any("SPREAD" in e for e in report.schema_errors)


def test_unsorted_dates_are_a_schema_error():
    data = frame(["2024-01-02", "2024-01-01"], A_Index_PX_LAST=[1.0, 2.0])
    report = validate(data)
    assert any("ascending" in e for e in report.schema_errors)


def test_schema_errors_is_empty_for_a_clean_file():
    data = frame(["2024-01-02", "2024-01-03"], A_Index_PX_LAST=[100.0, 101.0])
    assert schema_errors(data) == []


def test_schema_errors_names_a_bad_value():
    data = frame(["2024-01-01", "2024-01-02"], A_Index_TOT_RETURN_INDEX=[100.0, -5.0])
    assert any("TOT_RETURN_INDEX" in e for e in schema_errors(data))


def test_schema_errors_matches_validate_report():
    # validate() delegates to schema_errors() for this field — same file,
    # same result, whichever entry point a caller uses.
    data = frame(["2024-01-01", "2024-01-02"], A_Index_TOT_RETURN_INDEX=[100.0, -5.0])
    assert validate(data).schema_errors == schema_errors(data)


def test_a_timestamp_failure_case_is_shown_as_dd_mm_yyyy_with_no_time():
    assert _format_failure(pd.Timestamp("2017-02-23")) == "23/02/2017"


def test_a_non_date_failure_case_is_shown_as_its_repr():
    assert _format_failure(False) == "False"


# --- guards carried over from the retired quality/outliers.py ---------------
#
# That module duplicated this one's MAD scoring and is removed in the sprint
# plan's Phase 3. Its tests were the only ones covering these three behaviours,
# so they move here, pointed at the implementation that is actually live.


def test_a_lone_jump_in_a_stale_feed_is_still_flagged():
    # More than half the daily changes are zero - a pegged signal or a frozen
    # feed - so the median absolute deviation is zero. Scaling by it would call
    # the one real move unremarkable; the mean-absolute-deviation fallback
    # catches it.
    values = [5.0] * 20 + [90.0] + [5.0] * 20
    data = frame(pd.date_range("2024-01-01", periods=len(values)), A_Index_PX_LAST=values)

    report = validate(data)

    assert len(report.big_moves) == 1


def test_the_score_is_not_inflated_by_the_outlier_it_is_measuring():
    # Why MAD and not standard deviation: one huge value drags a standard
    # deviation up far enough to hide itself.
    values = pd.Series([1.0, -1.0] * 30 + [200.0])
    classic = abs((values.iloc[-1] - values.mean()) / values.std())

    assert classic < 8
    assert abs(_robust_z(values).iloc[-1]) > 8


def test_a_crash_of_consecutive_same_direction_moves_is_kept_whole():
    # Only an opposite-direction rebound is collapsed into its first flag. A
    # crash is several real days in a row, and each one is kept.
    before = [100.0, 101.0] * 15
    after = [40.0, 41.0] * 15
    values = before + [80.0, 60.0, 40.0] + after
    data = frame(pd.date_range("2024-01-01", periods=len(values)), A_Index_PX_LAST=values)

    report = validate(data)

    assert len(report.big_moves) == 3


# --- bounded messages and the PX_ field check ----------------------------------


def test_a_column_failing_on_every_row_gives_one_short_message_with_a_count():
    dates = pd.date_range("2024-01-01", periods=2_000)
    data = frame(dates, A_Index_SPREAD=[50_000.0 + i for i in range(2_000)])

    (message,) = schema_errors(data)

    assert message.startswith("A_Index_SPREAD: ")
    assert "50000.0" in message
    assert "2,000 values in all" in message
    assert len(message) < 200


def test_a_column_whose_security_merely_contains_px_is_still_range_checked():
    # SPX_ contains "PX_", but this column's field is CUR_MKT_CAP, not a PX_ field.
    data = frame(["2024-01-01", "2024-01-02"], SPX_Index_CUR_MKT_CAP=[5.1e7, 5.2e7])

    assert any("SPX_Index_CUR_MKT_CAP" in e for e in schema_errors(data))


# --- prices are scored on their percent change ----------------------------


def _rising_index(crash_at: int, crash_pct: float, late_pct: float) -> list[float]:
    # An index that rises about sixfold over 400 days, with ordinary days of
    # about 1%, a crash early on while it is low and a smaller move on its last
    # day, once it is high.
    daily = np.random.default_rng(0).normal(0.0045, 0.01, 400)
    daily[crash_at], daily[-1] = np.log1p(crash_pct), np.log1p(late_pct)
    return list(1_000.0 * np.exp(daily.cumsum()))


def test_a_crash_while_a_price_index_was_low_is_flagged_over_an_ordinary_day_once_it_is_high():
    # The S&P 500 rose about five times between 2008 and 2026. In points, its
    # +11.6% on 13 Oct 2008 scored 5.9 and an ordinary -1.8% day in Oct 2026
    # scored 8.1, so the crash went unflagged and the ordinary day was flagged.
    values = _rising_index(crash_at=50, crash_pct=0.116, late_pct=-0.04)
    dates = pd.bdate_range("2008-01-01", periods=len(values))
    assert values[-1] / values[0] > 5

    report = validate(frame(dates, SPX_Index_PX_LAST=values))

    assert list(report.big_moves["date"]) == [dates[50]]


def test_a_total_return_index_is_scored_on_its_percent_change_whatever_its_ticker():
    values = _rising_index(crash_at=50, crash_pct=0.116, late_pct=-0.04)
    dates = pd.bdate_range("2008-01-01", periods=len(values))

    report = validate(frame(dates, NEW_Index_TOT_RETURN_INDEX_GROSS_DVDS=values))

    assert list(report.big_moves["date"]) == [dates[50]]


def test_a_flagged_price_move_is_still_reported_in_points():
    # The analyst checks a flag against the file, which holds points.
    values = _rising_index(crash_at=50, crash_pct=0.116, late_pct=-0.04)
    dates = pd.bdate_range("2008-01-01", periods=len(values))

    report = validate(frame(dates, SPX_Index_PX_LAST=values))

    assert report.big_moves.iloc[0]["change"] == values[50] - values[49]


def test_a_rate_near_zero_is_still_scored_on_its_change_in_points():
    # A 10-year yield falling from 5% to 0.3%, moving a few hundredths a day.
    # Near the end those moves are several percent of the yield, so scored on a
    # percent change the late days would flag. Rates keep the plain change.
    steps = np.random.default_rng(0).normal(0, 0.02, 400)
    values = list(np.linspace(5.0, 0.3, 400) + steps)
    dates = pd.bdate_range("2019-01-01", periods=len(values))
    assert min(values) > 0

    report = validate(frame(dates, USGG10YR_Index_PX_LAST=values))

    assert report.big_moves.empty


def test_a_price_column_with_a_value_at_or_below_zero_falls_back_to_its_change_in_points():
    # A log change is undefined there; scoring the plain change still flags it.
    values = [100.0 + (i % 2) for i in range(60)]
    values[30] = -100.0
    dates = pd.bdate_range("2020-01-01", periods=len(values))

    report = validate(frame(dates, SPX_Index_PX_LAST=values))

    assert list(report.big_moves["date"]) == [dates[30]]
