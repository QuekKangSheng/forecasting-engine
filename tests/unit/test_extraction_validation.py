"""The extraction package's pandera schema and plain-pandas report."""

import pandas as pd

from forecasting_engine.extraction.validation import (
    _format_failure,
    _robust_z,
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
