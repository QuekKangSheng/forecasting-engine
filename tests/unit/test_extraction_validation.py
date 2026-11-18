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


def test_a_new_high_after_a_rebound_is_flagged_and_not_taken_for_a_second_rebound():
    # VIX in March 2020: up 21.6 on 12 Mar, down 17.6 on 13 Mar, up 24.9 on 16 Mar
    # to a new high. 13 Mar is 12 Mar's rebound. 16 Mar is not 13 Mar's: 13 Mar was
    # never reported, and 16 Mar went past where 12 Mar had been.
    dates = pd.date_range("2024-01-01", periods=60)
    values = [20.0 + (i % 2) * 0.5 for i in range(60)]
    values[30:33] = [41.6, 24.0, 48.9]
    data = frame(dates, A_Index_PX_LAST=values)

    report = validate(data)

    assert list(report.big_moves["date"]) == [dates[30], dates[32]]


def test_two_bad_values_a_day_apart_are_each_reported_once():
    # Each bad value makes a move on and a move off: four alternating flags in a
    # row. The two rows holding a bad value are reported, their rebounds are not.
    dates = pd.date_range("2024-01-01", periods=60)
    values = [100.0 + i for i in range(60)]
    values[30], values[32] = 300.0, 300.0
    data = frame(dates, A_Index_PX_LAST=values)

    report = validate(data)

    assert list(report.big_moves["date"]) == [dates[30], dates[32]]


# --- a price is scored on its percentage move, a rate on its move in points -----


def _grown_index() -> np.ndarray:
    """A price index that rises about fourfold on the same percentage wobble
    throughout and falls 5% on its eleventh day: the S&P 500 from 2008 to 2026 in
    miniature. In points that fall is smaller than an ordinary day at the end."""
    returns = np.random.default_rng(0).normal(0.004, 0.005, 400)
    returns[10] = -0.05
    return 1000.0 * np.cumprod(1 + returns)


def test_an_early_crash_in_a_total_return_index_is_flagged_after_the_index_has_grown():
    dates = pd.date_range("2024-01-01", periods=400)
    data = frame(dates, A_Index_TOT_RETURN_INDEX_GROSS_DVDS=_grown_index())

    report = validate(data)

    assert list(report.big_moves["date"]) == [dates[10]]


def test_a_flagged_price_move_carries_its_percentage_change():
    dates = pd.date_range("2024-01-01", periods=400)
    values = _grown_index()
    data = frame(dates, A_Index_TOT_RETURN_INDEX_GROSS_DVDS=values)

    move = validate(data).big_moves.iloc[0]

    assert np.isclose(move["percent_change"], -0.05)
    assert np.isclose(move["change"], values[10] - values[9])


def test_a_price_field_is_scored_on_percentage_moves_when_the_caller_names_it():
    # SPX's last price carries the field every rate and gauge carries too, so only
    # the caller, who knows the security, can say it is a price.
    dates = pd.date_range("2024-01-01", periods=400)
    data = frame(dates, SPX_Index_PX_LAST=_grown_index())

    named = validate(data, ratio_columns={"SPX_Index_PX_LAST"})
    unnamed = validate(data)

    assert list(named.big_moves["date"]) == [dates[10]]
    assert unnamed.big_moves.empty


def test_a_rate_is_scored_in_points_however_low_it_falls():
    # A yield falling from 5 to 0.3 on the same wobble in points: late on, an
    # ordinary day is a large percentage of the yield, and none of them is news.
    points = np.random.default_rng(0).normal(-0.015, 0.02, 300)
    data = frame(
        pd.date_range("2024-01-01", periods=300), USGG10YR_Index_PX_LAST=5.0 + np.cumsum(points)
    )

    in_points = validate(data)
    in_percent = validate(data, ratio_columns={"USGG10YR_Index_PX_LAST"})

    assert in_points.big_moves.empty
    assert in_points.big_moves.columns.tolist() == [
        "column",
        "date",
        "change",
        "percent_change",
        "robust_score",
    ]
    assert not in_percent.big_moves.empty


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
