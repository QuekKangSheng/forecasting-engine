"""Robustness checks for the active generic Bloomberg extraction path."""

import pandas as pd
import pytest

from forecasting_engine.extraction.bloomberg_csv import (
    BloombergCsvError,
    ColumnSource,
    column_sources,
    drop_empty_columns,
    join_split_files,
    merge,
    missing_row_report,
    read_export,
    with_display_dates,
)


def test_a_missing_value_on_an_unreadable_date_is_reported_without_crashing():
    frame = pd.DataFrame({"Date": [pd.NaT], "A_Index_PX_LAST": [None]})

    report = missing_row_report(frame)

    assert len(report) == 1
    assert report.iloc[0]["Likely reason"] == "Unreadable date"


def test_a_missing_value_on_a_weekend_is_labelled_a_weekend():
    # 2024-01-06 is a Saturday.
    frame = pd.DataFrame({"Date": [pd.Timestamp("2024-01-06")], "A_Index_PX_LAST": [None]})

    report = missing_row_report(frame)

    assert len(report) == 1
    assert report.iloc[0]["Likely reason"] == "Weekend"


def test_a_missing_value_on_an_nyse_holiday_is_labelled_a_market_holiday():
    # 2024-01-01: New Year's Day, closed on the NYSE calendar.
    frame = pd.DataFrame({"Date": [pd.Timestamp("2024-01-01")], "A_Index_PX_LAST": [None]})

    report = missing_row_report(frame)

    assert len(report) == 1
    assert report.iloc[0]["Likely reason"] == "Likely NYSE market holiday"


def test_a_missing_value_on_an_ordinary_trading_day_is_an_unexplained_gap():
    # 2024-01-03: an ordinary Wednesday, no holiday.
    frame = pd.DataFrame({"Date": [pd.Timestamp("2024-01-03")], "A_Index_PX_LAST": [None]})

    report = missing_row_report(frame)

    assert len(report) == 1
    assert report.iloc[0]["Likely reason"] == "Unexplained gap"


def test_merging_exports_with_different_dates_keeps_every_date_as_an_outer_join():
    spx = read_export("spx.csv", b"Security,SPX Index\n\nDate,PX_LAST\n2024-01-02,100\n")
    vix = read_export(
        "vix.csv", b"Security,VIX Index\n\nDate,PX_LAST\n2024-01-02,10\n2024-01-03,11\n"
    )

    merged = merge([spx, vix])

    assert len(merged) == 2
    row = merged[merged["Date"] == pd.Timestamp("2024-01-03")].iloc[0]
    assert pd.isna(row["SPX_Index_PX_LAST"])
    assert row["VIX_Index_PX_LAST"] == 11


def test_drop_empty_columns_removes_a_column_with_no_values_and_names_it():
    frame = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")],
            "VIX_Index_PX_LAST": [10.0, 11.0],
            "VIX_Index_PX_VOLUME": [None, None],
        }
    )

    kept, dropped = drop_empty_columns(frame)

    assert "VIX_Index_PX_VOLUME" not in kept.columns
    assert "VIX_Index_PX_LAST" in kept.columns
    assert dropped == ["VIX_Index_PX_VOLUME"]


def test_a_trailing_empty_metadata_field_does_not_leak_into_the_security():
    # Real exports pad the row with a trailing empty field
    # ("Security,SPX Index,") rather than leaving it bare.
    data = b"Security,SPX Index,\nPeriod,D,\n,,\nDate,PX_LAST\n2024-01-02,100.0\n"

    export = read_export("spx.csv", data)

    assert export.security == "SPX Index"


def _export(security: str, filename: str, fields: str):
    data = f"Security,{security}\n\nDate,{fields}\n2024-01-02,1,2\n"
    return read_export(filename, data.encode())


def test_column_sources_name_each_merged_column_by_security_and_field():
    spx = _export("SPX Index", "spx.csv", "TOT_RETURN_INDEX_GROSS_DVDS,PX_BID")
    vix = _export("VIX Index", "vix.csv", "PX_LAST,PX_BID")
    merged = merge([spx, vix])

    sources = column_sources([spx, vix], merged.columns)

    assert sources["SPX_Index_PX_BID"] == ColumnSource("SPX Index", "PX_BID")
    assert sources["VIX_Index_PX_LAST"].ticker == "VIX"
    assert len(sources) == 4


def test_column_sources_follow_a_column_relabelled_by_its_file():
    # Two files sharing a security are relabelled by filename in merge().
    price = _export("SPX Index", "spx_price.csv", "PX_LAST,PX_BID")
    total = _export("SPX Index", "spx_tr.csv", "TOT_RETURN_INDEX_GROSS_DVDS,PX_BID")
    merged = merge([price, total])

    sources = column_sources([price, total], merged.columns)

    assert sources["spx_tr_TOT_RETURN_INDEX_GROSS_DVDS"] == ColumnSource(
        "SPX Index", "TOT_RETURN_INDEX_GROSS_DVDS"
    )
    assert sources["spx_price_PX_LAST"].security == "SPX Index"


def _with(meta: str, rows: list[str], *, security="TEST Index", filename="test.csv"):
    """An export with ``meta`` lines under its Security line and ``rows`` under
    a ``Date,PX_LAST`` header."""
    data = f"Security,{security}\n{meta}\nDate,PX_LAST\n" + "".join(f"{r}\n" for r in rows)
    return read_export(filename, data.encode())


def _dates(export) -> list[str]:
    return [d.date().isoformat() for d in export.frame["Date"]]


# --- dates: the order is settled from the whole file, never guessed ----------


def test_a_real_export_reads_its_dates_month_first_as_before():
    meta = "Start Date,8/31/2016\nEnd Date,9/30/2026\nPeriod,D\n"
    export = _with(meta, ["9/30/2026,16.34", "9/29/2026,16.04", "9/1/2016,13.48"])

    assert _dates(export) == ["2026-09-30", "2026-09-29", "2016-09-01"]


def test_a_daily_file_whose_dates_all_read_either_way_is_read_the_way_that_steps_by_days():
    # 1/2/2020 and 1/3/2020 are a day apart read month first and a month apart
    # read day first; the file says it is daily.
    export = _with("Period,Daily\n", ["1/2/2020,100.0", "1/3/2020,101.0"])

    assert _dates(export) == ["2020-01-02", "2020-01-03"]


def test_a_day_first_daily_file_whose_days_are_all_twelve_or_less_is_read_day_first():
    export = _with("Period,D\n", ["03/01/2020,2.0", "02/01/2020,1.0"])

    assert _dates(export) == ["2020-01-03", "2020-01-02"]


def test_a_day_first_monthly_file_whose_days_are_all_twelve_or_less_is_read_day_first():
    # Every day is the 5th, so each date reads either way. Month first they
    # would step a day at a time, which a monthly file does not.
    rows = [f"05/{month:02d}/2025,{100 + month}" for month in range(1, 13)]

    export = _with("Period,M\n", rows)

    assert _dates(export)[:3] == ["2025-01-05", "2025-02-05", "2025-03-05"]


def test_one_date_that_reads_only_one_way_settles_the_order_for_the_whole_file():
    # The first row could be 9 October or 10 September; the second can only be
    # read day first, so the first is 9 October too.
    export = _with("", ["09/10/2026,1.0", "30/09/2026,2.0"])

    assert _dates(export) == ["2026-10-09", "2026-09-30"]


def test_the_start_and_end_date_lines_settle_the_order():
    meta = "Start Date,31/12/2019\nEnd Date,31/01/2020\n"

    export = _with(meta, ["2/1/2020,1.0", "3/1/2020,2.0"])

    assert _dates(export) == ["2020-01-02", "2020-01-03"]


def test_dates_nothing_in_the_file_can_settle_are_refused_rather_than_guessed():
    # No Period, no Start or End Date: 1/2/2020 is 2 January or 1 February.
    with pytest.raises(BloombergCsvError, match="month first or day first"):
        _with("", ["1/2/2020,1.0", "1/3/2020,2.0"])


def test_a_file_mixing_month_first_and_day_first_dates_is_refused():
    with pytest.raises(BloombergCsvError, match="13/01/2020"):
        _with("Period,D\n", ["1/14/2020,1.0", "13/01/2020,2.0"])


def test_excel_serial_numbers_in_place_of_dates_are_refused_with_how_to_fix_it():
    with pytest.raises(BloombergCsvError, match="format the Date column as a date"):
        _with("Period,D\n", ["46295,1.0", "46294,2.0"])


def test_dates_written_with_two_digit_years_are_refused_rather_than_guessed():
    # 26-09-28 is 28 September 2026 written year first, or 26 September 2028.
    with pytest.raises(BloombergCsvError, match="four-digit years"):
        _with("Period,D\n", ["26-09-28,1.0", "26-09-29,2.0"])
    with pytest.raises(BloombergCsvError, match="four-digit years"):
        _with("Period,D\n", ["9/29/2026,1.0", "9/30/26,2.0"])


def test_a_date_in_year_zero_is_refused_as_not_a_date():
    with pytest.raises(BloombergCsvError, match="1/1/0000 in the Date column is not a date"):
        _with("", ["1/1/0000,1.0", "1/2/0000,2.0"])


def test_an_iso_date_in_year_zero_is_read_as_no_date_for_the_schema_to_refuse():
    export = read_export("x.csv", b"Security,X Index\n\nDate,PX_LAST\n0000-01-13,1.0\n")

    assert export.frame["Date"].isna().all()


def test_rows_ending_in_a_trailing_comma_are_read_under_their_header():
    # The header is Date,PX_LAST but every row is 9/30/2026,16.3, - one empty
    # cell more than the header names.
    export = _with("Period,D\n", ["9/30/2026,16.3,", "9/29/2026,16.1,"])

    assert _dates(export) == ["2026-09-30", "2026-09-29"]
    assert export.frame["TEST_Index_PX_LAST"].tolist() == [16.3, 16.1]


def test_a_row_with_more_values_than_the_header_names_is_refused():
    with pytest.raises(BloombergCsvError, match="more values than the header"):
        _with("Period,D\n", ["1/2/2024,1,2,3", "1/3/2024,1"])


def test_a_cell_too_long_to_read_is_refused_with_a_plain_message():
    with pytest.raises(BloombergCsvError, match="could not be read"):
        _with("Period,D\n", ["1/2/2024," + "1" * 200_000])


def test_a_header_with_no_rows_under_it_is_refused():
    with pytest.raises(BloombergCsvError, match="no rows"):
        read_export("hdr.csv", b"Security,SPX Index\nPeriod,D\n\nDate,PX_LAST\n")


# --- numbers written with thousands separators --------------------------------


def test_numbers_written_with_thousands_separators_are_read_as_numbers():
    export = _with("Period,D\n", ['1/2/2024,"4,512.30"', '1/3/2024,"998.50"'])

    assert export.frame["TEST_Index_PX_LAST"].tolist() == [4512.30, 998.50]


def test_a_comma_that_could_be_a_decimal_comma_is_not_read_as_a_thousands_separator():
    # "4,512" alone is 4512 or 4.512; nothing in the column says which.
    export = _with("Period,D\n", ['1/2/2024,"4,512"', '1/3/2024,"4,498"'])

    assert export.frame["TEST_Index_PX_LAST"].tolist() == ["4,512", "4,498"]


# --- a file with no Security line ---------------------------------------------


def test_a_file_with_several_columns_and_no_security_line_is_refused():
    data = b"Period,D\n\nDate,PX_LAST,PX_BID\n2024-01-02,1.0,2.0\n"

    with pytest.raises(BloombergCsvError, match="no Security line"):
        read_export("mystery.csv", data)


def test_the_merged_download_cannot_be_uploaded_again_and_read_as_one_security():
    spx = _export("SPX Index", "spx.csv", "PX_LAST,PX_BID")
    vix = _export("VIX Index", "vix.csv", "PX_LAST,PX_BID")
    download = with_display_dates(merge([spx, vix])).to_csv(index=False).encode()

    with pytest.raises(BloombergCsvError, match="no Security line"):
        read_export("bloomberg_merged.csv", download)


def test_a_one_column_file_with_no_security_line_still_reads_labelled_by_its_name():
    export = read_export("my export (1).csv", b"Period,D\n\nDate,PX_LAST\n2024-01-02,1.0\n")

    assert list(export.frame.columns) == ["Date", "my_export_1_PX_LAST"]


def test_a_one_column_file_with_no_security_line_and_a_trailing_comma_still_reads():
    # The header ends in a comma, so pandas reads a third column, "Unnamed: 2",
    # empty on every row: padding, not a second field.
    export = read_export("my export.csv", b"Date,PX_LAST,\n9/29/2026,1,\n9/30/2026,2,\n")

    assert export.frame["my_export_PX_LAST"].tolist() == [1, 2]


# --- one security split over several files ------------------------------------


def _vix(filename: str, rows: list[str]):
    return _with("Period,D\n", rows, security="VIX Index", filename=filename)


def test_a_security_split_over_two_files_is_joined_into_one_series_and_said_so():
    late = _vix("vix_late.csv", ["1/7/2020,18.0", "1/6/2020,17.0"])
    early = _vix("vix_early.csv", ["1/3/2020,16.0", "1/2/2020,15.0"])

    exports, problems = join_split_files([late, early])

    assert problems == []
    (joined,) = exports
    assert joined.frame["VIX_Index_PX_LAST"].tolist() == [15.0, 16.0, 17.0, 18.0]
    assert [part.filename for part in joined.parts] == ["vix_late.csv", "vix_early.csv"]
    assert "vix_late.csv" in joined.notes[0] and "vix_early.csv" in joined.notes[0]
    assert list(merge(exports).columns) == ["Date", "VIX_Index_PX_LAST"]


def test_split_files_that_agree_where_they_overlap_keep_each_date_once():
    late = _vix("vix_late.csv", ["1/6/2020,17.0", "1/3/2020,16.0"])
    early = _vix("vix_early.csv", ["1/3/2020,16.0", "1/2/2020,15.0"])

    (joined,), problems = join_split_files([late, early])

    assert problems == []
    assert joined.frame["VIX_Index_PX_LAST"].tolist() == [15.0, 16.0, 17.0]


def test_a_blank_in_one_split_file_is_filled_from_the_other_not_counted_as_a_clash():
    # The earlier export was taken before 3 January's value was in, so Bloomberg
    # wrote #N/A N/A there; the later export has it. They do not disagree.
    late = _vix("vix_late.csv", ["1/6/2020,17.0", "1/3/2020,16.0"])
    early = _vix("vix_early.csv", ["1/3/2020,#N/A N/A", "1/2/2020,15.0"])

    (joined,), problems = join_split_files([late, early])

    assert problems == []
    assert joined.frame["VIX_Index_PX_LAST"].tolist() == [15.0, 16.0, 17.0]


def test_split_files_that_disagree_where_they_overlap_are_named_and_left_out():
    late = _vix("vix_late.csv", ["1/6/2020,17.0", "1/3/2020,16.5"])
    early = _vix("vix_early.csv", ["1/3/2020,16.0", "1/2/2020,15.0"])
    other = _with("Period,D\n", ["1/2/2020,1.0", "1/3/2020,2.0"], filename="other.csv")

    exports, problems = join_split_files([late, early, other])

    assert [e.filename for e in exports] == ["other.csv"]
    (problem,) = problems
    assert "VIX Index" in problem
    assert "03/01/2020" in problem


def test_two_files_of_one_security_with_different_fields_are_not_joined():
    price = _export("SPX Index", "spx_price.csv", "PX_LAST,PX_BID")
    total = _export("SPX Index", "spx_tr.csv", "TOT_RETURN_INDEX_GROSS_DVDS,PX_BID")

    exports, problems = join_split_files([price, total])

    assert exports[0] is price and exports[1] is total
    assert problems == []


def test_a_joined_security_beside_another_file_of_it_is_labelled_by_its_first_part():
    # The two price parts join; the total return file shares their security, so
    # merge labels both by file, the joined one as its first part on its own.
    first = _with("", ["1/14/2020,10", "1/15/2020,11"], security="SPX Index", filename="spx_a1.csv")
    second = _with("", ["1/15/2020,11", "1/16/2020,12"], security="SPX Index", filename="spx_2.csv")
    total = _export("SPX Index", "spx_tr.csv", "TOT_RETURN_INDEX_GROSS_DVDS,PX_BID")

    exports, _ = join_split_files([first, second, total])
    merged = merge(exports)

    assert list(merged.columns) == [
        "Date",
        "spx_a1_PX_LAST",
        "spx_tr_TOT_RETURN_INDEX_GROSS_DVDS",
        "spx_tr_PX_BID",
    ]
    assert column_sources(exports, merged.columns)["spx_a1_PX_LAST"] == ColumnSource(
        "SPX Index", "PX_LAST"
    )
