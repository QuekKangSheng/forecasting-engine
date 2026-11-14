"""Robustness checks for the active generic Bloomberg extraction path."""

import pandas as pd

from forecasting_engine.extraction.bloomberg_csv import (
    ColumnSource,
    column_sources,
    drop_empty_columns,
    merge,
    missing_row_report,
    read_export,
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
