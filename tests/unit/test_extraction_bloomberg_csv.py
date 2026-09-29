"""Robustness checks for the active generic Bloomberg extraction path."""

import pandas as pd

from forecasting_engine.extraction.bloomberg_csv import (
    ColumnSource,
    column_sources,
    merge,
    missing_row_report,
    read_export,
)


def test_a_missing_value_on_an_unreadable_date_is_reported_without_crashing():
    frame = pd.DataFrame({"Date": [pd.NaT], "A_Index_PX_LAST": [None]})

    report = missing_row_report(frame)

    assert len(report) == 1
    assert report.iloc[0]["Likely reason"] == "Unreadable date"


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
