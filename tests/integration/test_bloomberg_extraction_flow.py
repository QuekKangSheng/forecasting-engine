"""End-to-end: raw Bloomberg CSV export bytes -> parsed per file -> merged on
Date -> validated, the path the Data page takes before anything reaches a model."""

from __future__ import annotations

import pytest

from forecasting_engine.extraction.bloomberg_csv import BloombergCsvError, merge, read_export
from forecasting_engine.extraction.validation import validate


def _export(security: str, filename: str, field: str, rows: str):
    data = f"Security,{security}\n\nDate,{field}\n{rows}"
    return read_export(filename, data.encode())


def test_two_exports_flow_from_raw_bytes_through_merge_to_a_clean_validation_report():
    spx = _export(
        "SPX Index",
        "spx.csv",
        "TOT_RETURN_INDEX_GROSS_DVDS",
        "2024-01-02,4750.50\n2024-01-03,4761.20\n2024-01-04,4739.80\n",
    )
    vix = _export(
        "VIX Index",
        "vix.csv",
        "PX_LAST",
        "2024-01-02,13.20\n2024-01-03,12.90\n2024-01-04,14.10\n",
    )

    merged = merge([spx, vix])

    assert list(merged.columns) == [
        "Date",
        "SPX_Index_TOT_RETURN_INDEX_GROSS_DVDS",
        "VIX_Index_PX_LAST",
    ]
    assert len(merged) == 3

    report = validate(merged)
    assert report.is_clean


def test_a_negative_total_return_value_survives_merge_and_is_still_caught_by_validation():
    spx = _export(
        "SPX Index",
        "spx.csv",
        "TOT_RETURN_INDEX_GROSS_DVDS",
        "2024-01-02,4750.50\n2024-01-03,-1.0\n2024-01-04,4739.80\n",
    )
    vix = _export(
        "VIX Index",
        "vix.csv",
        "PX_LAST",
        "2024-01-02,13.20\n2024-01-03,12.90\n2024-01-04,14.10\n",
    )

    merged = merge([spx, vix])
    report = validate(merged)

    # The labelled column name must still match validate's field check after
    # going through the full read -> merge chain, not just in isolation.
    assert not report.is_clean
    assert any("SPX_Index_TOT_RETURN" in e for e in report.schema_errors)


def test_a_file_with_no_date_header_is_refused_before_it_can_reach_merge():
    not_bloomberg = b"just,some,columns\n1,2,3\n"

    with pytest.raises(BloombergCsvError, match="no 'Date,...' header row found"):
        read_export("not_an_export.csv", not_bloomberg)
