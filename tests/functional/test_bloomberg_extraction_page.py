"""The Bloomberg extraction page, driven through the real Streamlit page."""

import sys
from datetime import date
from io import BytesIO
from pathlib import Path

import openpyxl
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.ingest.upload import MAX_UPLOAD_BYTES
from forecasting_engine.store.uploads import recent_uploads

REPO_ROOT = Path(__file__).resolve().parents[2]
PAGE = REPO_ROOT / "app" / "app_pages" / "1_Data.py"
APP_DIR = REPO_ROOT / "app"

def export(security, rows, fields="PX_LAST"):
    lines = [f"Security,{security}", "Period,Daily", "", f"Date,{fields}", *rows]
    return ("\n".join(lines) + "\n").encode()


def xlsx_export(security, rows, fields=("PX_LAST",)):
    """A workbook shaped like a real Bloomberg .xlsx export: a Data sheet and
    a Metadata sheet, as raw bytes ready for the uploader."""
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = "Data"
    sheet.append(["Date", *fields])
    for row_date, *values in rows:
        sheet.append([row_date, *values])
    meta = book.create_sheet("Metadata")
    meta.append(["Field", "Value"])
    meta.append(["Security", security])
    buffer = BytesIO()
    book.save(buffer)
    return buffer.getvalue()


SPX = export("SPX Index", ["1/2/2020,100.0", "1/3/2020,101.0"])
VIX = export("VIX Index", ["1/2/2020,15.0", "1/3/2020,16.0"])

_XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


@pytest.fixture(autouse=True)
def app_on_path():
    sys.path.insert(0, str(APP_DIR))


@pytest.fixture
def page(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(str(PAGE), default_timeout=30)
    app.run()
    return app


def upload(page, files):
    """Upload through the **signal** uploader (index 1) — used by tests about
    generic Bloomberg merge/validate behaviour, unrelated to target-role
    detection. The securities these use (SPX, VIX, A Index, ...) are picked
    as arbitrary examples, not because their ticker matters."""
    page.file_uploader[1].set_value(files)
    return page.run()


def upload_targets(page, files):
    """Upload through the **target** uploader (index 0) — used by tests about
    target role/field detection specifically."""
    page.file_uploader[0].set_value(files)
    return page.run()


def texts(elements):
    return " ".join(e.value for e in elements)


def test_nothing_is_shown_before_a_file_is_uploaded(page):
    assert not page.success
    assert not page.error


def test_multiple_files_are_merged_into_one_frame(page):
    result = upload(page, [("spx.csv", SPX, "text/csv"), ("vix.csv", VIX, "text/csv")])

    assert not result.error
    assert "Merged 2 file(s) into 2 rows, 2 data columns" in texts(result.success)


def test_an_xlsx_export_merges_the_same_as_a_csv_one(page):
    spx_xlsx = xlsx_export(
        "SPX Index", [(date(2020, 1, 2), 100.0), (date(2020, 1, 3), 101.0)]
    )

    result = upload(page, [("spx.xlsx", spx_xlsx, _XLSX_MIME)])

    assert not result.error
    assert "Merged 1 file(s) into 2 rows, 1 data columns" in texts(result.success)
    merged = result.session_state["signal_merged"]
    assert "SPX_Index_PX_LAST" in merged.columns


def test_a_mixed_csv_and_xlsx_upload_merges_into_one_frame(page):
    vix_xlsx = xlsx_export("VIX Index", [(date(2020, 1, 2), 15.0), (date(2020, 1, 3), 16.0)])

    result = upload(page, [("spx.csv", SPX, "text/csv"), ("vix.xlsx", vix_xlsx, _XLSX_MIME)])

    assert not result.error
    assert "Merged 2 file(s) into 2 rows, 2 data columns" in texts(result.success)


def test_original_exports_with_trailing_cells_and_any_field_merge_unchanged(page):
    total_return = (
        b"Security,SPX Index,\nPeriod,D,\n,,\n"
        b"Date,TOT_RETURN_INDEX_GROSS_DVDS,PX_BID\n"
        b"2024-01-02,100.0,99.9\n2024-01-03,101.0,100.9\n"
    )

    result = upload(page, [("spx-total-return.csv", total_return, "text/csv")])

    assert not result.error
    assert "Merged 1 file(s)" in texts(result.success)
    merged = result.session_state["signal_merged"]
    assert "SPX_Index_TOT_RETURN_INDEX_GROSS_DVDS" in merged.columns


def test_a_file_with_no_date_header_is_reported_as_an_error_not_a_crash(page):
    result = upload(page, [("bad.csv", b"not,a,bloomberg,file\n1,2,3\n", "text/csv")])

    assert not result.exception
    assert "no 'Date,...' header row" in texts(result.error)


def test_a_file_that_fails_the_schema_alone_is_reported_and_nothing_merges(page):
    bad = export("BAD Index", ["1/2/2020,restricted", "1/3/2020,101.0"])
    result = upload(page, [("bad.csv", bad, "text/csv")])

    assert not result.success
    assert "PX_LAST" in texts(result.error)
    assert "Upload target index files and/or signal files above to get started" in texts(
        result.info
    )


def test_a_bad_type_file_among_good_files_is_excluded_while_good_ones_merge(page):
    bad = export("BAD Index", ["1/2/2020,restricted", "1/3/2020,101.0"])
    result = upload(page, [("spx.csv", SPX, "text/csv"), ("bad.csv", bad, "text/csv")])

    assert "PX_LAST" in texts(result.error)
    assert "Merged 1 file(s) into 2 rows, 1 data columns" in texts(result.success)


def test_the_coverage_metrics_are_shown_for_a_clean_file(page):
    result = upload(page, [("spx.csv", SPX, "text/csv")])

    labels = {m.label: m.value for m in result.metric}
    assert labels["Rows"] == "2"
    assert labels["Signals"] == "1"
    assert labels["Flagged"] == "0"
    assert "Nothing flagged" in texts(result.success)
    assert "02/01/2020 to 03/01/2020" in texts(result.caption)


def test_duplicate_dates_are_reported_and_named(page):
    # Same value both rows, so this exercises duplicate-date detection only —
    # not also the day-over-day move check.
    dup = export("A Index", ["1/2/2020,1.0", "1/2/2020,1.0"])
    result = upload(page, [("a.csv", dup, "text/csv")])

    labels = {m.label: m.value for m in result.metric}
    assert labels["Flagged"] == "1"
    assert any("Duplicate dates" in e.label for e in result.expander)
    assert "02/01/2020" in texts(result.markdown)


def test_the_data_page_has_no_fama_french_section_and_only_the_merged_download(page):
    result = upload(page, [("spx.csv", SPX, "text/csv")])

    assert [d.label for d in result.download_button] == ["Bloomberg merged (.csv)"]
    assert "Fama-French" not in texts(result.markdown)
    assert all("factor" not in b.label.lower() for b in result.button)


def test_both_uploaders_are_always_multi_file_without_a_toggle(page):
    assert not page.toggle
    assert page.file_uploader[0].accept_multiple_files is True
    assert page.file_uploader[1].accept_multiple_files is True


def test_an_oversized_export_is_rejected_before_merging(page):
    oversized = SPX + b"#" * (MAX_UPLOAD_BYTES - len(SPX) + 1)

    result = upload(page, [("oversized.csv", oversized, "text/csv")])

    assert "25.0 MB" in texts(result.error)
    assert not result.success


def test_a_column_with_no_data_at_all_is_dropped_and_noted(page):
    # Every JPMVXYGL_Index_PX_BID cell is Bloomberg's "#N/A N/A" — no data.
    rows = ["1/2/2020,6.5,#N/A N/A", "1/3/2020,6.6,#N/A N/A"]
    csv = export("JPMVXYGL Index", rows, fields="PX_LAST,PX_BID")
    result = upload(page, [("jpm.csv", csv, "text/csv")])

    merged = result.session_state["signal_merged"]
    assert "JPMVXYGL_Index_PX_BID" not in merged.columns
    assert "Dropped 1 column" in texts(result.caption)


# --- explaining and excluding gap rows -------------------------------------


def gappy_csv():
    # 2016-09-05 is Labor Day; LF98TRUU has no row for it, LEGATRUU does —
    # merging on Date leaves that cell blank for LF98TRUU.
    lf98truu = export("LF98TRUU Index", ["9/6/2016,1772.0", "9/2/2016,1771.03"])
    legatruu = export(
        "LEGATRUU Index", ["9/6/2016,486.5722", "9/5/2016,482.7748", "9/2/2016,481.9552"]
    )
    return lf98truu, legatruu


def test_no_missing_section_when_nothing_is_missing(page):
    result = upload(page, [("spx.csv", SPX, "text/csv")])
    assert "Rows missing a value" not in texts(result.markdown)


def test_a_gap_is_listed_and_committed_blank_never_filled(page):
    lf98truu, legatruu = gappy_csv()
    result = upload(
        page, [("lf98truu.csv", lf98truu, "text/csv"), ("legatruu.csv", legatruu, "text/csv")]
    )
    assert "Rows missing a value" in texts(result.markdown)
    assert all("fill" not in n.label.lower() for n in result.number_input)

    commit_button = next(b for b in result.button if b.label == "Use Updated Data")
    result = commit_button.click().run()

    committed = result.session_state["extraction_committed"].set_index("Date")
    assert len(committed) == 3
    assert pd.isna(committed.loc[pd.Timestamp("2016-09-05"), "LF98TRUU_Index_PX_LAST"])


def test_committing_records_each_columns_security_and_field(page):
    upload_targets(page, [("spx.csv", SPX, "text/csv")])
    result = upload(page, [("vix.csv", VIX, "text/csv")])

    commit_button = next(b for b in result.button if b.label == "Use Updated Data")
    result = commit_button.click().run()

    sources = result.session_state["extraction_committed_sources"]
    assert sources["SPX_Index_PX_LAST"].security == "SPX Index"
    assert sources["VIX_Index_PX_LAST"].field == "PX_LAST"


def test_a_target_column_is_never_filled_even_for_a_short_gap(page):
    # SPX (target) is missing exactly one row — short enough that a signal
    # would auto-fill it, but targets never are: a filled price on a day the
    # target's own market was shut would fabricate a return that never
    # happened.
    spx = export("SPX Index", ["1/2/2020,100.0", "1/6/2020,103.0"])
    vix = export("VIX Index", ["1/2/2020,15.0", "1/3/2020,15.5", "1/6/2020,16.0"])

    upload_targets(page, [("spx.csv", spx, "text/csv")])
    result = upload(page, [("vix.csv", vix, "text/csv")])

    commit_button = next(b for b in result.button if b.label == "Use Updated Data")
    result = commit_button.click().run()

    committed = result.session_state["extraction_committed"]
    by_date = committed.set_index("Date")
    assert pd.isna(by_date.loc[pd.Timestamp("2020-01-03"), "SPX_Index_PX_LAST"])


# --- the merge is kept and logged like any other upload -----------------------


def test_the_merge_is_logged_to_duckdb_once(page, tmp_path):
    result = upload(page, [("spx.csv", SPX, "text/csv"), ("vix.csv", VIX, "text/csv")])
    result.run()  # a rerun must not log it again

    (row,) = recent_uploads(db_path=tmp_path / "data" / "forecasting.duckdb")
    assert row.filename == "bloomberg_merged.csv"
    assert row.row_count == 2
    assert "content hash" in " ".join(c.value for c in result.caption)


def test_the_merged_bytes_are_stored_under_their_hash(page, tmp_path):
    upload(page, [("spx.csv", SPX, "text/csv")])

    (stored,) = (tmp_path / "data" / "uploads").glob("*.csv")
    assert stored.read_bytes().startswith(b"Date,SPX_Index_PX_LAST\n2020-01-02,")


# --- target vs. signal split (Phase 4.1) ------------------------------------


def test_target_role_is_pre_filled_from_the_recognised_equity_ticker(page):
    spx = export("SPX Index", ["1/2/2020,100.0", "1/3/2020,101.0"])
    result = upload_targets(page, [("spx.csv", spx, "text/csv")])

    assert not result.error
    assert "1 target file(s) merged into 2 rows" in texts(result.success)
    assert result.selectbox[0].value == TargetRole.EQUITY


def test_bond_ticker_is_pre_filled_as_the_bond_role(page):
    bond = export("LBUSTRUU Index", ["1/2/2020,100.0", "1/3/2020,101.0"])
    result = upload_targets(page, [("agg.csv", bond, "text/csv")])

    assert result.selectbox[0].value == TargetRole.BOND


def test_an_unrecognised_ticker_leaves_the_role_unset_but_does_not_break_the_page(page):
    mystery = export("ZZZ Index", ["1/2/2020,1.0", "1/3/2020,2.0"])
    result = upload_targets(page, [("mystery.csv", mystery, "text/csv")])

    assert not result.error
    assert not result.exception
    assert result.selectbox[0].value is None
    # The rest of the page still works even though no role was resolved.
    labels = {m.label: m.value for m in result.metric}
    assert labels["Rows"] == "2"


def test_an_unassigned_target_file_is_warned_about(page):
    mystery = export("ZZZ Index", ["1/2/2020,1.0", "1/3/2020,2.0"])
    result = upload_targets(page, [("mystery.csv", mystery, "text/csv")])

    assert "no role picked" in texts(result.warning)
    assert "mystery.csv" in texts(result.warning)


def test_the_field_defaults_to_total_return_when_present(page):
    total_return = (
        b"Security,SPX Index,\nPeriod,D,\n,,\n"
        b"Date,PX_LAST,TOT_RETURN_INDEX_GROSS_DVDS\n"
        b"2024-01-02,100.0,105.0\n2024-01-03,101.0,106.0\n"
    )
    result = upload_targets(page, [("spx-tr.csv", total_return, "text/csv")])

    # Selectbox 0 is the role, selectbox 1 is the field.
    assert result.selectbox[1].value == "TOT_RETURN_INDEX_GROSS_DVDS"


def test_a_target_and_signal_resolving_to_the_same_column_is_renamed_by_file(page):
    # Different filenames, same security and field — the filename check
    # can't catch this, but the resulting column name still collides. Rather
    # than rejecting it, the signal's column is relabelled by its own
    # filename instead, the same fallback bloomberg_csv.merge() uses when two
    # target files collide.
    spx = export("SPX Index", ["1/2/2020,100.0", "1/3/2020,101.0"])
    spx_again = export("SPX Index", ["1/2/2020,105.0", "1/3/2020,106.0"])

    upload_targets(page, [("spx-target.csv", spx, "text/csv")])
    result = upload(page, [("spx-signal-copy.csv", spx_again, "text/csv")])

    assert not result.error
    assert "Renamed 1 signal column" in texts(result.caption)
    assert "SPX_Index_PX_LAST" in texts(result.caption)

    commit_button = next(b for b in result.button if b.label == "Use Updated Data")
    result = commit_button.click().run()

    committed = result.session_state["extraction_committed"]
    assert "SPX_Index_PX_LAST" in committed.columns
    assert "spx_signal_copy_PX_LAST" in committed.columns


def test_a_chosen_role_and_field_are_remembered_outside_the_widget(page):
    # Stored in plain session state, not just the widget's own key, so a
    # fresh script run on a different page (which never saw this widget
    # instance before) still has it to fall back on.
    legatruu = export(
        "LEGATRUU Index", ["1/2/2020,100.0", "1/3/2020,101.0"], fields="PX_BID"
    )
    result = upload_targets(page, [("bond.csv", legatruu, "text/csv")])
    result = result.selectbox[0].select(TargetRole.BOND).run()

    assert result.session_state["_target_role_choices"]["bond.csv"] == TargetRole.BOND
    assert result.session_state["_target_field_choices"]["bond.csv"] == "PX_BID"


def test_clear_data_also_clears_stale_model_results(page):
    # Model results are namespaced per target role in app_pages/3_Models.py
    # (e.g. "polynomial_result_equity") — a result fit on data that's about
    # to be cleared shouldn't keep showing on the Model Metrics page.
    page.session_state["polynomial_result_equity"] = "stale"
    page.session_state["ml_result_bond"] = "stale"

    clear_button = next(b for b in page.button if b.label == "Clear Data")
    result = clear_button.click().run()

    assert "polynomial_result_equity" not in result.session_state.filtered_state
    assert "ml_result_bond" not in result.session_state.filtered_state


def test_two_files_set_to_the_same_role_is_reported_as_an_error(page):
    spx = export("SPX Index", ["1/2/2020,100.0", "1/3/2020,101.0"])
    other = export("ZZZ Index", ["1/2/2020,1.0", "1/3/2020,2.0"])
    result = upload_targets(
        page, [("spx.csv", spx, "text/csv"), ("zzz.csv", other, "text/csv")]
    )

    # ZZZ's role is unrecognised (selectbox index 2 — file 2's role) —
    # force a collision by selecting Equity for it too, same as file 1.
    result = result.selectbox[2].select(TargetRole.EQUITY).run()

    assert "same role" in texts(result.error)


def test_the_same_security_cannot_fill_both_target_roles(page):
    # Two files for the same security (SPX price and SPX total return) —
    # assigning one to Equity and the other to Bond would silently make
    # "the bond target" just be SPX again, so it's rejected instead.
    spx_price = export("SPX Index", ["1/2/2020,100.0", "1/3/2020,101.0"], fields="PX_LAST")
    spx_total_return = export(
        "SPX Index", ["1/2/2020,99.0", "1/3/2020,100.0"], fields="TOT_RETURN_INDEX_GROSS_DVDS"
    )
    result = upload_targets(
        page,
        [
            ("spx_price.csv", spx_price, "text/csv"),
            ("spx_total_return.csv", spx_total_return, "text/csv"),
        ],
    )
    result = result.selectbox[2].select(TargetRole.BOND).run()

    assert "can't fill both roles" in texts(result.error)
    assert "SPX Index" in texts(result.error)


def test_the_same_filename_in_both_uploads_is_rejected(page):
    spx = export("SPX Index", ["1/2/2020,100.0", "1/3/2020,101.0"])

    upload_targets(page, [("spx.csv", spx, "text/csv")])
    result = upload(page, [("spx.csv", spx, "text/csv")])

    assert "both a target and a signal" in texts(result.error)
    assert "spx.csv" in texts(result.error)


def test_target_and_signal_files_combine_and_commit_with_resolved_targets(page):
    spx = export("SPX Index", ["1/2/2020,100.0", "1/3/2020,101.0"])
    vix = export("VIX Index", ["1/2/2020,15.0", "1/3/2020,16.0"])

    upload_targets(page, [("spx.csv", spx, "text/csv")])
    result = upload(page, [("vix.csv", vix, "text/csv")])

    commit_button = next(b for b in result.button if b.label == "Use Updated Data")
    result = commit_button.click().run()

    committed = result.session_state["extraction_committed"]
    assert "SPX_Index_PX_LAST" in committed.columns
    assert "VIX_Index_PX_LAST" in committed.columns
    assert len(committed) == 2

    targets = result.session_state["extraction_committed_targets"]
    assert targets[TargetRole.EQUITY] == "SPX_Index_PX_LAST"
    assert TargetRole.BOND not in targets


def test_missing_one_target_role_does_not_block_committing(page):
    spx = export("SPX Index", ["1/2/2020,100.0", "1/3/2020,101.0"])
    result = upload_targets(page, [("spx.csv", spx, "text/csv")])

    commit_button = next(b for b in result.button if b.label == "Use Updated Data")
    result = commit_button.click().run()

    assert "now committed" in texts(result.success)
