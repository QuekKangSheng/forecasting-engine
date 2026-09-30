"""The Bloomberg extraction panel: file upload, merge, validate, download.

Rendering only. Parsing, merging and validation live in
``forecasting_engine.extraction`` and know nothing about Streamlit.

The validation report renderer is shared between this page and the Home
page's summary — that sharing is the whole integration between the two.

Targets and signals are uploaded separately (Phase 4.1 of the ingestion plan):
which files are the two forecasting targets is a structural choice made here,
at ingestion, not inferred later on the Models page and not left to a free
column picker — a target column can no longer end up in the signal set by
construction, whatever it's named.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass, field

import pandas as pd
import streamlit as st

import model_runs
import ui
from forecasting_engine.extraction import bloomberg_csv, bloomberg_xlsx, validation
from forecasting_engine.extraction.bloomberg_csv import BloombergCsvExport
from forecasting_engine.extraction.targets import PREFERRED_FIELD, TARGET_TICKERS, TargetRole
from forecasting_engine.extraction.validation import ValidationReport
from forecasting_engine.ingest.align import MAX_STALENESS
from forecasting_engine.ingest.upload import (
    MAX_UPLOAD_BYTES,
    AcceptedUpload,
    UploadError,
    accept_upload,
    check_extension,
    check_size,
)
from forecasting_engine.store.uploads import record_upload

#: The two uploaders' own working copies — refreshed on every new upload,
#: regardless of any commit below.
TARGET_MERGED_KEY = "target_merged"
TARGET_REPORT_KEY = "target_report"
SIGNAL_MERGED_KEY = "signal_merged"
SIGNAL_REPORT_KEY = "signal_report"
_TARGET_EXPORTS_KEY = "_target_exports"
_SIGNAL_EXPORTS_KEY = "_signal_exports"
_TARGET_DROPPED_KEY = "_target_dropped"
_SIGNAL_DROPPED_KEY = "_signal_dropped"
#: filename -> the Role/Field the user last picked for it, so navigating to
#: another page and back doesn't lose the choice (a fresh upload resets the
#: file_uploader widget itself, but this dict is plain session state).
_TARGET_ROLE_CHOICES_KEY = "_target_role_choices"
_TARGET_FIELD_CHOICES_KEY = "_target_field_choices"

#: The cleaned, combined frame (and its own report) that Home and Models
#: read. Only updated when the user clicks "Use Updated Data" —
#: not on every gap-review edit — so those pages stay stable while decisions
#: are still being made.
COMMITTED_KEY = "extraction_committed"
COMMITTED_REPORT_KEY = "extraction_committed_report"
#: role -> resolved column name in COMMITTED_KEY's frame, for whichever roles
#: were actually resolved at the last commit. Read by the Models page
#: (Phase 4.2) to know which columns are targets and must never be treated as
#: signals.
COMMITTED_TARGETS_KEY = "extraction_committed_targets"
#: column -> ColumnSource (security, field) for COMMITTED_KEY's frame, so the
#: Models page can leave out every field of a target's security.
COMMITTED_SOURCES_KEY = "extraction_committed_sources"
#: Columns dropped at ingest for holding no data, for the data quality report.
COMMITTED_DROPPED_KEY = "extraction_committed_dropped"

#: What the merged file is called in the upload log and under data/uploads.
MERGED_NAME = "bloomberg_merged.csv"
_LOGGED_KEY = "_logged_bloomberg_merge"

#: Bumped by "Clear Data" so both file_uploaders get a fresh widget key and
#: drop whatever files they were showing, instead of re-displaying them.
_UPLOADER_VERSION_KEY = "_bloomberg_uploader_version"

def _fmt(date) -> str:
    """A date for display: dd/mm/yyyy, no time component."""
    return pd.Timestamp(date).strftime(bloomberg_csv.DATE_DISPLAY_FORMAT)


@dataclass
class _UploadResult:
    """One uploader's parse-and-merge outcome."""

    exports: list[BloombergCsvExport] = field(default_factory=list)
    merged: pd.DataFrame | None = None
    report: ValidationReport | None = None
    errors: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)


def _parse_and_merge(files) -> _UploadResult:
    """Read every uploaded file, merge the good ones, validate the result.

    Shared between the target and signal uploaders — parsing, merging and
    validating a batch of Bloomberg exports doesn't depend on what role
    they'll play afterward.
    """
    result = _UploadResult()
    for file in files:
        try:
            data = file.getvalue()
            check_extension(file.name)
            check_size(len(data), filename=file.name)
            reader = (
                bloomberg_xlsx.read_export
                if file.name.lower().endswith(".xlsx")
                else bloomberg_csv.read_export
            )
            export = reader(file.name, data)
        except (
            UploadError,
            bloomberg_csv.BloombergCsvError,
            bloomberg_xlsx.BloombergXlsxError,
        ) as exc:
            result.errors.append(str(exc))
            continue

        # A file whose own data fails the schema (wrong type, an impossible
        # value) is excluded the same way a file that fails to parse is —
        # good files still merge, this one does not ride along with a bad
        # value in it. Sorted by date first: a single export's own row order
        # isn't guaranteed ascending (only the final merge() output is), so
        # checking the raw order here would flag a fine file as broken.
        sorted_frame = export.frame.sort_values(bloomberg_csv.DATE_COLUMN)
        file_errors = validation.schema_errors(sorted_frame)
        if file_errors:
            result.errors.append(f"{file.name}: " + "; ".join(file_errors))
            continue
        result.exports.append(export)

    if result.exports:
        merged = bloomberg_csv.merge(result.exports)
        merged, dropped = bloomberg_csv.drop_empty_columns(merged)
        result.merged = merged
        result.dropped = dropped
        result.report = validation.validate(merged)
    return result


def render() -> None:
    ui.inject()
    st.header("Bloomberg data extraction", anchor=False, divider="grey")

    if st.button(
        "Clear Data",
        icon=":material/delete_sweep:",
        help="Remove the current upload so you can start over.",
    ):
        for key in (
            TARGET_MERGED_KEY,
            TARGET_REPORT_KEY,
            _TARGET_EXPORTS_KEY,
            _TARGET_DROPPED_KEY,
            _TARGET_ROLE_CHOICES_KEY,
            _TARGET_FIELD_CHOICES_KEY,
            SIGNAL_MERGED_KEY,
            SIGNAL_REPORT_KEY,
            _SIGNAL_EXPORTS_KEY,
            _SIGNAL_DROPPED_KEY,
            COMMITTED_KEY,
            COMMITTED_REPORT_KEY,
            COMMITTED_TARGETS_KEY,
            COMMITTED_SOURCES_KEY,
            COMMITTED_DROPPED_KEY,
            _LOGGED_KEY,
        ):
            st.session_state.pop(key, None)
        # Old results are stale once the data they were fit on is cleared.
        st.session_state.pop(model_runs.RUNS_KEY, None)
        st.session_state[_UPLOADER_VERSION_KEY] = (
            st.session_state.get(_UPLOADER_VERSION_KEY, 0) + 1
        )
        st.rerun()

    version = st.session_state.get(_UPLOADER_VERSION_KEY, 0)

    st.markdown(ui.eyebrow("Target index files"), unsafe_allow_html=True)
    st.caption(
        "The two series being forecast — the S&P 500 and the US Aggregate bond "
        "index, both total-return. Upload each as its own Bloomberg export; the "
        "security named in the file is used to guess which role it fills, shown "
        "below for you to confirm or change."
    )
    target_files = st.file_uploader(
        "Target exports (.csv or .xlsx)",
        type=["csv", "xlsx"],
        accept_multiple_files=True,
        key=f"target_uploader_{version}",
    )
    target_columns, target_merged = _render_target_uploader(target_files, version)

    st.divider()
    st.markdown(ui.eyebrow("Signal files"), unsafe_allow_html=True)
    st.caption(
        f"Every other market or macro signal, up to {MAX_UPLOAD_BYTES // 1_000_000} MB "
        "each. All selected files are merged on Date with an outer join — none of "
        "them can become a target, whatever column they carry."
    )
    signal_files = st.file_uploader(
        "Signal exports (.csv or .xlsx)",
        type=["csv", "xlsx"],
        accept_multiple_files=True,
        key=f"signal_uploader_{version}",
    )
    if signal_files:
        with st.spinner("Reading and merging the signal files…"):
            result = _parse_and_merge(signal_files)
        for message in result.errors:
            st.error(message, icon=":material/error:")
        if result.merged is not None:
            st.success(
                f"Merged {len(result.exports)} file(s) into {len(result.merged):,} rows, "
                f"{len(result.merged.columns) - 1} data columns.",
                icon=":material/check_circle:",
            )
            if result.dropped:
                st.caption(
                    f"Dropped {len(result.dropped)} column(s) with no data at all (the "
                    f"security has nothing for that field): {', '.join(result.dropped)}."
                )
            st.dataframe(
                bloomberg_csv.with_display_dates(result.merged.head(10)), width="stretch"
            )
            st.session_state[SIGNAL_MERGED_KEY] = result.merged
            st.session_state[SIGNAL_REPORT_KEY] = result.report
            st.session_state[_SIGNAL_EXPORTS_KEY] = result.exports
            st.session_state[_SIGNAL_DROPPED_KEY] = result.dropped

    signal_merged = st.session_state.get(SIGNAL_MERGED_KEY)

    # Uploading the same file to both merges its data in twice, once under
    # each role's own label — caught here before it happens.
    target_names = {e.filename for e in st.session_state.get(_TARGET_EXPORTS_KEY, [])}
    signal_names = {e.filename for e in st.session_state.get(_SIGNAL_EXPORTS_KEY, [])}
    overlapping_names = sorted(target_names & signal_names)
    if overlapping_names:
        st.error(
            "The same file is uploaded as both a target and a signal: "
            f"{', '.join(overlapping_names)}. Remove it from one side — uploading it to "
            "both merges its data in twice.",
            icon=":material/error:",
        )
        st.stop()

    # A target and a signal can also collide under different filenames, if
    # they resolve to the same column name (e.g. the same security's field,
    # re-exported separately, or a field left over in the target file that
    # wasn't picked for the target role). Uncaught, pandas silently suffixes
    # both to _x/_y instead of keeping either. Relabelled by filename instead
    # — the same fallback bloomberg_csv.merge() already uses when two target
    # files share a security.
    if target_merged is not None and signal_merged is not None:
        column_overlap = sorted(
            (set(target_merged.columns) & set(signal_merged.columns))
            - {bloomberg_csv.DATE_COLUMN}
        )
        if column_overlap:
            signal_merged = _deconflict_signal_columns(
                signal_merged, st.session_state.get(_SIGNAL_EXPORTS_KEY, []), column_overlap
            )
            st.caption(
                f"Renamed {len(column_overlap)} signal column(s) that matched a target "
                f"column name, using the file name instead: {', '.join(column_overlap)}."
            )

    # Falling back to session state (rather than returning when nothing was
    # just uploaded) is what keeps this page showing the last merge instead
    # of going blank when the user navigates here from another tab.
    if target_merged is None and signal_merged is None:
        st.info(
            "Upload target index files and/or signal files above to get started.",
            icon=":material/upload_file:",
        )
        return

    combined = _combine(target_merged, signal_merged)

    file_id = "|".join(
        getattr(f, "file_id", f.name) for f in (list(target_files or []) + list(signal_files or []))
    )
    if file_id:
        _keep_and_log(combined, file_id=file_id)

    # Reserved here so the summary stays in its usual position, but filled in
    # further down — after the "Use Updated Data" button has had a chance to
    # run in this same script pass. That lets a same-click commit show up
    # without an st.rerun(), which would tear down and repaint the whole page
    # (the "flash" a full rerun causes here).
    summary_slot = st.container()

    _render_gap_review(combined)

    st.divider()
    st.markdown(ui.eyebrow("Using this data"), unsafe_allow_html=True)
    st.caption(
        "This dataset must be uploaded here before it can be used anywhere else in "
        "the application — every other page reads whichever dataset was last "
        "committed here, and none of them update on every gap-review edit. Make "
        "your include/clean choices above, then click below to push them through."
    )
    if st.button("Use Updated Data", icon=":material/publish:"):
        with st.spinner("Validating the cleaned dataset…"):
            st.session_state[COMMITTED_KEY] = combined
            st.session_state[COMMITTED_REPORT_KEY] = validation.validate(combined)
            st.session_state[COMMITTED_TARGETS_KEY] = target_columns
            st.session_state[COMMITTED_SOURCES_KEY] = bloomberg_csv.column_sources(
                [
                    *st.session_state.get(_TARGET_EXPORTS_KEY, []),
                    *st.session_state.get(_SIGNAL_EXPORTS_KEY, []),
                ],
                combined.columns,
            )
            st.session_state[COMMITTED_DROPPED_KEY] = _dropped()
        st.success(
            "This cleaned dataset is now committed and available throughout the application.",
            icon=":material/check_circle:",
        )
    elif COMMITTED_KEY in st.session_state:
        st.caption("A dataset is committed for Home and Models.")
    else:
        st.caption("Nothing committed yet — Home and Models have no data yet.")

    # Filled in now (not where reserved above) so this reflects a commit made
    # by the button just above it, in this same run — no rerun needed.
    with summary_slot:
        committed_report = st.session_state.get(COMMITTED_REPORT_KEY, validation.validate(combined))
        committed_merged = st.session_state.get(COMMITTED_KEY, combined)
        committed_dropped = st.session_state.get(COMMITTED_DROPPED_KEY, _dropped())
        _render_report(committed_report, committed_merged, committed_dropped)

    st.download_button(
        "Bloomberg merged (.csv)",
        data=bloomberg_csv.with_display_dates(combined).to_csv(index=False).encode(),
        file_name="bloomberg_merged.csv",
        mime="text/csv",
        icon=":material/download:",
    )


def _render_target_uploader(
    files, version: int
) -> tuple[dict[TargetRole, str], pd.DataFrame | None]:
    """Parse, merge and validate the target files; let the user confirm or
    override which role (equity/bond) and which field each one fills.

    Returns the resolved role -> column-name mapping (only for roles actually
    resolved this render) and the merged target frame, ``None``/empty when
    nothing is uploaded and nothing was committed before.
    """
    if files:
        with st.spinner("Reading and merging the target files…"):
            result = _parse_and_merge(files)
        for message in result.errors:
            st.error(message, icon=":material/error:")
        if result.merged is not None:
            st.session_state[TARGET_MERGED_KEY] = result.merged
            st.session_state[TARGET_REPORT_KEY] = result.report
            st.session_state[_TARGET_EXPORTS_KEY] = result.exports
            st.session_state[_TARGET_DROPPED_KEY] = result.dropped
            if result.dropped:
                st.caption(
                    f"Dropped {len(result.dropped)} column(s) with no data at all: "
                    f"{', '.join(result.dropped)}."
                )

    merged = st.session_state.get(TARGET_MERGED_KEY)
    exports: list[BloombergCsvExport] = st.session_state.get(_TARGET_EXPORTS_KEY, [])
    if merged is None:
        return {}, None

    st.success(
        f"{len(exports)} target file(s) merged into {len(merged):,} rows.",
        icon=":material/check_circle:",
    )

    # Mirrors bloomberg_csv.merge()'s own collision handling: when two target
    # files share a security (SPX price and SPX total return are both "SPX
    # Index"), merge() relabels both by filename instead — so the column this
    # loop looks up in the merged frame has to be computed the same way, or a
    # two-file security collision reports "couldn't find" instead of letting
    # the same-security-two-roles check below ever see it.
    own_labels = [bloomberg_csv.label(e.security, e.filename) for e in exports]
    label_counts = Counter(own_labels)

    role_choices: dict[str, TargetRole | None] = st.session_state.setdefault(
        _TARGET_ROLE_CHOICES_KEY, {}
    )
    field_choices: dict[str, str] = st.session_state.setdefault(_TARGET_FIELD_CHOICES_KEY, {})

    resolved: dict[TargetRole, str] = {}
    resolved_by: dict[TargetRole, str] = {}
    role_by_security: dict[str, TargetRole] = {}
    for export, lbl in zip(exports, own_labels, strict=True):
        fields = [
            c[len(lbl) + 1 :]
            for c in export.frame.columns
            if c != bloomberg_csv.DATE_COLUMN and c.startswith(f"{lbl}_")
        ]
        if not fields:
            continue

        remembered_role = role_choices.get(
            export.filename, TARGET_TICKERS.get(export.security)
        )
        role_options = list(TargetRole)
        role_idx = role_options.index(remembered_role) if remembered_role is not None else None

        name_col, role_col, field_col = st.columns([2, 1, 1])
        name_col.caption(
            f"**{export.filename}**  \nSecurity: {export.security or 'not found in the file'}"
        )
        role = role_col.selectbox(
            "Role",
            role_options,
            index=role_idx,
            format_func=lambda r: "Equity target" if r == TargetRole.EQUITY else "Bond target",
            key=f"target_role_{export.filename}_{version}",
        )
        role_choices[export.filename] = role

        remembered_field = field_choices.get(export.filename)
        if remembered_field in fields:
            default_field_idx = fields.index(remembered_field)
        elif PREFERRED_FIELD in fields:
            default_field_idx = fields.index(PREFERRED_FIELD)
        else:
            default_field_idx = 0
        field_choice = field_col.selectbox(
            "Field",
            fields,
            index=default_field_idx,
            key=f"target_field_{export.filename}_{version}",
        )
        field_choices[export.filename] = field_choice

        if role is None:
            st.warning(
                f"{export.filename!r} has no role picked — it still merges into the committed "
                "data and would ride along as an ordinary signal. Pick a role or remove the file.",
                icon=":material/warning:",
            )
            continue
        merged_label = bloomberg_csv.label("", export.filename) if label_counts[lbl] > 1 else lbl
        resolved_col = f"{merged_label}_{field_choice}"
        if resolved_col not in merged.columns:
            st.error(
                f"{export.filename}: couldn't find {resolved_col!r} in the merged target "
                "data — two target files may share a security label."
            )
            continue
        if role in resolved_by:
            st.error(
                f"Both {resolved_by[role]!r} and {export.filename!r} are set as the same "
                f"role ({role.value}) — pick one role per file."
            )
            resolved.pop(role, None)
            continue
        security_key = export.security.strip().casefold()
        prior_role = role_by_security.get(security_key)
        if security_key and prior_role is not None and prior_role != role:
            st.error(
                f"{export.security!r} is set as both the {prior_role.value} and {role.value} "
                "target — a security can't fill both roles. Upload the actual bond/equity "
                "index instead of reusing the same file."
            )
            continue
        if security_key:
            role_by_security[security_key] = role
        resolved_by[role] = export.filename
        resolved[role] = resolved_col

    return resolved, merged


def _deconflict_signal_columns(
    signal_merged: pd.DataFrame,
    signal_exports: list[BloombergCsvExport],
    colliding: Collection[str],
) -> pd.DataFrame:
    """Rename signal columns that collide with a target column, using the
    file's own name instead of its security — the same fallback
    ``bloomberg_csv.merge()`` uses when two files share a security label.
    """
    renames = {}
    for export in signal_exports:
        lbl = bloomberg_csv.label(export.security, export.filename)
        distinct = bloomberg_csv.label("", export.filename)
        for col in colliding:
            if col in signal_merged.columns and col.startswith(f"{lbl}_"):
                renames[col] = distinct + col[len(lbl) :]
    return signal_merged.rename(columns=renames)


def _combine(target: pd.DataFrame | None, signal: pd.DataFrame | None) -> pd.DataFrame:
    """Outer-join the target and signal merges on Date into one frame.

    Modelling needs one indexed frame with both target and signal columns
    together; keeping the two merges separate up to this point is what makes
    "which columns are targets" a structural fact instead of an inferred one.
    """
    if target is None:
        return signal
    if signal is None:
        return target
    return (
        target.merge(signal, on=bloomberg_csv.DATE_COLUMN, how="outer")
        .sort_values(bloomberg_csv.DATE_COLUMN)
        .reset_index(drop=True)
    )


def _keep_and_log(merged: pd.DataFrame, *, file_id: str) -> AcceptedUpload:
    """Store the merged file under its content hash and log it to DuckDB.

    The merge is an upload like any other: the bytes go under data/uploads and
    a row goes in the uploads table, once per set of files rather than once
    per Streamlit rerun. Dates are written ISO so the stored copy reads back
    unambiguously.
    """
    # A fixed line ending, so the same merge hashes the same on every machine.
    csv_bytes = merged.to_csv(index=False, date_format="%Y-%m-%d", lineterminator="\n").encode(
        "utf-8"
    )
    accepted = accept_upload(MERGED_NAME, csv_bytes)
    if st.session_state.get(_LOGGED_KEY) != file_id:
        record_upload(accepted)
        st.session_state[_LOGGED_KEY] = file_id
    st.caption(f"Stored as {MERGED_NAME}, content hash {accepted.source.short_hash}.")
    return accepted


def _render_gap_review(merged: pd.DataFrame) -> None:
    """List every row missing a value. Nothing here is filled: each model
    aligns its signals to its own target's calendar on the Models page."""
    remaining = bloomberg_csv.missing_row_report(merged)
    if remaining.empty:
        return

    st.markdown(ui.eyebrow("Rows missing a value"), unsafe_allow_html=True)
    st.caption(
        "Nothing is filled here. On the Models page each target keeps only the dates "
        "it has a price, and each signal takes its latest value on or before each of "
        f"those dates, for up to {MAX_STALENESS} rows."
    )
    st.dataframe(
        bloomberg_csv.with_display_dates(remaining),
        width="stretch",
        hide_index=True,
        column_config={
            "Date": st.column_config.TextColumn(width="small"),
            "Missing columns": st.column_config.TextColumn(width="large"),
            "Likely reason": st.column_config.TextColumn(width="medium"),
        },
    )


def render_summary() -> None:
    """The Home page's data quality report, read back from session state.

    Reflects whichever dataset was last committed on the Data page via
    "Use Updated Data" — the same commit the Models page reads — not
    every upload or gap-review edit.
    """
    ui.inject()
    st.subheader("Data quality report", divider="grey")

    report: ValidationReport | None = st.session_state.get(COMMITTED_REPORT_KEY)
    merged = st.session_state.get(COMMITTED_KEY)
    if report is None or merged is None:
        _render_awaiting_upload()
        return

    _render_report(report, merged, st.session_state.get(COMMITTED_DROPPED_KEY, []))


def _dropped() -> list[str]:
    """Columns either uploader dropped because they held no data at all."""
    return [
        *st.session_state.get(_TARGET_DROPPED_KEY, []),
        *st.session_state.get(_SIGNAL_DROPPED_KEY, []),
    ]


# --- shared between the Data page and the Home summary ---------------------


def _render_awaiting_upload() -> None:
    st.info(
        "No data committed yet. Upload Bloomberg CSV or Excel exports on the **Data** page and "
        "click **Use Updated Data**.",
        icon=":material/hourglass_empty:",
    )
    st.caption("This report fills in once a cleaned dataset is committed.")
    badge = ui.lozenge("Pending", "neutral")
    rows = "".join(
        ui.status_row(title, badge)
        for title in ("Schema validation", "Duplicate dates", "Weekend rows", "Day-over-day moves")
    )
    st.markdown(rows, unsafe_allow_html=True)


def _render_report(report: ValidationReport, merged: pd.DataFrame, dropped: list[str]) -> None:
    _render_verdict(report)
    _render_coverage(report, merged)
    _render_dropped(dropped)
    _render_completeness(report)
    _render_breakdown(report)


def _render_dropped(dropped: list[str]) -> None:
    if not dropped:
        return
    st.markdown(ui.eyebrow("Columns dropped at ingest"), unsafe_allow_html=True)
    st.caption(
        "These fields had no value on any date (the security has no data for that "
        "field), so they were removed before merging."
    )
    st.dataframe([{"Column": name} for name in dropped], width="stretch", hide_index=True)


def _flagged_count(report: ValidationReport) -> int:
    return report.duplicate_dates + report.weekend_rows + len(report.big_moves)


def _render_verdict(report: ValidationReport) -> None:
    if report.schema_errors:
        st.error(
            f"{len(report.schema_errors)} schema issue"
            f"{'s' if len(report.schema_errors) != 1 else ''} — see the columns "
            "listed below.",
            icon=":material/error:",
        )
        for message in report.schema_errors:
            st.caption(message)
        return

    flagged = _flagged_count(report)
    if flagged:
        st.success(
            f"Merged and validated. {flagged} observation"
            f"{'s' if flagged != 1 else ''} flagged below for information — "
            "none of them stop a run.",
            icon=":material/check_circle:",
        )
    else:
        st.success("Merged and validated. Nothing flagged.", icon=":material/check_circle:")


def _render_coverage(report: ValidationReport, merged: pd.DataFrame) -> None:
    rows, signals, flag_col = st.columns(3)
    rows.metric("Rows", f"{len(merged):,}", icon=":material/table_rows:")
    signals.metric("Signals", len(merged.columns) - 1, icon=":material/show_chart:")
    flag_col.metric("Flagged", _flagged_count(report), icon=":material/flag:")

    dates = merged["Date"].dropna()
    if not dates.empty:
        st.caption(f"Covering {_fmt(dates.min())} to {_fmt(dates.max())}")


def _render_completeness(report: ValidationReport) -> None:
    incomplete = {k: v for k, v in report.missing_pct.items() if v > 0}
    if not incomplete:
        return
    st.markdown(ui.eyebrow("Completeness by column"), unsafe_allow_html=True)
    st.dataframe(
        [
            {"Column": name, "Missing %": pct}
            for name, pct in sorted(incomplete.items(), key=lambda kv: -kv[1])
        ],
        width="stretch",
        hide_index=True,
        column_config={
            "Column": st.column_config.TextColumn(width="medium"),
            "Missing %": st.column_config.NumberColumn("Missing %", format="%.1f%%"),
        },
    )


def _render_breakdown(report: ValidationReport) -> None:
    has_findings = (
        report.duplicate_date_values or report.weekend_date_values or not report.big_moves.empty
    )
    if not has_findings:
        return

    st.markdown(ui.eyebrow("Flagged observations"), unsafe_allow_html=True)

    if report.duplicate_date_values:
        row = _dates_row(report.duplicate_dates, report.duplicate_date_values)
        with st.expander(f"Duplicate dates — {report.duplicate_dates}"):
            st.markdown(row, unsafe_allow_html=True)

    if report.weekend_date_values:
        row = _dates_row(report.weekend_rows, report.weekend_date_values)
        with st.expander(f"Weekend rows — {report.weekend_rows}"):
            st.markdown(row, unsafe_allow_html=True)

    for column, moves in _by_column(report.big_moves):
        with st.expander(
            f"{column} — {len(moves)} statistically unusual "
            f"move{'s' if len(moves) != 1 else ''}"
        ):
            rows_html = "".join(
                ui.finding_row(
                    ui.lozenge("Info", "info"),
                    _fmt(move["date"]),
                    f"{move['change']:+,.4g} day-over-day change, "
                    f"{move['robust_score']:.0f}x the typical move",
                )
                for move in moves
            )
            st.markdown(rows_html, unsafe_allow_html=True)


def _dates_row(count: int, values: tuple[str, ...]) -> str:
    # ``values`` are ISO strings from the report; reformatted only for
    # display, never re-parsed back into anything that drives a decision.
    shown = ", ".join(_fmt(v) for v in values)
    remaining = count - len(values)
    if remaining > 0:
        shown += f" and {remaining} more"
    return ui.finding_row(ui.lozenge("Info", "info"), "", shown)


def _by_column(big_moves: pd.DataFrame) -> list[tuple[str, list]]:
    grouped: dict[str, list] = {}
    for _, row in big_moves.iterrows():
        grouped.setdefault(row["column"], []).append(row)
    return sorted(grouped.items(), key=lambda kv: -len(kv[1]))
