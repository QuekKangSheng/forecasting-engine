"""Run history page: every model fit recorded in DuckDB (``store/runs``), newest
first, and any one opened to see its settings, metrics and forecasts, and the data
it was fitted on; and every distinct optimiser-and-backtest result the Portfolio
page has shown (``store/portfolio_runs``), once each, numbered, nameable and opened
from its saved results.

Reads the log only: nothing here fits a model, solves the optimiser or sets an active
model. The Models page's tables show the latest result per setting; this page keeps
every fit, including the ones that failed and the ones whose settings have since
changed.
"""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

import charts
import glossary
import saved_run
import ui
from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.reporting.model_metrics import display_model_name
from forecasting_engine.store import active_model, lineage, portfolio_runs, runs, uploads

ui.page_header(
    "Run history",
    "Every model fit and every distinct portfolio result, newest first: when it ran, on "
    "which data, under which settings, and what came out. Nothing here refits a model.",
)

OPEN_RUN_KEY = "open_portfolio_run"
#: A stored metric's key -> the name the Models page's table gives it, in the order
#: a run is read: the two gate figures first.
METRIC_NAMES = {
    "signal_rank_ic": "Signal Rank IC",
    "signal_t": "t",
    "oos_rank_ic": "OOS Rank IC",
    "ic": "IC",
    "rmse": "RMSE",
    "rows_scored": "Rows scored",
    "crash_recall": "Crash recall",
    "crash_precision": "Crash precision",
    "crash_f1": "Crash F1",
    "crash_tail_days": "Crash tail days",
    "signal_rank_ic_se": "Signal Rank IC s.e. (h−1 lags)",
    "signal_rank_ic_se_test": "Signal Rank IC s.e. (test-window lags)",
    "oos_rank_ic_se": "OOS Rank IC s.e. (h−1 lags)",
    "oos_rank_ic_se_test": "OOS Rank IC s.e. (test-window lags)",
}


def _metric_value(value: float | None) -> str:
    """A dash for a metric that wasn't computed, rather than the word "None"."""
    if value is None or value != value:
        return "—"
    return f"{value:,.0f}" if float(value).is_integer() and abs(value) >= 1 else f"{value:.4g}"
RUNS_PER_PAGE = 20
RUN_NAME_MAX_CHARS = 80
#: Run, name, first shown, equity model, bond model, risk level, then the row's buttons.
#: The buttons had a line of their own while the page was a narrow column; it is the
#: full width now, and the extra line made the list twice as long, so an opened run
#: landed a screen or two below the button that opened it.
RUN_COLUMNS = [0.5, 2.4, 1.5, 1.7, 1.5, 0.6, 2.1]
RUN_HEADINGS = ["Run", "Name", "First shown", "Equity model", "Bond model", "Risk", ""]


# Each tab is a fragment: a control on one redraws that tab alone. Run as one page,
# opening a portfolio run also read the fits table back, a fit's forecasts and its
# chart, and choosing a fit redrew every portfolio run row.
@st.fragment
def _model_fits() -> None:
    TARGETS = {"All": None, "Equity": TargetRole.EQUITY, "Bond": TargetRole.BOND}
    target = st.segmented_control("Target", list(TARGETS), default="All", required=True)
    table = runs.runs_table(role=TARGETS[target])
    if table.empty:
        st.info(
            "No runs recorded yet. Run a model on the Models page and it appears here.",
            icon=":material/info:",
        )
        return

    st.markdown(
        ui.eyebrow(f"{len(table):,} runs", glossary.term("Run history")), unsafe_allow_html=True
    )
    # Which runs the active models came from, so the table can say which are in use.
    in_use = {
        record.run_id: role.value.title()
        for role in TargetRole
        if (record := active_model.get_active_model(role)) is not None and record.run_id
    }
    shown = pd.DataFrame(
        {
            "Active": table["run_id"].map(in_use).fillna(""),
            "Started": table["started_at"],
            "Target": table["role"].str.title(),
            "Model": table["model_name"].map(display_model_name),
            "Status": table["status"].str.title(),
            "Seconds": table["seconds"],
            "Signal Rank IC": table["signal_rank_ic"],
            "t": table["signal_t"],
            "OOS Rank IC": table["oos_rank_ic"],
            "RMSE": table["rmse"],
            "Data": table["dataset_sha256"].str[:12],
            "Code": table["code_version"].fillna("—"),
        }
    )
    # Formatted through the frame's own styler so a figure a run doesn't have (the
    # naive baseline has no gate figures, a failed run has none at all) is a dash. A
    # number column's format left those cells reading "None".
    figures = {
        "Seconds": "{:.1f}",
        "Signal Rank IC": "{:+.3f}",
        "t": "{:.2f}",
        "OOS Rank IC": "{:+.3f}",
        "RMSE": "{:.4f}",
    }
    st.dataframe(
        shown.style.format(figures, na_rep="—"),
        hide_index=True,
        width="stretch",
        column_config={
            "Started": st.column_config.DatetimeColumn(format="D MMM YYYY, HH:mm"),
            "Data": st.column_config.TextColumn(help=glossary.term("Dataset hash")),
            "Active": st.column_config.TextColumn(
                help="The target whose active model this run produced, if any."
            ),
        },
    )

    labels = {
        row.run_id: f"{row.started_at:%d %b %Y %H:%M} · {row.role.title()} · "
        f"{display_model_name(row.model_name)} · {row.status}"
        + (" · active" if row.run_id in in_use else "")
        for row in table.itertuples()
    }
    chosen = st.selectbox("Open a run", list(labels), format_func=labels.get)
    run = runs.get_run(chosen)

    st.markdown(ui.eyebrow("Run"), unsafe_allow_html=True)
    same = int((table["config_key"] == run.config_key).sum())
    cols = st.columns(4)
    cols[0].metric("Status", run.status.title())
    cols[1].metric("Took", f"{run.seconds:.1f} s")
    cols[2].metric("Code version", run.code_version or "—")
    cols[3].metric("Runs with these settings", f"{same}", help=glossary.term("Config key"))
    if run.error:
        st.error(run.error, icon=":material/error:")

    # Lineage: the dataset hash leads to the upload that holds those exact bytes. The
    # file and date are what a reader wants; the hashes and paths that prove it are
    # one click away, for an audit, rather than leading the run.
    upload = uploads.find_upload(run.dataset_sha256)
    summary = (
        f"{html.escape(upload.filename)}, uploaded {upload.uploaded_at:%d %b %Y %H:%M}, "
        f"{upload.row_count:,} rows"
        if upload is not None
        else "no upload with this hash is in the upload log"
    )
    st.markdown(
        ui.eyebrow("Data", glossary.term("Dataset hash"))
        + f'<p style="margin:0">{summary}</p>',
        unsafe_allow_html=True,
    )
    lineage_box = st.expander("Lineage: hashes and the exports this data came from")
    lineage_box.markdown(
        f'<p style="margin:0">SHA-256 <code>{run.dataset_sha256}</code><br>'
        f'<span style="color: var(--fe-muted)">stored as '
        f"data/uploads/{run.dataset_sha256}.csv</span></p>",
        unsafe_allow_html=True,
    )

    # One step further back: the Bloomberg exports the dataset was merged from.
    sources = lineage.sources_of(run.dataset_sha256)
    if sources:
        lineage_box.markdown(
            ui.eyebrow(f"Merged from {len(sources)} export(s)", glossary.term("Lineage")),
            unsafe_allow_html=True,
        )
        lineage_box.dataframe(
            pd.DataFrame(
                {
                    "Role": [s.role.title() for s in sources],
                    "File": [s.filename or "not in the upload log" for s in sources],
                    "Rows": [s.row_count for s in sources],
                    "Uploaded": [s.uploaded_at for s in sources],
                    "SHA-256": [s.sha256[:12] for s in sources],
                }
            ),
            hide_index=True,
            width="stretch",
            column_config={"Uploaded": st.column_config.DatetimeColumn(format="D MMM YYYY, HH:mm")},
        )
    else:
        lineage_box.caption(
            "Which exports this dataset was merged from wasn't recorded: it was committed "
            "before exports were logged one by one."
        )

    with st.expander("Settings"):
        st.json(dict(run.settings))

    metrics = runs.run_metrics(run.run_id)
    if metrics:
        st.markdown(ui.eyebrow("Metrics"), unsafe_allow_html=True)
        # The gate's two figures first, then the rest, under the names the Models
        # page's table uses rather than their storage keys.
        ordered = [k for k in METRIC_NAMES if k in metrics] + [
            k for k in metrics if k not in METRIC_NAMES
        ]
        st.dataframe(
            pd.DataFrame(
                {
                    "Metric": [
                        METRIC_NAMES.get(k, k.replace("_", " ").capitalize()) for k in ordered
                    ],
                    "Value": [_metric_value(metrics[k]) for k in ordered],
                }
            ),
            hide_index=True,
            width="stretch",
        )

    forecasts = runs.run_forecasts(run.run_id)
    if not forecasts.empty:
        st.markdown(
            ui.eyebrow(
                "Out-of-sample forecasts",
                "Each test date's forecast of the return over the horizon, beside its fold's "
                "training-mean baseline and the return that followed.",
            ),
            unsafe_allow_html=True,
        )
        # The same chart as every other line chart in the app, in the same colours:
        # realised muted behind the two lines a model drew. Drawn on a canvas like
        # them too, since three lines of every test date were thousands of page
        # elements as SVG.
        shown = forecasts.rename(
            columns={"predicted": "Forecast", "baseline": "Baseline", "realised": "Realised"}
        )
        ordered = [c for c in ("Realised", "Forecast", "Baseline") if c in shown.columns]
        st.altair_chart(
            charts.line_chart(shown[ordered], "Return", title="Return over the horizon"),
            width="stretch",
        )


def _rename_run(run_key: str, widget: str) -> None:
    portfolio_runs.rename_portfolio_run(run_key, st.session_state[widget])


def _open_run(run_key: str) -> None:
    st.session_state[OPEN_RUN_KEY] = run_key


def _close_run() -> None:
    st.session_state.pop(OPEN_RUN_KEY, None)


def _model_cell(name: object) -> str:
    return display_model_name(name) if pd.notna(name) else "—"


def _run_row(row, *, is_open: bool) -> None:
    cols = st.columns(RUN_COLUMNS, vertical_alignment="center")
    cols[0].markdown(f"**{row.run_number}**")
    widget = f"run_name_{row.run_key}"
    cols[1].text_input(
        "Run name",
        value=row.name if pd.notna(row.name) else "",
        max_chars=RUN_NAME_MAX_CHARS,
        key=widget,
        placeholder="Add a name",
        label_visibility="collapsed",
        on_change=_rename_run,
        args=(row.run_key, widget),
    )
    cols[2].text(f"{row.recorded_at:%d %b %Y %H:%M}")
    cols[3].text(_model_cell(row.equity_model))
    cols[4].text(_model_cell(row.bond_model))
    cols[5].text(f"{row.risk_level:.0f}" if pd.notna(row.risk_level) else "—")
    buttons = cols[6].container(horizontal=True, horizontal_alignment="right")
    if is_open:
        # The run is shown under this row; the same button puts it away again.
        buttons.button("Close", key=f"close_run_{row.run_key}", on_click=_close_run)
    else:
        buttons.button(
            "Open",
            key=f"open_run_{row.run_key}",
            disabled=not row.has_results,
            help=None if row.has_results else NOT_SAVED,
            on_click=_open_run,
            args=(row.run_key,),
        )
    if row.has_results:
        saved_run.report_button(buttons, row.run_key, label="Report", number=row.run_number)
    else:
        buttons.button("Report", key=f"no_report_{row.run_key}", disabled=True, help=NOT_SAVED)


NOT_SAVED = (
    "Recorded before results were saved. Show the same settings on the Portfolio page, on "
    "the same code, and they are saved under this run."
)


@st.fragment
def _portfolio_runs() -> None:
    table = portfolio_runs.portfolio_runs_table()
    if table.empty:
        st.info(
            "No portfolio results recorded yet. Open the Portfolio Optimizer with both "
            "active models set and its result appears here.",
            icon=":material/info:",
        )
        return
    st.markdown(
        ui.eyebrow(
            f"{len(table):,} distinct result{'' if len(table) == 1 else 's'}",
            glossary.term("Portfolio run"),
        ),
        unsafe_allow_html=True,
    )
    pages = (len(table) - 1) // RUNS_PER_PAGE + 1
    page = (
        st.number_input("Page", min_value=1, max_value=pages, value=1, width=120)
        if pages > 1
        else 1
    )
    # An opened run is shown straight under its own row, where the click was. At the
    # foot of the list it was out of sight, so "Open" looked as if it had done nothing.
    open_key = st.session_state.get(OPEN_RUN_KEY)
    opened = None if open_key is None else portfolio_runs.get_saved_run(open_key)
    shown_inline = False
    with st.container(border=True):
        header = st.columns(RUN_COLUMNS)
        for col, heading in zip(header, RUN_HEADINGS, strict=True):
            col.caption(heading)
        start = (page - 1) * RUNS_PER_PAGE
        for row in table.iloc[start : start + RUNS_PER_PAGE].itertuples():
            is_open = opened is not None and row.run_key == opened.run_key
            _run_row(row, is_open=is_open)
            if is_open:
                shown_inline = True
                with st.container(border=True):
                    saved_run.render(opened)

    with st.expander("Compare runs"):
        _compare_runs(table)

    # A run opened from another page of the list has no row here to sit under.
    if opened is not None and not shown_inline:
        st.divider()
        saved_run.render(opened)


def _compare_runs(table: pd.DataFrame) -> None:
    st.dataframe(
        pd.DataFrame(
            {
                "Run": table["run_number"],
                "First shown": table["recorded_at"],
                "Risk level": table["risk_level"],
                "Minimum in each": table["min_weight"],
                "Net Sharpe": table["net_sharpe"],
                "50/50 net Sharpe": table["baseline_net_sharpe"],
                "Information ratio": table["information_ratio"],
                "Backtest": table["start_date"].astype(str)
                + " to "
                + table["end_date"].astype(str),
                "Key": table["run_key"].str[:12],
                "Code": table["code_version"].fillna("—"),
            }
        ),
        hide_index=True,
        width="stretch",
        column_config={
            "First shown": st.column_config.DatetimeColumn(format="D MMM YYYY, HH:mm"),
            "Minimum in each": st.column_config.NumberColumn(format="%.2f"),
            "Net Sharpe": st.column_config.NumberColumn(format="%.2f"),
            "50/50 net Sharpe": st.column_config.NumberColumn(format="%.2f"),
            "Information ratio": st.column_config.NumberColumn(format="%.2f"),
            "Key": st.column_config.TextColumn(help=glossary.term("Portfolio run")),
        },
    )
    st.caption(
        "One row per distinct set of inputs, however many times the Portfolio page showed "
        "it: the key is the hash of the data, both active models, the windows, the risk "
        "level and limits, the trading costs and the code version."
    )


fits, portfolios = st.tabs(["Model fits", "Portfolio runs"])
with fits:
    _model_fits()
with portfolios:
    _portfolio_runs()
