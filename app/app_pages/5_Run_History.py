"""Run history page: every model fit recorded in DuckDB (``store/runs``), newest
first, and any one opened to see its settings, metrics and forecasts, and the data
it was fitted on.

Reads the log only: nothing here fits a model. The Models page's tables show the
latest result per setting; this page keeps every fit, including the ones that
failed and the ones whose settings have since changed.
"""

from __future__ import annotations

import html

import altair as alt
import pandas as pd
import streamlit as st

import glossary
import ui
from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.store import active_model, lineage, runs, uploads

ui.inject()

st.title("Run history")
st.caption(
    "Every model fit, newest first: when it ran, on which data, under which settings, "
    "and what came out. Nothing here refits a model."
)

TARGETS = {"All": None, "Equity": TargetRole.EQUITY, "Bond": TargetRole.BOND}
target = st.segmented_control("Target", list(TARGETS), default="All", required=True)
table = runs.runs_table(role=TARGETS[target])
if table.empty:
    st.info(
        "No runs recorded yet. Run a model on the Models page and it appears here.",
        icon=":material/info:",
    )
    st.stop()

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
        "Model": table["model_name"],
        "Status": table["status"].str.title(),
        "Seconds": table["seconds"],
        "Signal Rank IC": table["signal_rank_ic"],
        "t": table["signal_t"],
        "OOS Rank IC": table["oos_rank_ic"],
        "RMSE": table["rmse"],
        "Data": table["dataset_sha256"].str[:12],
        "Code": table["code_version"],
    }
)
st.dataframe(
    shown,
    hide_index=True,
    width="stretch",
    column_config={
        "Started": st.column_config.DatetimeColumn(format="D MMM YYYY, HH:mm"),
        "Seconds": st.column_config.NumberColumn(format="%.1f"),
        "Signal Rank IC": st.column_config.NumberColumn(format="%+.3f"),
        "t": st.column_config.NumberColumn(format="%.2f"),
        "OOS Rank IC": st.column_config.NumberColumn(format="%+.3f"),
        "RMSE": st.column_config.NumberColumn(format="%.4f"),
        "Data": st.column_config.TextColumn(help=glossary.term("Dataset hash")),
        "Active": st.column_config.TextColumn(
            help="The target whose active model this run produced, if any."
        ),
    },
)

labels = {
    row.run_id: f"{row.started_at:%d %b %Y %H:%M} · {row.role.title()} · {row.model_name} "
    f"· {row.status}" + (" · active" if row.run_id in in_use else "")
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

# Lineage: the dataset hash leads to the upload that holds those exact bytes.
upload = uploads.find_upload(run.dataset_sha256)
source = (
    f"{html.escape(upload.filename)}, uploaded {upload.uploaded_at:%d %b %Y %H:%M}, "
    f"{upload.row_count:,} rows, stored as data/uploads/{run.dataset_sha256}.csv"
    if upload is not None
    else "no upload with this hash is in the upload log"
)
st.markdown(
    ui.eyebrow("Data", glossary.term("Dataset hash"))
    + f'<p style="margin:0">SHA-256 <code>{run.dataset_sha256}</code><br>'
    f'<span style="color: var(--fe-muted)">{source}</span></p>',
    unsafe_allow_html=True,
)

# One step further back: the Bloomberg exports the dataset was merged from.
sources = lineage.sources_of(run.dataset_sha256)
if sources:
    st.markdown(
        ui.eyebrow(f"Merged from {len(sources)} export(s)", glossary.term("Lineage")),
        unsafe_allow_html=True,
    )
    st.dataframe(
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
        column_config={
            "Uploaded": st.column_config.DatetimeColumn(format="D MMM YYYY, HH:mm")
        },
    )
else:
    st.caption(
        "Which exports this dataset was merged from wasn't recorded: it was committed "
        "before exports were logged one by one."
    )

with st.expander("Settings"):
    st.json(dict(run.settings))

metrics = runs.run_metrics(run.run_id)
if metrics:
    st.markdown(ui.eyebrow("Metrics"), unsafe_allow_html=True)
    st.dataframe(
        pd.DataFrame({"Metric": list(metrics), "Value": list(metrics.values())}),
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
    long = (
        forecasts.rename(
            columns={"predicted": "Forecast", "baseline": "Baseline", "realised": "Realised"}
        )
        .rename_axis("date")
        .reset_index()
        .melt("date", var_name="Series", value_name="Return")
    )
    chart = (
        alt.Chart(long)
        .mark_line(strokeWidth=1.2)
        .encode(
            # One tick a year, labelled with the year: left to itself the axis labels
            # yearly ticks with the month alone.
            x=alt.X("date:T", title="Date", axis=alt.Axis(format="%Y", tickCount="year")),
            y=alt.Y("Return:Q", title="Return over the horizon", axis=alt.Axis(format="%")),
            color=alt.Color(
                "Series:N",
                # Realised muted behind the two lines a model drew.
                scale=alt.Scale(
                    domain=["Realised", "Forecast", "Baseline"],
                    range=["#9198A1", "#4C8DF6", "#E8912D"],
                ),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            tooltip=["date:T", "Series:N", alt.Tooltip("Return:Q", format=".2%")],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)
