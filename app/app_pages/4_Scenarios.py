"""Scenarios page: shock one or more signals the active polynomials read and see
what each index's forecast becomes (FYP-32).

The active functions are evaluated again on the shocked inputs; nothing is
refitted. Each shock replaces the signal's last observed value and goes through
the same alignment, transform and lag as the fit (``scenario.shock``). Shocks are
validated before the scenario can run, and a shocked input outside the range the
model was trained on is flagged as an extrapolation.
"""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

import bloomberg_extraction_panel
import glossary
import ui
from forecasting_engine.extraction.bloomberg_csv import DATE_COLUMN
from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.ingest.align import transform_for
from forecasting_engine.reporting.factor_labels import labeller
from forecasting_engine.reporting.polynomial_function import dataset_fingerprint
from forecasting_engine.scenario import shock
from forecasting_engine.store import active_model
from forecasting_engine.validation.splitters import TUNING_ROWS, PurgedWalkForward
from model_settings import EMBARGO_DAYS

ROLES = (TargetRole.EQUITY, TargetRole.BOND)
RESULT_KEY = "scenario_result"
KIND_LABELS = {shock.ShockKind.ABSOLUTE: "New value", shock.ShockKind.PERCENT: "% change"}

ui.inject()

st.title("Scenarios")
st.caption(
    "Shock one or more signals the active polynomials read, and see how each index's "
    "forecast changes. Nothing is refitted: the active functions are evaluated again on "
    "the shocked inputs."
)

merged = st.session_state.get(bloomberg_extraction_panel.COMMITTED_KEY)
target_columns = st.session_state.get(bloomberg_extraction_panel.COMMITTED_TARGETS_KEY, {})
if merged is None or not target_columns:
    st.info(
        "No committed data with a target resolved — visit the Data page first.",
        icon=":material/info:",
    )
    st.stop()

fingerprint = str(dataset_fingerprint(merged))
numeric_cols = [c for c in merged.columns if c != DATE_COLUMN]
label = labeller(numeric_cols)
sources = st.session_state.get(bloomberg_extraction_panel.COMMITTED_SOURCES_KEY, {})
frame = merged.set_index(DATE_COLUMN)

# Per index: its active model, and the function a scenario evaluates, or why not.
records, functions, unavailable = {}, {}, {}
for role in ROLES:
    name = label(target_columns[role]) if role in target_columns else role.value.title()
    record = active_model.get_active_model(role)
    if role not in target_columns:
        unavailable[role] = f"No {role.value} target is in the committed data."
    elif record is None:
        unavailable[role] = f"No active model is set for {name}."
    elif record.dataset_fingerprint != fingerprint:
        unavailable[role] = (
            f"The active {name} model was set on another dataset. Run it on this data on "
            "the Models page and set it active again."
        )
    else:
        function = active_model.get_active_function(role)
        if function is None:
            unavailable[role] = (
                f"The active {name} model, {record.model_name}, has no stored polynomial "
                "function: a scenario can re-evaluate a polynomial only. Set a polynomial "
                "active on the Models page (setting one again stores its function)."
            )
        else:
            records[role], functions[role] = record, function

if not functions:
    st.info(
        "A scenario needs an active polynomial for at least one index. "
        + " ".join(unavailable.values()),
        icon=":material/info:",
    )
    st.stop()

# A signal's "last observed" value is its last observation on or before the earliest
# last price among the indices evaluated: after that, no function reads anything.
as_of = min(frame[target_columns[role]].dropna().index[-1] for role in functions)
inputs = list(dict.fromkeys(s for f in functions.values() for s in f.inputs))

st.markdown(
    ui.eyebrow("Signals to shock", glossary.term("Signal shock")), unsafe_allow_html=True
)
selected = st.multiselect(
    "Signals to shock",
    inputs,
    format_func=label,
    placeholder="Choose the signals the scenario moves",
    label_visibility="collapsed",
)

shocks, pending = [], []
for signal in selected:
    date, value = shock.last_observed(frame, signal, as_of)
    cols = st.columns([3, 2, 3], vertical_alignment="bottom")
    cols[0].markdown(
        f"**{html.escape(label(signal))}**<br>"
        f'<span style="color: var(--fe-muted)">Last observed {value:,.4g} on '
        f"{date:%d %b %Y}</span>",
        unsafe_allow_html=True,
    )
    kind = cols[1].segmented_control(
        "Shock as",
        list(KIND_LABELS),
        default=shock.ShockKind.ABSOLUTE,
        format_func=KIND_LABELS.get,
        required=True,
        key=f"shock_kind_{signal}",
        help=glossary.term("Shock as"),
    )
    entered = cols[2].text_input(
        KIND_LABELS[kind],
        key=f"shock_value_{signal}",
        placeholder=f"{value:.4g}" if kind is shock.ShockKind.ABSOLUTE else "e.g. +50",
    )
    if entered.strip():
        shocks.append(shock.Shock(signal, kind, entered))
    else:
        pending.append(signal)

positive = {s for s in inputs if transform_for(sources.get(s)) in shock.POSITIVE_TRANSFORMS}
levels, errors = shock.validate(shocks, frame, as_of=as_of, positive=positive, label=label)
for message in errors.values():
    st.error(message, icon=":material/error:")
if pending and selected:
    st.caption(
        f"Enter a value for {', '.join(label(s) for s in pending)} to run the scenario."
    )

# Everything a result depends on: one from other inputs is not shown as this one's.
inputs_key = (
    fingerprint,
    tuple((role, records[role].set_at) for role in functions),
    tuple(sorted(levels.items())),
)
blocked = not selected or bool(pending) or bool(errors)
if st.button("Run scenario", type="primary", disabled=blocked):
    results, failures = {}, {}
    for role, function in functions.items():
        record = records[role]
        splitter = PurgedWalkForward(
            train=record.train_window,
            test=record.test_window,
            embargo=EMBARGO_DAYS,
            tuning_rows=TUNING_ROWS,
        )
        try:
            results[role] = shock.run_scenario(
                function,
                frame,
                target_columns[role],
                levels,
                horizon=record.horizon,
                splitter=splitter,
            )
        except shock.ScenarioError as exc:
            failures[role] = str(exc)
    st.session_state[RESULT_KEY] = (inputs_key, results, failures)

stored = st.session_state.get(RESULT_KEY)
if stored is None or stored[0] != inputs_key:
    if stored is not None and not blocked:
        st.info(
            "The shocks have changed since the last run. Run the scenario again to see "
            "their effect.",
            icon=":material/info:",
        )
    st.stop()

_, results, failures = stored


def _pct(value: float, signed: bool = True) -> str:
    return "—" if value != value else f"{value * 100:{'+' if signed else ''}.2f}%"


def _cell(text: str, term: str | None = None) -> str:
    if term is None:
        return f"<td>{html.escape(text)}</td>"
    hint = html.escape(glossary.term(term), quote=True)
    return (
        f'<td title="{hint}">{html.escape(text)}'
        f'<span class="fe-eyebrow-help" title="{hint}">i</span></td>'
    )


st.markdown(
    ui.eyebrow("Forecast impact", glossary.term("Shocked forecast")), unsafe_allow_html=True
)
header = "".join(
    f"<th>{h}</th>"
    for h in ("Index", "Active model", "Base case", "Shocked", "Change", "% change")
)
rows = []
for role in ROLES:
    index_name = label(target_columns[role]) if role in target_columns else role.value.title()
    if role in results:
        result = results[role]
        rows.append(
            _cell(index_name)
            + _cell(records[role].model_name)
            + _cell(_pct(result.base, signed=True), "Base-case forecast")
            + _cell(_pct(result.shocked, signed=True))
            + _cell(f"{result.change * 100:+.2f} pts")
            + _cell(_pct(result.relative_change))
        )
    else:
        reason = failures.get(role) or unavailable.get(role, "")
        rows.append(
            _cell(index_name)
            + f'<td colspan="5" style="color: var(--fe-muted)">{html.escape(reason)}</td>'
        )
st.markdown(
    f'<div class="fe-table-wrap"><table class="fe-table"><thead><tr>{header}</tr></thead>'
    f"<tbody>{''.join(f'<tr>{r}</tr>' for r in rows)}</tbody></table></div>",
    unsafe_allow_html=True,
)
horizons = sorted({records[role].horizon for role in results})
st.caption(
    f"Returns over the next {' or '.join(map(str, horizons))} trading days after "
    f"{as_of:%d %b %Y}. The base case is each active function on the latest data; the "
    "shocked forecast is the same function with the shocks applied. Change is shocked "
    "less base in percentage points, and % change is that change relative to the base "
    "case."
)

for role, result in results.items():
    index_name = label(target_columns[role])
    for change in result.changes:
        if not change.extrapolated:
            continue
        low, high = change.trained_on
        clipped = (
            f" The model clips this input to {change.clipped_to:.4g} before its terms "
            "apply, so the forecast moves no further than it would at that value."
            if change.clipped_to is not None
            else ""
        )
        st.warning(
            f"Extrapolation: {label(change.signal)} enters the {index_name} model as "
            f"{change.shocked:.4g}, outside the {low:.4g} to {high:.4g} it was trained on, so "
            f"this forecast goes beyond observed data.{clipped}",
            icon=":material/warning:",
        )

with st.expander("What each model reads"):
    detail = []
    for role, result in results.items():
        transforms = functions[role].transforms
        for change in result.changes:
            low, high = change.trained_on
            detail.append(
                {
                    "Index": label(target_columns[role]),
                    "Signal": label(change.signal),
                    "Read as": str(transforms[change.signal]),
                    "Base input": change.base,
                    "Shocked input": change.shocked,
                    "Trained on": f"{low:.4g} to {high:.4g}",
                }
            )
    if detail:
        st.dataframe(pd.DataFrame(detail), hide_index=True, use_container_width=True)
    st.caption(
        "Each input after the transform the model was fitted on (a level, a one-day "
        "change, or a one-day log return) and the one-day lag. Trained on is the input's "
        "range over the latest walk-forward fold's training rows, the fit the function is."
    )
