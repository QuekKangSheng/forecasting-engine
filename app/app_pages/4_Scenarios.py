"""Scenarios page: shock one or more signals the active polynomials read and see
what each index's forecast becomes (FYP-32), then what that does to the
optimised portfolio (FYP-33).

The active functions are evaluated again on the shocked inputs; nothing is
refitted. Each shock replaces the signal's last observed value and goes through
the same alignment, transform and lag as the fit (``scenario.shock``). Shocks are
validated before the scenario can run, and a shocked input outside the range the
model was trained on is flagged as an extrapolation.

The portfolio impact (``scenario.portfolio_impact``) carries the shocked
forecasts through the Monte Carlo simulation at the latest optimised weights,
held as they are, beside the same simulation on the base-case forecasts. Only if
the user switches on "Re-optimise under shock" (FYP-246) are the weights also
re-solved from the shocked forecasts, shown in a column of their own.
"""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

import bloomberg_extraction_panel
import glossary
import portfolio_summary
import ui
from forecasting_engine.extraction.bloomberg_csv import DATE_COLUMN
from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.ingest.align import transform_for
from forecasting_engine.portfolio import optimize
from forecasting_engine.reporting.factor_labels import labeller
from forecasting_engine.reporting.model_metrics import display_model_name
from forecasting_engine.reporting.polynomial_function import dataset_fingerprint
from forecasting_engine.risk import monte_carlo
from forecasting_engine.scenario import portfolio_impact, shock
from forecasting_engine.store import active_model
from forecasting_engine.validation.splitters import TUNING_ROWS, PurgedWalkForward
from model_settings import EMBARGO_DAYS, TRAIN_WINDOW

ROLES = (TargetRole.EQUITY, TargetRole.BOND)
RESULT_KEY = "scenario_result"
#: The Portfolio page's Monte Carlo settings, so both pages simulate the same way.
METHOD_KEY = "monte_carlo_method"
PATHS_KEY = "monte_carlo_paths"
REOPTIMISE_KEY = "scenario_reoptimise"
KIND_LABELS = {shock.ShockKind.ABSOLUTE: "New value", shock.ShockKind.PERCENT: "% change"}

ui.page_header(
    "Scenarios",
    "Shock one or more signals the active polynomials read, and see how each index's "
    "forecast changes. Nothing is refitted: the active functions are evaluated again on "
    "the shocked inputs.",
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
                f"The active {name} model, {display_model_name(record.model_name)}, has no "
                "stored polynomial function: a scenario can re-evaluate a polynomial only. "
                "Set a polynomial active on the Models page (setting one again stores its "
                "function)."
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
    cols = st.columns([4, 4, 3], vertical_alignment="bottom")
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


def _cell(text: str) -> str:
    return f"<td>{html.escape(text)}</td>"


def _heading(text: str, term: str | None = None) -> str:
    if term is None:
        return f"<th>{html.escape(text)}</th>"
    hint = html.escape(glossary.term(term), quote=True)
    return (
        f'<th title="{hint}">{html.escape(text)}'
        f'<span class="fe-eyebrow-help" title="{hint}">i</span></th>'
    )


def _index_cell(role: TargetRole) -> str:
    """The index, with the active model behind its figures beneath it."""
    name = label(target_columns[role]) if role in target_columns else role.value.title()
    model = (
        f'<br><span style="color: var(--fe-muted)">'
        f"{html.escape(display_model_name(records[role].model_name))}</span>"
        if role in records
        else ""
    )
    # Wraps, unlike the table's other cells: an index named after its export file
    # can be long enough to push the figures off the side.
    return f'<td style="white-space: normal">{html.escape(name)}{model}</td>'


st.markdown(
    ui.eyebrow("Forecast impact", glossary.term("Shocked forecast")), unsafe_allow_html=True
)
header = (
    _heading("Index")
    + _heading("Base case", "Base-case forecast")
    + _heading("Shocked", "Shocked forecast")
    + _heading("Change")
    + _heading("% change")
)
rows = []
for role in ROLES:
    if role in results:
        result = results[role]
        rows.append(
            _index_cell(role)
            + _cell(_pct(result.base))
            + _cell(_pct(result.shocked))
            + _cell(f"{result.change * 100:+.2f} pts")
            + _cell(_pct(result.relative_change))
        )
    else:
        reason = failures.get(role) or unavailable.get(role, "")
        rows.append(
            _index_cell(role)
            + f'<td colspan="4" style="color: var(--fe-muted)">{html.escape(reason)}</td>'
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
        low, high = change.trained_on
        # A clipped input caps the shock's effect, extrapolation or not: without
        # saying so, every shock beyond the bound reads as the same forecast.
        clipped = (
            f"The model clips this input to {change.clipped_to:.4g} before its terms "
            "apply, so any shock beyond that gives the same forecast."
            if change.clipped_to is not None
            else ""
        )
        if change.extrapolated:
            st.warning(
                f"Extrapolation: {label(change.signal)} enters the {index_name} model as "
                f"{change.shocked:.4g}, outside the {low:.4g} to {high:.4g} it was trained "
                f"on, so this forecast goes beyond observed data. {clipped}".rstrip(),
                icon=":material/warning:",
            )
        elif clipped:
            st.info(
                f"{label(change.signal)} enters the {index_name} model as "
                f"{change.shocked:.4g}. {clipped}",
                icon=":material/info:",
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


# --- FYP-33: what the shock does to the optimised portfolio ---------------------------


def _pts(value: float) -> str:
    return "—" if value != value else f"{value * 100:+.2f} pts"


def _base_forecast(role: TargetRole) -> float | None:
    """The index's base-case forecast for an index the scenario didn't evaluate:
    its active model's latest saved forecast, as the Portfolio page centres the
    simulation on. ``None`` if the index has no usable active model."""
    record = active_model.get_active_model(role)
    if record is None or record.dataset_fingerprint != fingerprint:
        return None
    saved = active_model.get_active_model_forecast(role)
    if saved is None or saved.dropna().empty:
        return None
    return float(saved.dropna().iloc[-1])


def _portfolio_forecasts() -> tuple[dict, dict, str | None]:
    """Each index's base-case and shocked forecast for the simulation, or why the
    portfolio can't be simulated. An index the scenario evaluated takes its own
    base and shocked forecasts; one it didn't takes its active model's latest
    saved forecast for both, so the shock leaves it unmoved."""
    base, shocked = {}, {}
    for role in ROLES:
        if role in results:
            base[role], shocked[role] = results[role].base, results[role].shocked
            continue
        latest = _base_forecast(role)
        if latest is None:
            name = label(target_columns[role]) if role in target_columns else role.value
            return {}, {}, (
                f"The portfolio can't be simulated without a forecast for {name}: "
                + (failures.get(role) or unavailable.get(role, ""))
            )
        base[role] = shocked[role] = latest
    return base, shocked, None


summary = st.session_state.get(portfolio_summary.SUMMARY_KEY)
st.markdown(
    ui.eyebrow("Portfolio impact", glossary.term("Portfolio impact")), unsafe_allow_html=True
)
if summary is None or summary.dataset_fingerprint != fingerprint:
    st.info(
        "Open the Portfolio Optimizer page first, so the latest optimised weights are "
        "known. The shocked forecasts are then carried through to the portfolio at "
        "those weights.",
        icon=":material/info:",
    )
    st.stop()

base_forecasts, shocked_forecasts, blocked_reason = _portfolio_forecasts()
if blocked_reason:
    st.info(blocked_reason, icon=":material/info:")
    st.stop()

# FYP-246: off by default, so the held weights are what a risk manager sees first.
reoptimise = st.toggle(
    "Re-optimise under shock",
    key=REOPTIMISE_KEY,
    disabled=summary.solve_inputs is None,
    help=glossary.term("Re-optimise under shock"),
)
if summary.solve_inputs is None:
    st.caption(
        "The latest rebalance's inputs couldn't be rebuilt, so the weights can't be "
        "re-optimised. Open the Portfolio Optimizer page again to refresh them."
    )

weights = {TargetRole.EQUITY: summary.equity_weight, TargetRole.BOND: summary.bond_weight}
horizon = max(records[role].horizon for role in results)
method = st.session_state.get(METHOD_KEY, monte_carlo.BOOTSTRAP)
paths = int(st.session_state.get(PATHS_KEY, monte_carlo.DEFAULT_PATHS))
prices = {role: frame[target_columns[role]].dropna() for role in ROLES}
try:
    impact = portfolio_impact.portfolio_impact(
        monte_carlo.history_window(prices, summary.as_of, window=TRAIN_WINDOW),
        base_forecasts,
        shocked_forecasts,
        weights,
        horizon=horizon,
        method=method,
        paths=paths,
        reoptimise=summary.solve_inputs if reoptimise else None,
    )
except (portfolio_impact.PortfolioImpactError, monte_carlo.MonteCarloDataError) as exc:
    st.warning(str(exc), icon=":material/warning:")
    st.stop()

equity_name = label(target_columns[TargetRole.EQUITY])
bond_name = label(target_columns[TargetRole.BOND])
resolved = impact.reoptimised


def _metric_row(name: str, term: str, read, *, signed: bool = True) -> str:
    """One metric, ``read`` off each case's ``PortfolioMetrics``: base case, shocked with
    the weights held, the change between them, then shocked and re-optimised if asked."""
    held_change = _pts(read(impact.shocked) - read(impact.base))
    resolved_cell = _cell(_pct(read(resolved.metrics), signed)) if resolved else ""
    return (
        _heading(name, term)
        + _cell(_pct(read(impact.base), signed))
        + _cell(_pct(read(impact.shocked), signed))
        + _cell(held_change)
        + resolved_cell
    )


held_equity = _pct(impact.weights[TargetRole.EQUITY], signed=False)
impact_header = (
    _heading("Metric")
    + _heading("Base case")
    + _heading("Shocked, weights held")
    + _heading("Change")
    + (_heading("Shocked, re-optimised", "Re-optimise under shock") if resolved else "")
)
impact_rows = [
    # FYP-248: the weights move only in the re-optimised column.
    _heading(f"{equity_name} weight")
    + _cell(held_equity)
    + _cell(held_equity)
    + _cell(_pts(0.0))
    + (_cell(_pct(resolved.weights[TargetRole.EQUITY], signed=False)) if resolved else ""),
    _metric_row("Return", "Shocked return", lambda m: m.mean_return),
    _metric_row("Volatility", "Shocked volatility", lambda m: m.volatility, signed=False),
]
for confidence in impact.base.max_drawdown:
    impact_rows.append(
        _metric_row(
            f"Max drawdown {confidence:.0%}",
            "Shocked drawdown",
            lambda m, c=confidence: m.max_drawdown[c],
        )
    )
st.markdown(
    f'<div class="fe-table-wrap"><table class="fe-table"><thead><tr>{impact_header}</tr>'
    f"</thead><tbody>{''.join(f'<tr>{r}</tr>' for r in impact_rows)}</tbody></table></div>",
    unsafe_allow_html=True,
)
untouched = [label(target_columns[role]) for role in ROLES if role not in results]
st.caption(
    f"Over the next {impact.horizon} trading days, at the latest optimised weights "
    f"({impact.weights[TargetRole.EQUITY]:.0%} {equity_name}, "
    f"{impact.weights[TargetRole.BOND]:.0%} {bond_name}, as of {summary.as_of:%d %b %Y}), "
    "held as they are: the weights are not re-optimised for the shock. The base case "
    f"and the shocked case are simulated on the same {impact.paths:,} {method} paths, so "
    "the change is the shock's alone. Volatility is the spread of the simulated horizon "
    "returns; max drawdown is the fall only the worst 5% or 1% of paths exceed."
    + (
        f" {' and '.join(untouched)} had no scenario result, so its base-case "
        "forecast is used in both cases."
        if untouched
        else ""
    )
)

if resolved:
    solved_from = summary.solve_inputs
    if solved_from.risk_level == optimize.RISK_LEVELS[-1]:
        why_still = (
            " Risk level 5 ignores the forecasts, so re-optimising can't move the weights."
        )
    elif solved_from.trust == 0:
        why_still = (
            " The forecasts aren't yet statistically reliable enough to move any weight, so "
            "re-optimising can't move the weights."
        )
    else:
        why_still = ""
    st.caption(
        f"Shocked, re-optimised solves the {summary.as_of:%d %b %Y} rebalance again with "
        f"each signal moved by as much as the shock moved its forecast, and the same risk "
        f"level ({solved_from.risk_level}), weight limits ({solved_from.bounds[0]:.0%} to "
        f"{solved_from.bounds[1]:.0%}), covariance and forecast trust: "
        f"{resolved.weights[TargetRole.EQUITY]:.0%} {equity_name}, "
        f"{resolved.weights[TargetRole.BOND]:.0%} {bond_name}. It is simulated under the "
        "shocked forecasts on the same paths as the held weights. With no shock it would "
        f"give {resolved.unshocked.equity:.0%} {equity_name}, the rebalance's target before "
        "the no-trade band, which can differ from the weights held."
        + why_still
        + " This is a what-if: the weights on the Portfolio Optimizer page don't change.",
        help=glossary.term("Re-optimise under shock"),
    )
