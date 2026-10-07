"""Portfolio Optimizer page: combine the active equity and bond models' saved
forecasts into a mean-variance weight schedule, then backtest that schedule
against the equal-weight benchmark (FYP-19).

Reads whichever settings (horizon, walk-forward windows) produced the active
models on the Models page — it never lets the two be re-chosen here, since
the forecasts were already fixed under those settings. Only the optimiser's
own parameters (risk aversion, weight bounds) are configurable on this page.
"""

from __future__ import annotations

import html

import altair as alt
import pandas as pd
import streamlit as st

import bloomberg_extraction_panel
import glossary
import ui
from forecasting_engine.extraction.bloomberg_csv import DATE_COLUMN
from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.portfolio import optimize
from forecasting_engine.portfolio.backtest import (
    REBALANCE_FREQUENCY,
    BacktestDataError,
    run_backtest,
)
from forecasting_engine.reporting.factor_labels import labeller
from forecasting_engine.reporting.polynomial_function import dataset_fingerprint
from forecasting_engine.reporting.portfolio_comparison import (
    PORTFOLIO_LABELS,
    comparison_rows,
    cumulative_paths,
    tail_rows,
)
from forecasting_engine.store import active_model
from model_settings import EMBARGO_DAYS

EQUITY, BOND = TargetRole.EQUITY, TargetRole.BOND

ui.inject()

st.title("Portfolio Optimizer")
st.caption(
    "A long-only, two-asset mean-variance allocation between the active equity "
    "and bond models, re-derived at every rebalance."
)

equity_active = active_model.get_active_model(EQUITY)
bond_active = active_model.get_active_model(BOND)
missing = [
    name for name, record in (("Equity", equity_active), ("Bond", bond_active)) if record is None
]
if missing:
    st.info(
        f"Set an active model for {' and '.join(missing)} on the Models page before "
        "running the optimiser. Once both are set, the optimised allocation and its "
        "performance against the equal-weight benchmark appear here.",
        icon=":material/info:",
    )
    st.stop()

if not active_model.settings_match(equity_active, bond_active):
    st.warning(
        "The active equity and bond models were run under different settings — "
        f"equity: {equity_active.horizon}-day horizon, {equity_active.train_window}/"
        f"{equity_active.test_window}-day train/test window; bond: "
        f"{bond_active.horizon}-day horizon, {bond_active.train_window}/"
        f"{bond_active.test_window}-day train/test window. Re-run both under the same "
        "settings on the Models page, then set them active again.",
        icon=":material/warning:",
    )
    st.stop()

merged = st.session_state.get(bloomberg_extraction_panel.COMMITTED_KEY)
target_columns = st.session_state.get(bloomberg_extraction_panel.COMMITTED_TARGETS_KEY, {})
if merged is None or EQUITY not in target_columns or BOND not in target_columns:
    st.info(
        "No committed data with both an equity and a bond target resolved — visit the "
        "Data page first.",
        icon=":material/info:",
    )
    st.stop()

# Each active model's saved forecast belongs to the dataset it was run on. One
# set on an earlier upload (other dates, another bond index) would otherwise be
# combined with today's prices without a word.
stale = [
    name
    for name, record in (("equity", equity_active), ("bond", bond_active))
    if record.dataset_fingerprint != str(dataset_fingerprint(merged))
]
if stale:
    st.warning(
        f"The active {' and '.join(stale)} model{'s were' if len(stale) > 1 else ' was'} "
        "set from a run on a different dataset than the one committed now, so "
        f"{'their forecasts' if len(stale) > 1 else 'its forecast'} can't be combined with "
        "these prices. Run the models on this data on the Models page and set them active "
        "again.",
        icon=":material/warning:",
    )
    st.stop()

numeric_cols = [c for c in merged.columns if c != DATE_COLUMN]
label = labeller(numeric_cols)
indexed = merged.set_index(DATE_COLUMN)
prices = {
    EQUITY: indexed[target_columns[EQUITY]].dropna(),
    BOND: indexed[target_columns[BOND]].dropna(),
}

forecasts = {
    EQUITY: active_model.get_active_model_forecast(EQUITY),
    BOND: active_model.get_active_model_forecast(BOND),
}
if forecasts[EQUITY] is None or forecasts[BOND] is None:
    st.info(
        "The active model's forecast wasn't saved — set it active again on the Models "
        "page to capture it.",
        icon=":material/info:",
    )
    st.stop()

with st.expander("Settings"):
    risk_aversion = st.number_input(
        "Risk aversion (λ)",
        min_value=0.1,
        value=optimize.DEFAULT_RISK_AVERSION,
        step=0.5,
        help=glossary.term("Risk aversion (λ)"),
    )
    st.caption(
        "Higher keeps the allocation near the lowest-risk mix whatever the forecasts "
        "say; lower follows the forecasts. About 1–2 is risk-seeking, 3–5 moderate, "
        "8 or more conservative."
    )
    # Equity and bond always sum to 100%, so one floor bounds both: at least this
    # much in each means at most 100% minus it in the other.
    lower = st.slider(
        "Minimum in each index",
        0.0,
        0.5,
        optimize.DEFAULT_WEIGHT_BOUNDS[0],
        step=0.05,
        format="%.2f",
        help=glossary.term("Weight bounds"),
    )
    upper = 1 - lower
    st.caption(f"Each index stays between {lower:.0%} and {upper:.0%}.")

schedule = optimize.weight_schedule(
    forecasts,
    prices,
    horizon=equity_active.horizon,
    train_window=equity_active.train_window,
    test_window=equity_active.test_window,
    embargo=EMBARGO_DAYS,
    risk_aversion=risk_aversion,
    bounds=(lower, upper),
)

if schedule.empty:
    st.info("No rebalance date has both a forecast and enough price history yet.")
    st.stop()

equity_name = label(target_columns[EQUITY])
bond_name = label(target_columns[BOND])

latest_date = schedule.index[-1]
latest = schedule.loc[latest_date]
st.markdown(
    ui.eyebrow("Latest weights", f"As of {latest_date:%d %b %Y}"), unsafe_allow_html=True
)
cols = st.columns(2)
cols[0].metric(equity_name, f"{latest[EQUITY]:.0%}")
cols[1].metric(bond_name, f"{latest[BOND]:.0%}")

latest_forecast = optimize.expected_returns(pd.DatetimeIndex([latest_date]), forecasts).iloc[0]
breakdown = optimize.weight_breakdown(
    latest_forecast,
    optimize.covariance_at_rebalance(
        latest_date,
        prices,
        horizon=equity_active.horizon,
        train_window=equity_active.train_window,
        embargo=EMBARGO_DAYS,
    ),
    risk_aversion=risk_aversion,
    bounds=(lower, upper),
)
direction = "adds" if breakdown.forecast_tilt >= 0 else "takes"
cap = (
    f", which the {lower:.0%}–{upper:.0%} limits cap at {breakdown.equity:.0%}"
    if breakdown.capped
    else ""
)
st.caption(
    f"How this was set: the lowest-risk mix is {breakdown.lowest_risk:.0%} {equity_name}. "
    f"The forecasts ({equity_name} {latest_forecast[EQUITY]:+.2%}, {bond_name} "
    f"{latest_forecast[BOND]:+.2%} over {equity_active.horizon} days) {direction} "
    f"{abs(breakdown.forecast_tilt):.0%} at risk aversion {risk_aversion:g}, giving "
    f"{breakdown.unconstrained:.0%}{cap}."
)

st.markdown(
    ui.eyebrow(
        "Weights per rebalance",
        "Each rebalance's allocation, re-derived from that date's forecasts and risk.",
    ),
    unsafe_allow_html=True,
)
renamed = schedule.rename(columns={EQUITY: equity_name, BOND: bond_name})
long = renamed.rename_axis("date").reset_index().melt("date", var_name="Asset", value_name="Weight")
chart = (
    alt.Chart(long)
    .mark_bar()
    .encode(
        # Ordinal, not temporal: rebalances are sparse relative to the full date
        # range, so a time axis squeezes every bar into a sliver with long empty
        # gaps between them. Each rebalance instead gets an equal-width slot.
        x=alt.X("yearmonthdate(date):O", title="Rebalance date", axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("Weight:Q", stack="zero", axis=alt.Axis(format="%")),
        color=alt.Color("Asset:N", legend=alt.Legend(title=None)),
        tooltip=["date:T", "Asset:N", alt.Tooltip("Weight:Q", format=".0%")],
    )
    .properties(height=600)
)
st.altair_chart(chart, use_container_width=True)


# --- FYP-19: the optimised allocation against the equal-weight benchmark -----------

BASES = {"After costs": "net", "Before costs": "gross"}

st.markdown(
    ui.eyebrow(
        "Performance vs equal-weight benchmark", glossary.term("Equal-weight benchmark")
    ),
    unsafe_allow_html=True,
)
try:
    backtest = run_backtest(
        prices,
        schedule,
        active_models=(
            f"{equity_name}: {equity_active.model_name}",
            f"{bond_name}: {bond_active.model_name}",
        ),
    )
except BacktestDataError as exc:
    st.warning(f"The allocation could not be backtested: {exc}", icon=":material/warning:")
    st.stop()

costs = backtest.costs_bps
st.caption(
    f"Backtest {backtest.start:%d %b %Y} to {backtest.end:%d %b %Y}. The optimised "
    f"portfolio is rebalanced at each of the {len(schedule)} rebalances above (every "
    f"{equity_active.test_window} trading days); the benchmark is reset to 50/50 "
    f"{REBALANCE_FREQUENCY}. Trading costs: {costs[EQUITY]:g} bp equity, "
    f"{costs[BOND]:g} bp bond. Forecasts from {'; '.join(backtest.active_models)}."
)
basis_label = st.segmented_control(
    "Returns",
    list(BASES),
    default="After costs",
    required=True,
    key="backtest_basis",
)
basis = BASES[basis_label]


def _metric_cell(metric: str) -> str:
    hint = html.escape(glossary.term(metric), quote=True)
    return (
        f'<td title="{hint}">{html.escape(metric)}'
        f'<span class="fe-eyebrow-help" title="{hint}">i</span></td>'
    )


header = "".join(
    f"<th>{html.escape(h)}</th>"
    for h in ("Metric", PORTFOLIO_LABELS["optimised"], PORTFOLIO_LABELS["baseline"], "Difference")
)
body = "".join(
    f"<tr>{_metric_cell(row.metric)}<td>{row.optimised}</td><td>{row.baseline}</td>"
    f"<td>{row.difference}</td></tr>"
    for row in comparison_rows(backtest, basis)
)
st.markdown(
    f'<div class="fe-table-wrap"><table class="fe-table"><thead><tr>{header}</tr></thead>'
    f"<tbody>{body}</tbody></table></div>",
    unsafe_allow_html=True,
)
st.caption(
    "Difference is optimised minus benchmark: positive favours the optimised portfolio "
    "on every row, drawdown included."
)

paths = cumulative_paths(backtest, basis) * 100
st.line_chart(paths, x_label="Date", y_label="Cumulative return (%)")


# --- FYP-56: historical VaR and CVaR, beside max drawdown ---------------------------

TAIL_TERMS = {"1-day VaR": "1-day VaR", "1-day CVaR": "1-day CVaR", "Max drawdown": "Max drawdown"}

st.markdown(
    ui.eyebrow(
        f"Tail risk {ui.lozenge('Historical', 'neutral')}", glossary.term("Historical tail risk")
    ),
    unsafe_allow_html=True,
)


def _tail_cell(metric: str) -> str:
    term = next((t for prefix, t in TAIL_TERMS.items() if metric.startswith(prefix)), None)
    term = term or "Breach rate"
    hint = html.escape(glossary.term(term), quote=True)
    return (
        f'<td title="{hint}">{html.escape(metric)}'
        f'<span class="fe-eyebrow-help" title="{hint}">i</span></td>'
    )


tail_header = "".join(
    f"<th>{html.escape(h)}</th>"
    for h in ("Metric", PORTFOLIO_LABELS["optimised"], PORTFOLIO_LABELS["baseline"])
)
tail_body = "".join(
    f"<tr>{_tail_cell(row.metric)}<td>{row.optimised}</td><td>{row.baseline}</td></tr>"
    for row in tail_rows(backtest, basis)
)
st.markdown(
    f'<div class="fe-table-wrap"><table class="fe-table"><thead><tr>{tail_header}</tr>'
    f"</thead><tbody>{tail_body}</tbody></table></div>",
    unsafe_allow_html=True,
)
tail = backtest.tail_risk[("optimised", basis)]
st.caption(
    f"{basis_label}, over the same {tail.days:,} trading days as the table above. VaR and "
    "CVaR are one-day losses read from the realised returns, shown as positive numbers: "
    "lower is better. Max drawdown is as in the table above. Each "
    f"breach rate tests every day against the VaR of the {tail.window} days before it."
    + (
        ""
        if tail.days > tail.window
        else f" The backtest is too short for that ({tail.window} days are needed first)."
    )
)
