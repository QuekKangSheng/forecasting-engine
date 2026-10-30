"""Portfolio Optimizer page: combine the active equity and bond models' saved
signals into a 50/50-anchored weight schedule, then backtest that schedule
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
import portfolio_summary
import ui
from forecasting_engine.extraction.bloomberg_csv import DATE_COLUMN
from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.portfolio import optimize
from forecasting_engine.portfolio.backtest import (
    REBALANCE_FREQUENCY,
    BacktestDataError,
    run_backtest,
)
from forecasting_engine.portfolio.performance import ROLLING_VOL_WINDOW
from forecasting_engine.reporting.factor_labels import labeller
from forecasting_engine.reporting.polynomial_function import dataset_fingerprint
from forecasting_engine.reporting.portfolio_comparison import (
    NOT_RUN,
    PORTFOLIO_LABELS,
    comparison_rows,
    cumulative_paths,
    drawdown_paths,
    rolling_volatility_paths,
    significance_summary,
    tail_rows,
)
from forecasting_engine.risk.tail import HISTORICAL, MONTE_CARLO, MONTE_CARLO_GARCH, TAIL_METHODS
from forecasting_engine.store import active_model
from model_settings import EMBARGO_DAYS, TRAIN_WINDOW

EQUITY, BOND = TargetRole.EQUITY, TargetRole.BOND

ui.inject()

st.title("Portfolio Optimizer")
st.caption(
    "A long-only, two-asset allocation that starts from 50/50 and tilts towards "
    "whichever index the active models' signals favour, re-derived every forecast "
    "horizon."
)

equity_active = active_model.get_active_model(EQUITY)
bond_active = active_model.get_active_model(BOND)
missing = [
    name for name, record in (("Equity", equity_active), ("Bond", bond_active)) if record is None
]
if missing:
    st.info(
        f"Set an active model for {' and '.join(missing)} on the Models page before "
        "running the optimiser. Once both are set, the optimised allocation, its "
        "performance against the equal-weight benchmark and whether that outperformance "
        "is statistically significant appear here.",
        icon=":material/info:",
    )
    st.stop()

if not active_model.settings_match(equity_active, bond_active):
    st.warning(
        "The active equity and bond models were run under different settings — "
        f"equity: {equity_active.horizon}-day horizon, {equity_active.test_window}-day "
        f"test window; bond: {bond_active.horizon}-day horizon, {bond_active.test_window}-"
        "day test window. Re-run both under the same settings on the Models page, then "
        "set them active again.",
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

signals = {
    EQUITY: active_model.get_active_model_signal(EQUITY),
    BOND: active_model.get_active_model_signal(BOND),
}
if signals[EQUITY] is None or signals[BOND] is None:
    st.info(
        "The active model's signal wasn't saved — it was set active before signals were "
        "captured. Run it again on the Models page and set it active again.",
        icon=":material/info:",
    )
    st.stop()

with st.expander("Settings"):
    risk_level = st.select_slider(
        "Risk aversion",
        options=list(optimize.RISK_AVERSION_SCALE),
        value=optimize.DEFAULT_RISK_LEVEL,
        format_func=lambda level: {1: "1 · risk-loving", 5: "5 · risk-averse"}.get(
            level, str(level)
        ),
        help=glossary.term("Risk aversion"),
    )
    risk_aversion = optimize.risk_aversion_for(risk_level)
    st.caption(
        "1 follows the signals as far as the limits below allow; 5 stays close to "
        f"50/50 whatever they say. Level {risk_level} is λ = {risk_aversion:g}."
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

try:
    breakdowns = optimize.schedule_breakdowns(
        signals,
        prices,
        horizon=equity_active.horizon,
        train_window=TRAIN_WINDOW,
        test_window=equity_active.test_window,
        embargo=EMBARGO_DAYS,
        risk_aversion=risk_aversion,
        bounds=(lower, upper),
    )
except BacktestDataError as exc:
    st.warning(f"The allocation could not be set: {exc}", icon=":material/warning:")
    st.stop()
schedule = optimize.schedule_weights(breakdowns)

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

latest_signal = optimize.expected_returns(pd.DatetimeIndex([latest_date]), signals).iloc[0]
breakdown = breakdowns[latest_date]
direction = "adds" if breakdown.forecast_tilt >= 0 else "takes"
cap = (
    f", which the {lower:.0%}–{upper:.0%} limits cap at {breakdown.bounded:.0%}"
    if breakdown.capped
    else ""
)
st.caption(
    f"How this was set: the benchmark holds {breakdown.anchor:.0%} {equity_name}. The "
    f"signals — what each model adds to its training-mean return — ({equity_name} "
    f"{latest_signal[EQUITY]:+.2%}, {bond_name} {latest_signal[BOND]:+.2%} over "
    f"{equity_active.horizon} days) {direction} {abs(breakdown.forecast_tilt):.0%} at risk "
    f"aversion {risk_level}, giving {breakdown.unconstrained:.0%}{cap}."
)
if breakdown.held:
    st.caption(
        f"Held: within the no-trade band. Since the last rebalance the portfolio had "
        f"drifted to {breakdown.drifted:.1%} {equity_name}, within the ±{breakdown.band:.1%} "
        f"band around it, so it was left as it was rather than pay to trade.",
        help=glossary.term("No-trade band"),
    )
elif breakdown.drifted is not None:
    st.caption(
        f"Since the last rebalance the portfolio had drifted to {breakdown.drifted:.1%} "
        f"{equity_name}. Trading costs set a no-trade band of ±{breakdown.band:.1%} around "
        f"it, so it traded only to the band's edge, {breakdown.equity:.1%}.",
        help=glossary.term("No-trade band"),
    )
after_first = list(breakdowns.values())[1:]
if after_first:
    held = sum(b.held for b in after_first)
    st.caption(
        f"{held} of {len(after_first)} rebalances after the first ({held / len(after_first):.0%}) "
        "were held by the no-trade band.",
        help=glossary.term("No-trade band"),
    )

st.markdown(
    ui.eyebrow(
        "Weights per rebalance",
        "Each rebalance's allocation, re-derived from that date's signals and risk.",
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
st.altair_chart(chart, width="stretch")


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

st.session_state[portfolio_summary.SUMMARY_KEY] = portfolio_summary.PortfolioSummary(
    dataset_fingerprint=str(dataset_fingerprint(merged)),
    as_of=latest_date,
    equity_weight=float(latest[EQUITY]),
    bond_weight=float(latest[BOND]),
    net_sharpe=backtest.metrics[("optimised", "net")].sharpe,
    baseline_net_sharpe=backtest.metrics[("baseline", "net")].sharpe,
)

costs = backtest.costs_bps
st.caption(
    f"Backtest {backtest.start:%d %b %Y} to {backtest.end:%d %b %Y}. The optimised "
    f"portfolio is rebalanced at each of the {len(schedule)} rebalances above (every "
    f"{equity_active.horizon} trading days, the forecast horizon); the benchmark is reset "
    "to 50/50 "
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

def _line_chart(paths: pd.DataFrame, value: str) -> alt.LayerChart:
    """A line per column of ``paths`` on one time axis, in percent. Hovering or
    tapping anywhere marks the nearest day on a line and shows its date and value."""
    long = paths.rename_axis("date").reset_index().melt("date", var_name="Series", value_name=value)
    nearest = alt.selection_point(nearest=True, on="pointerover", clear="pointerout", empty=False)
    lines = alt.Chart(long).encode(
        x=alt.X("date:T", title="Date", axis=alt.Axis(tickCount="year")),
        y=alt.Y(f"{value}:Q", axis=alt.Axis(format="%")),
        color=alt.Color("Series:N", legend=alt.Legend(title=None, orient="bottom", labelLimit=0)),
    )
    # Vega only shows a line's tooltip within a pixel or two of one of its days,
    # too fine to tap; invisible points with a nearest-day selection catch any hover.
    points = (
        lines.mark_point(filled=True)
        .encode(
            opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
            tooltip=[
                alt.Tooltip("date:T", title="Date", format="%d %b %Y"),
                "Series:N",
                alt.Tooltip(f"{value}:Q", format=".2%"),
            ],
        )
        .add_params(nearest)
    )
    # Canvas, not Streamlit's default SVG: SVG makes every day's point a page element,
    # thousands per chart, which stalls scrolling and every redraw.
    canvas = {"embedOptions": {"renderer": "canvas"}}
    return (lines.mark_line() + points).properties(usermeta=canvas)


cumulative = cumulative_paths(backtest, basis)
st.altair_chart(_line_chart(cumulative, "Cumulative return"), width="stretch")

# --- FYP-254 / FYP-255: how risk has moved over the backtest ------------------------

st.markdown(
    ui.eyebrow("Rolling volatility", glossary.term("Rolling volatility")),
    unsafe_allow_html=True,
)
index_names = {EQUITY: equity_name, BOND: bond_name}
volatility = rolling_volatility_paths(backtest, basis, index_names, window=ROLLING_VOL_WINDOW)
if len(volatility) < 2:  # a line needs two points
    st.info(
        f"Rolling volatility needs more than {ROLLING_VOL_WINDOW} trading days of returns, "
        f"and this backtest has {len(backtest.optimised.gross)}, so there is no chart yet.",
        icon=":material/info:",
    )
else:
    st.altair_chart(_line_chart(volatility, "Rolling volatility"), width="stretch")
    st.caption(
        f"Annualised standard deviation of the previous {ROLLING_VOL_WINDOW} trading days' "
        f"returns, at each day. Each index is held alone, with no trading costs; the "
        f"optimised portfolio is {basis_label.lower()}, the same basis as the table above."
    )

# --- FYP-258 / FYP-259: how far below its peak each line has stood -------------------

st.markdown(ui.eyebrow("Drawdown", glossary.term("Drawdown")), unsafe_allow_html=True)
drawdowns = drawdown_paths(backtest, basis, index_names)
st.altair_chart(_line_chart(drawdowns, "Drawdown"), width="stretch")
st.caption(
    "Fall from the highest value reached so far, at each day, counting the start of the "
    "backtest as the first high. Each index is held alone, with no trading costs; the "
    f"optimised portfolio is {basis_label.lower()}, and its lowest point is the max "
    "drawdown in the table above."
)

# --- FYP-20 / FYP-52: is the outperformance more than noise? ------------------------

significance = significance_summary(backtest, basis)
st.markdown(
    ui.eyebrow("Is the outperformance genuine?", glossary.term("Spread t-statistic")),
    unsafe_allow_html=True,
)
st.markdown(
    ui.status_row(
        "Spread t-statistic",
        ui.lozenge(html.escape(significance.label), significance.tone),
        f"t = {significance.t}",
    ),
    unsafe_allow_html=True,
)
st.caption(f"{significance.detail} {basis_label}, the same basis as the table above.")


# --- FYP-56 / FYP-38: VaR and CVaR by method, beside max drawdown -------------------

TAIL_TERMS = {
    "1-day VaR": "1-day VaR",
    "1-day CVaR": "1-day CVaR",
    "Max drawdown": "Max drawdown",
    HISTORICAL: "Historical tail risk",
    MONTE_CARLO: MONTE_CARLO,
    MONTE_CARLO_GARCH: MONTE_CARLO_GARCH,
}

rows_by_method = tail_rows(backtest, basis)
ran = {row.method for row in rows_by_method if row.optimised != NOT_RUN}
lozenges = " ".join(
    ui.lozenge(html.escape(m), "success")
    if m in ran
    else ui.lozenge(html.escape(f"{m} · {NOT_RUN}"), "neutral")
    for m in TAIL_METHODS
)
st.markdown(
    ui.eyebrow(f"Tail risk {lozenges}", glossary.term("Historical tail risk")),
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


def _tail_value(value: str) -> str:
    if value == NOT_RUN:
        return f'<td title="{html.escape(glossary.term(NOT_RUN), quote=True)}">{NOT_RUN}</td>'
    return f"<td>{html.escape(value)}</td>"


tail_header = "".join(
    f"<th>{html.escape(h)}</th>"
    for h in ("Method", "Measure", PORTFOLIO_LABELS["optimised"], PORTFOLIO_LABELS["baseline"])
)
tail_body = "".join(
    f"<tr>{_tail_cell(row.method)}{_tail_cell(row.metric)}"
    f"{_tail_value(row.optimised)}{_tail_value(row.baseline)}</tr>"
    for row in rows_by_method
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
    + ' Methods marked "Not run" have not been computed for this backtest, so a missing '
    "figure is not a zero risk. Monte Carlo figures, once run, are simulated and answer a "
    "different question from the historical ones."
)
