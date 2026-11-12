"""Portfolio Optimizer page: combine the active equity and bond models' saved
signals into a weight schedule at the chosen risk level's target volatility,
then backtest it against the equal-weight benchmark (FYP-19) and that level's
no-forecast weights.

Reads whichever settings (horizon, walk-forward windows) produced the active
models on the Models page — it never lets the two be re-chosen here, since
the forecasts were already fixed under those settings. Only the optimiser's
own parameters (risk level, weight bounds) are configurable on this page.
"""

from __future__ import annotations

import html
import traceback

import pandas as pd
import streamlit as st

import bloomberg_extraction_panel
import charts
import glossary
import portfolio_summary
import saved_run
import ui
from forecasting_engine.extraction import validation
from forecasting_engine.extraction.bloomberg_csv import DATE_COLUMN
from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.ingest.upload import frame_sha256
from forecasting_engine.portfolio import optimize
from forecasting_engine.portfolio.backtest import (
    BASELINE_WEIGHTS,
    REBALANCE_FREQUENCY,
    BacktestDataError,
    run_backtest,
)
from forecasting_engine.portfolio.performance import ROLLING_SHARPE_WINDOW, ROLLING_VOL_WINDOW
from forecasting_engine.reporting.factor_labels import labeller
from forecasting_engine.reporting.model_metrics import display_model_name
from forecasting_engine.reporting.polynomial_function import dataset_fingerprint
from forecasting_engine.reporting.portfolio_comparison import (
    NOT_RUN,
    PORTFOLIO_LABELS,
    comparison_rows,
    cumulative_paths,
    drawdown_paths,
    forecast_value_summary,
    monte_carlo_rows,
    rolling_sharpe_paths,
    rolling_volatility_paths,
    significance_summary,
    tail_rows,
)
from forecasting_engine.reporting.run_results import (
    data_quality_summary,
    model_summary,
    run_results,
    simulation_results,
)
from forecasting_engine.risk import monte_carlo
from forecasting_engine.risk.tail import HISTORICAL, MONTE_CARLO, TAIL_METHODS
from forecasting_engine.store import active_model, portfolio_runs
from forecasting_engine.store import runs as run_history
from model_settings import EMBARGO_DAYS, TRAIN_WINDOW

EQUITY, BOND = TargetRole.EQUITY, TargetRole.BOND

ui.page_header(
    "Portfolio Optimizer",
    "A long-only split between the two indices that targets the chosen risk level's "
    "volatility and tilts towards whichever index the active models favour, re-set every "
    "forecast horizon.",
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
baselines = {
    EQUITY: active_model.get_active_model_baseline(EQUITY),
    BOND: active_model.get_active_model_baseline(BOND),
}

equity_name = label(target_columns[EQUITY])
bond_name = label(target_columns[BOND])

# --- FYP-53: the models every result below comes from ------------------------------
# Both active models are read afresh on every run, so one set active on the Models
# page re-solves the whole page on its next run. Naming them first says which.
st.markdown(
    ui.eyebrow(
        "Active models",
        "Every figure on this page comes from these two models' forecasts. Set "
        "another active on the Models page and the page re-solves with it.",
    )
    + "".join(
        ui.status_row(
            html.escape(name),
            ui.lozenge(html.escape(record.model_name), "info"),
            f"set active {record.set_at:%d %b %Y %H:%M}",
        )
        for name, record in ((equity_name, equity_active), (bond_name, bond_active))
    ),
    unsafe_allow_html=True,
)
# FYP-273: filled at the end of the page, once the run and any simulation are known.
report_slot = st.container()

with st.expander("Settings"):
    risk_level = st.select_slider(
        "Risk level",
        options=list(optimize.RISK_LEVELS),
        value=optimize.DEFAULT_RISK_LEVEL,
        format_func=lambda level: {1: "1 · risk-loving", 5: "5 · risk-averse"}.get(
            level, str(level)
        ),
        help=glossary.term("Risk level"),
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
    st.caption(
        "Each level targets a volatility evenly spaced between the lowest-risk and "
        "highest-risk mixes these limits allow, and the forecasts tilt from there."
    )

schedule_settings = {
    "baselines": baselines,
    "horizon": equity_active.horizon,
    "train_window": TRAIN_WINDOW,
    "test_window": equity_active.test_window,
    "embargo": EMBARGO_DAYS,
    "risk_level": risk_level,
    "bounds": (lower, upper),
}
try:
    breakdowns = optimize.schedule_breakdowns(signals, prices, **schedule_settings)
    no_forecast_breakdowns = optimize.schedule_breakdowns(
        signals, prices, trust=0.0, **schedule_settings
    )
except BacktestDataError as exc:
    st.warning(f"The allocation could not be set: {exc}", icon=":material/warning:")
    st.stop()
schedule = optimize.schedule_weights(breakdowns)

if schedule.empty:
    st.info("No rebalance date has both a forecast and enough price history yet.")
    st.stop()

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
# Level 5 ignores forecasts whatever their record, so their record isn't why.
unreliable = (
    ", with the forecasts not used because they are not yet statistically reliable"
    if risk_level != optimize.RISK_LEVELS[-1]
    and optimize.forecast_gate_shut(
        latest_date,
        optimize.trust_pairs(signals, baselines, prices, horizon=equity_active.horizon),
        horizon=equity_active.horizon,
    )
    else ""
)
how_set = [
    f"How this was set: level {risk_level} targets {breakdown.level_weight:.0%} {equity_name}. "
    f"The signals — what each model adds to its training-mean return — ({equity_name} "
    f"{latest_signal[EQUITY]:+.2%}, {bond_name} {latest_signal[BOND]:+.2%} over "
    f"{equity_active.horizon} days) {direction} {abs(breakdown.forecast_tilt):.0%}, giving "
    f"{breakdown.unconstrained:.0%}{cap}{unreliable}."
]
st.caption(how_set[0])
no_trade_lines = []
if breakdown.held:
    no_trade_lines.append(
        f"Held: within the no-trade band. Since the last rebalance the portfolio had "
        f"drifted to {breakdown.drifted:.1%} {equity_name}, within the ±{breakdown.band:.1%} "
        f"band around it, so it was left as it was rather than pay to trade."
    )
elif breakdown.drifted is not None:
    no_trade_lines.append(
        f"Since the last rebalance the portfolio had drifted to {breakdown.drifted:.1%} "
        f"{equity_name}. Trading costs set a no-trade band of ±{breakdown.band:.1%} around "
        f"it, so it traded only to the band's edge, {breakdown.equity:.1%}."
    )
after_first = list(breakdowns.values())[1:]
if after_first:
    held = sum(b.held for b in after_first)
    no_trade_lines.append(
        f"{held} of {len(after_first)} rebalances after the first ({held / len(after_first):.0%}) "
        "were held by the no-trade band."
    )
for line in no_trade_lines:
    st.caption(line, help=glossary.term("No-trade band"))
how_set += no_trade_lines

st.markdown(
    ui.eyebrow(
        "Weights per rebalance",
        "Each rebalance's allocation, re-derived from that date's signals and risk.",
    ),
    unsafe_allow_html=True,
)
renamed = schedule.rename(columns={EQUITY: equity_name, BOND: bond_name})
st.altair_chart(charts.weights_chart(renamed), width="stretch")


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
            f"{equity_name}: {display_model_name(equity_active.model_name)}",
            f"{bond_name}: {display_model_name(bond_active.model_name)}",
        ),
    )
except BacktestDataError as exc:
    st.warning(f"The allocation could not be backtested: {exc}", icon=":material/warning:")
    st.stop()

# The same level and no-trade band with no forecasts, so what the forecasts added
# can be read separately.
no_forecast_backtest = run_backtest(prices, optimize.schedule_weights(no_forecast_breakdowns))

# FYP-246: what the latest target was solved from, so the Scenarios page can solve it
# again on shocked forecasts with everything else held the same.
try:
    latest_solve_inputs = optimize.solve_inputs_at(
        latest_date,
        signals,
        prices,
        baselines=baselines,
        horizon=equity_active.horizon,
        train_window=TRAIN_WINDOW,
        embargo=EMBARGO_DAYS,
        risk_level=risk_level,
        bounds=(lower, upper),
    )
except optimize.OptimizeDataError:
    latest_solve_inputs = None

st.session_state[portfolio_summary.SUMMARY_KEY] = portfolio_summary.PortfolioSummary(
    dataset_fingerprint=str(dataset_fingerprint(merged)),
    as_of=latest_date,
    equity_weight=float(latest[EQUITY]),
    bond_weight=float(latest[BOND]),
    net_sharpe=backtest.metrics[("optimised", "net")].sharpe,
    baseline_net_sharpe=backtest.metrics[("baseline", "net")].sharpe,
    solve_inputs=latest_solve_inputs,
)

# Recorded once per distinct set of inputs (store/portfolio_runs): the key is the
# hash of everything the result depends on, so recording on every view is safe,
# and a failed write is reported without stopping the page.
run_inputs = portfolio_runs.PortfolioInputs(
    dataset_sha256=frame_sha256(merged),
    equity_run_id=equity_active.run_id,
    bond_run_id=bond_active.run_id,
    equity_set_at=equity_active.set_at,
    bond_set_at=bond_active.set_at,
    horizon=equity_active.horizon,
    train_window=TRAIN_WINDOW,
    test_window=equity_active.test_window,
    embargo=EMBARGO_DAYS,
    risk_level=risk_level,
    bounds=(lower, upper),
    costs_bps={role.value: cost for role, cost in backtest.costs_bps.items()},
    code_version=run_history.code_version(),
)
try:
    # FYP-275: built only the first time, since every later view of these inputs is the
    # same run and would be thrown away.
    results = (
        None
        if portfolio_runs.has_results(run_inputs.run_key)
        else run_results(
            backtest,
            no_forecast_backtest,
            names={EQUITY: equity_name, BOND: bond_name},
            settings={
                "risk_level": risk_level,
                "bounds": [lower, upper],
                "costs_bps": {role.value: cost for role, cost in backtest.costs_bps.items()},
                "risk_free_rate": backtest.risk_free_rate,
                "horizon": equity_active.horizon,
                "train_window": TRAIN_WINDOW,
                "test_window": equity_active.test_window,
                "embargo": EMBARGO_DAYS,
                "rebalance_frequency": REBALANCE_FREQUENCY,
            },
            latest_date=latest_date,
            latest_weights={EQUITY: latest[EQUITY], BOND: latest[BOND]},
            how_set=how_set,
            windows={
                "rolling_volatility": ROLLING_VOL_WINDOW,
                "rolling_sharpe": ROLLING_SHARPE_WINDOW,
            },
            data_quality=data_quality_summary(
                st.session_state.get(bloomberg_extraction_panel.COMMITTED_REPORT_KEY)
                or validation.validate(merged),
                merged,
                st.session_state.get(bloomberg_extraction_panel.COMMITTED_DROPPED_KEY, []),
            ),
            models=[
                model_summary(
                    name,
                    record,
                    display_name=display_model_name(record.model_name),
                    signal_t=(
                        run_history.run_metrics(record.run_id).get("signal_t")
                        if record.run_id
                        else None
                    ),
                    # By set_at: the models this page read, even if another session
                    # has set a newer one since.
                    function=active_model.get_active_function(
                        record.role, set_at=record.set_at
                    ),
                    attribution=active_model.get_active_attribution(
                        record.role, set_at=record.set_at
                    ),
                    label=label,
                    target=target_columns[record.role],
                    columns=numeric_cols,
                )
                for name, record in ((equity_name, equity_active), (bond_name, bond_active))
            ],
        )
    )
    portfolio_runs.record_portfolio_run(run_inputs, backtest, schedule, results=results)
except Exception:  # noqa: BLE001 - a failed log must not stop the page
    traceback.print_exc()

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


NO_FORECAST_COL = f"Level {risk_level}, no forecasts"
header = "".join(
    f"<th>{html.escape(h)}</th>"
    for h in (
        "Metric",
        PORTFOLIO_LABELS["optimised"],
        PORTFOLIO_LABELS["baseline"],
        NO_FORECAST_COL,
        "Difference",
    )
)
no_forecast_rows = comparison_rows(no_forecast_backtest, basis)
body = "".join(
    f"<tr>{_metric_cell(row.metric)}<td>{row.optimised}</td><td>{row.baseline}</td>"
    f"<td>{level_row.optimised}</td><td>{row.difference}</td></tr>"
    for row, level_row in zip(comparison_rows(backtest, basis), no_forecast_rows, strict=True)
)
st.markdown(
    f'<div class="fe-table-wrap"><table class="fe-table"><thead><tr>{header}</tr></thead>'
    f"<tbody>{body}</tbody></table></div>",
    unsafe_allow_html=True,
)
st.caption(
    f"{NO_FORECAST_COL} holds level {risk_level}'s target weight with no forecast, under the "
    "same no-trade band, so the forecasts are the only difference between it and the "
    "optimised portfolio. Difference is optimised minus benchmark: positive "
    "favours the optimised portfolio "
    "on every row, drawdown included."
)

cumulative = cumulative_paths(backtest, basis)
st.altair_chart(charts.line_chart(cumulative, "Cumulative return"), width="stretch")

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
    st.altair_chart(charts.line_chart(volatility, "Rolling volatility"), width="stretch")
    st.caption(
        f"Annualised standard deviation of the previous {ROLLING_VOL_WINDOW} trading days' "
        f"returns, at each day. Each index is held alone, with no trading costs; the "
        f"optimised portfolio is {basis_label.lower()}, the same basis as the table above."
    )

# --- FYP-258 / FYP-259: how far below its peak each line has stood -------------------

st.markdown(ui.eyebrow("Drawdown", glossary.term("Drawdown")), unsafe_allow_html=True)
drawdowns = drawdown_paths(backtest, basis, index_names)
st.altair_chart(charts.line_chart(drawdowns, "Drawdown"), width="stretch")
st.caption(
    "Fall from the highest value reached so far, at each day, counting the start of the "
    "backtest as the first high. Each index is held alone, with no trading costs; the "
    f"optimised portfolio is {basis_label.lower()}, and its lowest point is the max "
    "drawdown in the table above."
)

# --- FYP-279 / FYP-280: how risk-adjusted return has moved over the backtest ----------

st.markdown(ui.eyebrow("Rolling Sharpe", glossary.term("Rolling Sharpe")), unsafe_allow_html=True)
rolling_sharpes = rolling_sharpe_paths(backtest, basis, index_names, window=ROLLING_SHARPE_WINDOW)
if len(rolling_sharpes) < 2:  # a line needs two points
    st.info(
        f"Rolling Sharpe needs more than {ROLLING_SHARPE_WINDOW} trading days of returns, and "
        f"this backtest has {len(backtest.optimised.gross)}, so there is no chart yet.",
        icon=":material/info:",
    )
else:
    st.altair_chart(
        charts.line_chart(
            rolling_sharpes,
            "Rolling Sharpe",
            percent=False,
            title=f"Rolling Sharpe, {ROLLING_SHARPE_WINDOW}-day window",
        ),
        width="stretch",
    )
    st.caption(
        f"Annualised Sharpe of the previous {ROLLING_SHARPE_WINDOW} trading days' returns, at "
        "each day. Each index is held alone, with no trading costs; the optimised portfolio "
        f"is {basis_label.lower()}, the same basis as the table above."
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

forecast_value = forecast_value_summary(backtest, no_forecast_backtest, basis)
st.markdown(
    ui.status_row(
        f"Did the forecasts add value? Against level {risk_level}, no forecasts",
        ui.lozenge(html.escape(forecast_value.label), forecast_value.tone),
        f"t = {forecast_value.t}",
    ),
    unsafe_allow_html=True,
)
st.caption(
    f"{forecast_value.detail} {basis_label}. The test above, against the benchmark, also "
    f"reflects level {risk_level}'s equity weight. This one has that weight and the no-trade "
    "band on both sides, so the forecasts are the only difference.",
    help=glossary.term("Forecast value t-statistic"),
)


# --- FYP-56 / FYP-38: VaR and CVaR by method, beside max drawdown -------------------

TAIL_TERMS = {
    "1-day VaR": "1-day VaR",
    "1-day CVaR": "1-day CVaR",
    "Max drawdown": "Max drawdown",
    HISTORICAL: "Historical tail risk",
    MONTE_CARLO: MONTE_CARLO,
}

# FYP-36: the Monte Carlo rows of this table come from the simulation run below it.
# Its button runs the simulation in a callback, before this script, so the table
# shows a new result on the same click that asked for it.
SIMULATION_KEY = "monte_carlo_result"
SIMULATION_ERROR_KEY = "monte_carlo_error"
METHOD_KEY = "monte_carlo_method"
PATHS_KEY = "monte_carlo_paths"

# Everything the figures depend on: a result simulated from any other inputs is
# not shown as if it were this one's.
allocation = (
    str(dataset_fingerprint(merged)),
    equity_active.set_at,
    bond_active.set_at,
    latest_date,
    float(latest[EQUITY]),
)


def _simulation_inputs() -> tuple:
    return (
        *allocation,
        st.session_state.get(METHOD_KEY, monte_carlo.BOOTSTRAP),
        int(st.session_state.get(PATHS_KEY, monte_carlo.DEFAULT_PATHS)),
    )


def _simulate() -> None:
    st.session_state.pop(SIMULATION_ERROR_KEY, None)
    inputs = _simulation_inputs()
    forecasts = {role: active_model.get_active_model_forecast(role) for role in (EQUITY, BOND)}
    try:
        base_case = optimize.expected_returns(pd.DatetimeIndex([latest_date]), forecasts).iloc[0]
        result = monte_carlo.simulate(
            monte_carlo.history_window(prices, latest_date, window=TRAIN_WINDOW),
            base_case,
            {"optimised": latest, "baseline": pd.Series(BASELINE_WEIGHTS)},
            horizon=equity_active.horizon,
            method=inputs[-2],
            paths=inputs[-1],
        )
    except monte_carlo.MonteCarloDataError as exc:
        st.session_state[SIMULATION_ERROR_KEY] = str(exc)
        return
    st.session_state[SIMULATION_KEY] = (inputs, result)
    try:
        portfolio_runs.save_simulation(run_inputs.run_key, simulation_results(backtest, result))
    except Exception:  # noqa: BLE001 - a failed log must not stop the page
        traceback.print_exc()


stored_simulation = st.session_state.get(SIMULATION_KEY)
simulated = (
    stored_simulation[1]
    if stored_simulation is not None and stored_simulation[0] == _simulation_inputs()
    else None
)

rows_by_method = tail_rows(
    backtest,
    basis,
    # No trading inside a simulated path, so before and after costs are the same.
    simulated=None
    if simulated is None
    else {
        MONTE_CARLO: {
            (portfolio, b): monte_carlo.as_tail_risk(simulated, portfolio)
            for portfolio in PORTFOLIO_LABELS
            for b in BASES.values()
        }
    },
)
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
    "different question from the historical ones"
    + (
        ": they are the first day of each simulated path below, which trades nothing, so "
        "they are the same before and after costs."
        if simulated is not None
        else "."
    )
)


# --- FYP-36: Monte Carlo VaR, CVaR and drawdown over the next horizon ---------------

SAMPLING_LABELS = {
    monte_carlo.BOOTSTRAP: "Bootstrap (historical days)",
    monte_carlo.PARAMETRIC: "Parametric (fitted normal)",
}
SIMULATED_TERMS = {
    "Mean": "Mean simulated return",
    f"{equity_active.horizon}-day VaR": "Simulated VaR",
    f"{equity_active.horizon}-day CVaR": "Simulated CVaR",
    "Max drawdown": "Simulated max drawdown",
}

st.markdown(
    ui.eyebrow(
        f"Monte Carlo over the next {equity_active.horizon} days",
        glossary.term("Simulated tail risk"),
    ),
    unsafe_allow_html=True,
)
st.segmented_control(
    "Sampling method",
    list(SAMPLING_LABELS),
    default=monte_carlo.BOOTSTRAP,
    format_func=SAMPLING_LABELS.get,
    required=True,
    key=METHOD_KEY,
    help=glossary.term("Sampling method"),
)
with st.expander("Advanced settings"):
    st.number_input(
        "Simulated paths",
        min_value=monte_carlo.MIN_PATHS,
        max_value=monte_carlo.MAX_PATHS,
        value=monte_carlo.DEFAULT_PATHS,
        step=10_000,
        key=PATHS_KEY,
        help=glossary.term("Simulated paths"),
    )
st.button("Run simulation", type="primary", on_click=_simulate)

if SIMULATION_ERROR_KEY in st.session_state:
    st.warning(
        f"The simulation could not run: {st.session_state[SIMULATION_ERROR_KEY]}",
        icon=":material/warning:",
    )
elif stored_simulation is None:
    st.caption(
        f"Simulates the next {equity_active.horizon} trading days from the latest weights, "
        "for both portfolios, and fills in the Monte Carlo rows above. Choose a sampling "
        "method, then run it."
    )
elif simulated is None:
    st.info(
        "The settings or the allocation have changed since the last simulation. Run it "
        "again to see figures for these.",
        icon=":material/info:",
    )
else:

    def _simulated_cell(metric: str) -> str:
        term = next(t for prefix, t in SIMULATED_TERMS.items() if metric.startswith(prefix))
        hint = html.escape(glossary.term(term), quote=True)
        return (
            f'<td title="{hint}">{html.escape(metric)}'
            f'<span class="fe-eyebrow-help" title="{hint}">i</span></td>'
        )

    simulated_header = "".join(
        f"<th>{html.escape(h)}</th>"
        for h in ("Measure", PORTFOLIO_LABELS["optimised"], PORTFOLIO_LABELS["baseline"])
    )
    simulated_body = "".join(
        f"<tr>{_simulated_cell(row.metric)}<td>{row.optimised}</td><td>{row.baseline}</td></tr>"
        for row in monte_carlo_rows(simulated)
    )
    st.markdown(
        f'<div class="fe-table-wrap"><table class="fe-table"><thead><tr>{simulated_header}'
        f"</tr></thead><tbody>{simulated_body}</tbody></table></div>",
        unsafe_allow_html=True,
    )
    base = simulated.forecasts
    st.caption(
        f"{simulated.paths:,} paths of the next {simulated.horizon} trading days from "
        f"{simulated.as_of:%d %b %Y}, each portfolio held at its weights then and left to "
        f"drift, with no trading costs. The paths centre on the active models' forecasts "
        f"({equity_name} {base[EQUITY]:+.2%}, {bond_name} {base[BOND]:+.2%} over "
        f"{simulated.horizon} days) and draw their daily moves "
        + (
            "by resampling whole days"
            if simulated.method == monte_carlo.BOOTSTRAP
            else "from a normal distribution fitted to the days"
        )
        + f" from {simulated.history_start:%d %b %Y} to {simulated.as_of:%d %b %Y} "
        f"({simulated.history_days} trading days). VaR and CVaR here are losses over the "
        f"whole {simulated.horizon} days; the one-day Monte Carlo rows above are the first "
        "day of these same paths. Both portfolios are simulated on the same paths."
    )


# --- FYP-273: the same run as a file to read offline -----------------------------------

# Only once its results are saved: the report is written from them, not from this page,
# so generating it solves nothing. A simulation counts only if the page is showing it.
if portfolio_runs.has_results(run_inputs.run_key):
    saved_run.report_button(
        report_slot,
        run_inputs.run_key,
        simulation=None if simulated is None else (simulated.method, simulated.paths),
        latest_saved=False,
    )
