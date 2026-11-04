"""One portfolio run opened from the Run history page: what the Portfolio page showed
for it, read back from ``store/portfolio_runs``. Nothing here fits a model, solves the
optimiser or sets an active model."""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence

import pandas as pd
import streamlit as st

import charts
import glossary
import ui
from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.reporting.portfolio_comparison import (
    PORTFOLIO_LABELS,
    cumulative_paths,
    drawdown_paths,
    rolling_sharpe_paths,
    rolling_volatility_paths,
)
from forecasting_engine.reporting.run_results import BASIS_LABELS, tail_with_simulation
from forecasting_engine.store import portfolio_runs

EQUITY, BOND = TargetRole.EQUITY, TargetRole.BOND


def title(run: portfolio_runs.SavedRun) -> str:
    return f"Run {run.run_number}" + (f": {run.name}" if run.name else "")


def _status(name: str, summary: Mapping[str, str]) -> None:
    st.markdown(
        ui.status_row(
            name,
            ui.lozenge(html.escape(summary["label"]), summary["tone"]),
            f"t = {summary['t']}",
        ),
        unsafe_allow_html=True,
    )
    st.caption(summary["detail"])


def _rows(rows: Sequence[Mapping[str, str]], columns: Mapping[str, str]) -> None:
    st.dataframe(
        pd.DataFrame([{shown: row[field] for field, shown in columns.items()} for row in rows]),
        hide_index=True,
        width="stretch",
    )


def _line(paths: pd.DataFrame, value: str, *, window: int | None = None, percent=True) -> None:
    if len(paths) < 2:  # a line needs two points
        st.caption(f"{value}: this backtest is too short for a chart.")
        return
    st.markdown(ui.eyebrow(value, glossary.term(value)), unsafe_allow_html=True)
    chart_title = None if window is None else f"{value}, {window}-day window"
    st.altair_chart(
        charts.line_chart(paths, value, percent=percent, title=chart_title), width="stretch"
    )


def render(run: portfolio_runs.SavedRun) -> None:
    results = run.results
    names = {role: results["names"][role.value] for role in (EQUITY, BOND)}
    backtest = results["backtest"]

    st.subheader(title(run))
    st.caption(
        f"First shown {run.recorded_at:%d %b %Y %H:%M}, code {run.code_version or 'not recorded'}. "
        "These are the saved results: nothing is refitted or solved again, and the active "
        "models stay as they are."
    )
    st.caption(f"Forecasts from {'; '.join(results['active_models'])}.")

    latest = results["latest"]
    st.markdown(
        ui.eyebrow("Latest weights", f"As of {pd.Timestamp(latest['date']):%d %b %Y}"),
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for col, role in zip(cols, (EQUITY, BOND), strict=True):
        col.metric(names[role], f"{latest['weights'][role.value]:.0%}")
    for line in results["how_set"]:
        st.caption(line)

    paths = portfolio_runs.saved_paths(run.run_key)
    st.markdown(ui.eyebrow("Weights per rebalance"), unsafe_allow_html=True)
    st.altair_chart(
        charts.weights_chart(
            paths.optimised.weights.rename(columns={role: names[role] for role in (EQUITY, BOND)})
        ),
        width="stretch",
    )

    basis_label = st.segmented_control(
        "Returns",
        list(BASIS_LABELS.values()),
        default=BASIS_LABELS["net"],
        required=True,
        key=f"saved_run_basis_{run.run_key}",
    )
    basis = next(key for key, label in BASIS_LABELS.items() if label == basis_label)
    shown = results["bases"][basis]

    st.markdown(
        ui.eyebrow(
            "Performance vs equal-weight benchmark", glossary.term("Equal-weight benchmark")
        ),
        unsafe_allow_html=True,
    )
    risk_level = results["settings"]["risk_level"]
    _rows(
        shown["comparison"],
        {
            "metric": "Metric",
            "optimised": PORTFOLIO_LABELS["optimised"],
            "baseline": PORTFOLIO_LABELS["baseline"],
            "no_forecast": f"Level {risk_level}, no forecasts",
            "difference": "Difference",
        },
    )
    st.caption(
        f"{basis_label}, backtest {pd.Timestamp(backtest['start']):%d %b %Y} to "
        f"{pd.Timestamp(backtest['end']):%d %b %Y}."
    )

    st.markdown(
        ui.eyebrow("Is the outperformance genuine?", glossary.term("Spread t-statistic")),
        unsafe_allow_html=True,
    )
    _status("Spread t-statistic", shown["significance"])
    _status(
        f"Did the forecasts add value? Against level {risk_level}, no forecasts",
        shown["forecast_value"],
    )

    simulation = portfolio_runs.latest_simulation(run.run_key)
    st.markdown(
        ui.eyebrow("Tail risk", glossary.term("Historical tail risk")), unsafe_allow_html=True
    )
    tail_columns = {
        "method": "Method",
        "metric": "Measure",
        "optimised": PORTFOLIO_LABELS["optimised"],
        "baseline": PORTFOLIO_LABELS["baseline"],
    }
    _rows(tail_with_simulation(shown["tail"], simulation), tail_columns)
    if simulation is not None:
        st.markdown(
            ui.eyebrow(
                f"Monte Carlo over the next {simulation['horizon']} days",
                glossary.term("Simulated tail risk"),
            ),
            unsafe_allow_html=True,
        )
        _rows(simulation["rows"], {k: v for k, v in tail_columns.items() if k != "method"})
        st.caption(
            f"{simulation['paths']:,} paths, {simulation['method']} sampling, from "
            f"{pd.Timestamp(simulation['as_of']):%d %b %Y}."
        )

    windows = results["windows"]
    st.altair_chart(
        charts.line_chart(cumulative_paths(paths, basis), "Cumulative return"), width="stretch"
    )
    _line(
        rolling_volatility_paths(paths, basis, names, window=windows["rolling_volatility"]),
        "Rolling volatility",
    )
    _line(drawdown_paths(paths, basis, names), "Drawdown")
    _line(
        rolling_sharpe_paths(paths, basis, names, window=windows["rolling_sharpe"]),
        "Rolling Sharpe",
        window=windows["rolling_sharpe"],
        percent=False,
    )
