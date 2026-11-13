"""One portfolio run opened from the Run history page: what the Portfolio page showed
for it, read back from ``store/portfolio_runs``. Nothing here fits a model, solves the
optimiser or sets an active model. The same saved results make its report."""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from functools import partial

import pandas as pd
import streamlit as st

import charts
import glossary
import page_cache
import ui
from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.reporting.portfolio_comparison import (
    PORTFOLIO_LABELS,
    cumulative_paths,
    drawdown_paths,
    rolling_sharpe_paths,
    rolling_volatility_paths,
)
from forecasting_engine.reporting.run_report import report_html
from forecasting_engine.reporting.run_results import BASIS_LABELS, tail_with_simulation
from forecasting_engine.store import portfolio_runs

EQUITY, BOND = TargetRole.EQUITY, TargetRole.BOND


def report_bytes(
    run_key: str, *, simulation: tuple[str, int] | None = None, latest_saved: bool = True
) -> bytes:
    """A run's report, written from what was saved for it. ``simulation`` is the method
    and paths to include; otherwise ``latest_saved`` says whether to use the run's most
    recent simulation (Run history) or show Monte Carlo as not run (a Portfolio page
    that isn't showing one)."""
    run = portfolio_runs.get_saved_run(run_key)
    if simulation is not None:
        simulated = portfolio_runs.saved_simulation(run_key, *simulation)
    else:
        simulated = portfolio_runs.latest_simulation(run_key) if latest_saved else None
    return report_html(run, portfolio_runs.saved_paths(run_key), simulated).encode("utf-8")


def report_button(
    container,
    run_key: str,
    *,
    simulation: tuple[str, int] | None = None,
    latest_saved: bool = True,
    label: str = "Download run report",
    number: int | None = None,
) -> None:
    """Generated on the click, not on every rerun of the page. ``on_click="ignore"``
    since downloading changes nothing the page shows."""
    if number is None:
        number = portfolio_runs.run_number(run_key)
    container.download_button(
        label,
        data=partial(report_bytes, run_key, simulation=simulation, latest_saved=latest_saved),
        file_name=f"run-{number}-report.html",
        mime="text/html",
        key=f"report_{run_key}",
        on_click="ignore",
        icon=":material/download:",
        help="One HTML file to read offline: open it in any browser, or print it to PDF.",
    )


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


@st.fragment
def render(run: portfolio_runs.SavedRun) -> None:
    """The opened run, as the Portfolio page showed it. A fragment, so switching
    between after and before costs redraws the run alone, not the list above it."""
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

    paths = page_cache.saved_paths(run.run_key)
    st.markdown(ui.eyebrow("Weights per rebalance"), unsafe_allow_html=True)
    # A run saved before its weight limits were kept is drawn without them.
    limits = results["settings"].get("bounds")
    st.altair_chart(
        charts.weights_chart(
            paths.optimised.weights.rename(columns={role: names[role] for role in (EQUITY, BOND)}),
            limits=None if limits is None else (limits[0], limits[1]),
            until=paths.end,
        ),
        width="stretch",
    )
    st.caption(
        f"The line is {names[EQUITY]}'s weight and the rest is {names[BOND]}; each step is "
        "a rebalance."
        + ("" if limits is None else " Dashed lines mark the weight limits.")
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
