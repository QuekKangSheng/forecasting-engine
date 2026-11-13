"""What the pages keep between reruns, so a click doesn't redo work it can't change.

Streamlit runs a page's whole script again on every click. Left alone, each rerun
read the same rows back from DuckDB and solved and backtested the same allocation
again: seconds per click on the Portfolio page, for any control at all, including the
ones that move no weight (after or before costs, the Monte Carlo method).

Everything here is keyed on the contents of what it is computed from, never on a
name or a position, so a result is reused exactly where it would have come out the
same and nowhere else: other data, another active model or another setting is another
key. Nothing is stored that the app doesn't already compute, and the entries are
capped, so the cache can only cost memory up to a known size.

The work is split by what it depends on:

- ``schedule_inputs``: each rebalance's covariance and forecast trust. The data and
  the two active models decide these; the risk level does not. Estimated once, then
  every level and weight limit is solved from the same estimates.
- ``schedules``: the allocation at one risk level and weight limit, with and without
  forecasts.
- ``backtest``: one weight schedule run through the prices.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import streamlit as st

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.ingest.upload import frame_sha256
from forecasting_engine.portfolio import backtest as backtesting
from forecasting_engine.portfolio import optimize
from forecasting_engine.store import active_model, portfolio_runs

#: Distinct datasets-and-model-pairs whose estimates are kept: the one in use and the
#: few before it, for a model switched and switched back.
_INPUT_ENTRIES = 4
#: Distinct (risk level, weight limit) results kept: every level at a handful of limits.
_SOLVE_ENTRIES = 64

_SERIES = {
    "signal": active_model.get_active_model_signal,
    "baseline": active_model.get_active_model_baseline,
    "forecast": active_model.get_active_model_forecast,
}


@st.cache_data(show_spinner=False, max_entries=32)
def _active_series(kind: str, role: TargetRole, set_at: datetime) -> pd.Series | None:
    return _SERIES[kind](role, set_at=set_at)


def active_series(kind: str, record: active_model.ActiveModelRecord) -> pd.Series | None:
    """``record``'s saved ``"signal"``, ``"baseline"`` or ``"forecast"``, read from the
    store the first time and kept. The active-model log is append-only, so what is
    stored for a model set at a given time never changes."""
    return _active_series(kind, record.role, record.set_at)


@st.cache_data(show_spinner=False, max_entries=8)
def dataset_sha256(frame: pd.DataFrame) -> str:
    """``frame_sha256`` of the committed data, worked out once per dataset."""
    return frame_sha256(frame)


@dataclass(frozen=True)
class Schedules:
    """One risk level's allocation over the backtest, and the same level with no
    forecasts."""

    breakdowns: Mapping[pd.Timestamp, optimize.WeightBreakdown]
    no_forecast: Mapping[pd.Timestamp, optimize.WeightBreakdown]
    latest_inputs: optimize.SolveInputs | None
    """What the latest rebalance's target was solved from, or ``None`` with no rebalance."""
    forecasts_gated: bool
    """Whether, at the latest rebalance, the forecasts had a track record long enough
    to test and it was not statistically reliable (``optimize.forecast_gate_shut``)."""


@st.cache_resource(show_spinner=False, max_entries=_INPUT_ENTRIES)
def _schedule_inputs(
    signals: Mapping[TargetRole, pd.Series],
    baselines: Mapping[TargetRole, pd.Series],
    prices: Mapping[TargetRole, pd.Series],
    horizon: int,
    train_window: int,
    test_window: int,
    embargo: int,
) -> optimize.ScheduleInputs:
    # Kept as the object itself, not a copy per read: it is only ever read, and it
    # holds a small frame per rebalance, which would be copied on every rerun.
    return optimize.schedule_inputs(
        signals,
        prices,
        baselines=baselines,
        horizon=horizon,
        train_window=train_window,
        test_window=test_window,
        embargo=embargo,
    )


# The one slow step a visitor can meet, the first time a level is solved on new data
# or models: it says what it is doing rather than leaving the page half drawn.
@st.cache_data(show_spinner="Solving the allocation…", max_entries=_SOLVE_ENTRIES)
def schedules(
    signals: Mapping[TargetRole, pd.Series],
    baselines: Mapping[TargetRole, pd.Series],
    prices: Mapping[TargetRole, pd.Series],
    *,
    horizon: int,
    train_window: int,
    test_window: int,
    embargo: int,
    risk_level: int,
    bounds: tuple[float, float],
) -> Schedules:
    """The weight schedule at ``risk_level`` within ``bounds``, and that level's
    schedule with no forecasts. Raises what ``optimize.schedule_breakdowns`` raises."""
    inputs = _schedule_inputs(
        signals, baselines, prices, horizon, train_window, test_window, embargo
    )
    breakdowns = optimize.breakdowns_from(inputs, risk_level=risk_level, bounds=bounds)
    no_forecast = optimize.breakdowns_from(
        inputs, risk_level=risk_level, bounds=bounds, trust=0.0
    )
    latest = next(reversed(breakdowns), None)
    return Schedules(
        breakdowns=breakdowns,
        no_forecast=no_forecast,
        latest_inputs=None
        if latest is None
        else inputs.solve_inputs(latest, risk_level=risk_level, bounds=bounds),
        forecasts_gated=latest is not None and inputs.rebalances[latest].gate_shut,
    )


@st.cache_data(show_spinner=False, max_entries=16)
def _saved_paths(run_key: str) -> backtesting.BacktestResult:
    paths = portfolio_runs.saved_paths(run_key)
    if paths is None:
        # Not kept: a run recorded before its daily returns were saved gets them the
        # next time the Portfolio page shows it, and must then be found.
        raise LookupError(run_key)
    return paths


def saved_paths(run_key: str) -> backtesting.BacktestResult | None:
    """``portfolio_runs.saved_paths``, read from the store once per run: a saved run's
    daily returns and weights never change, and reading thousands of rows back on
    every rerun was most of what made an opened run slow to respond."""
    try:
        return _saved_paths(run_key)
    except LookupError:
        return None


@st.cache_data(show_spinner=False, max_entries=2 * _SOLVE_ENTRIES)
def backtest(
    prices: Mapping[TargetRole, pd.Series],
    weights: pd.DataFrame,
    active_models: tuple[str, ...] = (),
) -> backtesting.BacktestResult:
    """``backtest.run_backtest`` of ``weights``, kept for the next rerun that asks for
    the same schedule on the same prices."""
    return backtesting.run_backtest(prices, weights, active_models=active_models)
