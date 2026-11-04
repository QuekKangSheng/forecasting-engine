"""FYP-275: a portfolio run's results as they are saved, so a run can be opened later
without refitting a model or solving the optimiser again.

The figures are the Portfolio page's own: the rows here come from the same
``reporting.portfolio_comparison`` functions the page shows, on both bases, already
formatted. Keeping the text rather than the numbers behind it means an old run keeps
reading as it did, even after a later change to how a figure is computed or shown.
Everything is JSON-safe, for ``store/portfolio_runs``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict

import pandas as pd

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.portfolio.backtest import BASES, BacktestResult
from forecasting_engine.reporting.portfolio_comparison import (
    PORTFOLIO_LABELS,
    comparison_rows,
    forecast_value_summary,
    monte_carlo_rows,
    significance_summary,
    tail_rows,
)
from forecasting_engine.risk.monte_carlo import MonteCarloResult, as_tail_risk
from forecasting_engine.risk.tail import MONTE_CARLO

BASIS_LABELS: dict[str, str] = {"net": "After costs", "gross": "Before costs"}


def run_results(
    backtest: BacktestResult,
    no_forecast: BacktestResult,
    *,
    names: Mapping[TargetRole, str],
    settings: Mapping[str, object],
    latest_date: pd.Timestamp,
    latest_weights: Mapping[TargetRole, float],
    how_set: Sequence[str],
    windows: Mapping[str, int],
) -> dict:
    """Everything an opened run shows that its daily returns can't give back.

    ``windows`` are the rolling charts' windows, so a run's charts keep the windows it
    was shown with."""
    return {
        "names": {role.value: name for role, name in names.items()},
        "settings": dict(settings),
        "latest": {
            "date": latest_date.date().isoformat(),
            "weights": {role.value: float(weight) for role, weight in latest_weights.items()},
        },
        "how_set": list(how_set),
        "active_models": list(backtest.active_models),
        "backtest": {
            "start": backtest.start.date().isoformat(),
            "end": backtest.end.date().isoformat(),
        },
        "windows": dict(windows),
        "bases": {basis: _basis_results(backtest, no_forecast, basis) for basis in BASES},
    }


def _basis_results(backtest: BacktestResult, no_forecast: BacktestResult, basis: str) -> dict:
    rows = zip(comparison_rows(backtest, basis), comparison_rows(no_forecast, basis), strict=True)
    return {
        "comparison": [
            {**asdict(row), "no_forecast": level_row.optimised} for row, level_row in rows
        ],
        "significance": asdict(significance_summary(backtest, basis)),
        "forecast_value": asdict(forecast_value_summary(backtest, no_forecast, basis)),
        "tail": [asdict(row) for row in tail_rows(backtest, basis)],
    }


def simulation_results(backtest: BacktestResult, simulated: MonteCarloResult) -> dict:
    """A Monte Carlo simulation of a run: its own table and, since no path trades, the
    one-day rows it fills in on either basis of the tail table."""
    risks = {
        (portfolio, basis): as_tail_risk(simulated, portfolio)
        for portfolio in PORTFOLIO_LABELS
        for basis in BASES
    }
    one_day = tail_rows(backtest, BASES[0], simulated={MONTE_CARLO: risks})
    return {
        "method": simulated.method,
        "paths": simulated.paths,
        "horizon": simulated.horizon,
        "as_of": simulated.as_of.date().isoformat(),
        "history_start": simulated.history_start.date().isoformat(),
        "history_days": simulated.history_days,
        "forecasts": {TargetRole(k).value: float(v) for k, v in simulated.forecasts.items()},
        "rows": [asdict(row) for row in monte_carlo_rows(simulated)],
        "one_day": [asdict(row) for row in one_day if row.method == MONTE_CARLO],
    }


def tail_with_simulation(
    tail: Sequence[Mapping[str, str]], simulation: Mapping[str, object] | None
) -> list[Mapping[str, str]]:
    """A run's saved tail rows, with the Monte Carlo ones filled in if it was simulated."""
    if simulation is None:
        return list(tail)
    return [row for row in tail if row["method"] != MONTE_CARLO] + list(simulation["one_day"])
