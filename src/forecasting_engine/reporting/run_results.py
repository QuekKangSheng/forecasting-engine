"""FYP-275: a portfolio run's results as they are saved, so a run can be opened later
without refitting a model or solving the optimiser again.

The figures are the Portfolio page's own: the rows here come from the same
``reporting.portfolio_comparison`` functions the page shows, on both bases, already
formatted. Keeping the text rather than the numbers behind it means an old run keeps
reading as it did, even after a later change to how a figure is computed or shown.
Everything is JSON-safe, for ``store/portfolio_runs``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import asdict

import pandas as pd

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.extraction.validation import ValidationReport
from forecasting_engine.portfolio.backtest import BASES, BacktestResult
from forecasting_engine.portfolio.optimize import SolveInputs, WeightBreakdown
from forecasting_engine.portfolio.performance import TRADING_DAYS_PER_YEAR
from forecasting_engine.reporting.polynomial_function import (
    from_description,
    standardisation_lines,
    term_rows,
)
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
from forecasting_engine.store.active_model import ActiveFunction, ActiveModelRecord
from forecasting_engine.validation.gates import evaluate_candidate

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
    data_quality: Mapping[str, object] | None = None,
    models: Sequence[Mapping[str, object]] = (),
    conditions: Mapping[str, object] | None = None,
) -> dict:
    """Everything an opened run or its report shows that its daily returns can't give
    back.

    ``windows`` are the rolling charts' windows, so a run's charts keep the windows it
    was shown with. ``data_quality`` is ``data_quality_summary`` and ``models`` one
    ``model_summary`` per active model, and ``conditions`` is ``market_conditions`` for
    the latest rebalance."""
    return {
        "data_quality": None if data_quality is None else dict(data_quality),
        "models": list(models),
        "names": {role.value: name for role, name in names.items()},
        "settings": dict(settings),
        "latest": {
            "date": latest_date.date().isoformat(),
            "weights": {role.value: float(weight) for role, weight in latest_weights.items()},
        },
        "how_set": list(how_set),
        "conditions": None if conditions is None else dict(conditions),
        "active_models": list(backtest.active_models),
        "backtest": {
            "start": backtest.start.date().isoformat(),
            "end": backtest.end.date().isoformat(),
        },
        "windows": dict(windows),
        "bases": {basis: _basis_results(backtest, no_forecast, basis) for basis in BASES},
    }


def market_conditions(
    breakdown: WeightBreakdown, inputs: SolveInputs, *, horizon: int, gate_shut: bool
) -> dict:
    """What the latest rebalance's weights rest on, for a report to put in words: the
    risk level's own equity weight and the volatility it targets, how far the forecasts
    moved it and whether their record was trusted, and each index's volatility over
    the training window. Volatilities are annualised from the solve's ``horizon``-day
    covariance."""
    per_year = TRADING_DAYS_PER_YEAR / horizon
    covariance = inputs.covariance
    equity, bond = covariance.index
    w = breakdown.level_weight
    variance = (
        w**2 * covariance.loc[equity, equity]
        + (1 - w) ** 2 * covariance.loc[bond, bond]
        + 2 * w * (1 - w) * covariance.loc[equity, bond]
    )
    return {
        "level_weight": float(w),
        "forecast_tilt": float(breakdown.forecast_tilt),
        "equity": float(breakdown.equity),
        "capped": bool(breakdown.capped),
        "trust": float(inputs.trust),
        "gate_shut": bool(gate_shut),
        "target_volatility": math.sqrt(max(float(variance), 0.0) * per_year),
        "volatility": {
            TargetRole(role).value: math.sqrt(float(covariance.loc[role, role]) * per_year)
            for role in covariance.index
        },
    }


def data_quality_summary(
    report: ValidationReport, merged: pd.DataFrame, dropped: Collection[str]
) -> dict:
    """What the Data page's quality report says about the run's dataset. Nothing here
    is corrected; these are the flags as they stand."""
    dates = merged["Date"].dropna()
    return {
        "rows": len(merged),
        "signals": len(merged.columns) - 1,
        "first_date": None if dates.empty else dates.min().date().isoformat(),
        "last_date": None if dates.empty else dates.max().date().isoformat(),
        "schema_errors": list(report.schema_errors),
        "duplicate_dates": int(report.duplicate_dates),
        "weekend_rows": int(report.weekend_rows),
        "big_moves": len(report.big_moves),
        "incomplete": {
            column: float(pct) for column, pct in report.missing_pct.items() if pct > 0
        },
        "dropped": list(dropped),
    }


def model_summary(
    name: str,
    record: ActiveModelRecord,
    *,
    display_name: str,
    signal_t: float | None,
    function: ActiveFunction | None,
    attribution: Mapping[str, float] | None,
    holdout: Mapping[str, float] | None = None,
    label: Callable[[str], str],
    target: str,
    columns: Collection[str],
) -> dict:
    """One active model as a report shows it: its gate results, its polynomial function
    if it is one, its per-signal SHAP if that was saved, and a research model's pre-2016
    hold-out (``signal_rank_ic``, ``t``, ``windows``) if its run recorded one. ``None`` for what the
    model doesn't have or an older record didn't keep, so nothing is passed off as zero.
    """
    fn = (
        None
        if function is None
        else from_description(
            function.description,
            origin=function.origin,
            target=target,
            horizon=record.horizon,
            columns=columns,
        )
    )
    known = record.signal_rank_ic is not None and signal_t is not None
    outcome = evaluate_candidate(record.signal_rank_ic, signal_t) if known else None
    return {
        "name": name,
        "model": display_name,
        "set_at": record.set_at.isoformat(),
        "high_risk": bool(record.high_risk),
        "signal_rank_ic": record.signal_rank_ic,
        "signal_t": signal_t,
        "gate_met": None if outcome is None else outcome.promoted,
        "gates_failed": [] if outcome is None else list(outcome.failed_gates),
        "origin": None if function is None else function.origin.value,
        "formula": None if fn is None else fn.shape or fn.formula,
        "terms": None if fn is None else term_rows(fn, label),
        "standardisation": [] if fn is None else standardisation_lines(fn, label),
        "attribution": None
        if attribution is None
        else {label(signal): float(value) for signal, value in attribution.items()},
        "holdout": None if holdout is None else dict(holdout),
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
