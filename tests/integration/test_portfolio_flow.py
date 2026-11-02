"""End-to-end: equity/bond signals -> an optimised weight schedule -> a backtest
against the 50/50 baseline -> the shared comparison and tail-risk display rows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.portfolio.backtest import BacktestDataError, run_backtest
from forecasting_engine.portfolio.optimize import (
    schedule_breakdowns,
    schedule_weights,
    weight_schedule,
)
from forecasting_engine.portfolio.performance import relative
from forecasting_engine.reporting.portfolio_comparison import (
    NOT_RUN,
    comparison_rows,
    forecast_value_summary,
    tail_rows,
)
from forecasting_engine.risk.tail import HISTORICAL, MONTE_CARLO

EQUITY, BOND = TargetRole.EQUITY, TargetRole.BOND


def _prices(dates: pd.DatetimeIndex, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    equity = pd.Series(100 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(dates))), index=dates)
    bond = pd.Series(100 * np.cumprod(1 + rng.normal(0.0001, 0.004, len(dates))), index=dates)
    return {EQUITY: equity, BOND: bond}


def _signals(dates: pd.DatetimeIndex, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    equity = pd.Series(rng.normal(0, 0.01, len(dates)), index=dates)
    bond = pd.Series(rng.normal(0, 0.005, len(dates)), index=dates)
    return {EQUITY: equity, BOND: bond}


@pytest.fixture(autouse=True)
def no_tuning_period(monkeypatch):
    """Same fixture-sizing workaround as ``test_optimize.py``: a real model run
    reserves a 504-row tuning period before its first out-of-sample day, too
    large for this flow's small synthetic history."""
    monkeypatch.setattr("forecasting_engine.portfolio.optimize.TUNING_ROWS", 0)


def test_optimised_weights_flow_from_signals_through_the_backtest_to_the_comparison_table():
    dates = pd.bdate_range("2024-01-02", periods=150)
    prices = _prices(dates, seed=1)
    signals = _signals(dates, seed=2)

    schedule = weight_schedule(
        signals,
        prices,
        baselines=signals,
        horizon=5,
        train_window=20,
        test_window=20,
        embargo=5,
    )
    assert not schedule.empty

    result = run_backtest(prices, schedule, active_models=("Polynomial", "FF5 Benchmark"))

    assert result.active_models == ("Polynomial", "FF5 Benchmark")

    rows = comparison_rows(result, "net")
    assert {row.metric for row in rows} == {
        "Annual return",
        "Sharpe",
        "Sortino",
        "Calmar",
        "Max drawdown",
    }

    tail = tail_rows(result, "net")
    assert any(row.metric.startswith("1-day VaR") for row in tail)
    assert any(row.metric.startswith("1-day CVaR") for row in tail)
    assert any("breach rate" in row.metric for row in tail)
    historical = [row for row in tail if row.method == HISTORICAL]
    assert all(row.optimised.endswith("%") for row in historical if row.metric.startswith("1-day"))
    assert {(row.optimised, row.baseline) for row in tail if row.method == MONTE_CARLO} == {
        (NOT_RUN, NOT_RUN)
    }


ORACLE_HORIZON = 20
ORACLE_SETTINGS = {
    "horizon": ORACLE_HORIZON,
    "train_window": 504,
    "test_window": 20,
    "embargo": 20,
    "risk_level": 3,
}


def _oracle_inputs(baseline: float = 0.0, equity_share: float = 1.0) -> tuple[dict, dict, dict]:
    """Signals whose difference is the equity-minus-bond return over the horizon that
    follows, worked out here from the prices rather than through project code, less the
    training-mean difference. ``equity_share`` is how much of it the equity signal holds;
    the bond signal holds the rest, with the opposite sign. The equity premium is clearly
    positive, since with none the optimiser has no scale to tilt by."""
    dates = pd.bdate_range("2015-01-02", periods=1500)
    rng = np.random.default_rng(1)
    prices = {
        EQUITY: pd.Series(100 * np.cumprod(1 + rng.normal(0.0008, 0.01, len(dates))), index=dates),
        BOND: pd.Series(100 * np.cumprod(1 + rng.normal(0.0001, 0.004, len(dates))), index=dates),
    }
    following = (prices[EQUITY].shift(-ORACLE_HORIZON) / prices[EQUITY] - 1) - (
        prices[BOND].shift(-ORACLE_HORIZON) / prices[BOND] - 1
    )
    baselines = {EQUITY: pd.Series(baseline, index=dates), BOND: pd.Series(0.0, index=dates)}
    difference = following.fillna(0.0) - baseline
    signals = {EQUITY: equity_share * difference, BOND: (equity_share - 1) * difference}
    return prices, signals, baselines


def _optimised_and_no_forecast(prices, signals, baselines, **kwargs):
    settings = {**ORACLE_SETTINGS, "baselines": baselines, **kwargs}
    return (
        run_backtest(prices, schedule_weights(schedule_breakdowns(signals, prices, **settings))),
        run_backtest(
            prices,
            schedule_weights(schedule_breakdowns(signals, prices, **{**settings, "trust": 0.0})),
        ),
    )


@pytest.mark.parametrize(
    ("baseline", "equity_share"),
    [(0.0, 1.0), (0.0, 0.0), (0.0, 0.5), (0.004, 1.0), (-0.004, 0.5)],
)
def test_an_oracle_forecast_beats_the_same_level_with_no_forecasts(baseline, equity_share):
    prices, signals, baselines = _oracle_inputs(baseline, equity_share)

    optimised, no_forecast = _optimised_and_no_forecast(prices, signals, baselines)

    spread = relative(optimised.optimised.net, no_forecast.optimised.net)
    assert spread.spread_t > 0
    assert (
        optimised.metrics[("optimised", "net")].annual_return
        > no_forecast.metrics[("optimised", "net")].annual_return
    )
    assert "Forecasts added value" in forecast_value_summary(optimised, no_forecast, "net").label


def test_an_oracle_read_backwards_loses_to_the_same_level_with_no_forecasts():
    prices, signals, baselines = _oracle_inputs()
    backwards = {role: -signal for role, signal in signals.items()}

    optimised, no_forecast = _optimised_and_no_forecast(prices, backwards, baselines, trust=1.0)

    assert relative(optimised.optimised.net, no_forecast.optimised.net).spread_t < 0


def test_a_real_schedule_fed_a_price_series_missing_an_index_raises_before_the_comparison_table():
    dates = pd.bdate_range("2024-01-02", periods=150)
    prices = _prices(dates, seed=1)
    signals = _signals(dates, seed=2)

    schedule = weight_schedule(
        signals,
        prices,
        baselines=signals,
        horizon=5,
        train_window=20,
        test_window=20,
        embargo=5,
    )
    assert not schedule.empty

    incomplete_prices = {EQUITY: prices[EQUITY]}

    with pytest.raises(BacktestDataError, match="bond"):
        run_backtest(incomplete_prices, schedule)
