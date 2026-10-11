"""End-to-end: equity/bond signals -> an optimised weight schedule -> a backtest
against the 50/50 baseline -> the shared comparison and tail-risk display rows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.portfolio.backtest import BacktestDataError, run_backtest
from forecasting_engine.portfolio.optimize import weight_schedule
from forecasting_engine.reporting.portfolio_comparison import comparison_rows, tail_rows

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
        signals, prices, horizon=5, train_window=20, test_window=20, embargo=5
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


def test_a_real_schedule_fed_a_price_series_missing_an_index_raises_before_the_comparison_table():
    dates = pd.bdate_range("2024-01-02", periods=150)
    prices = _prices(dates, seed=1)
    signals = _signals(dates, seed=2)

    schedule = weight_schedule(
        signals, prices, horizon=5, train_window=20, test_window=20, embargo=5
    )
    assert not schedule.empty

    incomplete_prices = {EQUITY: prices[EQUITY]}

    with pytest.raises(BacktestDataError, match="bond"):
        run_backtest(incomplete_prices, schedule)
