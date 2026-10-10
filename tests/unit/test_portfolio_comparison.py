"""FYP-19: the optimised portfolio beside the equal-weight benchmark."""

import numpy as np
import pandas as pd
import pytest

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.portfolio.backtest import month_end_rebalance_dates, run_backtest
from forecasting_engine.reporting.portfolio_comparison import (
    METRICS,
    comparison_rows,
    cumulative_paths,
    tail_rows,
)

EQUITY, BOND = TargetRole.EQUITY, TargetRole.BOND


def _backtest(equity_weight: float = 0.8):
    """Rebalanced on the benchmark's own dates — the first day, then each month end
    strictly inside the period — so only the weights differ between the two."""
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2024-01-31", periods=120)
    prices = {
        EQUITY: pd.Series(100 * np.cumprod(1 + rng.normal(0.001, 0.01, 120)), index=idx),
        BOND: pd.Series(100 * np.cumprod(1 + rng.normal(0.0002, 0.003, 120)), index=idx),
    }
    month_ends = month_end_rebalance_dates(idx)
    dates = idx[:1].append(month_ends[(month_ends > idx[0]) & (month_ends < idx[-1])])
    schedule = pd.DataFrame(
        {EQUITY: [equity_weight] * len(dates), BOND: [1 - equity_weight] * len(dates)},
        index=dates,
    )
    return run_backtest(prices, schedule)


def test_the_four_ratios_the_story_asks_for_are_all_there():
    names = [row.metric for row in comparison_rows(_backtest(), "net")]

    assert {"Sharpe", "Sortino", "Calmar", "Max drawdown"} <= set(names)
    assert names == list(METRICS)


def test_each_row_shows_both_portfolios_from_the_backtest():
    result = _backtest()
    sharpe = next(r for r in comparison_rows(result, "net") if r.metric == "Sharpe")

    assert sharpe.optimised == f"{result.metrics[('optimised', 'net')].sharpe:.2f}"
    assert sharpe.baseline == f"{result.metrics[('baseline', 'net')].sharpe:.2f}"


def test_the_difference_is_optimised_minus_benchmark_and_signed():
    result = _backtest()
    sharpe = next(r for r in comparison_rows(result, "gross") if r.metric == "Sharpe")
    expected = (
        result.metrics[("optimised", "gross")].sharpe - result.metrics[("baseline", "gross")].sharpe
    )

    assert sharpe.difference == f"{expected:+.2f}"


def test_drawdown_is_a_percentage_and_its_difference_is_in_points():
    result = _backtest()
    row = next(r for r in comparison_rows(result, "net") if r.metric == "Max drawdown")

    assert row.optimised.endswith("%") and row.optimised.startswith("-")
    assert row.difference.endswith(" pts")


def test_an_identical_allocation_differs_by_nothing():
    rows = comparison_rows(_backtest(equity_weight=0.5), "gross")

    assert all(row.difference in ("+0.00", "+0.00 pts", "-0.00", "-0.00 pts") for row in rows)


def test_costs_lower_the_net_return_below_the_gross():
    result = _backtest()
    gross = cumulative_paths(result, "gross")
    net = cumulative_paths(result, "net")

    assert list(gross.columns) == ["Optimised", "Equal-weight benchmark"]
    assert (net.iloc[-1] < gross.iloc[-1]).all()


def test_an_unknown_basis_is_refused():
    with pytest.raises(ValueError):
        comparison_rows(_backtest(), "after tax")


# --- FYP-56: historical tail risk beside max drawdown ---------------------------------


def test_tail_rows_show_var_and_cvar_at_both_levels_then_drawdown_and_breaches():
    rows = tail_rows(_backtest(), "net")

    assert [r.metric for r in rows] == [
        "1-day VaR 95%",
        "1-day CVaR 95%",
        "1-day VaR 99%",
        "1-day CVaR 99%",
        "Max drawdown",
        "VaR 95% breach rate (expected 5%)",
        "VaR 99% breach rate (expected 1%)",
    ]


def test_tail_rows_read_the_backtests_own_figures_for_each_portfolio():
    result = _backtest()
    rows = {r.metric: r for r in tail_rows(result, "gross")}
    optimised = result.tail_risk[("optimised", "gross")]
    baseline = result.tail_risk[("baseline", "gross")]

    assert rows["1-day VaR 95%"].optimised == f"{optimised.levels[0].var * 100:.2f}%"
    assert rows["1-day CVaR 99%"].baseline == f"{baseline.levels[1].cvar * 100:.2f}%"
    assert rows["Max drawdown"].optimised == comparison_rows(result, "gross")[-1].optimised


def test_a_backtest_shorter_than_the_var_window_has_no_breach_rate():
    rows = {r.metric: r for r in tail_rows(_backtest(), "net")}  # 120 days < 252

    assert rows["VaR 95% breach rate (expected 5%)"].optimised == "—"


def test_tail_rows_refuse_an_unknown_basis():
    with pytest.raises(ValueError):
        tail_rows(_backtest(), "after tax")
