"""Chaining an optimised allocation and the 50/50 baseline through the same days."""

import math

import numpy as np
import pandas as pd
import pytest

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.portfolio.backtest import (
    BASELINE_WEIGHTS,
    DEFAULT_COSTS_BPS,
    BacktestDataError,
    chain,
    month_end_rebalance_dates,
    run_backtest,
)
from forecasting_engine.risk.tail import historical_var

EQUITY, BOND = TargetRole.EQUITY, TargetRole.BOND


def _prices(equity_returns, bond_returns, start="2024-01-31") -> dict[TargetRole, pd.Series]:
    """Index levels starting at 100 on ``start``, then compounding the given returns."""
    idx = pd.bdate_range(start, periods=len(equity_returns) + 1)
    return {
        EQUITY: pd.Series(100 * np.cumprod([1.0, *(1 + np.array(equity_returns))]), index=idx),
        BOND: pd.Series(100 * np.cumprod([1.0, *(1 + np.array(bond_returns))]), index=idx),
    }


def _schedule(dates, equity_weights) -> pd.DataFrame:
    return pd.DataFrame(
        {EQUITY: equity_weights, BOND: [1 - w for w in equity_weights]},
        index=pd.DatetimeIndex(dates),
    )


NO_COSTS = {EQUITY: 0.0, BOND: 0.0}


# --- chaining ------------------------------------------------------------------


def test_an_all_equity_allocation_earns_exactly_the_equity_return():
    prices = _prices([0.01, -0.02, 0.03], [0.001, 0.002, -0.001])
    weights = _schedule([prices[EQUITY].index[0]], [1.0])

    result = run_backtest(prices, weights, costs_bps=NO_COSTS)

    expected = prices[EQUITY].pct_change().iloc[1:]
    np.testing.assert_allclose(result.optimised.gross, expected)


def test_weights_drift_with_returns_between_rebalances():
    # Equity +10% on day one, so a 50/50 split drifts to 0.55/0.50 of a 1.05
    # whole. Day two must be earned on the drifted weights; holding 50/50 fixed
    # every day would quietly be a daily rebalance.
    prices = _prices([0.10, 0.02], [0.0, 0.0])
    weights = _schedule([prices[EQUITY].index[0]], [0.5])

    gross = run_backtest(prices, weights, costs_bps=NO_COSTS).optimised.gross

    assert gross.iloc[0] == pytest.approx(0.05)
    assert gross.iloc[1] == pytest.approx(0.55 / 1.05 * 0.02)


def test_turnover_is_measured_against_the_drifted_weights():
    prices = _prices([0.10, 0.0], [0.0, 0.0])
    days = prices[EQUITY].index
    weights = _schedule([days[0], days[1]], [0.5, 0.5])

    path = run_backtest(prices, weights, costs_bps=NO_COSTS).optimised

    drifted = 0.55 / 1.05
    assert path.turnover.loc[days[1]] == pytest.approx(2 * (drifted - 0.5))


def test_the_first_allocation_from_cash_is_full_turnover():
    prices = _prices([0.0, 0.0], [0.0, 0.0])
    weights = _schedule([prices[EQUITY].index[0]], [0.6])

    path = run_backtest(prices, weights, costs_bps=NO_COSTS).optimised

    assert path.turnover.iloc[0] == pytest.approx(1.0)


# --- costs ---------------------------------------------------------------------


def test_each_index_is_charged_its_own_rate_on_what_was_traded():
    prices = _prices([0.0, 0.0], [0.0, 0.0])
    weights = _schedule([prices[EQUITY].index[0]], [0.6])

    path = run_backtest(prices, weights).optimised

    expected = 0.6 * DEFAULT_COSTS_BPS[EQUITY] / 1e4 + 0.4 * DEFAULT_COSTS_BPS[BOND] / 1e4
    assert path.costs.iloc[0] == pytest.approx(expected)
    # Flat prices: the only thing the net return loses is the cost of buying in.
    assert path.net.iloc[0] == pytest.approx(-expected)


def test_costs_scale_with_turnover():
    prices = _prices([0.0] * 3, [0.0] * 3)
    days = prices[EQUITY].index
    small = run_backtest(prices, _schedule([days[0], days[1]], [0.5, 0.6])).optimised
    large = run_backtest(prices, _schedule([days[0], days[1]], [0.5, 0.7])).optimised

    assert large.costs.loc[days[1]] == pytest.approx(2 * small.costs.loc[days[1]])


def test_rebalancing_to_where_the_weights_already_are_costs_nothing():
    prices = _prices([0.0] * 3, [0.0] * 3)
    days = prices[EQUITY].index
    path = run_backtest(prices, _schedule([days[0], days[1]], [0.5, 0.5])).optimised

    assert path.costs.loc[days[1]] == 0.0


def test_cost_rates_are_configurable():
    prices = _prices([0.0], [0.0])
    weights = _schedule([prices[EQUITY].index[0]], [1.0])

    path = run_backtest(prices, weights, costs_bps={EQUITY: 10.0, BOND: 0.0}).optimised

    assert path.costs.iloc[0] == pytest.approx(10.0 / 1e4)


def test_a_cost_is_compounded_into_the_return_not_subtracted_from_it():
    prices = _prices([0.10, 0.0], [0.10, 0.0])
    weights = _schedule([prices[EQUITY].index[0]], [1.0])

    path = run_backtest(prices, weights, costs_bps={EQUITY: 100.0, BOND: 0.0}).optimised

    assert path.net.iloc[0] == pytest.approx(1.10 * (1 - 0.01) - 1)


# --- the baseline and like-for-like ------------------------------------------------


def test_both_portfolios_cover_exactly_the_same_days():
    rng = np.random.default_rng(0)
    prices = _prices(rng.normal(0, 0.01, 80), rng.normal(0, 0.003, 80))
    first = prices[EQUITY].index[5]
    weights = _schedule([first], [0.7])

    result = run_backtest(prices, weights)

    assert result.optimised.gross.index.equals(result.baseline.gross.index)
    assert result.optimised.net.index.equals(result.baseline.net.index)
    assert result.start == first
    assert result.optimised.gross.index[0] > first, "returns begin the day after allocating"
    assert result.end == result.optimised.gross.index[-1]


def test_the_baseline_is_fifty_fifty_rebalanced_at_each_month_end():
    rng = np.random.default_rng(1)
    prices = _prices(rng.normal(0, 0.01, 70), rng.normal(0, 0.003, 70), start="2024-01-31")
    result = run_backtest(prices, _schedule([prices[EQUITY].index[0]], [0.7]))

    rebalances = result.baseline.weights.index
    month_ends = month_end_rebalance_dates(prices[EQUITY].index)
    inside = month_ends[(month_ends > result.start) & (month_ends < result.end)]
    assert len(inside) >= 2, "fixture should span several month ends"
    assert rebalances[0] == result.start
    assert list(rebalances[1:]) == list(inside)
    assert (result.baseline.weights == pd.Series(BASELINE_WEIGHTS)).all().all()


def test_month_end_rebalances_fall_on_each_months_last_trading_day():
    calendar = pd.DatetimeIndex(
        ["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-28", "2024-03-01"]
    )
    assert list(month_end_rebalance_dates(calendar)) == list(
        pd.DatetimeIndex(["2024-01-31", "2024-02-28", "2024-03-01"])
    )


# --- the two markets' calendars ------------------------------------------------------


def test_a_market_holiday_moves_its_return_to_the_next_day_without_losing_it():
    equity_days = pd.bdate_range("2024-01-31", periods=5)
    bond_days = equity_days.delete(2)  # bond market shut on the third day
    prices = {
        EQUITY: pd.Series([100, 101, 102, 103, 104], index=equity_days, dtype=float),
        BOND: pd.Series([100, 100.5, 101.5, 102.0], index=bond_days, dtype=float),
    }
    weights = _schedule([equity_days[0]], [0.0])

    gross = run_backtest(prices, weights, costs_bps=NO_COSTS).optimised.gross

    assert gross.loc[equity_days[2]] == 0.0
    assert gross.loc[equity_days[3]] == pytest.approx(101.5 / 100.5 - 1)
    assert (1 + gross).prod() == pytest.approx(102.0 / 100)


def test_a_gap_longer_than_a_holiday_is_refused_not_treated_as_flat():
    equity_days = pd.bdate_range("2024-01-31", periods=12)
    bond_days = equity_days.delete(range(2, 8))
    prices = {
        EQUITY: pd.Series(100.0, index=equity_days),
        BOND: pd.Series(100.0, index=bond_days),
    }

    with pytest.raises(BacktestDataError, match="BOND|bond"):
        run_backtest(prices, _schedule([equity_days[0]], [0.5]))


def test_the_period_ends_when_either_index_runs_out():
    equity_days = pd.bdate_range("2024-01-31", periods=10)
    prices = {
        EQUITY: pd.Series(100.0, index=equity_days),
        BOND: pd.Series(100.0, index=equity_days[:7]),
    }

    result = run_backtest(prices, _schedule([equity_days[0]], [0.5]))

    assert result.end == equity_days[6]


# --- refusing a schedule that can't be honoured ----------------------------------------


@pytest.mark.parametrize(
    ("dates", "weights", "message"),
    [
        (["2024-02-01"], [[0.6, 0.6]], "sum to 1"),
        (["2024-02-03"], [[0.5, 0.5]], "trading day"),  # a Saturday
        (["2023-12-01"], [[0.5, 0.5]], "before"),
        ([], [], "at least one"),
        (["2024-02-01"], [[float("nan"), 1.0]], "sum to 1"),
    ],
)
def test_a_schedule_that_cant_be_honoured_is_refused(dates, weights, message):
    prices = _prices([0.0] * 10, [0.0] * 10)
    schedule = pd.DataFrame(weights, index=pd.DatetimeIndex(dates), columns=[EQUITY, BOND])

    with pytest.raises(BacktestDataError, match=message):
        run_backtest(prices, schedule)


# --- the one object downstream reads ---------------------------------------------------


def test_the_result_carries_every_metric_both_ways_for_both_portfolios():
    rng = np.random.default_rng(2)
    prices = _prices(rng.normal(0, 0.01, 60), rng.normal(0, 0.003, 60))
    result = run_backtest(
        prices, _schedule([prices[EQUITY].index[0]], [0.7]), active_models=("Polynomial",)
    )

    assert set(result.metrics) == {
        ("optimised", "gross"),
        ("optimised", "net"),
        ("baseline", "gross"),
        ("baseline", "net"),
    }
    assert set(result.relative) == {"gross", "net"}
    assert result.rebalance_frequency == "monthly"
    assert result.active_models == ("Polynomial",)
    assert dict(result.costs_bps) == dict(DEFAULT_COSTS_BPS)
    for metrics in result.metrics.values():
        assert not math.isnan(metrics.sharpe)


def test_net_never_beats_gross():
    rng = np.random.default_rng(3)
    prices = _prices(rng.normal(0, 0.01, 60), rng.normal(0, 0.003, 60))
    days = prices[EQUITY].index
    result = run_backtest(prices, _schedule([days[0], days[20], days[40]], [0.8, 0.2, 0.6]))

    for path in (result.optimised, result.baseline):
        assert (path.net <= path.gross + 1e-15).all()
        assert ((1 + path.net).prod()) < ((1 + path.gross).prod())


def test_chain_is_usable_on_its_own_returns():
    # The same chaining serves the baseline and the optimised portfolio, so it
    # is exposed for the optimiser's own checks.
    days = pd.bdate_range("2024-01-31", periods=3)
    returns = pd.DataFrame({EQUITY: [0.0, 0.1, 0.0], BOND: [0.0, 0.0, 0.0]}, index=days)

    path = chain(returns, _schedule([days[0]], [0.5]), NO_COSTS)

    assert list(path.gross) == pytest.approx([0.05, 0.0])


def test_each_path_carries_its_historical_tail_risk_beside_its_drawdown():
    rng = np.random.default_rng(4)
    prices = _prices(rng.normal(0, 0.01, 300), rng.normal(0, 0.003, 300))
    result = run_backtest(prices, _schedule([prices[EQUITY].index[0]], [0.7]))

    assert set(result.tail_risk) == set(result.metrics)
    for key, risk in result.tail_risk.items():
        portfolio, basis = key
        path = getattr(getattr(result, portfolio), basis)
        assert risk.method == "Historical"
        assert risk.max_drawdown == result.metrics[key].max_drawdown
        assert risk.days == len(path)
        assert [level.var for level in risk.levels] == [
            historical_var(path, level.confidence)[0] for level in risk.levels
        ]
