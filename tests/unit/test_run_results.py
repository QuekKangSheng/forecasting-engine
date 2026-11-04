"""A portfolio run's saved results: JSON-safe, and the page's own rows on both bases."""

import json

import numpy as np
import pandas as pd

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.portfolio.backtest import run_backtest
from forecasting_engine.reporting.portfolio_comparison import (
    NOT_RUN,
    comparison_rows,
    significance_summary,
    tail_rows,
)
from forecasting_engine.reporting.run_results import (
    run_results,
    simulation_results,
    tail_with_simulation,
)
from forecasting_engine.risk.monte_carlo import history_window, simulate
from forecasting_engine.risk.tail import HISTORICAL, MONTE_CARLO

EQUITY, BOND = TargetRole.EQUITY, TargetRole.BOND


def _prices():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2024-01-31", periods=300)
    return {
        EQUITY: pd.Series(100 * np.cumprod(1 + rng.normal(0.001, 0.01, 300)), index=idx),
        BOND: pd.Series(100 * np.cumprod(1 + rng.normal(0.0002, 0.003, 300)), index=idx),
    }


def _schedule(index, equity):
    dates = index[::20]
    return pd.DataFrame(
        {EQUITY: [equity] * len(dates), BOND: [1 - equity] * len(dates)}, index=dates
    )


PRICES = _prices()
INDEX = PRICES[EQUITY].index
BACKTEST = run_backtest(PRICES, _schedule(INDEX, 0.7), active_models=("S&P 500: Polynomial",))
NO_FORECAST = run_backtest(PRICES, _schedule(INDEX, 0.6))


def _results():
    return run_results(
        BACKTEST,
        NO_FORECAST,
        names={EQUITY: "S&P 500", BOND: "US Aggregate"},
        settings={"risk_level": 3, "bounds": [0.2, 0.8]},
        latest_date=INDEX[-1],
        latest_weights={EQUITY: np.float64(0.7), BOND: np.float64(0.3)},
        how_set=["How this was set: level 3."],
        windows={"rolling_volatility": 63, "rolling_sharpe": 252},
    )


def _simulation():
    simulated = simulate(
        history_window(PRICES, INDEX[-1], window=200),
        pd.Series({EQUITY: 0.01, BOND: 0.002}),
        {
            "optimised": pd.Series({EQUITY: 0.7, BOND: 0.3}),
            "baseline": pd.Series({EQUITY: 0.5, BOND: 0.5}),
        },
        horizon=5,
        method="bootstrap",
        paths=1_000,
    )
    return simulation_results(BACKTEST, simulated)


def test_results_survive_a_json_round_trip_unchanged():
    results = _results()

    assert json.loads(json.dumps(results)) == results


def test_results_hold_the_pages_rows_for_both_bases():
    results = _results()

    for basis in ("net", "gross"):
        saved = results["bases"][basis]
        page = comparison_rows(BACKTEST, basis)
        assert [r["optimised"] for r in saved["comparison"]] == [r.optimised for r in page]
        assert [r["difference"] for r in saved["comparison"]] == [r.difference for r in page]
        assert [r["no_forecast"] for r in saved["comparison"]] == [
            r.optimised for r in comparison_rows(NO_FORECAST, basis)
        ]
        assert saved["significance"]["t"] == significance_summary(BACKTEST, basis).t
        tail = tail_rows(BACKTEST, basis)
        assert [r["metric"] for r in saved["tail"]] == [r.metric for r in tail]


def test_each_basis_keeps_its_own_t_statistics():
    # Heavy trading at a high cost, so before and after costs clearly differ.
    weights = [0.8, 0.2] * (len(DATES) // 2) + [0.8] * (len(DATES) % 2)
    swinging = pd.DataFrame(
        {EQUITY: weights, BOND: [1 - w for w in weights]}, index=DATES
    )
    costs = {EQUITY: 100.0, BOND: 100.0}
    backtest = run_backtest(PRICES, swinging, costs_bps=costs)
    no_forecast = run_backtest(PRICES, _schedule(INDEX, 0.6), costs_bps=costs)

    results = run_results(
        backtest,
        no_forecast,
        names={EQUITY: "S&P 500", BOND: "US Aggregate"},
        settings={},
        latest_date=INDEX[-1],
        latest_weights={EQUITY: 0.8, BOND: 0.2},
        how_set=[],
        windows={},
    )

    net, gross = results["bases"]["net"], results["bases"]["gross"]
    assert net["significance"]["t"] != gross["significance"]["t"]
    assert net["forecast_value"]["t"] != gross["forecast_value"]["t"]
    assert net["significance"]["t"] == significance_summary(backtest, "net").t
    assert gross["significance"]["t"] == significance_summary(backtest, "gross").t


def test_a_simulation_survives_a_json_round_trip_unchanged():
    simulation = _simulation()

    assert json.loads(json.dumps(simulation)) == simulation
    assert simulation["forecasts"] == {"equity": 0.01, "bond": 0.002}


def test_a_simulation_fills_in_only_the_monte_carlo_rows():
    tail = _results()["bases"]["net"]["tail"]
    assert {r["optimised"] for r in tail if r["method"] == MONTE_CARLO} == {NOT_RUN}

    filled = tail_with_simulation(tail, _simulation())

    monte_carlo = [r for r in filled if r["method"] == MONTE_CARLO]
    assert len(monte_carlo) == len([r for r in tail if r["method"] == MONTE_CARLO])
    assert NOT_RUN not in {r["optimised"] for r in monte_carlo}
    assert [r for r in filled if r["method"] == HISTORICAL] == [
        r for r in tail if r["method"] == HISTORICAL
    ]


def test_without_a_simulation_the_saved_rows_stand():
    tail = _results()["bases"]["net"]["tail"]

    assert tail_with_simulation(tail, None) == tail
