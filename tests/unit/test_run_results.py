"""A portfolio run's saved results: JSON-safe, and the page's own rows on both bases."""

import json

import numpy as np
import pandas as pd
import pytest

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.portfolio.backtest import run_backtest
from forecasting_engine.portfolio.optimize import SolveInputs, WeightBreakdown
from forecasting_engine.reporting.portfolio_comparison import (
    NOT_RUN,
    comparison_rows,
    significance_summary,
    tail_rows,
)
from forecasting_engine.reporting.run_results import (
    market_conditions,
    model_summary,
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
    dates = INDEX[::20]
    weights = np.where(np.arange(len(dates)) % 2 == 0, 0.8, 0.2)
    swinging = pd.DataFrame({EQUITY: weights, BOND: 1 - weights}, index=dates)
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


def test_the_latest_rebalances_market_conditions_are_annualised_from_its_covariance():
    # 20-day variances: equity 0.002, bond 0.0002, covariance 0.0001.
    roles = [EQUITY, BOND]
    covariance = pd.DataFrame([[0.002, 0.0001], [0.0001, 0.0002]], index=roles, columns=roles)
    inputs = SolveInputs(
        date=pd.Timestamp("2026-10-30"),
        signal=pd.Series({EQUITY: 0.0, BOND: 0.0}),
        covariance=covariance,
        premium=0.01,
        trust=0.0,
        risk_level=3,
        bounds=(0.2, 0.8),
    )
    breakdown = WeightBreakdown(level_weight=0.5, forecast_tilt=0.0, equity=0.5)

    conditions = market_conditions(breakdown, inputs, horizon=20, gate_shut=True)

    per_year = 252 / 20
    assert conditions["volatility"]["equity"] == pytest.approx((0.002 * per_year) ** 0.5)
    assert conditions["volatility"]["bond"] == pytest.approx((0.0002 * per_year) ** 0.5)
    variance = 0.25 * 0.002 + 0.25 * 0.0002 + 2 * 0.25 * 0.0001
    assert conditions["target_volatility"] == pytest.approx((variance * per_year) ** 0.5)
    assert conditions["gate_shut"] is True and conditions["trust"] == 0.0
    assert conditions["level_weight"] == 0.5 and conditions["capped"] is False
    json.dumps(conditions)  # saved as JSON with the run


def test_a_bond_derived_function_reads_its_built_inputs_and_names_its_fit():
    from datetime import datetime

    from forecasting_engine.ingest.align import Transform
    from forecasting_engine.models.base import ModelDescription
    from forecasting_engine.reporting.factor_labels import labeller
    from forecasting_engine.reporting.polynomial_function import Origin
    from forecasting_engine.store.active_model import ActiveFunction, ActiveModelRecord

    rate = "USGG10YR_Index_PX_LAST"
    level, longer = f"{rate}_LEVEL", f"{rate}_W2520"
    description = ModelDescription(
        "BondDerivedPolynomial",
        (level, f"{longer}^2"),
        (0.002, -0.001),
        0.001,
        {level: (2.0, 6.0), longer: (-0.2, 0.2)},
        {level: (4.0, 0.5), longer: (0.0, 0.05)},
    )
    function = ActiveFunction(
        Origin.DERIVED, description, {level: Transform.LEVEL, longer: Transform.DIFFERENCE}
    )
    record = ActiveModelRecord(
        BOND, "Polynomial (derived)", 0.0, 0.0, 0.0, False, datetime(2026, 11, 18), 20, 504, 20, ""
    )

    summary = model_summary(
        "US Aggregate Bond",
        record,
        display_name="Polynomial (derived)",
        signal_t=1.0,
        function=function,
        attribution=None,
        label=labeller([rate]),
        target="LBUSTRUU_Index_TOT_RETURN_INDEX_GROSS_DVDS",
        columns=[rate],
    )

    assert summary["fit"] == "BondDerivedPolynomial"
    terms = [row for row in summary["terms"] if row["Factor"] != "(intercept)"]
    assert [row["Factor"] for row in terms] == [
        "z(US 10y Treasury yield (level))",
        "z(US 10y Treasury yield (2,520-day fit))",
    ]
    assert [row["Exponent"] for row in terms] == ["1", "2"]
