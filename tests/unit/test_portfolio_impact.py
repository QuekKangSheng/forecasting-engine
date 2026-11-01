"""The shocked portfolio's return, volatility and drawdown against the base case
(FYP-33), checked on simulations with known answers."""

import math

import numpy as np
import pandas as pd
import pytest

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.risk import monte_carlo
from forecasting_engine.scenario.portfolio_impact import (
    PortfolioImpactError,
    portfolio_impact,
)

EQUITY, BOND = TargetRole.EQUITY, TargetRole.BOND
HORIZON = 20
WEIGHTS = {EQUITY: 0.62, BOND: 0.38}
BASE = {EQUITY: 0.015, BOND: 0.004}


def _history(n: int = 504, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    moves = rng.standard_t(4, size=(n, 2)) * [0.011, 0.003]
    return pd.DataFrame(
        {EQUITY: moves[:, 0], BOND: moves[:, 1]},
        index=pd.bdate_range("2024-01-01", periods=n),
    )


def _impact(shocked, *, weights=WEIGHTS, paths=20_000, **kwargs):
    return portfolio_impact(
        _history(), BASE, shocked, weights, horizon=HORIZON, paths=paths, **kwargs
    )


# --- a zero shock is the base case (FYP-248) ----------------------------------------


def test_a_zero_shock_gives_exactly_the_base_case_metrics():
    impact = _impact(BASE)

    assert impact.shocked == impact.base
    assert impact.return_change == 0.0
    assert impact.volatility_change == 0.0
    assert all(impact.drawdown_change(c) == 0.0 for c in impact.base.max_drawdown)


def test_the_weights_are_held_not_re_optimised():
    shocked = {EQUITY: -0.10, BOND: 0.004}  # a shock big enough to tempt a re-solve
    impact = _impact(shocked)

    assert impact.weights.to_dict() == pytest.approx(WEIGHTS)


# --- the shock reaches the portfolio --------------------------------------------------


def test_a_lower_equity_forecast_lowers_the_portfolios_return_by_its_weight():
    shocked = {EQUITY: BASE[EQUITY] - 0.05, BOND: BASE[BOND]}
    impact = _impact(shocked)

    # Each day's drift falls by 0.05 / h on equity alone; compounded over the
    # horizon with drifting weights the portfolio loses close to 0.62 × 5%.
    assert impact.return_change == pytest.approx(-0.05 * WEIGHTS[EQUITY], abs=0.004)


def test_a_lower_forecast_deepens_the_simulated_drawdown():
    shocked = {EQUITY: -0.10, BOND: 0.004}
    impact = _impact(shocked)

    for confidence in impact.base.max_drawdown:
        assert impact.drawdown_change(confidence) < 0


def test_both_cases_share_the_same_paths_so_only_the_forecast_differs():
    """With no spread in the history every path is identical, so the shocked
    return is the base return shifted by exactly the shock's compounded drift."""
    flat = pd.DataFrame(
        {EQUITY: np.zeros(30), BOND: np.zeros(30)},
        index=pd.bdate_range("2024-01-01", periods=30),
    )
    all_equity = {EQUITY: 1.0, BOND: 0.0}
    impact = portfolio_impact(
        flat, BASE, {EQUITY: -0.02, BOND: BASE[BOND]}, all_equity, horizon=HORIZON, paths=1_000
    )

    assert impact.base.mean_return == pytest.approx((1 + BASE[EQUITY] / HORIZON) ** HORIZON - 1)
    assert impact.shocked.mean_return == pytest.approx((1 - 0.02 / HORIZON) ** HORIZON - 1)
    assert impact.base.volatility == pytest.approx(0.0, abs=1e-12)


def test_volatility_is_the_spread_of_the_simulated_outcomes():
    impact = _impact(BASE)

    assert impact.base.volatility > 0
    # A shift in the forecast moves every path by the same amount: the spread
    # around the mean is unchanged to within sampling noise from compounding.
    shifted = _impact({EQUITY: BASE[EQUITY] - 0.03, BOND: BASE[BOND]})
    assert shifted.volatility_change == pytest.approx(0.0, abs=0.002)


def test_the_result_records_what_it_was_made_from():
    shocked = {EQUITY: 0.0, BOND: 0.004}
    impact = _impact(shocked, method=monte_carlo.PARAMETRIC, paths=2_000)

    assert impact.horizon == HORIZON
    assert impact.method == monte_carlo.PARAMETRIC
    assert impact.paths == 2_000
    assert impact.as_of == _history().index[-1]
    assert impact.forecasts.loc[EQUITY, "base"] == BASE[EQUITY]
    assert impact.forecasts.loc[EQUITY, "shocked"] == 0.0
    assert impact.forecasts.loc[BOND, "shocked"] == BASE[BOND]


# --- refusals --------------------------------------------------------------------------


def test_a_missing_shocked_forecast_is_refused_naming_the_index():
    with pytest.raises(PortfolioImpactError, match="shocked forecast for the bond index"):
        _impact({EQUITY: 0.01})


def test_a_missing_base_forecast_is_refused_naming_the_index():
    with pytest.raises(PortfolioImpactError, match="base-case forecast for the equity index"):
        portfolio_impact(_history(), {BOND: 0.004}, BASE, WEIGHTS, horizon=HORIZON, paths=1_000)


def test_weights_that_do_not_sum_to_one_are_refused():
    with pytest.raises(ValueError, match="sum to 1"):
        _impact(BASE, weights={EQUITY: 0.7, BOND: 0.7}, paths=1_000)


def test_volatility_is_reported_by_the_simulation_itself():
    result = monte_carlo.simulate(
        _history(), pd.Series(BASE), {"p": pd.Series(WEIGHTS)}, horizon=HORIZON, paths=5_000
    )
    outcomes_sd = result.risk["p"].volatility

    assert math.isfinite(outcomes_sd) and outcomes_sd > 0
