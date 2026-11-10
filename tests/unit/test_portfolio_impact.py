"""The shocked portfolio's return, volatility and drawdown against the base case
(FYP-33), checked on simulations with known answers."""

import math

import numpy as np
import pandas as pd
import pytest

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.portfolio.optimize import SolveInputs
from forecasting_engine.risk import monte_carlo
from forecasting_engine.scenario.portfolio_impact import (
    PortfolioImpactError,
    portfolio_impact,
    shocked_signal,
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


# --- re-optimising under the shock (FYP-246) -----------------------------------------


def _solve_inputs(trust: float = 1.0, risk_level: int = 3) -> SolveInputs:
    """A rebalance whose forecasts are fully trusted, sized against a covariance where
    equity is the riskier index."""
    roles = [EQUITY, BOND]
    return SolveInputs(
        date=pd.Timestamp("2025-12-31"),
        signal=pd.Series({EQUITY: 0.002, BOND: 0.0}),
        covariance=pd.DataFrame([[0.0016, 0.0001], [0.0001, 0.0004]], index=roles, columns=roles),
        premium=0.01,
        trust=trust,
        risk_level=risk_level,
        bounds=(0.0, 1.0),
    )


def test_nothing_is_re_optimised_unless_asked():
    assert _impact({EQUITY: -0.05, BOND: 0.004}).reoptimised is None


def test_asking_to_re_optimise_leaves_the_held_weights_results_as_they_were():
    shocked = {EQUITY: -0.05, BOND: 0.004}

    held = _impact(shocked, paths=2_000)
    both = _impact(shocked, paths=2_000, reoptimise=_solve_inputs())

    assert both.base == held.base
    assert both.shocked == held.shocked
    assert both.weights.equals(held.weights)


def test_the_shock_moves_each_signal_by_as_much_as_it_moved_the_forecast():
    forecasts = pd.DataFrame(
        {"base": [0.015, 0.004], "shocked": [0.005, 0.004]}, index=[EQUITY, BOND]
    )

    signal = shocked_signal(_solve_inputs(), forecasts)

    assert signal[EQUITY] == pytest.approx(0.002 - 0.010)
    assert signal[BOND] == 0.0


def test_a_zero_shock_re_solves_to_the_rebalance_target():
    reoptimised = _impact(BASE, paths=2_000, reoptimise=_solve_inputs()).reoptimised

    assert reoptimised.breakdown == reoptimised.unshocked
    assert reoptimised.weights[EQUITY] == pytest.approx(reoptimised.unshocked.equity)
    assert reoptimised.weights.sum() == pytest.approx(1.0)


def test_a_lower_equity_forecast_re_solves_to_less_equity():
    reoptimised = _impact(
        {EQUITY: 0.005, BOND: 0.004}, paths=2_000, reoptimise=_solve_inputs()
    ).reoptimised

    assert reoptimised.weights[EQUITY] < reoptimised.unshocked.equity


def test_untrusted_forecasts_or_level_5_leave_the_re_solved_weights_where_they_were():
    shocked = {EQUITY: 0.005, BOND: 0.004}
    for inputs in (_solve_inputs(trust=0.0), _solve_inputs(risk_level=5)):
        reoptimised = _impact(shocked, paths=2_000, reoptimise=inputs).reoptimised

        assert reoptimised.weights[EQUITY] == pytest.approx(reoptimised.unshocked.equity)


def test_re_optimised_weights_are_simulated_on_the_held_weights_paths():
    inputs = _solve_inputs()
    target = inputs.solve().equity
    shocked = {EQUITY: 0.005, BOND: 0.004}
    # Holding the weights a zero-shock re-solve gives makes the two portfolios one.
    held_at_target = {EQUITY: target, BOND: 1 - target}
    zero = _impact(BASE, weights=held_at_target, paths=2_000, reoptimise=inputs)
    assert zero.reoptimised.metrics == zero.shocked

    moved = _impact(shocked, weights=held_at_target, paths=2_000, reoptimise=inputs)
    assert moved.reoptimised.metrics != moved.shocked
