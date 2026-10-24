"""Monte Carlo VaR, CVaR and drawdown, checked against cases with known answers."""

import math

import numpy as np
import pandas as pd
import pytest

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.portfolio.performance import max_drawdown
from forecasting_engine.risk.monte_carlo import (
    BOOTSTRAP,
    DEFAULT_PATHS,
    MAX_PATHS,
    MIN_PATHS,
    MONTE_CARLO,
    PARAMETRIC,
    SAMPLING_METHODS,
    MonteCarloDataError,
    history_window,
    path_drawdowns,
    simulate,
)

EQUITY, BOND = TargetRole.EQUITY, TargetRole.BOND
ALL_EQUITY = {"equity": pd.Series({EQUITY: 1.0, BOND: 0.0})}


def _history(equity, bond=None) -> pd.DataFrame:
    equity = np.asarray(equity, dtype=float)
    bond = np.zeros_like(equity) if bond is None else np.asarray(bond, dtype=float)
    return pd.DataFrame(
        {EQUITY: equity, BOND: bond}, index=pd.bdate_range("2024-01-01", periods=len(equity))
    )


def _fat_tailed(n=504, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    moves = rng.standard_t(4, size=(n, 2)) * [0.011, 0.003]
    return _history(moves[:, 0], moves[:, 1])


FORECASTS = pd.Series({EQUITY: 0.015, BOND: 0.004})
PORTFOLIOS = {
    "optimised": pd.Series({EQUITY: 0.62, BOND: 0.38}),
    "baseline": pd.Series({EQUITY: 0.5, BOND: 0.5}),
}


def _run(history=None, forecasts=FORECASTS, portfolios=None, **kwargs):
    kwargs.setdefault("horizon", 20)
    kwargs.setdefault("paths", 20_000)
    return simulate(
        _fat_tailed() if history is None else history,
        forecasts,
        PORTFOLIOS if portfolios is None else portfolios,
        **kwargs,
    )


# --- the centre: the paths follow the forecasts --------------------------------


@pytest.mark.parametrize("method", SAMPLING_METHODS)
def test_with_no_spread_every_path_earns_exactly_the_forecast_compounded(method):
    # Every historical day the same, so the demeaned moves are all zero.
    flat = _history([0.01] * 50, [0.002] * 50)
    result = _run(flat, method=method, horizon=10, paths=MIN_PATHS)

    for name, weights in PORTFOLIOS.items():
        expected = float(((1 + FORECASTS / 10) ** 10 - 1) @ weights)
        risk = result.risk[name]
        assert risk.mean_return == pytest.approx(expected)
        for level in risk.levels:
            assert level.var == pytest.approx(-expected)
            assert level.cvar == pytest.approx(-expected)
            assert level.max_drawdown == 0  # every day a gain: never below the start


def test_a_negative_forecast_with_no_spread_draws_down_by_the_whole_path():
    flat = _history([0.0] * 50)
    result = _run(flat, pd.Series({EQUITY: -0.05, BOND: 0.0}), ALL_EQUITY, horizon=5)

    (level, *_) = result.risk["equity"].levels
    assert level.max_drawdown == pytest.approx((1 - 0.01) ** 5 - 1)


@pytest.mark.parametrize("method", SAMPLING_METHODS)
def test_the_mean_simulated_return_is_the_portfolios_forecast(method):
    result = _run(method=method, paths=100_000)

    for name, weights in PORTFOLIOS.items():
        forecast = float(FORECASTS @ weights)
        # Compounding adds about half the horizon's variance; the rest is sampling.
        assert result.risk[name].mean_return == pytest.approx(forecast, abs=0.001)


def test_the_historys_own_drift_is_removed_so_only_the_forecast_sets_direction():
    trending = _history(np.full(100, 0.004) + np.tile([0.01, -0.01], 50))
    result = _run(trending, pd.Series({EQUITY: 0.0, BOND: 0.0}), ALL_EQUITY, horizon=1)

    assert result.risk["equity"].mean_return == pytest.approx(0.0, abs=5e-4)


# --- the spread: what each sampling method draws -------------------------------


def test_bootstrap_only_draws_days_that_happened():
    days = np.array([-0.03, -0.01, 0.0, 0.02, 0.04])
    result = _run(_history(days), pd.Series({EQUITY: 0.0, BOND: 0.0}), ALL_EQUITY, horizon=1)

    possible = days - days.mean()
    for level in result.risk["equity"].levels:
        assert np.isclose(-level.var, possible).any()


def test_parametric_var_over_one_day_is_the_normal_quantile():
    history = _fat_tailed()
    result = _run(history, FORECASTS, ALL_EQUITY, horizon=1, method=PARAMETRIC, paths=400_000)

    sd = history[EQUITY].std(ddof=1)
    mean = FORECASTS[EQUITY]
    var95, var99 = (level.var for level in result.risk["equity"].levels)
    assert var95 == pytest.approx(1.6449 * sd - mean, rel=0.02)
    assert var99 == pytest.approx(2.3263 * sd - mean, rel=0.02)


def test_bootstrap_keeps_the_fat_tails_a_fitted_normal_thins():
    bootstrap = _run(method=BOOTSTRAP, horizon=1, paths=200_000)
    parametric = _run(method=PARAMETRIC, horizon=1, paths=200_000)

    at_99 = [r.risk["optimised"].levels[-1] for r in (bootstrap, parametric)]
    assert at_99[0].cvar > at_99[1].cvar


def test_parametric_survives_an_index_that_never_moved():
    still = _history(np.random.default_rng(0).normal(0, 0.01, 60), np.zeros(60))

    result = _run(still, method=PARAMETRIC, paths=MIN_PATHS)

    assert all(math.isfinite(level.var) for level in result.risk["optimised"].levels)


@pytest.mark.parametrize("method", SAMPLING_METHODS)
def test_equity_and_bond_are_drawn_together_keeping_their_correlation(method):
    moves = np.random.default_rng(3).normal(0, 0.01, 300)
    hedged = _history(moves, -moves)  # bond moves exactly against equity
    half = {"half": pd.Series({EQUITY: 0.5, BOND: 0.5})}
    zero = pd.Series({EQUITY: 0.0, BOND: 0.0})

    (level, *_) = _run(hedged, zero, half, horizon=1, method=method).risk["half"].levels

    assert level.var == pytest.approx(0.0, abs=1e-12)


# --- the tail ------------------------------------------------------------------


def test_cvar_is_never_below_var_and_99_is_never_milder_than_95():
    for risk in _run().risk.values():
        at_95, at_99 = risk.levels
        for level in risk.levels:
            assert level.cvar >= level.var
        assert at_99.var >= at_95.var
        assert at_99.max_drawdown <= at_95.max_drawdown


def test_every_portfolio_is_simulated_on_the_same_paths():
    twins = {"a": PORTFOLIOS["optimised"], "b": PORTFOLIOS["optimised"]}

    result = _run(portfolios=twins)

    assert result.risk["a"] == result.risk["b"]


def test_a_portfolio_holding_more_of_the_calmer_index_has_less_tail_risk():
    result = _run(paths=50_000)

    riskier, calmer = (result.risk[name].levels[0] for name in ("optimised", "baseline"))
    assert calmer.var < riskier.var


@pytest.mark.parametrize("method", SAMPLING_METHODS)
def test_chunking_never_changes_the_result(method):
    whole = _run(method=method, paths=10_000, chunk_paths=10_000)
    chunked = _run(method=method, paths=10_000, chunk_paths=777)

    assert whole.risk == chunked.risk


def test_the_same_seed_gives_the_same_figures_and_another_seed_others():
    assert _run(seed=7).risk == _run(seed=7).risk
    assert _run(seed=7).risk != _run(seed=8).risk


def test_path_drawdowns_agree_with_the_backtests_max_drawdown():
    rng = np.random.default_rng(5)
    daily = rng.normal(0, 0.02, size=(50, 30))
    values = np.cumprod(1 + daily, axis=1)

    expected = [max_drawdown(pd.Series(row)) for row in daily]
    assert path_drawdowns(values) == pytest.approx(expected)


def test_the_result_says_how_it_was_made():
    history = _fat_tailed()
    result = _run(history, method=PARAMETRIC, paths=MIN_PATHS, horizon=5)

    assert result.label == MONTE_CARLO
    assert (result.method, result.paths, result.horizon) == (PARAMETRIC, MIN_PATHS, 5)
    assert result.as_of == history.index[-1]
    assert result.history_start == history.index[0]
    assert result.history_days == len(history)
    assert result.forecasts.equals(FORECASTS)


# --- refused inputs --------------------------------------------------------------


def test_the_default_is_a_hundred_thousand_paths_within_the_allowed_range():
    assert MIN_PATHS <= DEFAULT_PATHS == 100_000 <= MAX_PATHS


@pytest.mark.parametrize("paths", [MIN_PATHS - 1, MAX_PATHS + 1])
def test_a_path_count_outside_the_range_is_refused(paths):
    with pytest.raises(MonteCarloDataError, match="number of paths"):
        _run(paths=paths)


def test_a_missing_forecast_is_refused_naming_the_asset():
    with pytest.raises(MonteCarloDataError, match="bond"):
        _run(forecasts=pd.Series({EQUITY: 0.01}))


def test_weights_that_do_not_sum_to_one_are_refused():
    with pytest.raises(ValueError, match="sum to 1"):
        _run(portfolios={"bad": pd.Series({EQUITY: 0.7, BOND: 0.7})})


def test_an_unknown_sampling_method_is_refused():
    with pytest.raises(ValueError, match="method"):
        _run(method="garch")


# --- the history window ------------------------------------------------------------


def _prices() -> dict[TargetRole, pd.Series]:
    dates = pd.bdate_range("2024-01-01", periods=40)
    equity = pd.Series(100 * 1.01 ** np.arange(40), index=dates)
    bond = pd.Series(100 * 1.002 ** np.arange(40), index=dates).drop(dates[[5, 6]])
    return {EQUITY: equity, BOND: bond}


def test_history_is_the_last_window_returns_up_to_and_including_the_as_of_day():
    prices = _prices()
    as_of = prices[EQUITY].index[30]

    history = history_window(prices, as_of, window=10)

    assert len(history) == 10
    assert history.index[-1] == as_of
    assert list(history.columns) == [EQUITY, BOND]
    assert history[EQUITY].to_numpy() == pytest.approx(0.01)


def test_history_uses_only_days_both_indices_traded():
    prices = _prices()
    as_of = prices[EQUITY].index[-1]

    history = history_window(prices, as_of, window=37)

    assert not history.index.isin(prices[EQUITY].index[[5, 6]]).any()
    # The return over a missing day spans both days, so it is still one return.
    assert history.notna().all().all()


def test_too_short_a_history_is_refused_with_how_much_there_is():
    prices = _prices()

    with pytest.raises(MonteCarloDataError, match="only 7 are available"):
        history_window(prices, prices[EQUITY].index[9], window=20)
