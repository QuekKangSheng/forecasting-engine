"""Turning each target's saved signal into rebalance dates, expected signals, the
covariance each rebalance is sized against, and a tilt from the risk level's
starting weight."""

import numpy as np
import pandas as pd
import pytest

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.portfolio.backtest import run_backtest
from forecasting_engine.portfolio.optimize import (
    DEFAULT_RISK_LEVEL,
    DEFAULT_WEIGHT_BOUNDS,
    RISK_LEVELS,
    TILT_STRENGTH,
    OptimizeDataError,
    bounds_for_level,
    common_rebalance_dates,
    covariance_at_rebalance,
    expected_returns,
    schedule_breakdowns,
    solve_weights,
    starting_weight_for,
    weight_breakdown,
    weight_schedule,
)

EQUITY, BOND = TargetRole.EQUITY, TargetRole.BOND


def _forecast(dates, values=None) -> pd.Series:
    dates = pd.DatetimeIndex(dates)
    return pd.Series(values if values is not None else np.arange(len(dates), dtype=float), dates)


def _prices(dates, seed) -> dict:
    rng = np.random.default_rng(seed)
    equity = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, len(dates))), index=dates)
    bond = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, len(dates))), index=dates)
    return {EQUITY: equity, BOND: bond}


def _forecasts(equity, bond) -> dict:
    return {EQUITY: equity, BOND: bond}


@pytest.fixture(autouse=True)
def no_tuning_period(monkeypatch):
    """Rebalance dates are built with the harness's own TUNING_ROWS (504) in
    the loop, same as a real model run — too large for these small fixtures."""
    monkeypatch.setattr("forecasting_engine.portfolio.optimize.TUNING_ROWS", 0)


def test_a_one_day_horizon_rebalances_on_every_out_of_sample_day():
    dates = pd.bdate_range("2024-01-02", periods=10)
    prices = _prices(dates, seed=0)

    rebalances = common_rebalance_dates(prices, horizon=1, train_window=2, test_window=2, embargo=0)

    # tuning_rows=0, train=2, embargo=0 -> test windows cover positions 2 to 9.
    assert list(rebalances) == list(dates[2:10])


def test_a_rebalance_every_horizon_days_holds_each_forecast_for_its_own_horizon():
    dates = pd.bdate_range("2024-01-02", periods=10)
    prices = _prices(dates, seed=0)

    rebalances = common_rebalance_dates(prices, horizon=3, train_window=2, test_window=2, embargo=0)

    # Out-of-sample positions 2 to 9, every third: 2, 5, 8 — across test windows.
    assert list(rebalances) == [dates[2], dates[5], dates[8]]


def test_a_price_gap_in_one_asset_shifts_the_whole_shared_calendar():
    """Building one shared calendar first (rather than splitting equity's and
    bond's own calendars independently) means a gap on one side removes a day
    from both targets' rebalance schedule together, instead of knocking the
    two out of phase with each other for every later rebalance."""
    dates = pd.bdate_range("2024-01-02", periods=10)
    prices = _prices(dates, seed=1)
    with_gap = prices[BOND].copy()
    with_gap.iloc[3] = float("nan")  # bond has no price on dates[3]
    prices_with_gap = {**prices, BOND: with_gap}

    rebalances = common_rebalance_dates(
        prices_with_gap, horizon=2, train_window=2, test_window=2, embargo=0
    )

    # The shared calendar drops dates[3], so every second shared date from
    # position 2 is dates[2], dates[5], dates[7] — every later rebalance shifts
    # one row together for both targets, not out of sync with each other.
    assert list(rebalances) == [dates[2], dates[5], dates[7]]


def test_rebalances_are_sorted_ascending():
    dates = pd.bdate_range("2024-01-02", periods=8)
    prices = _prices(dates, seed=2)

    rebalances = common_rebalance_dates(prices, horizon=1, train_window=2, test_window=1, embargo=0)

    assert list(rebalances) == sorted(rebalances)


def test_with_no_room_for_a_full_test_window_there_are_no_rebalances():
    dates = pd.bdate_range("2024-01-02", periods=3)
    prices = _prices(dates, seed=3)

    rebalances = common_rebalance_dates(prices, horizon=1, train_window=2, test_window=2, embargo=0)

    assert rebalances.empty


# --- expected_returns -----------------------------------------------------------


def test_expected_returns_reads_each_targets_forecast_on_each_rebalance_date():
    dates = pd.bdate_range("2024-01-02", periods=4)
    equity = _forecast(dates, [0.01, 0.02, 0.03, 0.04])
    bond = _forecast(dates, [0.10, 0.20, 0.30, 0.40])
    rebalances = pd.DatetimeIndex([dates[0], dates[2]])

    returns = expected_returns(rebalances, _forecasts(equity, bond))

    assert list(returns.index) == [dates[0], dates[2]]
    assert list(returns[EQUITY]) == [0.01, 0.03]
    assert list(returns[BOND]) == [0.10, 0.30]


def test_expected_returns_columns_are_the_two_target_roles():
    dates = pd.bdate_range("2024-01-02", periods=2)
    forecasts = _forecasts(_forecast(dates), _forecast(dates))

    returns = expected_returns(dates, forecasts)

    assert list(returns.columns) == [EQUITY, BOND]


def test_expected_returns_uses_the_most_recent_forecast_when_the_exact_date_is_missing():
    """Equity's and bond's own saved forecasts can fall on slightly different
    calendars; a rebalance date absent from one target's own index still gets
    that target's most recent known view as of that date."""
    dates = pd.bdate_range("2024-01-02", periods=5)
    sparse = _forecast([dates[0], dates[2], dates[4]], [0.01, 0.03, 0.05])
    rebalances = pd.DatetimeIndex([dates[1], dates[3]])

    returns = expected_returns(rebalances, _forecasts(sparse, sparse))

    assert list(returns[EQUITY]) == [0.01, 0.03]


def test_expected_returns_is_nan_before_a_targets_first_forecast():
    dates = pd.bdate_range("2024-01-02", periods=5)
    sparse = _forecast([dates[2], dates[4]], [0.03, 0.05])

    returns = expected_returns(pd.DatetimeIndex([dates[0]]), _forecasts(sparse, sparse))

    assert returns[EQUITY].isna().all()


def test_expected_returns_keeps_an_explicit_nan_as_is():
    """A NaN stored at a date that IS in the forecast is the model's own
    answer for that day (no confident prediction) — reindex fills gaps
    between known dates, not over one that already has a value."""
    dates = pd.bdate_range("2024-01-02", periods=3)
    forecast = _forecast(dates, [0.01, float("nan"), 0.03])

    returns = expected_returns(dates, _forecasts(forecast, forecast))

    assert pd.isna(returns[EQUITY].iloc[1])  # the model's own NaN, not 0.01


# --- covariance_at_rebalance -----------------------------------------------------


def test_covariance_uses_exactly_the_purged_training_window():
    dates = pd.bdate_range("2024-01-02", periods=20)
    prices = _prices(dates, seed=0)
    rebalance_date = dates[10]

    cov = covariance_at_rebalance(rebalance_date, prices, horizon=3, train_window=5, embargo=2)

    # train=5, embargo=2, horizon=3 at position 10: natural_end=8, start=3,
    # purge_boundary=min(8, 10-3)=7 -> dates[3:7].
    window = dates[3:7]
    expected = pd.DataFrame(
        {role: prices[role].reindex(window).pct_change() for role in (EQUITY, BOND)}
    ).dropna()
    pd.testing.assert_frame_equal(cov, expected.cov() * 3)


def test_covariance_never_uses_data_on_or_after_the_embargo_gap():
    dates = pd.bdate_range("2024-01-02", periods=20)
    prices = _prices(dates, seed=1)
    rebalance_date = dates[10]
    kwargs = {"horizon": 1, "train_window": 5, "embargo": 2}

    baseline = covariance_at_rebalance(rebalance_date, prices, **kwargs)

    altered = dict(prices)
    blown_up = prices[EQUITY].copy()
    blown_up.iloc[9:] *= 100  # every price from the embargo gap onward
    altered[EQUITY] = blown_up
    changed = covariance_at_rebalance(rebalance_date, altered, **kwargs)

    pd.testing.assert_frame_equal(baseline, changed)


def test_too_few_training_days_raises():
    dates = pd.bdate_range("2024-01-02", periods=20)
    prices = _prices(dates, seed=2)

    with pytest.raises(OptimizeDataError, match="Not enough trading days"):
        covariance_at_rebalance(dates[2], prices, horizon=1, train_window=5, embargo=2)


def test_a_rebalance_date_off_the_calendar_raises():
    dates = pd.bdate_range("2024-01-02", periods=20)
    prices = _prices(dates, seed=3)
    saturday = dates[0] + pd.Timedelta(days=4)

    with pytest.raises(OptimizeDataError, match="not a trading day"):
        covariance_at_rebalance(saturday, prices, horizon=1, train_window=5, embargo=2)


def test_only_dates_both_targets_have_a_price_on_are_used():
    dates = pd.bdate_range("2024-01-02", periods=20)
    prices = _prices(dates, seed=4)
    with_gap = prices[BOND].copy()
    with_gap.iloc[5] = float("nan")  # a bond holiday inside what would be the window
    prices_with_gap = {**prices, BOND: with_gap}

    cov = covariance_at_rebalance(dates[10], prices_with_gap, horizon=1, train_window=5, embargo=2)

    assert not cov.isna().any().any()


# --- solve_weights ----------------------------------------------------------------


def _covariance(var_equity, var_bond, cov_eb) -> pd.DataFrame:
    return pd.DataFrame(
        {EQUITY: {EQUITY: var_equity, BOND: cov_eb}, BOND: {EQUITY: cov_eb, BOND: var_bond}}
    )


def test_solve_weights_matches_the_analytic_equal_variance_formula():
    """For equal-variance, uncorrelated assets the closed form reduces to
    w_equity = 1/2 + (mu_equity - mu_bond) / (2 * tilt_strength * variance) —
    the two-asset tangency solution FYP-202 checks against."""
    mu = pd.Series({EQUITY: 0.03, BOND: 0.01})
    variance = 0.04
    covariance = _covariance(variance, variance, 0.0)

    weights = solve_weights(mu, covariance, tilt_strength=4.0, bounds=(0.0, 1.0))

    expected_equity = 0.5 + (mu[EQUITY] - mu[BOND]) / (2 * 4.0 * variance)
    assert weights[EQUITY] == pytest.approx(expected_equity)
    assert weights[BOND] == pytest.approx(1 - expected_equity)


def test_the_breakdown_is_the_starting_weight_plus_the_signals_tilt():
    covariance = _covariance(0.0005, 0.00002, 0.000001)
    equal = pd.Series({EQUITY: 0.001, BOND: 0.001})
    apart = pd.Series({EQUITY: 0.003, BOND: 0.001})

    flat = weight_breakdown(equal, covariance, tilt_strength=4.0, bounds=(0.0, 1.0))
    tilted = weight_breakdown(apart, covariance, tilt_strength=4.0, bounds=(0.0, 1.0))

    # Agreeing signals leave the starting weight (50/50 by default), however
    # much calmer bond is.
    assert flat.forecast_tilt == 0.0
    assert flat.anchor == 0.5 and flat.equity == 0.5
    var_diff = 0.0005 - 2 * 0.000001 + 0.00002
    assert tilted.forecast_tilt == pytest.approx(0.002 / (4.0 * var_diff))
    assert tilted.unconstrained == pytest.approx(0.5 + tilted.forecast_tilt)


# --- starting weight by risk level, fixed tilt strength --------------------------


def test_no_signal_means_the_starting_weight_at_every_risk_level():
    covariance = _covariance(0.0005, 0.00002, 0.000001)
    none = pd.Series({EQUITY: 0.0, BOND: 0.0})

    for level in RISK_LEVELS:
        start = starting_weight_for(level)
        weights = solve_weights(none, covariance, starting_weight=start, bounds=(0.0, 1.0))
        assert weights[EQUITY] == start and weights[BOND] == pytest.approx(1 - start)


def test_the_risk_scale_runs_from_1_risk_loving_to_5_risk_averse():
    assert list(RISK_LEVELS) == [1, 2, 3, 4, 5]
    starts = [starting_weight_for(level) for level in RISK_LEVELS]
    assert starts == sorted(starts, reverse=True) and len(set(starts)) == 5
    assert DEFAULT_RISK_LEVEL == 3
    assert starting_weight_for(DEFAULT_RISK_LEVEL) == 0.5


def test_a_higher_level_never_holds_more_equity_for_the_same_signal():
    """A more risk-averse level never ends up with more equity than a less
    risk-averse one, for the same signal and covariance."""
    covariance = _covariance(0.0005, 0.00002, 0.000001)
    signal = pd.Series({EQUITY: 0.0005, BOND: 0.0})

    equities = [
        weight_breakdown(
            signal, covariance, starting_weight=starting_weight_for(level), bounds=(0.0, 1.0)
        ).equity
        for level in RISK_LEVELS
    ]
    assert equities == sorted(equities, reverse=True)


@pytest.mark.parametrize("level", [0, 6])
def test_a_risk_level_off_the_scale_is_refused(level):
    with pytest.raises(ValueError, match="1 to 5"):
        starting_weight_for(level)


# --- bounds_for_level: risk-loving never ends bond-heavy, risk-averse never equity-heavy ---


@pytest.mark.parametrize("level", [1, 2])
def test_risk_loving_levels_get_a_50_percent_floor(level):
    assert bounds_for_level(level, (0.2, 0.8)) == (0.5, 0.8)


@pytest.mark.parametrize("level", [4, 5])
def test_risk_averse_levels_get_a_50_percent_ceiling(level):
    assert bounds_for_level(level, (0.2, 0.8)) == (0.2, 0.5)


def test_level_3_keeps_the_bounds_as_given():
    assert bounds_for_level(3, (0.2, 0.8)) == (0.2, 0.8)


@pytest.mark.parametrize("level", [1, 2, 4, 5])
def test_an_extreme_signal_never_crosses_50_percent_the_wrong_way(level):
    """A signal strong enough to want to cross 50% the wrong way still stops
    at it."""
    covariance = _covariance(0.0005, 0.00002, 0.000001)
    adversarial = pd.Series({EQUITY: -10.0, BOND: 10.0}) if level < 3 else pd.Series(
        {EQUITY: 10.0, BOND: -10.0}
    )

    weights = solve_weights(
        adversarial,
        covariance,
        starting_weight=starting_weight_for(level),
        bounds=bounds_for_level(level, (0.2, 0.8)),
    )

    if level < 3:
        assert weights[EQUITY] >= 0.5
    else:
        assert weights[EQUITY] <= 0.5


def test_the_breakdown_says_when_a_bound_capped_the_weight():
    covariance = _covariance(0.0005, 0.00002, 0.000001)
    mu = pd.Series({EQUITY: 0.003, BOND: 0.001})

    capped = weight_breakdown(mu, covariance, tilt_strength=4.0, bounds=(0.2, 0.8))

    assert capped.unconstrained > 0.8
    assert capped.capped and capped.equity == 0.8
    assert solve_weights(mu, covariance, tilt_strength=4.0, bounds=(0.2, 0.8))[EQUITY] == 0.8


@pytest.mark.parametrize("level", list(RISK_LEVELS))
def test_bounds_hold_whatever_the_starting_weight_and_signal(level):
    """An extreme signal at any level's starting weight still clips to the
    configured bounds, never beyond them."""
    covariance = _covariance(0.01, 0.01, 0.0)
    extreme = pd.Series({EQUITY: 10.0, BOND: -10.0})

    weights = solve_weights(
        extreme, covariance, starting_weight=starting_weight_for(level), bounds=(0.2, 0.8)
    )

    assert weights[EQUITY] == pytest.approx(0.8)
    assert weights[BOND] == pytest.approx(0.2)


def test_solve_weights_sums_to_one():
    mu = pd.Series({EQUITY: 0.02, BOND: -0.01})
    covariance = _covariance(0.05, 0.02, 0.01)

    weights = solve_weights(mu, covariance)

    assert weights[EQUITY] + weights[BOND] == pytest.approx(1.0)


def test_solve_weights_is_clipped_to_bounds():
    mu = pd.Series({EQUITY: 10.0, BOND: -10.0})  # an extreme, unrealistic spread
    covariance = _covariance(0.01, 0.01, 0.0)

    weights = solve_weights(mu, covariance, tilt_strength=0.1, bounds=(0.2, 0.8))

    assert weights[EQUITY] == pytest.approx(0.8)
    assert weights[BOND] == pytest.approx(0.2)


def test_solve_weights_keeps_the_starting_weight_when_assets_are_indistinguishable():
    mu = pd.Series({EQUITY: 0.05, BOND: 0.01})
    identical = _covariance(0.04, 0.04, 0.04)  # perfectly correlated, equal variance

    weights = solve_weights(mu, identical, bounds=(0.2, 0.8))

    assert weights[EQUITY] == pytest.approx(0.5)


def test_higher_tilt_strength_tilts_less_far_from_the_starting_weight():
    mu = pd.Series({EQUITY: 0.05, BOND: 0.01})
    covariance = _covariance(0.04, 0.03, 0.01)

    cautious = solve_weights(mu, covariance, tilt_strength=20.0, bounds=(0.0, 1.0))
    bold = solve_weights(mu, covariance, tilt_strength=1.0, bounds=(0.0, 1.0))

    assert abs(cautious[EQUITY] - 0.5) < abs(bold[EQUITY] - 0.5)


# --- weight_schedule ----------------------------------------------------------------


def _schedule_inputs(seed=0):
    dates = pd.bdate_range("2024-01-02", periods=60)
    prices = _prices(dates, seed)
    rng = np.random.default_rng(seed)
    forecasts = {
        role: pd.Series(rng.normal(0, 0.02, len(dates)), index=dates) for role in (EQUITY, BOND)
    }
    return dates, prices, forecasts


# train_window=15, embargo=2, tuning_rows=0 (patched) -> test windows cover
# positions 17 to 56; a rebalance every horizon (3) days: 17, 20, ..., 56.
_SCHEDULE_KWARGS = {"horizon": 3, "train_window": 15, "test_window": 10, "embargo": 2}


def test_weight_schedule_has_one_row_per_rebalance_summing_to_one_within_bounds():
    dates, prices, forecasts = _schedule_inputs()

    schedule = weight_schedule(forecasts, prices, **_SCHEDULE_KWARGS)

    assert list(schedule.index) == list(dates[17:57:3])
    assert list(schedule.columns) == [EQUITY, BOND]
    assert schedule.sum(axis=1).sub(1.0).abs().max() < 1e-9
    assert schedule[EQUITY].between(*DEFAULT_WEIGHT_BOUNDS).all()


def test_weight_schedule_skips_a_rebalance_with_a_missing_forecast():
    dates, prices, forecasts = _schedule_inputs()
    first_rebalance = dates[17]
    equity = forecasts[EQUITY].copy()
    equity.loc[first_rebalance] = float("nan")
    forecasts = {**forecasts, EQUITY: equity}

    schedule = weight_schedule(forecasts, prices, **_SCHEDULE_KWARGS)

    assert first_rebalance not in schedule.index
    assert len(schedule) == len(dates[17:57:3]) - 1


# --- FYP-49: the no-trade band ---------------------------------------------------------


def _unbanded(forecasts, prices) -> pd.Series:
    """Each rebalance's bounded target on its own, as the optimiser set it before the band."""
    dates = common_rebalance_dates(prices, horizon=3, train_window=15, test_window=10, embargo=2)
    expected = expected_returns(dates, forecasts)
    return pd.Series(
        {
            date: weight_breakdown(
                expected.loc[date],
                covariance_at_rebalance(date, prices, horizon=3, train_window=15, embargo=2),
                tilt_strength=TILT_STRENGTH,
            ).equity
            for date in dates
        }
    )


def test_zero_costs_reproduce_the_unbanded_schedule_exactly():
    _dates, prices, forecasts = _schedule_inputs()
    free = {EQUITY: 0.0, BOND: 0.0}

    schedule = weight_schedule(forecasts, prices, costs_bps=free, **_SCHEDULE_KWARGS)

    pd.testing.assert_series_equal(
        schedule[EQUITY], _unbanded(forecasts, prices), check_names=False, check_freq=False
    )
    breakdowns = schedule_breakdowns(forecasts, prices, costs_bps=free, **_SCHEDULE_KWARGS)
    assert not any(b.held for b in breakdowns.values())


def _band_inputs():
    """No signal at the first rebalance, a tiny one at the second, a large one at the
    third: 50/50, then a change too small to pay for, then one that is."""
    dates, prices, _ = _schedule_inputs()
    equity = pd.Series(0.0, index=dates)
    equity.loc[dates[20] :] = 1e-6
    equity.loc[dates[23] :] = 0.05
    return dates, prices, {EQUITY: equity, BOND: pd.Series(0.0, index=dates)}


def test_a_small_signal_change_is_held_and_a_large_one_moves_to_the_band_edge():
    dates, prices, forecasts = _band_inputs()

    breakdowns = schedule_breakdowns(forecasts, prices, **_SCHEDULE_KWARGS)
    first, small, large = (breakdowns[dates[i]] for i in (17, 20, 23))

    assert (first.equity, first.drifted, first.held) == (0.5, None, False)

    cov = covariance_at_rebalance(dates[20], prices, horizon=3, train_window=15, embargo=2)
    d = cov.loc[EQUITY, EQUITY] - 2 * cov.loc[EQUITY, BOND] + cov.loc[BOND, BOND]
    assert small.band == pytest.approx((3.0 + 5.0) / 1e4 / (TILT_STRENGTH * d))
    assert abs(small.bounded - small.drifted) <= small.band
    assert small.held and small.equity == small.drifted

    assert large.bounded == DEFAULT_WEIGHT_BOUNDS[1]
    assert large.bounded - large.drifted > large.band
    assert not large.held
    assert large.equity == pytest.approx(large.bounded - large.band)


def test_a_held_rebalance_trades_nothing_in_the_backtest():
    dates, prices, forecasts = _band_inputs()

    backtest = run_backtest(prices, weight_schedule(forecasts, prices, **_SCHEDULE_KWARGS))

    assert backtest.optimised.turnover[dates[20]] == pytest.approx(0.0, abs=1e-12)
    assert backtest.optimised.turnover[dates[23]] > 0


def test_drift_past_a_bound_is_traded_back_to_exactly_that_bound():
    # Equity rises while bond is flat, so 80/20 drifts to about 82% equity. The
    # signals still ask for 80%, within the band, but the bound wins.
    dates = pd.bdate_range("2024-01-02", periods=60)
    rng = np.random.default_rng(3)
    equity = 100 * np.cumprod(1 + rng.normal(0, 0.01, 60))
    equity[18:21] = equity[17] * np.array([1.05, 1.08, 1.1])
    prices = {
        EQUITY: pd.Series(equity, index=dates),
        BOND: pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 60)), index=dates),
    }
    forecasts = {EQUITY: pd.Series(0.05, index=dates), BOND: pd.Series(0.0, index=dates)}

    breakdowns = schedule_breakdowns(forecasts, prices, **_SCHEDULE_KWARGS)
    first, second = breakdowns[dates[17]], breakdowns[dates[20]]

    assert first.equity == DEFAULT_WEIGHT_BOUNDS[1]
    assert second.drifted > DEFAULT_WEIGHT_BOUNDS[1]
    assert abs(second.bounded - second.drifted) <= second.band  # the band alone would hold
    assert not second.held
    assert second.equity == DEFAULT_WEIGHT_BOUNDS[1]
    backtest = run_backtest(prices, weight_schedule(forecasts, prices, **_SCHEDULE_KWARGS))
    assert backtest.optimised.turnover[dates[20]] > 0
    assert backtest.optimised.costs[dates[20]] > 0
