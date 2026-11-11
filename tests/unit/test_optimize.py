"""Turning each target's saved signal into rebalance dates, expected signals, the
covariance each rebalance is sized against, and a risk level's Markowitz weight."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import weights_uniform

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.portfolio.backtest import run_backtest
from forecasting_engine.portfolio.optimize import (
    DEFAULT_RISK_LEVEL,
    DEFAULT_WEIGHT_BOUNDS,
    MIN_TRUST_HISTORY,
    RISK_LEVELS,
    OptimizeDataError,
    common_rebalance_dates,
    covariance_at_rebalance,
    expected_returns,
    forecast_gate_shut,
    forecast_trust,
    level_weight,
    long_run_premium,
    pairs_known_by,
    schedule_breakdowns,
    schedule_weights,
    solve_inputs_at,
    solve_weights,
    trust_pairs,
    weight_breakdown,
    weight_schedule,
)
from forecasting_engine.portfolio.performance import newey_west_lags
from forecasting_engine.reporting.portfolio_comparison import forecast_value_summary

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


# --- the weight at each risk level --------------------------------------------------


def _covariance(var_equity, var_bond, cov_eb) -> pd.DataFrame:
    return pd.DataFrame(
        {EQUITY: {EQUITY: var_equity, BOND: cov_eb}, BOND: {EQUITY: cov_eb, BOND: var_bond}}
    )


COVARIANCE = _covariance(0.0025, 0.0004, 0.00005)
D = 0.0025 - 2 * 0.00005 + 0.0004
W_MV = (0.0004 - 0.00005) / D  # 12.5%: below the default 20% floor, above a 0% one
PREMIUM = 0.01
NONE = pd.Series({EQUITY: 0.0, BOND: 0.0})


def _signal(difference: float) -> pd.Series:
    return pd.Series({EQUITY: difference, BOND: 0.0})


def _vol(weight: float, covariance: pd.DataFrame = COVARIANCE) -> float:
    held = np.array([weight, 1 - weight])
    return float(np.sqrt(held @ covariance.to_numpy() @ held))


def test_the_risk_scale_runs_from_1_to_5_defaulting_to_the_middle():
    assert list(RISK_LEVELS) == [1, 2, 3, 4, 5]
    assert DEFAULT_RISK_LEVEL == 3


@pytest.mark.parametrize("level", [0, 6])
def test_a_risk_level_off_the_scale_is_refused(level):
    with pytest.raises(ValueError, match="1 to 5"):
        level_weight(level, COVARIANCE)


@pytest.mark.parametrize("level", list(RISK_LEVELS))
def test_with_no_trust_every_level_holds_its_no_forecast_weight(level):
    breakdown = weight_breakdown(
        _signal(0.05), COVARIANCE, premium=PREMIUM, trust=0.0, risk_level=level
    )

    assert breakdown.forecast_tilt == 0.0
    assert breakdown.equity == breakdown.level_weight == level_weight(level, COVARIANCE)


@pytest.mark.parametrize("bounds", [(0.2, 0.8), (0.0, 1.0), (0.1, 0.9)])
@pytest.mark.parametrize("difference", [0.0, 0.002, 0.05, -0.005, -0.01])
@pytest.mark.parametrize("trust", [0.0, 0.5, 1.0])
def test_equity_never_increases_from_level_1_to_5_while_the_expected_return_is_not_negative(
    bounds, difference, trust
):
    assert PREMIUM + trust * difference >= 0
    equities = [
        weight_breakdown(
            _signal(difference),
            COVARIANCE,
            premium=PREMIUM,
            trust=trust,
            risk_level=level,
            bounds=bounds,
        ).equity
        for level in RISK_LEVELS
    ]

    assert all(a >= b - 1e-12 for a, b in zip(equities, equities[1:], strict=False))


def test_a_negative_expected_return_can_cut_a_riskier_level_below_a_safer_one():
    """The ordering above stops at zero by design: with equity expected to lose,
    the less risk-averse level acts on that harder. Markowitz, not a defect."""
    weights = {
        level: weight_breakdown(
            _signal(-0.03),
            COVARIANCE,
            premium=PREMIUM,
            trust=1.0,
            risk_level=level,
            bounds=(0.0, 1.0),
        ).equity
        for level in (1, 5)
    }

    assert weights[1] < weights[5]


@pytest.mark.parametrize("bounds", [(0.2, 0.8), (0.0, 1.0), (0.05, 0.95)])
@pytest.mark.parametrize("level", [1, 2, 3, 4])
def test_each_levels_weight_has_its_target_volatility(bounds, level):
    low = max(W_MV, bounds[0])
    target = _vol(low) + (5 - level) / 4 * (_vol(bounds[1]) - _vol(low))

    assert _vol(level_weight(level, COVARIANCE, bounds)) == pytest.approx(target, rel=1e-12)


@pytest.mark.parametrize(
    "bounds, lowest", [((0.2, 0.8), 0.2), ((0.0, 1.0), W_MV), ((0.05, 0.95), W_MV)]
)
def test_level_1_is_the_riskiest_mix_and_level_5_the_safest_the_bounds_allow(bounds, lowest):
    assert level_weight(1, COVARIANCE, bounds) == bounds[1]
    assert level_weight(5, COVARIANCE, bounds) == pytest.approx(lowest)


def test_level_3_is_not_pinned_to_50_50():
    assert level_weight(3, COVARIANCE, (0.2, 0.8)) != pytest.approx(0.5, abs=1e-3)


def test_the_tilt_is_the_trusted_signal_over_the_premium_times_the_distance_from_min_variance():
    breakdown = weight_breakdown(
        _signal(0.002), COVARIANCE, premium=PREMIUM, trust=0.5, risk_level=3, bounds=(0.0, 1.0)
    )

    expected = 0.5 * 0.002 * (breakdown.level_weight - W_MV) / PREMIUM
    assert breakdown.forecast_tilt == pytest.approx(expected)
    assert breakdown.equity == pytest.approx(breakdown.level_weight + expected)
    assert breakdown.unconstrained == pytest.approx(breakdown.equity)
    assert not breakdown.capped


@pytest.mark.parametrize("difference", [0.05, -0.05])
def test_level_5_ignores_the_signal(difference):
    flat = weight_breakdown(NONE, COVARIANCE, premium=PREMIUM, trust=1.0, risk_level=5)
    signalled = weight_breakdown(
        _signal(difference), COVARIANCE, premium=PREMIUM, trust=1.0, risk_level=5
    )

    assert signalled.forecast_tilt == 0.0
    assert signalled.equity == flat.equity


@pytest.mark.parametrize("premium", [0.0, -0.01])
@pytest.mark.parametrize("level", list(RISK_LEVELS))
def test_a_premium_at_or_below_zero_gives_no_tilt(premium, level):
    breakdown = weight_breakdown(
        _signal(0.05), COVARIANCE, premium=premium, trust=1.0, risk_level=level
    )

    assert breakdown.forecast_tilt == 0.0
    assert breakdown.equity == breakdown.level_weight


@pytest.mark.parametrize("level", list(RISK_LEVELS))
@pytest.mark.parametrize("difference", [10.0, -10.0, 0.001])
@pytest.mark.parametrize("bounds", [(0.2, 0.8), (0.0, 1.0), (0.45, 0.55), (0.5, 0.5)])
def test_weights_always_stay_inside_the_bounds(level, difference, bounds):
    weights = solve_weights(
        _signal(difference),
        COVARIANCE,
        premium=PREMIUM,
        trust=1.0,
        risk_level=level,
        bounds=bounds,
    )

    assert bounds[0] <= weights[EQUITY] <= bounds[1]
    assert weights[EQUITY] + weights[BOND] == pytest.approx(1.0)


def test_the_breakdown_says_when_a_bound_capped_the_weight():
    capped = weight_breakdown(_signal(1.0), COVARIANCE, premium=PREMIUM, trust=1.0)

    assert capped.unconstrained > 0.8
    assert capped.capped and capped.equity == 0.8


@pytest.mark.parametrize("level", list(RISK_LEVELS))
@pytest.mark.parametrize(
    "covariance",
    [_covariance(0.04, 0.04, 0.04), _covariance(0.04, 0.04, 0.04 + 1e-12)],
    ids=["zero", "negative"],
)
def test_where_d_is_not_positive_every_level_holds_the_lowest_risk_weight(level, covariance):
    breakdown = weight_breakdown(
        _signal(0.05), covariance, premium=PREMIUM, trust=1.0, risk_level=level, bounds=(0.3, 0.5)
    )

    assert breakdown.equity == breakdown.level_weight == pytest.approx(0.4)
    assert breakdown.forecast_tilt == 0.0


@pytest.mark.parametrize("level", list(RISK_LEVELS))
def test_a_minimum_variance_weight_at_or_above_the_upper_bound_holds_that_bound(level):
    bond_riskier = _covariance(0.0001, 0.0025, 0.0)

    breakdown = weight_breakdown(
        _signal(-0.05), bond_riskier, premium=PREMIUM, trust=1.0, risk_level=level
    )

    assert breakdown.equity == breakdown.level_weight == 0.8
    assert breakdown.forecast_tilt == 0.0


# --- the long-run premium -------------------------------------------------------------


def test_the_premium_is_the_horizon_scaled_mean_of_every_day_to_date():
    dates = pd.bdate_range("2024-01-02", periods=30)
    prices = _prices(dates, seed=5)
    daily = prices[EQUITY].pct_change() - prices[BOND].pct_change()

    premium = long_run_premium(prices, horizon=20)

    assert premium[dates[15]] == pytest.approx(20 * daily.iloc[1:16].mean())
    assert premium[dates[29]] == pytest.approx(20 * daily.iloc[1:].mean())


def test_the_premium_never_reads_a_later_day():
    dates = pd.bdate_range("2024-01-02", periods=30)
    prices = _prices(dates, seed=6)
    blown_up = prices[EQUITY].copy()
    blown_up.iloc[16:] *= 100

    before = long_run_premium(prices, horizon=20)
    after = long_run_premium({**prices, EQUITY: blown_up}, horizon=20)

    pd.testing.assert_series_equal(before.loc[: dates[15]], after.loc[: dates[15]])


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
_NO_BASELINES = {
    role: pd.Series(dtype=float, index=pd.DatetimeIndex([])) for role in (EQUITY, BOND)
}
_SCHEDULE_KWARGS = {
    "baselines": _NO_BASELINES,
    "horizon": 3,
    "train_window": 15,
    "test_window": 10,
    "embargo": 2,
}
_FULLY_TRUSTED = {**_SCHEDULE_KWARGS, "trust": 1.0}


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


def test_with_no_trust_yet_the_schedule_holds_each_levels_no_forecast_weight():
    _dates, prices, forecasts = _schedule_inputs()
    free = {EQUITY: 0.0, BOND: 0.0}

    breakdowns = schedule_breakdowns(forecasts, prices, costs_bps=free, **_SCHEDULE_KWARGS)

    assert all(b.forecast_tilt == 0.0 for b in breakdowns.values())
    assert all(b.equity == b.level_weight for b in breakdowns.values())


# --- FYP-49: the no-trade band ---------------------------------------------------------


def _rising_prices(dates, seed) -> dict:
    """A clearly positive equity premium, so the band and the tilt are both live."""
    rng = np.random.default_rng(seed)
    equity = pd.Series(100 * np.cumprod(1 + rng.normal(0.003, 0.01, len(dates))), index=dates)
    bond = pd.Series(100 * np.cumprod(1 + rng.normal(0.0, 0.003, len(dates))), index=dates)
    return {EQUITY: equity, BOND: bond}


def _unbanded(forecasts, prices) -> pd.Series:
    """Each rebalance's bounded target on its own, as the optimiser set it before the band."""
    dates = common_rebalance_dates(prices, horizon=3, train_window=15, test_window=10, embargo=2)
    expected = expected_returns(dates, forecasts)
    premiums = long_run_premium(prices, horizon=3)
    return pd.Series(
        {
            date: weight_breakdown(
                expected.loc[date],
                covariance_at_rebalance(date, prices, horizon=3, train_window=15, embargo=2),
                premium=premiums[date],
                trust=1.0,
            ).equity
            for date in dates
        }
    )


def _expected_band(date, prices, level: int) -> float:
    """``c·(w*_L − w_mv)/π̄``, worked out from the covariance and the premium."""
    cov = covariance_at_rebalance(date, prices, horizon=3, train_window=15, embargo=2)
    d = cov.loc[EQUITY, EQUITY] - 2 * cov.loc[EQUITY, BOND] + cov.loc[BOND, BOND]
    w_mv = (cov.loc[BOND, BOND] - cov.loc[EQUITY, BOND]) / d
    premium = long_run_premium(prices, horizon=3)[date]
    return (3.0 + 5.0) / 1e4 * (level_weight(level, cov) - w_mv) / premium


def test_zero_costs_reproduce_the_unbanded_schedule_exactly():
    dates = pd.bdate_range("2024-01-02", periods=60)
    prices = _rising_prices(dates, seed=0)
    _, _, forecasts = _schedule_inputs()
    free = {EQUITY: 0.0, BOND: 0.0}

    schedule = weight_schedule(forecasts, prices, costs_bps=free, **_FULLY_TRUSTED)

    pd.testing.assert_series_equal(
        schedule[EQUITY], _unbanded(forecasts, prices), check_names=False, check_freq=False
    )
    breakdowns = schedule_breakdowns(forecasts, prices, costs_bps=free, **_FULLY_TRUSTED)
    assert not any(b.held for b in breakdowns.values())


@pytest.mark.parametrize("level", [1, 2, 3, 4])
def test_the_band_is_the_cost_times_the_distance_from_min_variance_over_the_premium(level):
    dates = pd.bdate_range("2024-01-02", periods=60)
    prices = _rising_prices(dates, seed=0)
    forecasts = {EQUITY: pd.Series(0.0, index=dates), BOND: pd.Series(0.0, index=dates)}

    breakdowns = schedule_breakdowns(forecasts, prices, risk_level=level, **_SCHEDULE_KWARGS)

    for date, breakdown in list(breakdowns.items())[1:]:
        assert breakdown.band == pytest.approx(_expected_band(date, prices, level))
        assert breakdown.band > 0


def test_the_band_is_zero_at_level_5():
    dates = pd.bdate_range("2024-01-02", periods=60)
    prices = _rising_prices(dates, seed=0)
    forecasts = {EQUITY: pd.Series(0.0, index=dates), BOND: pd.Series(0.0, index=dates)}

    breakdowns = schedule_breakdowns(forecasts, prices, risk_level=5, **_SCHEDULE_KWARGS)

    assert all(b.band == 0.0 for b in breakdowns.values())


def test_a_premium_at_or_below_zero_gives_no_band():
    dates = pd.bdate_range("2024-01-02", periods=60)
    rng = np.random.default_rng(0)
    prices = {
        EQUITY: pd.Series(100 * np.cumprod(1 + rng.normal(-0.005, 0.01, 60)), index=dates),
        BOND: pd.Series(100 * np.cumprod(1 + rng.normal(0.0, 0.003, 60)), index=dates),
    }
    forecasts = {EQUITY: pd.Series(0.05, index=dates), BOND: pd.Series(0.0, index=dates)}

    breakdowns = schedule_breakdowns(forecasts, prices, **_FULLY_TRUSTED)

    premiums = long_run_premium(prices, horizon=3)
    assert (premiums[list(breakdowns)] <= 0).all()
    assert all(b.band == 0.0 and b.forecast_tilt == 0.0 for b in breakdowns.values())


def _band_inputs():
    """No signal at the first rebalance, a tiny one at the second, a large one at the
    third: the level's weight, then a change too small to pay for, then one that is."""
    dates = pd.bdate_range("2024-01-02", periods=60)
    prices = _rising_prices(dates, seed=0)
    equity = pd.Series(0.0, index=dates)
    equity.loc[dates[20] :] = 1e-6
    equity.loc[dates[23] :] = 0.05
    return dates, prices, {EQUITY: equity, BOND: pd.Series(0.0, index=dates)}


def test_a_small_signal_change_is_held_and_a_large_one_moves_to_the_band_edge():
    dates, prices, forecasts = _band_inputs()

    breakdowns = schedule_breakdowns(forecasts, prices, **_FULLY_TRUSTED)
    first, small, large = (breakdowns[dates[i]] for i in (17, 20, 23))

    assert (first.equity, first.drifted, first.held) == (first.level_weight, None, False)

    assert small.band == pytest.approx(_expected_band(dates[20], prices, 3))
    assert abs(small.bounded - small.drifted) <= small.band
    assert small.held and small.equity == small.drifted

    assert large.bounded == DEFAULT_WEIGHT_BOUNDS[1]
    assert large.bounded - large.drifted > large.band
    assert not large.held
    assert large.equity == pytest.approx(large.bounded - large.band)


def test_a_held_rebalance_trades_nothing_in_the_backtest():
    dates, prices, forecasts = _band_inputs()

    backtest = run_backtest(prices, weight_schedule(forecasts, prices, **_FULLY_TRUSTED))

    assert backtest.optimised.turnover[dates[20]] == pytest.approx(0.0, abs=1e-12)
    assert backtest.optimised.turnover[dates[23]] > 0


def test_drift_past_a_bound_is_traded_back_to_exactly_that_bound():
    # Equity rises while bond is flat, so 80/20 drifts to about 82% equity. The
    # signals still ask for 80%, within the band, but the bound wins.
    dates = pd.bdate_range("2024-01-02", periods=60)
    rng = np.random.default_rng(3)
    equity = 100 * np.cumprod(1 + rng.normal(0.003, 0.01, 60))
    equity[18:21] = equity[17] * np.array([1.05, 1.08, 1.1])
    prices = {
        EQUITY: pd.Series(equity, index=dates),
        BOND: pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.003, 60)), index=dates),
    }
    forecasts = {EQUITY: pd.Series(0.05, index=dates), BOND: pd.Series(0.0, index=dates)}

    breakdowns = schedule_breakdowns(forecasts, prices, **_FULLY_TRUSTED)
    first, second = breakdowns[dates[17]], breakdowns[dates[20]]

    assert first.equity == DEFAULT_WEIGHT_BOUNDS[1]
    assert second.drifted > DEFAULT_WEIGHT_BOUNDS[1]
    assert abs(second.bounded - second.drifted) <= second.band  # the band alone would hold
    assert not second.held
    assert second.equity == DEFAULT_WEIGHT_BOUNDS[1]
    backtest = run_backtest(prices, weight_schedule(forecasts, prices, **_FULLY_TRUSTED))
    assert backtest.optimised.turnover[dates[20]] > 0
    assert backtest.optimised.costs[dates[20]] > 0


# --- forecast trust --------------------------------------------------------------------


def _trust_inputs(make_y, *, n=400, seed=0):
    """A one-day horizon with bond flat, so each date's ``y`` is exactly equity's next
    day's return, which ``make_y`` sets from that date's signal difference ``x``."""
    dates = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.01, n)
    y = make_y(x, rng)
    equity = 100 * np.cumprod(np.r_[1.0, 1 + y[:-1]])
    prices = {EQUITY: pd.Series(equity, index=dates), BOND: pd.Series(100.0, index=dates)}
    signals = {EQUITY: pd.Series(x, index=dates), BOND: pd.Series(0.0, index=dates)}
    return dates, signals, prices


def _zero_baselines(dates) -> dict:
    return {EQUITY: pd.Series(0.0, index=dates), BOND: pd.Series(0.0, index=dates)}


def _trust_at_the_end(make_y, **kwargs) -> float:
    dates, signals, prices = _trust_inputs(make_y, **kwargs)
    return forecast_trust(
        dates[-1], trust_pairs(signals, _zero_baselines(dates), prices, horizon=1), horizon=1
    )


def test_returns_that_follow_the_signal_earn_a_trust_near_one():
    trust = _trust_at_the_end(lambda x, rng: x + rng.normal(0, 0.002, len(x)))

    assert trust == pytest.approx(1.0, abs=0.05)


def test_returns_unrelated_to_the_signal_earn_a_trust_near_zero():
    assert _trust_at_the_end(lambda x, rng: rng.normal(0, 0.01, len(x))) < 0.15


def test_returns_against_the_signal_clip_the_trust_to_zero():
    assert _trust_at_the_end(lambda x, rng: -x) == 0.0


def test_the_trust_never_exceeds_one():
    assert _trust_at_the_end(lambda x, rng: 3 * x) == 1.0


def test_the_regression_has_an_intercept():
    assert _trust_at_the_end(lambda x, rng: x + 0.005) == pytest.approx(1.0)


def test_a_signal_that_never_varies_earns_no_trust():
    dates, signals, prices = _trust_inputs(lambda x, rng: x)
    constant = {EQUITY: pd.Series(0.01, index=dates), BOND: pd.Series(0.0, index=dates)}

    pairs = trust_pairs(constant, _zero_baselines(dates), prices, horizon=1)

    assert forecast_trust(dates[-1], pairs, horizon=1) == 0.0


@pytest.mark.parametrize(("role", "share"), [(EQUITY, -0.5), (BOND, 0.5)])
def test_the_realised_return_is_net_of_the_same_training_means_as_the_signal(role, share):
    # y is half of x before the training means. Taking them off leaves y = x.
    dates, signals, prices = _trust_inputs(lambda x, rng: 0.5 * x)
    x = signals[EQUITY]
    baselines = {**_zero_baselines(dates), role: share * x}

    netted = forecast_trust(
        dates[-1], trust_pairs(signals, baselines, prices, horizon=1), horizon=1
    )
    gross = forecast_trust(
        dates[-1], trust_pairs(signals, _zero_baselines(dates), prices, horizon=1), horizon=1
    )

    assert netted == pytest.approx(1.0)
    assert gross == pytest.approx(0.5)


def test_a_pair_is_the_signal_difference_and_the_harness_label_less_the_training_means():
    dates = pd.bdate_range("2020-01-01", periods=80)
    prices = _prices(dates, seed=4)
    signals = {EQUITY: pd.Series(0.03, index=dates), BOND: pd.Series(0.01, index=dates)}
    baselines = {EQUITY: pd.Series(0.004, index=dates), BOND: pd.Series(0.001, index=dates)}

    pairs = trust_pairs(signals, baselines, prices, horizon=20)

    equity_return = prices[EQUITY].iloc[30] / prices[EQUITY].iloc[10] - 1
    bond_return = prices[BOND].iloc[30] / prices[BOND].iloc[10] - 1
    row = pairs.loc[dates[10]]
    assert row["x"] == pytest.approx(0.02)
    assert row["y"] == pytest.approx(equity_return - bond_return - 0.003)
    assert row["label_end"] == dates[30]
    assert len(pairs) == 60


def test_a_pair_is_known_only_once_the_later_of_its_two_returns_is():
    dates = pd.bdate_range("2020-01-01", periods=80)
    prices = _prices(dates, seed=4)
    prices[BOND] = prices[BOND].drop(dates[25])
    zeros = _zero_baselines(dates)

    pairs = trust_pairs(zeros, zeros, prices, horizon=20)

    assert pairs.loc[dates[10], "label_end"] == dates[31]


def test_a_pair_needs_a_stored_forecast_from_both_models():
    dates = pd.bdate_range("2020-01-01", periods=80)
    prices = _prices(dates, seed=4)
    zeros = _zero_baselines(dates)
    signals = {EQUITY: zeros[EQUITY], BOND: zeros[BOND].drop(dates[10])}

    pairs = trust_pairs(signals, zeros, prices, horizon=20)

    assert dates[10] not in pairs.index
    assert dates[11] in pairs.index


def _leakage_inputs():
    """Twenty-day returns the signal partly predicts, so the trust is neither clipped
    nor insensitive to any one return."""
    dates = pd.bdate_range("2020-01-01", periods=400)
    prices = _prices(dates, seed=5)
    forward = (prices[EQUITY].shift(-20) / prices[EQUITY] - 1) - (
        prices[BOND].shift(-20) / prices[BOND] - 1
    )
    rng = np.random.default_rng(6)
    signal = forward.fillna(0.0) * 2 + rng.normal(0, 0.05, len(dates))
    return dates, prices, {EQUITY: signal, BOND: pd.Series(0.0, index=dates)}


def test_returns_ending_on_or_after_the_rebalance_never_change_the_trust():
    dates, prices, signals = _leakage_inputs()
    t = dates[350]
    zeros = _zero_baselines(dates)
    before = forecast_trust(t, trust_pairs(signals, zeros, prices, horizon=20), horizon=20)
    assert 0 < before < 1

    extreme = prices[EQUITY].copy()
    extreme.loc[t:] *= 50
    pairs = trust_pairs(signals, zeros, {**prices, EQUITY: extreme}, horizon=20)

    assert pairs.loc[dates[330], "y"] > 10  # ends exactly on t, and is extreme
    assert forecast_trust(t, pairs, horizon=20) == before


def test_a_return_ending_the_day_before_the_rebalance_is_used():
    dates, prices, signals = _leakage_inputs()
    t = dates[350]
    zeros = _zero_baselines(dates)
    before = forecast_trust(t, trust_pairs(signals, zeros, prices, horizon=20), horizon=20)

    moved = prices[EQUITY].copy()
    moved.iloc[349] *= 1.5

    assert (
        forecast_trust(
            t, trust_pairs(signals, zeros, {**prices, EQUITY: moved}, horizon=20), horizon=20
        )
        != before
    )


def test_no_weight_up_to_a_rebalance_depends_on_anything_after_it():
    dates = pd.bdate_range("2020-01-01", periods=400)
    prices = _rising_prices(dates, seed=7)
    rng = np.random.default_rng(8)
    signals = {
        EQUITY: pd.Series(rng.normal(0, 0.02, 400), index=dates),
        BOND: pd.Series(0.0, index=dates),
    }
    kwargs = {**_SCHEDULE_KWARGS, "baselines": _zero_baselines(dates)}
    t = dates[350]
    before = schedule_breakdowns(signals, prices, **kwargs)

    extreme = prices[EQUITY].copy()
    extreme.loc[extreme.index > t] *= 50
    changed = signals[EQUITY].copy()
    changed.loc[changed.index > t] = 0.5
    after = schedule_breakdowns({**signals, EQUITY: changed}, {**prices, EQUITY: extreme}, **kwargs)

    assert {d: b for d, b in before.items() if d <= t} == {d: b for d, b in after.items() if d <= t}
    assert any(before[d] != after[d] for d in before if d > t)


def test_below_the_minimum_history_the_trust_is_zero():
    dates, signals, prices = _trust_inputs(lambda x, rng: x)
    pairs = trust_pairs(signals, _zero_baselines(dates), prices, horizon=1)

    # A one-day return from position i is known from position i + 1, so before
    # position p there are p - 1 pairs known.
    one_short, enough = dates[MIN_TRUST_HISTORY], dates[MIN_TRUST_HISTORY + 1]
    assert len(pairs_known_by(one_short, pairs)) == MIN_TRUST_HISTORY - 1
    assert forecast_trust(one_short, pairs, horizon=1) == 0.0
    assert len(pairs_known_by(enough, pairs)) == MIN_TRUST_HISTORY
    assert forecast_trust(enough, pairs, horizon=1) == pytest.approx(1.0)


def _weak_slope(x, rng):
    """A true slope of 0.05 with noise uncorrelated with ``x``, whatever the seed."""
    noise = rng.normal(0, 0.01, len(x))
    return 0.05 * x + noise - np.polyval(np.polyfit(x, noise, 1), x)


def _strong_slope(x, rng):
    return 0.5 * x + rng.normal(0, 0.01, len(x))


def _gated_and_not(make_y) -> tuple[float, float]:
    dates, signals, prices = _trust_inputs(make_y)
    pairs = trust_pairs(signals, _zero_baselines(dates), prices, horizon=1)
    return (
        forecast_trust(dates[-1], pairs, horizon=1),
        forecast_trust(dates[-1], pairs, horizon=1, gate=float("-inf")),
    )


def test_a_weak_noisy_slope_earns_no_trust():
    gated, slope = _gated_and_not(_weak_slope)

    assert slope > 0
    assert gated == 0.0


def test_a_strong_slope_keeps_its_value():
    gated, slope = _gated_and_not(_strong_slope)

    assert gated == pytest.approx(0.5, abs=0.1)
    assert gated == slope


def test_the_gate_reads_a_t_over_h_minus_1_overlapping_lags_and_is_strict():
    dates, prices, signals = _leakage_inputs()
    pairs = trust_pairs(signals, _zero_baselines(dates), prices, horizon=20)
    known = pairs_known_by(dates[-1], pairs)
    fit = sm.OLS(known["y"].to_numpy(), sm.add_constant(known["x"].to_numpy()))
    # h - 1 lags counted in full (Hansen-Hodrick), as the model metrics' overlap error.
    t = fit.fit(
        cov_type="HAC", cov_kwds={"maxlags": 19, "weights_func": weights_uniform}
    ).tvalues[1]
    newey_west_t = fit.fit(cov_type="HAC", cov_kwds={"maxlags": 19}).tvalues[1]
    spread_rule_t = fit.fit(
        cov_type="HAC", cov_kwds={"maxlags": newey_west_lags(len(known))}
    ).tvalues[1]
    assert t != pytest.approx(newey_west_t)
    assert t != pytest.approx(spread_rule_t)

    assert forecast_trust(dates[-1], pairs, horizon=20, gate=t - 1e-9) > 0
    assert forecast_trust(dates[-1], pairs, horizon=20, gate=t) == 0.0


def test_the_gate_is_shut_only_where_a_fit_fails_it():
    weak_dates, weak_signals, weak_prices = _trust_inputs(_weak_slope)
    weak = trust_pairs(weak_signals, _zero_baselines(weak_dates), weak_prices, horizon=1)
    strong_dates, strong_signals, strong_prices = _trust_inputs(_strong_slope)
    strong = trust_pairs(strong_signals, _zero_baselines(strong_dates), strong_prices, horizon=1)

    assert forecast_gate_shut(weak_dates[-1], weak, horizon=1)
    assert not forecast_gate_shut(strong_dates[-1], strong, horizon=1)
    # Too little history is not the gate.
    assert not forecast_gate_shut(weak_dates[MIN_TRUST_HISTORY], weak, horizon=1)


def _weak_schedule_inputs():
    dates, signals, prices = _trust_inputs(_weak_slope)
    rising = _rising_prices(dates, seed=7)
    rng = np.random.default_rng(8)
    noise = pd.Series(rng.normal(0, 0.01, len(dates)), index=dates)
    return (
        rising,
        {**signals, EQUITY: noise},
        {**_SCHEDULE_KWARGS, "baselines": _zero_baselines(dates)},
    )


def test_a_schedule_with_a_rejected_slope_is_the_no_forecast_level():
    prices, signals, kwargs = _weak_schedule_inputs()

    breakdowns = schedule_breakdowns(signals, prices, **kwargs)
    ungated = schedule_breakdowns(signals, prices, trust_gate=float("-inf"), **kwargs)

    assert breakdowns == schedule_breakdowns(signals, prices, trust=0.0, **kwargs)
    assert any(b.forecast_tilt != 0 for b in ungated.values())


def test_the_no_forecast_level_keeps_the_no_trade_band():
    prices, signals, kwargs = _weak_schedule_inputs()

    breakdowns = list(schedule_breakdowns(signals, prices, trust=0.0, **kwargs).values())

    assert all(b.forecast_tilt == 0 for b in breakdowns)
    assert any(b.held for b in breakdowns)


@pytest.mark.parametrize("bounds", [(0.2, 0.8), (0.1, 0.9)])
def test_at_level_5_the_optimised_and_no_forecast_series_are_identical(bounds):
    prices, signals, kwargs = _weak_schedule_inputs()
    kwargs = {**kwargs, "risk_level": 5, "bounds": bounds}

    optimised = run_backtest(
        prices, schedule_weights(schedule_breakdowns(signals, prices, trust=1.0, **kwargs))
    )
    no_forecast = run_backtest(
        prices, schedule_weights(schedule_breakdowns(signals, prices, trust=0.0, **kwargs))
    )

    pd.testing.assert_series_equal(optimised.optimised.net, no_forecast.optimised.net)
    summary = forecast_value_summary(optimised, no_forecast, "net")
    assert summary.t == "—"
    assert summary.label == "No forecast effect to test"


def test_the_pairs_used_never_fall_from_one_rebalance_to_the_next():
    dates, signals, prices = _trust_inputs(lambda x, rng: x)
    pairs = trust_pairs(signals, _zero_baselines(dates), prices, horizon=1)

    counts = [len(pairs_known_by(date, pairs)) for date in dates[::20]]

    assert counts == sorted(counts)
    assert counts[-1] > counts[0]
    assert pairs_known_by(dates[300], pairs).index[0] == dates[0]  # expanding, not rolling


def _skilful_inputs():
    """Signals equal to the three-day return that follows, so the trust is 1 as soon as
    there is enough history."""
    dates = pd.bdate_range("2020-01-01", periods=400)
    prices = _rising_prices(dates, seed=9)
    forward = (prices[EQUITY].shift(-3) / prices[EQUITY] - 1) - (
        prices[BOND].shift(-3) / prices[BOND] - 1
    )
    signals = {EQUITY: forward.fillna(0.0), BOND: pd.Series(0.0, index=dates)}
    return dates, prices, signals, {**_SCHEDULE_KWARGS, "baselines": _zero_baselines(dates)}


def test_the_schedule_tilts_only_once_the_forecasts_have_earned_trust():
    dates, prices, signals, kwargs = _skilful_inputs()
    pairs = trust_pairs(signals, kwargs["baselines"], prices, horizon=3)

    breakdowns = schedule_breakdowns(signals, prices, **kwargs)

    for date, breakdown in breakdowns.items():
        if len(pairs_known_by(date, pairs)) < MIN_TRUST_HISTORY:
            assert breakdown.forecast_tilt == 0.0
        elif signals[EQUITY][date] != 0:
            assert breakdown.forecast_tilt != 0.0
    assert any(b.forecast_tilt != 0.0 for b in breakdowns.values())
    assert any(b.forecast_tilt == 0.0 for b in breakdowns.values())


def test_a_trust_passed_in_replaces_the_estimate_at_every_rebalance():
    _dates, prices, signals, kwargs = _skilful_inputs()

    untrusted = schedule_breakdowns(signals, prices, trust=0.0, **kwargs)
    trusted = schedule_breakdowns(signals, prices, trust=1.0, **kwargs)

    assert all(b.forecast_tilt == 0.0 for b in untrusted.values())
    assert next(iter(trusted.values())).forecast_tilt != 0.0


def test_switching_an_equity_model_moves_the_weights_when_the_trust_is_given():
    _dates, prices, signals, kwargs = _skilful_inputs()
    kwargs = {**kwargs, "costs_bps": {EQUITY: 0.0, BOND: 0.0}}
    up = {EQUITY: signals[EQUITY] * 0 + 0.005, BOND: signals[BOND]}
    down = {EQUITY: signals[EQUITY] * 0 - 0.005, BOND: signals[BOND]}

    higher = weight_schedule(up, prices, trust=1.0, **kwargs)
    lower = weight_schedule(down, prices, trust=1.0, **kwargs)

    assert (higher[EQUITY] >= lower[EQUITY]).all()
    assert (higher[EQUITY] > lower[EQUITY]).any()


# --- FYP-246: solving one rebalance again on a different signal ----------------------


def test_solve_inputs_reproduce_each_rebalance_target_before_the_no_trade_band():
    dates, prices, signals, kwargs = _skilful_inputs()

    breakdowns = schedule_breakdowns(signals, prices, **kwargs)

    for date in list(breakdowns)[-3:]:
        inputs = solve_inputs_at(
            date,
            signals,
            prices,
            baselines=kwargs["baselines"],
            horizon=kwargs["horizon"],
            train_window=kwargs["train_window"],
            embargo=kwargs["embargo"],
        )
        assert inputs.solve().equity == pytest.approx(breakdowns[date].bounded)


def test_a_higher_equity_signal_re_solves_to_more_equity_once_trusted():
    dates, prices, signals, kwargs = _skilful_inputs()
    date = list(schedule_breakdowns(signals, prices, **kwargs))[-1]
    inputs = solve_inputs_at(
        date,
        signals,
        prices,
        baselines=kwargs["baselines"],
        horizon=3,
        train_window=15,
        embargo=2,
        bounds=(0.0, 1.0),
    )
    assert inputs.trust > 0

    higher = inputs.signal + pd.Series({EQUITY: 0.001, BOND: 0.0})

    assert inputs.solve(higher).unconstrained > inputs.solve().unconstrained


def test_solve_inputs_refuse_a_date_with_no_signal():
    dates, prices, signals, kwargs = _skilful_inputs()
    gappy = {**signals, BOND: signals[BOND][dates[-1] :]}

    with pytest.raises(OptimizeDataError, match="no signal"):
        solve_inputs_at(
            dates[-50],
            gappy,
            prices,
            baselines=kwargs["baselines"],
            horizon=3,
            train_window=15,
            embargo=2,
        )
