"""Turning each target's saved forecast into rebalance dates, expected returns and
the covariance each rebalance is sized against."""

import numpy as np
import pandas as pd
import pytest

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.portfolio.optimize import (
    DEFAULT_WEIGHT_BOUNDS,
    OptimizeDataError,
    common_rebalance_dates,
    covariance_at_rebalance,
    expected_returns,
    solve_weights,
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


def test_one_rebalance_per_test_window_on_a_shared_calendar():
    dates = pd.bdate_range("2024-01-02", periods=10)
    prices = _prices(dates, seed=0)

    rebalances = common_rebalance_dates(
        prices, horizon=1, train_window=2, test_window=2, embargo=0
    )

    # tuning_rows=0, train=2, embargo=0 -> first test window starts at
    # position 2; each later one two rows further: 2, 4, 6, 8.
    assert list(rebalances) == [dates[2], dates[4], dates[6], dates[8]]


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
        prices_with_gap, horizon=1, train_window=2, test_window=2, embargo=0
    )

    # The shared calendar drops dates[3], leaving dates[2] folded with
    # dates[4] instead of dates[3] — every later fold start shifts one
    # row earlier than the no-gap case above, not out of sync entirely.
    assert list(rebalances) == [dates[2], dates[5], dates[7]]


def test_rebalances_are_sorted_ascending():
    dates = pd.bdate_range("2024-01-02", periods=8)
    prices = _prices(dates, seed=2)

    rebalances = common_rebalance_dates(
        prices, horizon=1, train_window=2, test_window=1, embargo=0
    )

    assert list(rebalances) == sorted(rebalances)


def test_with_no_room_for_a_full_test_window_there_are_no_rebalances():
    dates = pd.bdate_range("2024-01-02", periods=3)
    prices = _prices(dates, seed=3)

    rebalances = common_rebalance_dates(
        prices, horizon=1, train_window=2, test_window=2, embargo=0
    )

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

    cov = covariance_at_rebalance(
        rebalance_date, prices, horizon=3, train_window=5, embargo=2
    )

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
    w_equity = 1/2 + (mu_equity - mu_bond) / (2 * risk_aversion * variance) —
    the two-asset tangency solution FYP-202 checks against."""
    mu = pd.Series({EQUITY: 0.03, BOND: 0.01})
    variance = 0.04
    covariance = _covariance(variance, variance, 0.0)

    weights = solve_weights(mu, covariance, risk_aversion=4.0, bounds=(0.0, 1.0))

    expected_equity = 0.5 + (mu[EQUITY] - mu[BOND]) / (2 * 4.0 * variance)
    assert weights[EQUITY] == pytest.approx(expected_equity)
    assert weights[BOND] == pytest.approx(1 - expected_equity)


def test_solve_weights_sums_to_one():
    mu = pd.Series({EQUITY: 0.02, BOND: -0.01})
    covariance = _covariance(0.05, 0.02, 0.01)

    weights = solve_weights(mu, covariance)

    assert weights[EQUITY] + weights[BOND] == pytest.approx(1.0)


def test_solve_weights_is_clipped_to_bounds():
    mu = pd.Series({EQUITY: 10.0, BOND: -10.0})  # an extreme, unrealistic spread
    covariance = _covariance(0.01, 0.01, 0.0)

    weights = solve_weights(mu, covariance, risk_aversion=0.1, bounds=(0.2, 0.8))

    assert weights[EQUITY] == pytest.approx(0.8)
    assert weights[BOND] == pytest.approx(0.2)


def test_solve_weights_falls_back_to_the_bounds_midpoint_when_assets_are_indistinguishable():
    mu = pd.Series({EQUITY: 0.05, BOND: 0.01})
    identical = _covariance(0.04, 0.04, 0.04)  # perfectly correlated, equal variance

    weights = solve_weights(mu, identical, bounds=(0.2, 0.8))

    assert weights[EQUITY] == pytest.approx(0.5)


def test_higher_risk_aversion_tilts_less_far_from_the_midpoint():
    mu = pd.Series({EQUITY: 0.05, BOND: 0.01})
    covariance = _covariance(0.04, 0.03, 0.01)

    cautious = solve_weights(mu, covariance, risk_aversion=20.0, bounds=(0.0, 1.0))
    bold = solve_weights(mu, covariance, risk_aversion=1.0, bounds=(0.0, 1.0))

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


# train_window=15, embargo=2, tuning_rows=0 (patched) -> first test window
# starts at position 17; each later one ten rows further: 17, 27, 37, 47.
_SCHEDULE_KWARGS = {"horizon": 3, "train_window": 15, "test_window": 10, "embargo": 2}


def test_weight_schedule_has_one_row_per_rebalance_summing_to_one_within_bounds():
    dates, prices, forecasts = _schedule_inputs()

    schedule = weight_schedule(forecasts, prices, **_SCHEDULE_KWARGS)

    assert list(schedule.index) == [dates[17], dates[27], dates[37], dates[47]]
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
    assert len(schedule) == 3
