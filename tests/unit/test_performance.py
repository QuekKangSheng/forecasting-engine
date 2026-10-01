"""Risk-adjusted performance of a daily return series, checked against hand-worked values."""

import math

import numpy as np
import pandas as pd
import pytest

from forecasting_engine.portfolio.performance import (
    TRADING_DAYS_PER_YEAR,
    max_drawdown,
    performance,
    relative,
)

ROOT_YEAR = math.sqrt(TRADING_DAYS_PER_YEAR)


def _series(values) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range("2024-01-01", periods=len(values)), dtype=float)


# --- Sharpe ------------------------------------------------------------------


def test_sharpe_is_the_annualised_mean_over_the_annualised_sample_deviation():
    r = _series([0.03, -0.01] * 2)  # mean 0.01; each deviation ±0.02
    sample_sd = 0.02 * math.sqrt(4 / 3)

    assert performance(r).sharpe == pytest.approx(0.01 / sample_sd * ROOT_YEAR)


def test_sharpe_subtracts_the_risk_free_rate_compounded_to_a_day():
    r = _series([0.03, -0.01] * 2)
    daily_rf = 1.05 ** (1 / TRADING_DAYS_PER_YEAR) - 1
    sample_sd = 0.02 * math.sqrt(4 / 3)

    assert performance(r, risk_free_rate=0.05).sharpe == pytest.approx(
        (0.01 - daily_rf) / sample_sd * ROOT_YEAR
    )


def test_a_series_that_never_varies_has_no_sharpe():
    assert math.isnan(performance(_series([0.001] * 10)).sharpe)


# --- Sortino -----------------------------------------------------------------


def test_sortino_takes_downside_deviation_over_every_day_not_just_the_losing_ones():
    # Squared shortfalls below zero: 0, 1e-4, 0, 1e-4 -> mean over all four days
    # is 5e-5. Averaging over the two losing days alone would give 1e-4, a
    # downside deviation 41% too large and a Sortino 29% too small.
    r = _series([0.02, -0.01, 0.02, -0.01])
    downside = math.sqrt(5e-5)

    assert performance(r).sortino == pytest.approx(0.005 / downside * ROOT_YEAR)


def test_a_series_with_no_losing_day_has_no_sortino():
    assert math.isnan(performance(_series([0.01, 0.02, 0.0])).sortino)


# --- returns, drawdown, Calmar -------------------------------------------------


def test_annual_return_is_compounded_not_averaged():
    r = _series([0.10, -0.10])  # ends at 0.99 of where it started
    expected = 0.99 ** (TRADING_DAYS_PER_YEAR / 2) - 1

    assert performance(r).annual_return == pytest.approx(expected)


def test_drawdown_is_measured_from_the_running_peak():
    # 1.0 -> 1.1 -> 0.88 -> 0.968: the worst fall is 1.1 to 0.88, i.e. -20%.
    assert max_drawdown(_series([0.10, -0.20, 0.10])) == pytest.approx(-0.20)


def test_a_loss_on_the_first_day_counts_as_drawdown():
    # Without the starting value of 1.0 as the first peak, the running peak
    # would be 0.9 itself and this loss would vanish.
    assert max_drawdown(_series([-0.10, 0.05])) == pytest.approx(-0.10)


def test_a_series_that_only_rises_has_no_drawdown():
    assert max_drawdown(_series([0.01, 0.02])) == 0.0


def test_calmar_is_annual_return_over_the_size_of_the_worst_drawdown():
    r = _series([0.10, -0.20, 0.10] * 3)
    metrics = performance(r)

    assert metrics.calmar == pytest.approx(metrics.annual_return / abs(metrics.max_drawdown))


def test_a_series_with_no_drawdown_has_no_calmar():
    assert math.isnan(performance(_series([0.01, 0.02])).calmar)


def test_annual_volatility_is_the_sample_deviation_scaled_to_a_year():
    r = _series([0.03, -0.01] * 2)
    assert performance(r).annual_volatility == pytest.approx(0.02 * math.sqrt(4 / 3) * ROOT_YEAR)


# --- against the baseline ----------------------------------------------------


def test_tracking_error_and_information_ratio_come_from_the_daily_difference():
    baseline = _series([0.01, -0.02, 0.005, 0.0])
    active = np.array([0.002, -0.001, 0.003, 0.0])  # mean 0.001
    portfolio = baseline + active
    active_sd = float(np.std(active, ddof=1))

    result = relative(portfolio, baseline)

    assert result.tracking_error == pytest.approx(active_sd * ROOT_YEAR)
    annual_active = 0.001 * TRADING_DAYS_PER_YEAR
    assert result.information_ratio == pytest.approx(annual_active / (active_sd * ROOT_YEAR))


def test_a_portfolio_identical_to_the_baseline_has_no_information_ratio():
    baseline = _series([0.01, -0.02, 0.005])
    result = relative(baseline.copy(), baseline)

    assert result.tracking_error == 0.0
    assert math.isnan(result.information_ratio)


def test_comparing_series_over_different_days_is_refused():
    with pytest.raises(ValueError, match="same days"):
        relative(_series([0.01, 0.02]), _series([0.01, 0.02, 0.03]))
