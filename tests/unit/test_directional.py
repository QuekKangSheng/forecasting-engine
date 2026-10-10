"""FYP-162: holding an index only when its forecast is positive, against holding it
throughout — checked on small forecasts and returns whose answers are known."""

import math

import numpy as np
import pandas as pd
import pytest

from forecasting_engine.portfolio.directional import (
    DirectionalDataError,
    cumulative,
    directional_pnl,
)


def _series(values, start="2024-01-01") -> pd.Series:
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


# --- the long/cash strategy (FYP-183) ----------------------------------------------


def test_the_strategy_earns_the_realised_return_only_when_the_forecast_is_positive():
    forecast = _series([0.01, -0.02, 0.03, -0.01])
    realised = _series([0.02, -0.01, -0.03, 0.04])

    result = directional_pnl(forecast, realised, horizon=1)

    assert result.strategy.tolist() == [0.02, 0.0, -0.03, 0.0]
    assert result.invested.tolist() == [True, False, True, False]


def test_only_the_forecasts_sign_matters_not_its_size():
    realised = _series([0.02, -0.01, 0.03])
    small = directional_pnl(_series([1e-9, -1e-9, 1e-9]), realised, horizon=1)
    large = directional_pnl(_series([5.0, -5.0, 5.0]), realised, horizon=1)

    pd.testing.assert_series_equal(small.strategy, large.strategy)


def test_a_zero_forecast_is_not_a_rise_so_the_strategy_holds_cash():
    result = directional_pnl(_series([0.0]), _series([0.05]), horizon=1)

    assert result.strategy.tolist() == [0.0]
    assert not result.invested.iloc[0]


def test_a_day_without_a_forecast_is_held_in_cash_and_counted():
    forecast = _series([0.01, np.nan, 0.01])
    realised = _series([0.01, 0.05, 0.01])

    result = directional_pnl(forecast, realised, horizon=1)

    assert result.strategy.tolist() == [0.01, 0.0, 0.01]
    assert result.no_forecast == 1


def test_days_with_no_realised_return_yet_are_left_out():
    forecast = _series([0.01, 0.01, 0.01])
    realised = _series([0.02, 0.03, np.nan])

    result = directional_pnl(forecast, realised, horizon=1)

    assert result.calls == 2


# --- buy-and-hold and compounding (FYP-185) ---------------------------------------


def test_buy_and_hold_is_the_realised_return_every_step():
    realised = _series([0.02, -0.01, 0.03])
    result = directional_pnl(_series([-1.0, -1.0, -1.0]), realised, horizon=1)

    assert result.buy_and_hold.tolist() == realised.tolist()


def test_cumulative_returns_compound_rather_than_add():
    growth = cumulative(_series([0.10, 0.10]))

    assert growth.iloc[-1] == pytest.approx(0.21)


def test_a_forecast_that_is_always_positive_matches_buy_and_hold():
    realised = _series([0.02, -0.01, 0.03, -0.02])
    result = directional_pnl(_series([0.1] * 4), realised, horizon=1)

    pd.testing.assert_series_equal(
        result.strategy_cumulative, result.buy_and_hold_cumulative, check_names=False
    )


def test_a_forecast_that_sidesteps_the_falls_beats_buy_and_hold():
    realised = _series([0.02, -0.05, 0.03])
    result = directional_pnl(_series([0.1, -0.1, 0.1]), realised, horizon=1)

    assert result.strategy_cumulative.iloc[-1] == pytest.approx(1.02 * 1.03 - 1)
    assert result.buy_and_hold_cumulative.iloc[-1] == pytest.approx(1.02 * 0.95 * 1.03 - 1)


# --- hit rate and share invested (FYP-186) ----------------------------------------


def test_hit_rate_counts_steps_whose_direction_matched():
    forecast = _series([0.01, -0.01, 0.01, -0.01])
    realised = _series([0.02, -0.02, -0.02, 0.02])

    result = directional_pnl(forecast, realised, horizon=1)

    assert result.hit_rate == pytest.approx(0.5)


def test_a_correct_call_to_stay_out_is_a_hit():
    result = directional_pnl(_series([-0.01]), _series([-0.03]), horizon=1)

    assert result.hit_rate == 1.0


def test_steps_without_a_forecast_are_left_out_of_the_hit_rate():
    forecast = _series([0.01, np.nan])
    realised = _series([0.02, -0.02])

    result = directional_pnl(forecast, realised, horizon=1)

    assert result.hit_rate == 1.0


def test_share_invested_is_the_share_of_steps_in_the_index():
    forecast = _series([0.01, -0.01, 0.01, 0.01])
    result = directional_pnl(forecast, _series([0.01] * 4), horizon=1)

    assert result.share_invested == pytest.approx(0.75)


def test_no_forecast_at_all_has_no_hit_rate():
    result = directional_pnl(_series([np.nan, np.nan]), _series([0.01, 0.02]), horizon=1)

    assert math.isnan(result.hit_rate)
    assert result.share_invested == 0.0


# --- h = 5 in non-overlapping steps (FYP-184) -------------------------------------


def test_a_five_day_horizon_steps_five_rows_at_a_time():
    n = 20
    forecast = _series(np.arange(n, dtype=float) + 1)
    realised = _series(np.arange(n, dtype=float) / 100)

    result = directional_pnl(forecast, realised, horizon=5)

    assert result.calls == 4
    assert result.buy_and_hold.tolist() == [0.0, 0.05, 0.10, 0.15]
    assert list(result.strategy.index) == list(forecast.index[::5])


def test_one_and_five_day_results_differ_on_the_same_series():
    rng = np.random.default_rng(0)
    forecast = _series(rng.normal(size=30))
    realised = _series(rng.normal(scale=0.01, size=30))

    daily = directional_pnl(forecast, realised, horizon=1)
    weekly = directional_pnl(forecast, realised, horizon=5)

    assert daily.calls == 30
    assert weekly.calls == 6


# --- the whole out-of-sample period ----------------------------------------------


def test_every_out_of_sample_date_is_replayed():
    forecast = _series(np.ones(300))
    realised = _series(np.full(300, 0.001))

    result = directional_pnl(forecast, realised, horizon=1)

    assert result.days == 300
    assert result.calls == 300
    assert result.start == forecast.index[0]
    assert result.end == forecast.index[-1]


def test_a_window_is_replayed_as_if_it_were_the_whole_period():
    forecast = _series(np.arange(30, dtype=float) + 1)
    realised = _series(np.arange(30, dtype=float) / 100)
    start, end = forecast.index[7], forecast.index[21]

    result = directional_pnl(forecast, realised, horizon=5, start=start, end=end)

    assert result.start == start and result.end == forecast.index[17]
    assert result.days == 15
    assert result.buy_and_hold.tolist() == [0.07, 0.12, 0.17]
    assert result.strategy_cumulative.iloc[0] == pytest.approx(0.07)


def test_a_window_with_no_realised_return_says_so():
    forecast = _series([0.01] * 5)
    with pytest.raises(DirectionalDataError, match="chosen window"):
        directional_pnl(
            forecast,
            _series([0.01] * 5),
            horizon=1,
            start=pd.Timestamp("2030-01-01"),
            end=pd.Timestamp("2030-02-01"),
        )


def test_dates_awaiting_a_realised_return_are_not_counted_as_replayed():
    realised = _series([0.01] * 8 + [np.nan] * 2)

    result = directional_pnl(_series(np.ones(10)), realised, horizon=1)

    assert result.days == 8
    assert result.end == realised.index[7]


def test_only_the_forecasts_dates_are_used_never_other_realised_dates():
    forecast = _series([0.01, 0.01], start="2024-01-03")
    realised = _series([0.5, 0.5, 0.01, 0.01, 0.5], start="2024-01-01")

    result = directional_pnl(forecast, realised, horizon=1)

    assert result.buy_and_hold.tolist() == [0.01, 0.01]


# --- errors -----------------------------------------------------------------------


def test_no_realised_return_at_all_fails_with_a_readable_message():
    with pytest.raises(DirectionalDataError, match="no out-of-sample day"):
        directional_pnl(_series([0.01]), _series([np.nan]), horizon=1)


@pytest.mark.parametrize("horizon", [0, -1])
def test_a_nonsense_horizon_is_refused(horizon):
    with pytest.raises(DirectionalDataError):
        directional_pnl(_series([0.01]), _series([0.01]), horizon=horizon)
