"""Historical VaR, CVaR and breach counts, checked against hand-counted series."""

import math

import numpy as np
import pandas as pd
import pytest

from forecasting_engine.portfolio.performance import max_drawdown
from forecasting_engine.risk.tail import (
    HISTORICAL,
    VAR_CONFIDENCES,
    VAR_WINDOW,
    breaches,
    historical_tail_risk,
    historical_var,
)


def _series(values) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range("2020-01-01", periods=len(values)), dtype=float)


#: -5.0% up to +4.9% in steps of 0.1%: the five worst days are -5.0% ... -4.6%.
LADDER = _series([(i - 50) / 1000 for i in range(100)])


# --- VaR and CVaR --------------------------------------------------------------


def test_var_at_95_is_the_loss_on_the_fifth_worst_of_a_hundred_days():
    var, _ = historical_var(LADDER, 0.95)
    assert var == pytest.approx(0.046)


def test_cvar_at_95_is_the_mean_loss_over_those_five_days():
    _, cvar = historical_var(LADDER, 0.95)
    assert cvar == pytest.approx((0.050 + 0.049 + 0.048 + 0.047 + 0.046) / 5)


def test_at_99_one_day_in_a_hundred_is_the_worst_day_itself():
    assert historical_var(LADDER, 0.99) == pytest.approx((0.050, 0.050))


def test_a_fractional_tail_rounds_up_to_whole_days():
    # 30 days at 95% is 1.5 days of tail: the tail is the two worst days.
    thirty = _series([(i - 15) / 1000 for i in range(30)])
    assert historical_var(thirty, 0.95) == pytest.approx((0.014, 0.0145))


def test_the_order_of_the_days_does_not_matter():
    shuffled = LADDER.sample(frac=1, random_state=0).set_axis(LADDER.index)
    assert historical_var(shuffled, 0.95) == pytest.approx(historical_var(LADDER, 0.95))


def test_cvar_is_never_smaller_than_var():
    rng = np.random.default_rng(0)
    returns = _series(rng.standard_t(3, 500) / 100)
    for confidence in VAR_CONFIDENCES:
        var, cvar = historical_var(returns, confidence)
        assert cvar >= var


def test_losses_are_reported_as_positive_numbers():
    var, cvar = historical_var(_series([-0.03, -0.02, 0.01, 0.02]), 0.75)
    assert var == pytest.approx(0.03)
    assert cvar == pytest.approx(0.03)


def test_a_tail_of_gains_is_a_negative_loss_not_hidden():
    # If even the worst days made money, VaR is a negative loss, not zero.
    var, _ = historical_var(_series([0.01, 0.02, 0.03, 0.04]), 0.75)
    assert var == pytest.approx(-0.01)


def test_no_returns_means_no_var():
    var, cvar = historical_var(_series([]), 0.95)
    assert math.isnan(var) and math.isnan(cvar)


# --- breaches: tested against a VaR built only from earlier days ------------------


def test_a_day_is_tested_against_the_var_of_the_days_before_it():
    # Twenty calm days, then a 10% loss. Estimated from the calm days alone, the
    # VaR is tiny, so the loss is a breach. Had the VaR window included the day
    # being tested, the loss would set its own VaR and could never breach it.
    calm = [0.001 * ((-1) ** i) for i in range(20)]
    returns = _series([*calm, -0.10])

    count, tested = breaches(returns, 0.95, window=20)

    assert (count, tested) == (1, 1)


def test_a_loss_equal_to_the_var_is_not_a_breach():
    window = [-0.02] + [0.0] * 19  # at 95%, one tail day: VaR is exactly 2%
    returns = _series([*window, -0.02, -0.0201])

    count, tested = breaches(returns, 0.95, window=20)

    assert tested == 2
    assert count == 1, "-2.00% equals the VaR; only -2.01% exceeds it"


def test_nothing_is_tested_until_a_full_window_has_passed():
    assert breaches(_series([0.0] * 10), 0.95, window=20) == (0, 0)


def test_the_window_rolls_so_an_old_loss_stops_counting():
    # Two 5% losses, then 20 calm days, then a 1% loss. By the last day the
    # rolling window holds only calm days, so its VaR is 0 and the 1% loss
    # breaches it. An expanding window would still hold both old losses (two
    # tail days out of 22), keep its VaR at 5%, and miss the breach.
    returns = _series([-0.05, -0.05, *([0.0] * 20), -0.01])
    count, tested = breaches(returns, 0.95, window=20)
    assert tested == 3
    assert count == 1


# --- the result as a whole ----------------------------------------------------------


def test_the_result_is_tagged_historical_and_carries_every_level():
    rng = np.random.default_rng(1)
    returns = _series(rng.normal(0, 0.01, 600))

    risk = historical_tail_risk(returns)

    assert risk.method == HISTORICAL == "Historical"
    assert [level.confidence for level in risk.levels] == list(VAR_CONFIDENCES)
    assert risk.window == VAR_WINDOW
    assert risk.days == 600
    for level in risk.levels:
        assert (level.var, level.cvar) == historical_var(returns, level.confidence)
        assert level.days_tested == 600 - VAR_WINDOW
        assert level.expected_breach_rate == pytest.approx(1 - level.confidence)
        assert level.breach_rate == pytest.approx(level.breaches / level.days_tested)


def test_max_drawdown_sits_alongside_and_is_the_same_figure():
    returns = _series([0.01, -0.05, 0.02, -0.03, 0.04] * 60)
    assert historical_tail_risk(returns).max_drawdown == max_drawdown(returns)


def test_a_series_too_short_to_test_reports_no_breach_rate():
    level = historical_tail_risk(_series([0.01, -0.01] * 10)).levels[0]
    assert level.days_tested == 0
    assert math.isnan(level.breach_rate)


def test_an_unknown_confidence_is_refused():
    with pytest.raises(ValueError, match="between 0 and 1"):
        historical_var(LADDER, 1.5)
