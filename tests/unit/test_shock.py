"""Signal shocks: validation, inputs built as the fit builds them, and evaluation that
reproduces each fitted model without refitting it."""

import math

import numpy as np
import pandas as pd
import pytest

from forecasting_engine.ingest.align import Transform, align_and_lag
from forecasting_engine.models.base import ModelDescription
from forecasting_engine.models.polynomial import DerivedPolynomial, UserPolynomial
from forecasting_engine.models.sign_ruled import SignRuledPolynomial
from forecasting_engine.reporting.polynomial_function import Origin
from forecasting_engine.scenario.shock import (
    MAX_RANGE_MULTIPLE,
    ScenarioError,
    Shock,
    ShockKind,
    evaluate,
    function_inputs,
    last_observed,
    next_inputs,
    run_scenario,
    shocked_frame,
    validate,
)
from forecasting_engine.store.active_model import ActiveFunction
from forecasting_engine.validation.splitters import PurgedWalkForward

PRICE, VIX, RATE, BOND = "SPX", "VIX", "USGG10YR", "LEGATRUU"
TRANSFORMS = {VIX: Transform.LEVEL, RATE: Transform.DIFFERENCE, BOND: Transform.LOG_RETURN}
HORIZON = 5


def _frame(n=320, seed=0) -> pd.DataFrame:
    """SPX whose next-week return falls as VIX rises, beside a rate and a bond index."""
    rng = np.random.default_rng(seed)
    vix = 18 + np.cumsum(rng.normal(0, 0.6, n)).clip(-10, 30)
    rate = 3 + np.cumsum(rng.normal(0, 0.03, n))
    bond = 100 * np.cumprod(1 + rng.normal(0, 0.003, n))
    # Each day's return follows the day before's VIX, which the one-day lag shows.
    daily = np.r_[0.0, -0.001 * (vix[:-1] - 18)] + rng.normal(0, 0.003, n)
    spx = 4000 * np.cumprod(1 + daily)
    return pd.DataFrame(
        {PRICE: spx, VIX: vix, RATE: rate, BOND: bond},
        index=pd.bdate_range("2023-01-02", periods=n),
    )


def _panel(frame, transforms):
    return align_and_lag(frame, list(transforms), PRICE, horizon=HORIZON, transforms=transforms)


def _next_row(frame, transforms):
    """The panel row the forecast for the day after the last price reads, built by
    hand: the frame extended by that day, then aligned and lagged."""
    following = frame.index[-1] + pd.offsets.BDay(1)
    extended = pd.concat([frame, pd.DataFrame({PRICE: [frame[PRICE].iloc[-1]]}, index=[following])])
    return _panel(extended, transforms), pd.DatetimeIndex([following])


def _fitted(model, transforms, origin):
    frame = _frame()
    panel = _panel(frame, transforms)
    model.fit(panel, panel.frame.index[30:280])
    description = model.describe()
    inputs = function_inputs(description, origin, list(transforms))
    function = ActiveFunction(origin, description, {s: transforms[s] for s in inputs})
    return frame, model, function


# --- a zero shock reproduces each fitted model -------------------------------------


@pytest.mark.parametrize(
    ("model", "transforms", "origin"),
    [
        (DerivedPolynomial(degree=2), TRANSFORMS, Origin.DERIVED),
        (DerivedPolynomial(degree=1), TRANSFORMS, Origin.DERIVED),
        (
            UserPolynomial("x * y + x", {"x": VIX, "y": RATE}),
            TRANSFORMS,
            Origin.USER_SUPPLIED,
        ),
        (UserPolynomial("x / y", {"x": VIX, "y": BOND}), TRANSFORMS, Origin.USER_SUPPLIED),
        (
            SignRuledPolynomial({VIX: -1, RATE: 1}),
            dict.fromkeys((VIX, RATE), Transform.LEVEL),
            Origin.DERIVED,
        ),
    ],
    ids=["derived-2", "derived-1", "user", "user-ratio", "research"],
)
def test_the_stored_function_forecasts_what_the_fitted_model_predicts(model, transforms, origin):
    frame, model, function = _fitted(model, transforms, origin)
    panel, following = _next_row(frame, transforms)

    expected = model.predict(panel, following).iloc[0]
    actual = evaluate(function, next_inputs(frame, PRICE, function.transforms, horizon=HORIZON))

    assert math.isfinite(expected)
    assert actual == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_the_derived_fit_kept_terms_so_the_check_above_is_not_trivial():
    _, _, function = _fitted(DerivedPolynomial(degree=2), TRANSFORMS, Origin.DERIVED)

    assert function.description.terms
    assert VIX in function.inputs


def test_a_zero_shock_leaves_the_forecast_where_it_was():
    frame, _, function = _fitted(DerivedPolynomial(degree=2), TRANSFORMS, Origin.DERIVED)
    _, vix = last_observed(frame, VIX)

    result = run_scenario(function, frame, PRICE, {VIX: vix}, horizon=HORIZON, splitter=_splitter())

    assert result.shocked == result.base
    assert result.changes == ()


# --- a shock moves a degree-1 function by its slope ---------------------------------

LINEAR = ActiveFunction(
    Origin.DERIVED,
    ModelDescription(
        name="DerivedPolynomial",
        terms=(VIX,),
        coefficients=(-0.002,),
        intercept=0.01,
        input_bounds={VIX: (5.0, 35.0)},
        standardisation={VIX: (20.0, 5.0)},
    ),
    {VIX: Transform.LEVEL},
)


def _splitter():
    return PurgedWalkForward(train=200, test=20, embargo=HORIZON)


def _flat_vix(level=20.0) -> pd.DataFrame:
    frame = _frame()
    frame[VIX] = np.linspace(10, 30, len(frame))
    frame.iloc[-1, frame.columns.get_loc(VIX)] = level
    return frame


def test_a_degree_1_shock_moves_the_forecast_by_its_coefficient_times_the_standardised_shock():
    result = run_scenario(LINEAR, _flat_vix(20.0), PRICE, {VIX: 25.0}, horizon=HORIZON,
                          splitter=_splitter())

    assert result.base == pytest.approx(0.01)
    assert result.change == pytest.approx(-0.002 * (25 - 20) / 5)
    assert result.relative_change == pytest.approx(result.change / 0.01)


def test_a_shock_beyond_the_clipping_bound_moves_the_forecast_no_further_and_says_so():
    result = run_scenario(LINEAR, _flat_vix(20.0), PRICE, {VIX: 60.0}, horizon=HORIZON,
                          splitter=_splitter())

    assert result.change == pytest.approx(-0.002 * (35 - 20) / 5)
    (change,) = result.changes
    assert change.clipped_to == 35.0
    assert change.extrapolated
    assert change.trained_on[1] <= 30.0


def test_a_shock_within_the_training_range_is_not_an_extrapolation():
    (change,) = run_scenario(LINEAR, _flat_vix(20.0), PRICE, {VIX: 22.0}, horizon=HORIZON,
                             splitter=_splitter()).changes

    assert not change.extrapolated and change.clipped_to is None


def test_a_shock_to_a_signal_the_function_does_not_read_changes_nothing():
    frame = _flat_vix()

    result = run_scenario(LINEAR, frame, PRICE, {RATE: 9.0}, horizon=HORIZON, splitter=_splitter())

    assert result.shocked == result.base and result.changes == ()


# --- the shock passes through the same transform and lag as the fit ---------------


def test_a_differenced_input_moves_by_the_shock_less_the_last_level():
    frame = _frame()
    base = next_inputs(frame, PRICE, {RATE: Transform.DIFFERENCE}, horizon=HORIZON)[RATE]
    _, last = last_observed(frame, RATE)

    shocked = next_inputs(
        shocked_frame(frame, {RATE: last + 1.0}), PRICE, {RATE: Transform.DIFFERENCE},
        horizon=HORIZON,
    )[RATE]

    assert shocked - base == pytest.approx(1.0)


def test_a_log_return_input_reads_the_shock_against_the_day_before():
    frame = _frame()
    before = frame[BOND].iloc[-2]

    shocked = next_inputs(
        shocked_frame(frame, {BOND: before * 1.05}), PRICE, {BOND: Transform.LOG_RETURN},
        horizon=HORIZON,
    )[BOND]

    assert shocked == pytest.approx(math.log(1.05))


def test_the_next_forecast_reads_the_last_days_values_after_the_one_day_lag():
    frame = _frame()

    inputs = next_inputs(frame, PRICE, {VIX: Transform.LEVEL}, horizon=HORIZON)

    assert inputs[VIX] == frame[VIX].iloc[-1]


def test_a_signal_observed_after_the_last_price_is_shocked_on_or_before_that_price():
    frame = _frame()
    after = frame.index[-1] + pd.offsets.BDay(1)
    frame = pd.concat([frame, pd.DataFrame({VIX: [99.0]}, index=[after])])
    as_of = frame[PRICE].dropna().index[-1]

    date, value = last_observed(frame, VIX, as_of)

    assert date == as_of and value != 99.0
    shocked = shocked_frame(frame, {VIX: 40.0}, as_of)
    assert shocked.loc[as_of, VIX] == 40.0 and shocked.loc[after, VIX] == 99.0


def test_a_missing_input_on_the_last_day_is_refused_by_name():
    frame = _flat_vix()
    frame.iloc[-6:, frame.columns.get_loc(VIX)] = np.nan  # older than MAX_STALENESS

    with pytest.raises(ScenarioError, match="VIX has no value"):
        evaluate(LINEAR, next_inputs(frame, PRICE, LINEAR.transforms, horizon=HORIZON))


# --- validation ----------------------------------------------------------------------


def _validate(*shocks, positive=(VIX, BOND)):
    return validate(list(shocks), _flat_vix(20.0), positive=positive)


def test_a_valid_absolute_shock_is_the_new_level():
    levels, errors = _validate(Shock(VIX, ShockKind.ABSOLUTE, "45"))

    assert errors == {} and levels == {VIX: 45.0}


def test_a_percent_shock_is_a_change_from_the_last_observed_value():
    levels, errors = _validate(Shock(VIX, ShockKind.PERCENT, "+50"))

    assert errors == {} and levels[VIX] == pytest.approx(30.0)


def test_thousands_separators_are_accepted():
    levels, _ = _validate(Shock(VIX, ShockKind.ABSOLUTE, " 1,00 "))

    assert levels == {VIX: 100.0}


@pytest.mark.parametrize("entered", ["abc", "", "nan", "inf", "4o"])
def test_anything_but_a_number_is_refused_naming_the_signal(entered):
    levels, errors = _validate(Shock(VIX, ShockKind.ABSOLUTE, entered))

    assert levels == {}
    assert errors[VIX].startswith("VIX:") and "not a number" in errors[VIX]


@pytest.mark.parametrize(
    "shock", [Shock(VIX, ShockKind.ABSOLUTE, -5), Shock(VIX, ShockKind.PERCENT, -100)]
)
def test_a_series_always_above_zero_cannot_be_shocked_to_zero_or_below(shock):
    _, errors = _validate(shock)

    assert "always above zero" in errors[VIX]


def test_a_rate_may_go_negative():
    frame = _flat_vix()
    frame[RATE] = np.linspace(-0.5, 5, len(frame))

    levels, errors = validate([Shock(RATE, ShockKind.ABSOLUTE, -0.25)], frame, positive=(VIX,))

    assert errors == {} and levels == {RATE: -0.25}


def test_a_move_beyond_ten_times_the_whole_observed_range_is_refused():
    vix = _flat_vix(20.0)[VIX]
    limit = 20.0 + MAX_RANGE_MULTIPLE * (vix.max() - vix.min())
    assert MAX_RANGE_MULTIPLE == 10
    _, inside = _validate(Shock(VIX, ShockKind.ABSOLUTE, limit - 0.01))
    _, beyond = _validate(Shock(VIX, ShockKind.ABSOLUTE, limit + 0.01))

    assert inside == {}
    assert "more than 10 times the whole range" in beyond[VIX]


def test_each_invalid_signal_is_named_and_valid_ones_still_parse():
    levels, errors = _validate(
        Shock(VIX, ShockKind.ABSOLUTE, "x"), Shock(RATE, ShockKind.PERCENT, "10")
    )

    assert set(errors) == {VIX} and set(levels) == {RATE}


# --- which inputs a function reads ----------------------------------------------------


def test_a_derived_functions_inputs_are_the_columns_its_terms_multiply():
    description = ModelDescription(
        name="DerivedPolynomial",
        terms=(f"{VIX}^2", f"{RATE} {VIX}"),
        coefficients=(0.1, 0.2),
        intercept=0.0,
    )

    assert function_inputs(description, Origin.DERIVED, [BOND, RATE, VIX]) == (VIX, RATE)


def test_a_user_functions_inputs_are_the_columns_its_formula_names():
    description = ModelDescription(
        name="UserPolynomial", terms=(f"{BOND} / {VIX}",), coefficients=(1.0,), intercept=0.0
    )

    assert function_inputs(description, Origin.USER_SUPPLIED, [VIX, BOND]) == (BOND, VIX)
