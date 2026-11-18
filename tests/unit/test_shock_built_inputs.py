"""Scenarios on a function that reads inputs built from the committed columns: the bond
derived polynomial's rate levels, its longer window's copies and its calendar counts."""

import numpy as np
import pandas as pd
import pytest

from forecasting_engine.ingest.align import Transform, align_and_lag
from forecasting_engine.models import hybrid
from forecasting_engine.models.base import ModelDescription
from forecasting_engine.models.calendar_inputs import label_window_calendar
from forecasting_engine.models.hybrid import (
    bond_derived_folds,
    bond_derived_panel,
    function_transforms,
    level_input,
    long_window_input,
)
from forecasting_engine.reporting.polynomial_function import Origin
from forecasting_engine.scenario.shock import (
    ScenarioError,
    evaluate,
    function_inputs,
    next_inputs,
    run_scenario,
    signals_read,
    training_ranges,
)
from forecasting_engine.store.active_model import ActiveFunction, ActiveML
from forecasting_engine.validation.splitters import PurgedWalkForward, capped

RATE, GAUGE, FX = "USGG10YR_Index_PX_LAST", "VIX_Index_PX_LAST", "DXY_Curncy_PX_LAST"
PRICE = "LBUSTRUU_Index_TOT_RETURN_INDEX_GROSS_DVDS"
SIGNALS = (FX, RATE, GAUGE)
TRANSFORMS = {FX: Transform.LOG_RETURN, RATE: Transform.DIFFERENCE, GAUGE: Transform.LEVEL}
H = 5
RATE_LEVEL, RATE_LONG = level_input(RATE), long_window_input(RATE)
RATE_LEVEL_LONG = long_window_input(RATE_LEVEL)


def _frame(n: int = 420, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-02", periods=n)
    rate = 4 + np.cumsum(rng.normal(0, 0.05, n))
    gauge = 20 + np.cumsum(rng.normal(0, 0.5, n)).clip(-10, 10)
    fx = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    daily = 0.0003 * rate - 0.06 * np.r_[0.0, np.diff(rate)] + rng.normal(0, 0.002, n)
    return pd.DataFrame(
        {RATE: rate, GAUGE: gauge, FX: fx, PRICE: 100 * np.exp(np.cumsum(daily))}, index=idx
    )


def _splitter() -> PurgedWalkForward:
    return PurgedWalkForward(train=120, test=20, embargo=20, tuning_rows=120)


def _by_hand() -> ActiveFunction:
    """A function in a rate's change under both windows, its level and a calendar count."""
    names = (RATE, RATE_LONG, RATE_LEVEL, "CAL_MONTH_END")
    description = ModelDescription(
        "BondDerivedPolynomial",
        names,
        (0.001, -0.002, 0.003, 0.0005),
        0.001,
        {
            RATE: (-0.2, 0.2),
            RATE_LONG: (-0.25, 0.25),
            RATE_LEVEL: (2.0, 6.0),
            "CAL_MONTH_END": (0, 2),
        },
        {
            RATE: (0.0, 0.05),
            RATE_LONG: (0.001, 0.06),
            RATE_LEVEL: (4.0, 0.5),
            "CAL_MONTH_END": (0.25, 0.4),
        },
    )
    transforms = {
        RATE: Transform.DIFFERENCE,
        RATE_LONG: Transform.DIFFERENCE,
        RATE_LEVEL: Transform.LEVEL,
        "CAL_MONTH_END": Transform.LEVEL,
    }
    return ActiveFunction(Origin.DERIVED, description, transforms)


def test_built_inputs_are_read_from_the_columns_they_are_built_from():
    frame = _frame()
    transforms = {
        RATE: Transform.DIFFERENCE,
        RATE_LONG: Transform.DIFFERENCE,
        RATE_LEVEL_LONG: Transform.LEVEL,
        "CAL_MONTH_END": Transform.LEVEL,
    }

    inputs = next_inputs(frame, PRICE, transforms, horizon=H)

    plain = next_inputs(frame, PRICE, {RATE: Transform.DIFFERENCE}, horizon=H)
    assert inputs[RATE] == inputs[RATE_LONG] == plain[RATE]
    assert inputs[RATE_LEVEL_LONG] == frame[RATE].iloc[-1]  # the last level, a row later
    following = frame.index[-1] + pd.offsets.BDay(1)
    counts = label_window_calendar(frame.index.append(pd.DatetimeIndex([following])), H)
    assert inputs["CAL_MONTH_END"] == counts["CAL_MONTH_END"].iloc[-1]


def test_only_committed_columns_are_offered_to_shock():
    function = _by_hand()

    assert signals_read(function, [RATE, GAUGE, FX, PRICE]) == (RATE,)
    ml = ActiveML("xgboost", {}, {GAUGE: Transform.LEVEL}, pd.Timestamp(0), pd.Timestamp(1), 1)
    assert signals_read(ml, [RATE, GAUGE]) == (GAUGE,)


def test_shocking_a_rate_moves_its_change_its_level_and_its_longer_windows_copy():
    frame = _frame()
    function = _by_hand()
    last = frame[RATE].iloc[-1]

    result = run_scenario(
        function, frame, PRICE, {RATE: last + 0.5}, horizon=H, splitter=_splitter()
    )

    moved = {change.signal: change for change in result.changes}
    assert set(moved) == {RATE, RATE_LONG, RATE_LEVEL}  # never the calendar count
    assert moved[RATE].shocked - moved[RATE].base == pytest.approx(0.5)
    assert moved[RATE_LONG].shocked == moved[RATE].shocked
    assert moved[RATE_LEVEL].shocked == pytest.approx(last + 0.5)
    # Each window's input is clipped to its own bound.
    assert moved[RATE].clipped_to == 0.2 and moved[RATE_LONG].clipped_to == 0.25
    assert result.shocked != result.base


def test_the_longer_windows_inputs_are_ranged_over_that_windows_rows(monkeypatch):
    monkeypatch.setattr(hybrid, "LONG_TRAIN_WINDOW", 250)
    frame = _frame()
    page = align_and_lag(frame, [RATE], PRICE, horizon=H, transforms={RATE: Transform.DIFFERENCE})
    short = list(_splitter().split(page))[-1][0]
    longer = list(capped(_splitter(), 250).split(page))[-1][0]

    ranges = training_ranges(frame, PRICE, _by_hand().transforms, horizon=H, splitter=_splitter())

    assert len(longer) == 250 > len(short)
    assert ranges[RATE] == (page.frame.loc[short, RATE].min(), page.frame.loc[short, RATE].max())
    assert ranges[RATE_LONG] == (
        page.frame.loc[longer, RATE].min(),
        page.frame.loc[longer, RATE].max(),
    )


def test_a_fitted_bond_derived_function_reproduces_its_forecasts_from_the_store(monkeypatch):
    monkeypatch.setattr(hybrid, "LONG_TRAIN_WINDOW", 200)
    frame = _frame(seed=3)
    panel = bond_derived_panel(frame, SIGNALS, TRANSFORMS, PRICE, H)
    last = bond_derived_folds(panel, _splitter(), 10)[-1]
    read_as = function_transforms(panel)
    inputs = function_inputs(last.description, Origin.DERIVED, read_as)
    function = ActiveFunction(Origin.DERIVED, last.description, {s: read_as[s] for s in inputs})

    assert any(name.endswith(hybrid.LONG_WINDOW_SUFFIX) for name in function.inputs)
    for date in last.test:
        row = panel.frame.loc[date, list(function.inputs)]
        assert evaluate(function, row) == pytest.approx(last.predicted[date], abs=1e-12)

    # And a scenario with no shock forecasts what the function gives on the latest inputs.
    result = run_scenario(function, frame, PRICE, {}, horizon=H, splitter=_splitter())
    assert result.shocked == result.base
    assert result.base == evaluate(
        function, next_inputs(frame, PRICE, function.transforms, horizon=H)
    )


def test_a_function_reading_a_column_the_data_lacks_says_so():
    frame = _frame().drop(columns=[RATE])

    with pytest.raises(ScenarioError, match="neither a column of the committed data"):
        run_scenario(_by_hand(), frame, PRICE, {}, horizon=H, splitter=_splitter())
