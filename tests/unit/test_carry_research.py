"""The bond research model: a fixed set of levels, fitted with each input's innovation
over the label window taken out."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from forecasting_engine.extraction.bloomberg_csv import ColumnSource
from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.ingest.align import Transform, align_and_lag
from forecasting_engine.models.carry_research import (
    CARRY_INPUTS,
    RHO_CAP,
    CarryResearchDataError,
    CarryResearchInputsError,
    CarryResearchPolynomial,
    carry_inputs,
    corrected_kappa,
    corrected_rho,
    innovations,
    label_window_moves,
    lasso_given_controls,
    run_carry_research_polynomial,
)
from forecasting_engine.reporting.polynomial_function import Origin
from forecasting_engine.scenario.shock import evaluate as evaluate_function
from forecasting_engine.scenario.shock import function_inputs
from forecasting_engine.store.active_model import ActiveFunction
from forecasting_engine.validation.harness import evaluate, signal_forecasts
from forecasting_engine.validation.splitters import PurgedWalkForward, expanding

H = 5


def _carry_world(n: int = 700, seed: int = 11) -> pd.DataFrame:
    """A rate whose level earns a return and whose moves cost one, a persistent gauge,
    and a price, on business days."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-04", periods=n)
    rate = 2.5 + np.cumsum(rng.normal(scale=0.05, size=n))
    gauge = np.empty(n)
    gauge[0] = 18.0
    for t in range(1, n):
        gauge[t] = 18.0 + 0.97 * (gauge[t - 1] - 18.0) + rng.normal()
    daily = np.r_[
        0.0,
        0.002 * (rate[:-1] - 2.5) - 0.06 * np.diff(rate) + rng.normal(scale=0.001, size=n - 1),
    ]
    price = 100 * np.exp(np.cumsum(daily))
    return pd.DataFrame({"rate": rate, "gauge": gauge, "price": price}, index=idx)


def _splitter() -> PurgedWalkForward:
    return PurgedWalkForward(train=120, test=20, embargo=5, tuning_rows=200)


def _levels(frame: pd.DataFrame, horizon: int = H, signals=("gauge", "rate")):
    return align_and_lag(
        frame,
        list(signals),
        "price",
        horizon=horizon,
        transforms=dict.fromkeys(signals, Transform.LEVEL),
    )


def _fitted(panel, train, **kwargs) -> CarryResearchPolynomial:
    model = CarryResearchPolynomial(**kwargs)
    model.fit(panel, train)
    return model


def _zero_kappa(daily, x, moves, horizon, lag):
    return np.zeros(x.shape[1])


def _ols_kappa(daily, x, moves, horizon, lag):
    """Each input's plain least-squares slope of its window move on its level."""
    xc = x - x.mean(axis=0)
    return (xc * (moves - moves.mean(axis=0))).sum(axis=0) / (xc * xc).sum(axis=0)


# --- the inputs ----------------------------------------------------------------------


def test_the_inputs_are_found_by_ticker_in_the_fixed_order():
    columns = [f"{t}_Index_PX_LAST" for t in reversed(CARRY_INPUTS[TargetRole.BOND])]
    sources = {c: ColumnSource(c.split("_")[0] + " Index", "PX_LAST") for c in columns}

    found = carry_inputs(columns, sources, TargetRole.BOND)

    assert found == tuple(f"{t}_Index_PX_LAST" for t in CARRY_INPUTS[TargetRole.BOND])


def test_a_missing_input_is_refused_by_name():
    columns = [f"{t}_Index_PX_LAST" for t in CARRY_INPUTS[TargetRole.BOND] if t != "USGG10YR"]
    sources = {c: ColumnSource(c.split("_")[0] + " Index", "PX_LAST") for c in columns}

    with pytest.raises(CarryResearchInputsError, match="needs USGG10YR"):
        carry_inputs(columns, sources, TargetRole.BOND)


def test_the_equity_target_has_no_carry_inputs():
    with pytest.raises(CarryResearchInputsError, match="bond target only"):
        carry_inputs([], {}, TargetRole.EQUITY)


def test_the_run_reads_every_input_as_its_level_lagged_one_row():
    frame = _carry_world()
    _, _, panel = run_carry_research_polynomial(frame, ["gauge", "rate"], "price", H, _splitter())

    assert {a.transform for a in panel.alignment.values()} == {Transform.LEVEL}
    pd.testing.assert_series_equal(panel.frame["rate"], frame["rate"].shift(1), check_freq=False)


# --- the innovation --------------------------------------------------------------------


def test_a_rows_move_is_each_inputs_change_over_that_rows_own_label_window():
    frame = _carry_world()
    panel = _levels(frame)

    moves = label_window_moves(panel)

    expected = frame["rate"].shift(-H) - frame["rate"]
    pd.testing.assert_series_equal(
        moves["rate"].iloc[: -H - 1], expected.iloc[: -H - 1], check_freq=False, check_names=False
    )
    assert moves.iloc[-H - 1 :].isna().all().all()


def test_the_daily_ar1_coefficient_is_bias_corrected_as_amihud_and_hurvich_do():
    rng = np.random.default_rng(5)
    x = np.empty(400)
    x[0] = 0.0
    for t in range(1, len(x)):
        x[t] = 0.9 * x[t - 1] + rng.normal()
    rho = np.polyfit(x[:-1], x[1:], 1)[0]
    n = len(x) - 1

    assert corrected_rho(pd.Series(x)) == pytest.approx(
        rho + (1 + 3 * rho) / n + 3 * (1 + 3 * rho) / n**2
    )


def test_a_near_unit_root_is_capped_below_one():
    rng = np.random.default_rng(1)
    walk = pd.Series(np.cumsum(rng.normal(size=300)))

    assert corrected_rho(walk) == RHO_CAP


def test_kappa_compounds_the_daily_coefficient_from_the_level_the_forecast_reads():
    rng = np.random.default_rng(2)
    daily = pd.DataFrame({"a": rng.normal(size=300).cumsum(), "b": rng.normal(size=300)})
    rho = np.array([corrected_rho(daily["a"]), corrected_rho(daily["b"])])

    kappa = corrected_kappa(daily, np.empty((0, 2)), np.empty((0, 2)), 20, 1)

    np.testing.assert_allclose(kappa, rho * (rho**20 - 1))


def test_the_innovations_average_zero():
    rng = np.random.default_rng(3)
    x, moves = rng.normal(size=(50, 2)), rng.normal(size=(50, 2)) + 1.0

    np.testing.assert_allclose(
        innovations(x, moves, np.array([-0.1, 0.2])).mean(axis=0), 0.0, atol=1e-12
    )


# --- what the correction does ----------------------------------------------------------


def _one_input(frame: pd.DataFrame, horizon: int = H):
    panel = _levels(frame, horizon, signals=("rate",))
    train = panel.frame.index[1 : len(panel.frame) - horizon - 1]
    rows = panel.frame.loc[train]
    moves = label_window_moves(panel).loc[train]
    keep = rows.notna().all(axis=1) & moves.notna().all(axis=1)
    x = rows.loc[keep, ["rate"]].to_numpy(float)
    return x, moves.loc[keep].to_numpy(float), rows.loc[keep, panel.targets[0]].to_numpy(float)


def test_with_plain_ols_kappa_the_slope_is_the_plain_regressions():
    # The property that makes the bias correction necessary: plain-OLS innovations are
    # uncorrelated with the level in the training rows, so controlling for them leaves
    # the level's slope exactly where a regression without them puts it.
    x, moves, y = _one_input(_carry_world())
    z = (x - x.mean()) / x.std()
    surprise = innovations(x, moves, _ols_kappa(None, x, moves, H, 1))
    alphas = np.geomspace(1e-3, 1e-7, 20)

    with_innovation, _ = lasso_given_controls(
        z, np.column_stack([np.ones(len(y)), surprise]), y, alphas
    )
    without, _ = lasso_given_controls(z, np.ones((len(y), 1)), y, alphas)

    np.testing.assert_allclose(with_innovation, without, rtol=1e-6, atol=1e-12)


def test_with_kappa_zero_the_fit_is_the_raw_move_version():
    # Controlling for the whole window move, as the research's raw-move version did: the
    # same slopes, and a forecast that differs only by a constant (that version's
    # intercept is the return with no move; this one's is with an average move).
    panel = _levels(_carry_world())
    train = panel.frame.index[1:400]
    model = _fitted(panel, train, kappa=_zero_kappa)
    rows = panel.frame.loc[train]
    moves = label_window_moves(panel).loc[train]
    keep = rows.notna().all(axis=1) & moves.notna().all(axis=1)
    x = rows.loc[keep, list(panel.signals)]
    window = moves.loc[keep].to_numpy(float)
    raw_controls = np.column_stack([np.ones(len(window)), window / window.std(axis=0)])
    z = model._z(x)
    alphas = np.geomspace(1e-3, 1e-7, 20)
    y = rows.loc[keep, panel.targets[0]].to_numpy(float)
    ours = innovations(x.to_numpy(float), window, np.zeros(2))
    ours_controls = np.column_stack([np.ones(len(ours)), ours / ours.std(axis=0)])

    raw_slopes, _ = lasso_given_controls(z, raw_controls, y, alphas)
    our_slopes, _ = lasso_given_controls(z, ours_controls, y, alphas)

    np.testing.assert_allclose(our_slopes, raw_slopes, rtol=1e-6, atol=1e-12)


#: A yield in percent that is a random walk, earning 1/252 of itself a day and losing 6
#: (duration) per point it rises: the carry world of the November 2026 research.
CARRY_PER_DAY = 1 / 252 / 100
DURATION = 6.0


def _yield_world(seed: int, n: int = 1500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-04", periods=n)
    yld = 2.5 + np.cumsum(rng.normal(scale=0.055, size=n))
    daily = np.r_[
        0.0,
        CARRY_PER_DAY * yld[:-1]
        - DURATION * np.diff(yld) / 100
        + rng.normal(scale=0.0001, size=n - 1),
    ]
    price = 100 * np.exp(np.cumsum(daily))
    return pd.DataFrame({"yield": yld, "price": price}, index=idx)


def test_on_a_world_with_a_known_stambaugh_bias_the_corrected_slope_is_nearer_the_truth():
    horizon = 20
    truth = CARRY_PER_DAY * horizon  # h-day expected return per point of yield
    corrected, plain = [], []
    for seed in range(5):
        panel = _levels(_yield_world(seed), horizon, signals=("yield",))
        train = panel.frame.index[1 : len(panel.frame) - horizon - 1]
        description = _fitted(panel, train).describe()
        slopes = dict(zip(description.terms, description.coefficients, strict=True))
        corrected.append(slopes.get("yield", 0.0) / description.standardisation["yield"][1])
        rows = panel.frame.loc[train, ["yield", panel.targets[0]]].dropna()
        plain.append(np.polyfit(rows["yield"], rows[panel.targets[0]], 1)[0])

    corrected_error = np.mean(np.abs(np.array(corrected) - truth))
    plain_error = np.mean(np.abs(np.array(plain) - truth))
    assert corrected_error < plain_error / 3


# --- the fit -----------------------------------------------------------------------------


def test_a_fit_is_reproducible():
    panel = _levels(_carry_world())
    train = panel.frame.index[1:400]

    assert _fitted(panel, train).describe() == _fitted(panel, train).describe()


def test_the_description_reproduces_the_forecast_in_the_scenario_engine_even_when_clipped():
    panel = _levels(_carry_world())
    train = panel.frame.index[1:400]
    model = _fitted(panel, train)
    description = model.describe()
    assert description.terms  # the fit kept something to reproduce
    function = ActiveFunction(
        Origin.DERIVED,
        description,
        dict.fromkeys(function_inputs(description, Origin.DERIVED, panel.signals), Transform.LEVEL),
    )
    rows = panel.frame.iloc[400:460].copy()
    rows.iloc[:5, rows.columns.get_loc("rate")] = 50.0
    rows.iloc[5:10, rows.columns.get_loc("gauge")] = -100.0
    shocked = replace(panel, frame=rows)

    predicted = model.predict(shocked, rows.index)
    by_function = [evaluate_function(function, row.to_dict()) for _, row in rows.iterrows()]

    np.testing.assert_allclose(by_function, predicted.to_numpy(), rtol=1e-12, atol=1e-15)
    assert 50.0 > description.input_bounds["rate"][1]  # the first rows really were clipped


def test_a_row_missing_an_input_gets_no_forecast():
    panel = _levels(_carry_world())
    model = _fitted(panel, panel.frame.index[1:400])
    holed = panel.frame.copy()
    holed.iloc[450, holed.columns.get_loc("gauge")] = np.nan

    forecast = model.predict(replace(panel, frame=holed), holed.index[440:460])

    assert np.isnan(forecast.iloc[10])
    assert forecast.drop(holed.index[450]).notna().all()


def test_too_few_complete_training_rows_is_refused_with_a_clear_message():
    panel = _levels(_carry_world())

    with pytest.raises(CarryResearchDataError, match="not enough complete training rows"):
        _fitted(panel, panel.frame.index[1:20])


# --- no look-ahead -------------------------------------------------------------------------


def _fold(panel, number: int):
    return list(expanding(_splitter()).split(panel))[number]


def test_no_value_dated_on_or_after_the_test_window_changes_the_fit():
    frame = _carry_world()
    train, test = _fold(_levels(frame), 10)
    later = frame.index >= test[0]
    rng = np.random.default_rng(0)
    rewritten = frame.copy()
    rewritten.loc[later] *= rng.uniform(0.5, 1.5, size=(int(later.sum()), frame.shape[1]))

    before = _fitted(_levels(frame), train).describe()
    after = _fitted(_levels(rewritten), train).describe()

    assert after == before


@pytest.mark.parametrize("horizon", [1, 5, 20])
def test_no_test_window_row_of_the_panel_changes_the_fit(horizon):
    # Rewriting the raw data from the test window on leaves the panel's first test row
    # holding the level from the day before; this rewrites the panel's rows themselves.
    # The embargo equals the horizon, as on the Models page.
    panel = _levels(_carry_world(), horizon=horizon)
    splitter = PurgedWalkForward(train=120, test=20, embargo=horizon, tuning_rows=200)
    train, test = list(expanding(splitter).split(panel))[10]
    later = panel.frame.index >= test[0]
    rewritten = panel.frame.copy()
    rng = np.random.default_rng(0)
    rewritten.loc[later] *= rng.uniform(0.5, 1.5, size=(int(later.sum()), panel.frame.shape[1]))

    before = _fitted(panel, train).describe()
    after = _fitted(replace(panel, frame=rewritten), train).describe()

    assert after == before


def test_the_last_level_the_fit_may_read_does_change_the_fit():
    # The checks above would pass vacuously if the fit read nothing near its window's
    # end. The last training label ends the day before the test window; the level two
    # days before sits on the row before the test window, the last row the fit reads.
    frame = _carry_world()
    train, test = _fold(_levels(frame), 10)
    rewritten = frame.copy()
    day_before = frame.index[frame.index.get_loc(test[0]) - 2]
    rewritten.loc[day_before, "rate"] += 0.5

    before = _fitted(_levels(frame), train).describe()
    after = _fitted(_levels(rewritten), train).describe()

    assert after != before


def test_a_row_whose_label_or_innovation_is_not_yet_complete_is_not_used():
    panel = _levels(_carry_world())
    index = panel.frame.index
    # The last row kept has a complete label but no complete move; the fit drops it too.
    complete = index[: len(index) - H]
    assert panel.frame[panel.targets[0]].iloc[-H - 1 :].notna().iloc[0]

    everything = _fitted(panel, index, kappa=_zero_kappa).describe()
    only_complete = _fitted(panel, complete, kappa=_zero_kappa).describe()

    assert everything == only_complete


def test_every_fold_trains_on_every_earlier_row():
    frame = _carry_world()
    result, _, panel = run_carry_research_polynomial(
        frame, ["gauge", "rate"], "price", H, _splitter()
    )
    folds = list(expanding(_splitter()).split(panel))

    assert len(folds) > 1
    assert all(train[0] == panel.frame.index[0] for train, _ in folds)
    assert result.forecast.notna().any()


def test_its_result_carries_a_pre_2016_holdout_when_the_data_reaches_back_that_far():
    # Starting well before the research's own Aug 2016 cutoff, with a short enough
    # train/tuning period that the first test window opens before it too.
    rng = np.random.default_rng(13)
    n = 500
    idx = pd.bdate_range("2015-01-02", periods=n)
    rate = 2.5 + np.cumsum(rng.normal(scale=0.05, size=n))
    daily = np.r_[0.0, 0.002 * (rate[:-1] - 2.5) + rng.normal(scale=0.001, size=n - 1)]
    price = 100 * np.exp(np.cumsum(daily))
    frame = pd.DataFrame({"rate": rate, "price": price}, index=idx)
    splitter = PurgedWalkForward(train=60, test=10, embargo=H)

    result, _, _ = run_carry_research_polynomial(frame, ["rate"], "price", H, splitter)

    assert result.holdout is not None
    assert result.holdout.before == pd.Timestamp("2016-08-01")
    assert result.holdout.last_label_end < result.holdout.before
    assert result.holdout.windows > 0


# --- a fold that keeps nothing -------------------------------------------------------------


def _noise_world(n: int = 700, seed: int = 3) -> pd.DataFrame:
    """Rates and a gauge that move the price but whose levels earn nothing."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-04", periods=n)
    rate = 2.5 + np.cumsum(rng.normal(scale=0.05, size=n))
    gauge = 18 + np.cumsum(rng.normal(size=n))
    daily = np.r_[0.0, -0.06 * np.diff(rate) + rng.normal(scale=0.003, size=n - 1)]
    price = 100 * np.exp(np.cumsum(daily))
    return pd.DataFrame({"rate": rate, "gauge": gauge, "price": price}, index=idx)


def test_a_fold_that_keeps_no_term_forecasts_its_intercept_and_adds_exactly_no_signal():
    panel = _levels(_noise_world())
    folds = evaluate(CarryResearchPolynomial, panel, _splitter())
    empty = [f for f in folds if not f.description.terms]
    assert empty

    for fold in empty:
        assert (fold.predicted.dropna() == fold.description.intercept).all()
        assert (signal_forecasts((fold,)).dropna() == 0.0).all()
