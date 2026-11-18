"""The bond derived polynomial (``models/hybrid``): its members, its inputs, and the one
polynomial it reports. Synthetic data only."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from forecasting_engine.ingest.align import FeaturePanel, Transform, align_and_lag
from forecasting_engine.models import hybrid
from forecasting_engine.models.base import ModelDescription
from forecasting_engine.models.calendar_inputs import label_window_calendar
from forecasting_engine.models.hybrid import (
    FLOW_FITS,
    LONG_WINDOW_SUFFIX,
    MEMBERS,
    NAME,
    BondDerivedConfigError,
    HybridPolynomial,
    base_input,
    bond_derived_folds,
    bond_derived_inputs,
    bond_derived_panel,
    flow_forecaster,
    function_transforms,
    is_level,
    level_input,
    long_window_input,
    member_columns,
    member_folds,
    merge_descriptions,
    run_bond_derived_polynomial,
    source_column,
)
from forecasting_engine.models.polynomial import DerivedPolynomial
from forecasting_engine.validation import harness
from forecasting_engine.validation.harness import fold_baseline, summarize
from forecasting_engine.validation.splitters import TUNING_ROWS, PurgedWalkForward

DURATION = 6.0


# --- one member, on a hand-built panel (the research's checks) ----------------------


def make_panel(
    n=700, seed=0, carry=0.02, flow_beta=0.5, with_signals=True, horizon=5, rho=0.995
) -> FeaturePanel:
    """A bond-like price: its daily return is ``carry`` times yesterday's yield, less
    ``DURATION`` times today's yield change, plus ``flow_beta`` times an earlier ``FLOW``
    return, plus noise. ``YLD`` and ``GAUGE`` are persistent levels (``GAUGE`` predicts
    nothing); ``FLOW`` and ``NOISE`` are prices read as log returns."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    yld, gauge = np.empty(n), np.empty(n)
    yld[0], gauge[0] = 4.0, 20.0
    shocks, gshocks = rng.normal(0, 0.05, n), rng.normal(0, 1.0, n)
    for t in range(1, n):
        yld[t] = 4.0 + rho * (yld[t - 1] - 4.0) + shocks[t]
        gauge[t] = 20.0 + 0.98 * (gauge[t - 1] - 20.0) + gshocks[t]
    flow_ret, noise_ret = rng.normal(0, 0.01, n), rng.normal(0, 0.01, n)
    ret = np.zeros(n)
    for t in range(2, n):
        # Signals are lagged a row and the label starts the next close, so the flow that
        # predicts the return over (t-1, t] is the one dated t-2.
        ret[t] = (
            carry * yld[t - 2] / 100
            - DURATION * (yld[t] - yld[t - 1]) / 100
            + flow_beta * flow_ret[t - 2]
            + rng.normal(0, 0.0005)
        )
    frame = pd.DataFrame(
        {
            "YLD": yld,
            "GAUGE": gauge,
            "FLOW": 100 * np.exp(np.cumsum(flow_ret)),
            "NOISE": 100 * np.exp(np.cumsum(noise_ret)),
            "PRICE": 100 * np.exp(np.cumsum(ret)),
        },
        index=idx,
    )
    signals = ["YLD", "GAUGE", "FLOW", "NOISE"] if with_signals else []
    transforms = {
        "YLD": Transform.LEVEL,
        "GAUGE": Transform.LEVEL,
        "FLOW": Transform.LOG_RETURN,
        "NOISE": Transform.LOG_RETURN,
    }
    return align_and_lag(
        frame, signals, "PRICE", horizon=horizon, transforms={s: transforms[s] for s in signals}
    )


def evaluate_description(description: ModelDescription, raw: pd.DataFrame) -> np.ndarray:
    """The description's polynomial on ``raw``'s rows, written out by hand."""
    z = {}
    for name, (mean, sd) in description.standardisation.items():
        low, high = description.input_bounds[name]
        z[name] = (raw[name].clip(low, high) - mean) / sd
    total = np.full(len(raw), description.intercept, dtype=float)
    for term, coefficient in zip(description.terms, description.coefficients, strict=True):
        value = np.ones(len(raw))
        for factor in term.split(" "):
            name, _, power = factor.partition("^")
            value = value * z[name].to_numpy() ** (int(power) if power else 1)
        total += coefficient * value
    return total


def _member(fit: str) -> HybridPolynomial:
    return HybridPolynomial(flow_forecaster(fit, 10), fit)


def _with(panel: FeaturePanel, frame: pd.DataFrame) -> FeaturePanel:
    return replace(panel, frame=frame)


def _same(a: ModelDescription, b: ModelDescription) -> bool:
    return (a.terms, a.coefficients, a.intercept) == (b.terms, b.coefficients, b.intercept)


def test_kinds_follow_the_transform_not_the_name():
    panel = make_panel(200)
    renamed = align_and_lag(
        panel.frame[["PRICE"]].assign(RATE_LEVEL=np.linspace(1, 2, 200)),
        ["RATE_LEVEL"],
        "PRICE",
        horizon=5,
        transforms={"RATE_LEVEL": Transform.DIFFERENCE},
    )

    assert [is_level(panel, s) for s in panel.signals] == [True, True, False, False]
    # Named like a level, read as a change: a flow.
    assert not is_level(renamed, "RATE_LEVEL")
    # A calendar count has no alignment of its own: a flow.
    assert not is_level(panel, "CAL_MONTH_END")


@pytest.mark.parametrize("fit", FLOW_FITS)
def test_no_lookahead_past_the_last_training_label(fit):
    """Nothing after the last training label's end is read: the level part reads each
    level's move over the label window, which ends there."""
    panel = make_panel(520)
    train = panel.frame.index[20:400]
    first = _member(fit)
    first.fit(panel, train)
    reach = panel.label_end.loc[train].max()
    changed = panel.frame.copy()
    after = changed.index > reach
    changed.loc[after] = np.random.default_rng(99).normal(5.0, 3.0, size=changed.loc[after].shape)

    second = _member(fit)
    second.fit(_with(panel, changed), train)

    assert _same(first.describe(), second.describe())


@pytest.mark.parametrize("fit", ["lasso1", "shrinkage"])
def test_no_fold_reads_its_test_window(fit):
    """On the page's walk-forward, overwriting a fold's test window and everything after
    it leaves that fold's fit unchanged."""
    panel = make_panel(TUNING_ROWS + 120)
    splitter = PurgedWalkForward(train=504, test=20, embargo=20, tuning_rows=TUNING_ROWS)
    train, test = list(splitter.split(panel))[-1]
    first = _member(fit)
    first.fit(panel, train)
    changed = panel.frame.copy()
    after = changed.index >= test[0]
    changed.loc[after] = np.random.default_rng(7).normal(5.0, 3.0, size=changed.loc[after].shape)

    second = _member(fit)
    second.fit(_with(panel, changed), train)

    assert _same(first.describe(), second.describe())


@pytest.mark.parametrize("fit", FLOW_FITS)
def test_a_members_description_reproduces_its_forecast_where_inputs_are_clipped(fit):
    panel = make_panel(520, seed=3)
    model = _member(fit)
    model.fit(panel, panel.frame.index[20:400])
    idx = panel.frame.index[400:500]
    scaled = panel.frame.copy()
    scaled.loc[idx, ["FLOW", "NOISE"]] *= 6.0  # beyond the clip bounds
    scaled.loc[idx, ["YLD", "GAUGE"]] *= 1.8

    predicted = model.predict(_with(panel, scaled), idx).to_numpy()
    described = evaluate_description(model.describe(), scaled.loc[idx, list(panel.signals)])

    assert np.allclose(predicted, described, atol=1e-9, rtol=0)


@pytest.mark.parametrize("fit", FLOW_FITS)
def test_a_member_recovers_a_planted_carry_and_flow(fit):
    panel = make_panel(900, seed=1, carry=0.25, flow_beta=0.5)
    model = _member(fit)
    model.fit(panel, panel.frame.index[20:880])

    description = model.describe()
    coefficients = dict(zip(description.terms, description.coefficients, strict=True))
    assert coefficients.get("YLD", 0.0) > 0, coefficients
    assert coefficients.get("FLOW", 0.0) > 0, coefficients
    assert abs(coefficients.get("NOISE", 0.0)) < coefficients["FLOW"], coefficients


@pytest.mark.parametrize("fit", FLOW_FITS)
def test_a_member_with_no_inputs_forecasts_its_training_mean(fit):
    panel = make_panel(300, with_signals=False)
    model = _member(fit)
    train = panel.frame.index[:250]
    model.fit(panel, train)

    description = model.describe()
    expected = panel.frame.loc[train, panel.targets[0]].mean()
    assert description.terms == () and description.coefficients == ()
    assert description.intercept == pytest.approx(expected)
    assert np.allclose(model.predict(panel, panel.frame.index[:5]).to_numpy(), expected)


def test_a_member_with_only_levels_or_only_flows_is_that_part_alone():
    panel = make_panel(520, seed=5)
    train, idx = panel.frame.index[20:400], panel.frame.index[400:450]
    only_levels = replace(panel, signals=("YLD", "GAUGE"))
    only_flows = replace(panel, signals=("FLOW", "NOISE"))

    levels = _member("lasso1")
    levels.fit(only_levels, train)
    flows = _member("lasso1")
    flows.fit(only_flows, train)
    plain = DerivedPolynomial(1, "lasso", 10)
    plain.fit(only_flows, train)

    assert set(levels.describe().terms) <= {"YLD", "GAUGE"}
    assert levels.predict(only_levels, idx).notna().all()
    assert np.allclose(flows.predict(only_flows, idx), plain.predict(only_flows, idx))


def test_the_level_part_does_not_credit_a_random_walk_with_its_own_reversion():
    """Why the level part exists. A yield that is a random walk, with no carry, predicts
    nothing: the truth is a slope of zero. Its moves drive the price, so in one sample
    the plain regression's slope on its level is positive on average (Stambaugh), as if
    the level predicted its own fall. The level part's is nearer zero."""
    plain, fitted = [], []
    for seed in range(16):
        panel = make_panel(560, seed=100 + seed, carry=0.0, flow_beta=0.0, horizon=20, rho=1.0)
        train = panel.frame.index[20:524]
        levels = replace(panel, signals=("YLD",))
        model = _member("lasso1")
        model.fit(levels, train)
        description = model.describe()
        coefficients = dict(zip(description.terms, description.coefficients, strict=True))
        fitted.append(coefficients.get("YLD", 0.0))
        rows = panel.frame.loc[train, ["YLD", panel.targets[0]]].dropna()
        mean, sd = description.standardisation["YLD"]
        plain.append(float(np.polyfit((rows["YLD"] - mean) / sd, rows[panel.targets[0]], 1)[0]))

    assert np.mean(plain) > 0
    assert abs(np.mean(fitted)) < 0.5 * np.mean(plain), (np.mean(plain), np.mean(fitted))


# --- the 14 and their inputs -------------------------------------------------------

RATE, GAUGE, FX = "USGG10YR_Index_PX_LAST", "VIX_Index_PX_LAST", "DXY_Curncy_PX_LAST"
PRICE = "LBUSTRUU_Index_TOT_RETURN_INDEX_GROSS_DVDS"
SIGNALS = (FX, RATE, GAUGE)
TRANSFORMS = {FX: Transform.LOG_RETURN, RATE: Transform.DIFFERENCE, GAUGE: Transform.LEVEL}
H = 5


def committed(n: int = 420, seed: int = 0) -> pd.DataFrame:
    """Committed data, indexed by date: a yield that earns its own level and drives the
    price as it moves, a mean-reverting gauge, a currency, and the bond index."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-02", periods=n)
    rate = 4 + np.cumsum(rng.normal(0, 0.05, n))
    gauge = np.empty(n)
    gauge[0] = 20.0
    for t in range(1, n):
        gauge[t] = 20 + 0.95 * (gauge[t - 1] - 20) + rng.normal()
    fx = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    move = np.r_[0.0, np.diff(rate)]
    daily = (
        0.0003 * rate
        - 0.06 * move
        + 0.2 * np.r_[0.0, np.diff(np.log(fx))]
        + rng.normal(0, 0.002, n)
    )
    return pd.DataFrame(
        {RATE: rate, GAUGE: gauge, FX: fx, PRICE: 100 * np.exp(np.cumsum(daily))}, index=idx
    )


@pytest.fixture
def splitter() -> PurgedWalkForward:
    return PurgedWalkForward(train=120, test=20, embargo=20, tuning_rows=120)


def test_there_are_fourteen_members_and_no_degree_three_lasso_with_the_calendar():
    assert len(MEMBERS) == 14
    assert len(set(MEMBERS)) == 14
    assert not [m for m in MEMBERS if m.fit == "lasso3" and m.calendar]
    for calendar in (False, True):
        for long_window in (False, True):
            fits = [
                m.fit for m in MEMBERS if (m.calendar, m.long_window) == (calendar, long_window)
            ]
            assert fits == (["lasso1", "lasso2", "shrinkage"] if calendar else list(FLOW_FITS))


def test_the_inputs_are_found_by_transform_never_by_ticker():
    transforms = {
        "ODD_Index_PX_LAST": Transform.DIFFERENCE,
        GAUGE: Transform.LEVEL,
        FX: Transform.LOG_RETURN,
    }

    inputs = bond_derived_inputs(list(transforms), transforms, horizon=5)

    # An unknown series read as a change is a rate: it gets a level. A gauge and a price
    # don't.
    assert level_input("ODD_Index_PX_LAST") in inputs
    assert level_input(GAUGE) not in inputs and level_input(FX) not in inputs
    assert [n for n in inputs if n.startswith("CAL_") and LONG_WINDOW_SUFFIX not in n] == [
        "CAL_MONTH_END",
        "CAL_MONTH_START",
        "CAL_JOBS_FRIDAY",
    ]
    assert {n for n in inputs if n.endswith(LONG_WINDOW_SUFFIX)} == {
        long_window_input(n) for n in inputs if not n.endswith(LONG_WINDOW_SUFFIX)
    }


def test_the_weekday_flags_come_in_at_one_day_only():
    one_day = bond_derived_inputs(list(SIGNALS), TRANSFORMS, horizon=1)

    assert {"CAL_MON", "CAL_TUE", "CAL_WED", "CAL_THU"} <= set(one_day)
    assert "CAL_MON" not in bond_derived_inputs(list(SIGNALS), TRANSFORMS, horizon=5)


def test_the_panel_reads_each_input_as_its_fit_does():
    frame = committed()
    panel = bond_derived_panel(frame, SIGNALS, TRANSFORMS, PRICE, H)
    page = align_and_lag(frame, list(SIGNALS), PRICE, horizon=H, transforms=TRANSFORMS)
    levels = align_and_lag(frame, [RATE], PRICE, horizon=H, transforms={RATE: Transform.LEVEL})

    for signal in SIGNALS:
        pd.testing.assert_series_equal(panel.frame[signal], page.frame[signal])
        pd.testing.assert_series_equal(
            panel.frame[long_window_input(signal)], page.frame[signal], check_names=False
        )
    # A rate's level is lagged a row like any signal; a calendar count is not.
    pd.testing.assert_series_equal(
        panel.frame[level_input(RATE)], levels.frame[RATE], check_names=False
    )
    counts = label_window_calendar(page.frame.index, H)
    pd.testing.assert_series_equal(panel.frame["CAL_MONTH_END"], counts["CAL_MONTH_END"])
    assert panel.frame["CAL_MONTH_END"].notna().all()  # the latest rows too
    pd.testing.assert_series_equal(panel.frame[page.targets[0]], page.frame[page.targets[0]])
    kinds = {s for s in panel.signals if is_level(panel, s)}
    assert kinds == {
        GAUGE,
        level_input(RATE),
        long_window_input(GAUGE),
        long_window_input(level_input(RATE)),
    }


@pytest.mark.parametrize(
    "clash",
    ["VIX_Index_PX_LAST_LEVEL", "DXY_Curncy_PX_LAST_W2520", "CAL_MONTH_END"],
    ids=["level suffix", "window suffix", "calendar name"],
)
def test_a_column_named_like_a_built_input_is_refused_with_what_to_do(clash):
    frame = committed().rename(columns={GAUGE: clash})
    transforms = {FX: Transform.LOG_RETURN, RATE: Transform.DIFFERENCE, clash: Transform.LEVEL}

    with pytest.raises(BondDerivedConfigError, match="Rename that series' field in its export"):
        bond_derived_panel(frame, [FX, RATE, clash], transforms, PRICE, H)


def test_a_built_name_already_taken_by_another_column_is_refused():
    frame = committed().assign(**{level_input(RATE): 1.0})

    with pytest.raises(BondDerivedConfigError, match=level_input(RATE)):
        bond_derived_panel(frame, SIGNALS, TRANSFORMS, PRICE, H)


def test_every_input_name_maps_back_to_the_column_it_reads():
    columns = [RATE, GAUGE, FX]

    assert source_column(RATE, columns) == RATE
    assert source_column(long_window_input(RATE), columns) == RATE
    assert source_column(level_input(RATE), columns) == RATE
    assert source_column(long_window_input(level_input(RATE)), columns) == RATE
    assert source_column("CAL_MONTH_END", columns) is None
    assert source_column(long_window_input("CAL_JOBS_FRIDAY"), columns) is None
    assert base_input(long_window_input(level_input(RATE))) == level_input(RATE)
    # A committed column is always itself, whatever it is named.
    assert source_column("ODD_LEVEL", ["ODD_LEVEL"]) == "ODD_LEVEL"
    with pytest.raises(BondDerivedConfigError, match="neither a column of the committed data"):
        source_column("MISSING_Index_PX_LAST", columns)


def test_each_member_is_offered_its_windows_names_and_its_library():
    panel = bond_derived_panel(committed(), SIGNALS, TRANSFORMS, PRICE, H)
    plain = next(m for m in MEMBERS if not m.calendar and not m.long_window)
    longer = next(m for m in MEMBERS if m.calendar and m.long_window)

    assert member_columns(panel, plain) == (*SIGNALS, level_input(RATE))
    assert member_columns(panel, longer) == tuple(
        long_window_input(s)
        for s in (
            *SIGNALS,
            level_input(RATE),
            "CAL_MONTH_END",
            "CAL_MONTH_START",
            "CAL_JOBS_FRIDAY",
        )
    )


def test_function_transforms_record_how_every_input_is_read():
    panel = bond_derived_panel(committed(), SIGNALS, TRANSFORMS, PRICE, H)

    read = function_transforms(panel)

    assert (
        read[RATE] is Transform.DIFFERENCE and read[long_window_input(RATE)] is Transform.DIFFERENCE
    )
    assert read[level_input(RATE)] is Transform.LEVEL
    assert read["CAL_MONTH_END"] is Transform.LEVEL
    assert set(read) == set(panel.signals)


# --- the model: the mean of the 14 -------------------------------------------------

#: A longer window the data outgrows, so that the two windows really differ.
LONG = 200


@pytest.fixture(scope="module")
def walked() -> SimpleNamespace:
    """One walk-forward, and its members fitted both ways, shared by the tests below: a
    run of the 14 is slow."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(hybrid, "LONG_TRAIN_WINDOW", LONG)
        panel = bond_derived_panel(committed(n=300, seed=2), SIGNALS, TRANSFORMS, PRICE, H)
        splitter = PurgedWalkForward(train=120, test=20, embargo=20, tuning_rows=120)
        return SimpleNamespace(
            panel=panel,
            splitter=splitter,
            folds=bond_derived_folds(panel, splitter, 10),
            shared=member_folds(panel, splitter, 10),
            whole=member_folds(panel, splitter, 10, share_level_part=False),
        )


def _assert_same_members(shared, whole) -> None:
    for fast, slow in zip(shared, whole, strict=True):
        for a, b in zip(fast, slow, strict=True):
            assert a.train.equals(b.train) and a.test.equals(b.test)
            np.testing.assert_allclose(a.predicted, b.predicted, rtol=0, atol=1e-12)
            np.testing.assert_allclose(a.predicted_train, b.predicted_train, rtol=0, atol=1e-12)
            assert fold_baseline(a) == pytest.approx(fold_baseline(b), abs=1e-12)
            assert a.description.terms == b.description.terms
            np.testing.assert_allclose(
                a.description.coefficients, b.description.coefficients, rtol=0, atol=1e-12
            )
            assert a.description.intercept == pytest.approx(b.description.intercept, abs=1e-12)
            assert a.screening == b.screening


def test_the_forecast_is_the_mean_of_the_members_and_so_is_its_own_level(walked):
    assert len(walked.shared) == 14 and len(walked.folds) > 3
    for k, fold in enumerate(walked.folds):
        forecasts = pd.concat([m[k].predicted for m in walked.shared], axis=1)
        pd.testing.assert_series_equal(fold.predicted, forecasts.mean(axis=1), check_names=False)
        levels = [fold_baseline(m[k]) for m in walked.shared]
        assert fold.own_level == pytest.approx(np.mean(levels))
        assert fold.description.name == NAME
        assert fold.description == merge_descriptions([m[k].description for m in walked.shared])


def test_the_longer_window_trains_on_more_rows_and_in_sample_uses_the_pages(walked):
    short = MEMBERS.index(next(m for m in MEMBERS if not m.long_window))
    longer = MEMBERS.index(next(m for m in MEMBERS if m.long_window))
    last = walked.folds[-1]

    assert len(walked.shared[longer][-1].train) == LONG
    assert len(walked.shared[short][-1].train) == len(last.train) == 120
    assert walked.shared[short][-1].train.isin(walked.shared[longer][-1].train).all()
    assert last.predicted_train.index.equals(last.train)


def test_fitting_each_windows_level_part_once_gives_the_members_fitted_whole(walked):
    _assert_same_members(walked.shared, walked.whole)


def test_so_it_does_at_one_day_with_the_weekday_flags(monkeypatch, splitter):
    monkeypatch.setattr(hybrid, "LONG_TRAIN_WINDOW", LONG)
    panel = bond_derived_panel(committed(n=260, seed=5), SIGNALS, TRANSFORMS, PRICE, 1)

    shared = member_folds(panel, splitter, 10)

    assert "CAL_MON" in panel.signals
    _assert_same_members(shared, member_folds(panel, splitter, 10, share_level_part=False))


def test_a_member_whose_inputs_all_fail_screening_forecasts_its_training_mean(
    monkeypatch, splitter
):
    screen = harness.screen_over_folds
    monkeypatch.setattr(
        harness, "screen_over_folds", lambda panel, folds: screen(panel, folds, threshold=2.0)
    )
    panel = bond_derived_panel(committed(), SIGNALS, TRANSFORMS, PRICE, H)

    for fast, slow in zip(
        member_folds(panel, splitter, 10),
        member_folds(panel, splitter, 10, share_level_part=False),
        strict=True,
    ):
        for a, b in zip(fast, slow, strict=True):
            assert a.screening.fell_back and b.screening.fell_back
            pd.testing.assert_series_equal(a.predicted, b.predicted)
            assert a.description.terms == ()


def test_the_one_description_reproduces_the_forecast_across_both_windows(walked):
    last = walked.folds[-1]
    description = last.description
    names = {f for term in description.terms for f, _ in hybrid._factors(term)}

    # Both windows' inputs are in it, each with its own clip range and standardisation.
    assert any(n.endswith(LONG_WINDOW_SUFFIX) for n in names)
    assert any(not n.endswith(LONG_WINDOW_SUFFIX) for n in names)
    assert set(description.input_bounds) == set(description.standardisation) == names
    got = evaluate_description(description, walked.panel.frame.loc[last.test, list(names)])
    np.testing.assert_allclose(got, last.predicted.to_numpy(), rtol=0, atol=1e-9)


def test_the_one_description_reproduces_the_forecast_where_every_input_is_clipped(walked):
    """The latest fold's 14 fits again, on the inputs each one's screening kept, then
    asked for rows driven far past every clip bound."""
    panel, last = walked.panel, walked.folds[-1]
    bounds = last.description.input_bounds
    names = list(bounds)
    stretched = panel.frame.copy()
    beyond = [n for n in names if not n.startswith("CAL_")]
    for name in beyond:
        low, high = bounds[name]
        far = 1.0 + stretched.loc[last.test, name].abs().to_numpy()
        above = np.arange(len(last.test)) % 2 == 0
        stretched.loc[last.test, name] = np.where(above, high + far, low - far)
    wide = replace(panel, frame=stretched)
    forecasts = []
    for member, folds in zip(MEMBERS, walked.shared, strict=True):
        fold = folds[-1]
        if fold.screening.fell_back:
            forecasts.append(pd.Series(fold.description.intercept, index=last.test))
            continue
        model = HybridPolynomial(flow_forecaster(member.fit, 10), member.name)
        model.fit(replace(panel, signals=fold.screening.included), fold.train)
        assert _same(model.describe(), fold.description)
        forecasts.append(model.predict(replace(wide, signals=fold.screening.included), last.test))

    clipped = evaluate_description(last.description, stretched.loc[last.test, names])

    assert beyond, "the latest fold's equation reads no input that can be clipped"
    np.testing.assert_allclose(
        clipped, pd.concat(forecasts, axis=1).mean(axis=1).to_numpy(), rtol=0, atol=1e-9
    )


def test_merging_sums_a_shared_term_drops_a_cancelled_one_and_averages_the_intercept():
    a = ModelDescription(
        "a",
        ("X", "Y"),
        (0.2, 0.1),
        0.01,
        {"X": (-1.0, 1.0), "Y": (-2.0, 2.0)},
        {"X": (0.0, 0.5), "Y": (0.0, 1.0)},
    )
    b = ModelDescription(
        "b",
        ("X", "Y", "Z"),
        (0.4, -0.1, 0.0),
        0.03,
        {"X": (-1.0, 1.0), "Y": (-2.0, 2.0), "Z": (0.0, 1.0)},
        {"X": (0.0, 0.5), "Y": (0.0, 1.0), "Z": (0.5, 0.1)},
    )

    merged = merge_descriptions([a, b])

    assert merged.name == NAME
    assert merged.terms == ("X",)
    assert merged.coefficients == pytest.approx((0.3,))
    assert merged.intercept == pytest.approx(0.02)
    assert merged.input_bounds == {"X": (-1.0, 1.0)}
    assert merged.standardisation == {"X": (0.0, 0.5)}


def test_two_fits_reading_one_input_two_ways_are_not_passed_off_as_one_polynomial():
    a = ModelDescription("a", ("X",), (0.2,), 0.0, {"X": (-1.0, 1.0)}, {"X": (0.0, 0.5)})
    b = ModelDescription("b", ("X",), (0.2,), 0.0, {"X": (-1.0, 1.0)}, {"X": (0.1, 0.5)})

    with pytest.raises(RuntimeError, match="not one polynomial"):
        merge_descriptions([a, b])


def test_the_page_windows_terms_come_first_then_the_longer_windows():
    long_x = long_window_input("X")
    a = ModelDescription(
        "a",
        (f"{long_x}", "X^2", "X"),
        (0.1, 0.2, 0.3),
        0.0,
        {"X": (-1.0, 1.0), long_x: (-1.0, 1.0)},
        {"X": (0.0, 1.0), long_x: (0.0, 1.0)},
    )

    assert merge_descriptions([a]).terms == ("X", "X^2", long_x)


def test_a_run_is_deterministic_and_reports_the_latest_folds_equation(walked, monkeypatch):
    monkeypatch.setattr(hybrid, "LONG_TRAIN_WINDOW", LONG)

    result, description, panel = run_bond_derived_polynomial(
        committed(n=300, seed=2), SIGNALS, TRANSFORMS, PRICE, H, walked.splitter, max_terms=10
    )

    expected, latest = summarize(walked.folds)
    assert description == latest == walked.folds[-1].description
    assert panel.signals == walked.panel.signals
    pd.testing.assert_series_equal(result.forecast, expected.forecast)
    pd.testing.assert_series_equal(result.baseline, expected.baseline)
    assert result.signal_rank_ic == expected.signal_rank_ic
    assert result.screening is not None and result.screening.folds == len(walked.folds)
    assert set(result.screening.latest_included) <= set(panel.signals)


def test_too_short_a_history_says_so(splitter):
    with pytest.raises(BondDerivedConfigError, match="too few for one walk-forward fold"):
        run_bond_derived_polynomial(
            committed(n=150), SIGNALS, TRANSFORMS, PRICE, H, splitter, max_terms=10
        )


def test_the_term_cap_reaches_each_lasso_fit():
    capped = flow_forecaster("lasso2", 3)()
    shrinkage = flow_forecaster("shrinkage", 3)()

    assert isinstance(capped, DerivedPolynomial) and capped.max_terms == 3 and capped.degree == 2
    assert not isinstance(shrinkage, DerivedPolynomial)
    with pytest.raises(ValueError, match="no flow fit"):
        flow_forecaster("lasso4", 3)
