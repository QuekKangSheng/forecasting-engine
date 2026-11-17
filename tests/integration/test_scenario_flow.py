"""End-to-end: a saved active model -> a shocked scenario evaluated without
refitting -> the portfolio's simulated impact, through the real store and
optimizer (FYP-32, FYP-33, FYP-246)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.ingest.align import Transform, align_and_lag
from forecasting_engine.models.boosted import run_boosted
from forecasting_engine.models.carry_research import CarryResearchPolynomial
from forecasting_engine.models.polynomial import DerivedPolynomial
from forecasting_engine.portfolio.optimize import solve_inputs_at
from forecasting_engine.reporting.model_metrics import ModelRunResult
from forecasting_engine.reporting.polynomial_function import Origin
from forecasting_engine.scenario.portfolio_impact import portfolio_impact
from forecasting_engine.scenario.shock import (
    MAX_RANGE_MULTIPLE,
    Shock,
    ShockKind,
    evaluate,
    function_inputs,
    last_observed,
    ml_panel,
    next_inputs,
    refit_latest_fold,
    run_ml_scenario,
    run_scenario,
    validate,
)
from forecasting_engine.store.active_model import (
    ActiveFunction,
    ActiveML,
    get_active_function,
    get_active_ml,
    set_active_model,
)
from forecasting_engine.validation.crash import CrashDiagnostics
from forecasting_engine.validation.splitters import PurgedWalkForward

EQUITY, BOND = TargetRole.EQUITY, TargetRole.BOND
PRICE, VIX, RATE = "SPX", "VIX", "USGG10YR"
TRANSFORMS = {VIX: Transform.LEVEL, RATE: Transform.DIFFERENCE}
HORIZON = 5

BOND_PRICE = "AGG"
#: The bond research model's 7 fixed carry inputs; USGG10YR and VIX already exist
#: in ``_frame()``, so only the other 5 are added by ``_bond_research_frame()``.
CARRY_TICKERS = ("USGG10YR", "USGGBE10", "USYC2Y10", "LF98OAS", "LUACOAS", "VIX", "JPMVXYGL")


def _frame(n: int = 320, seed: int = 0) -> pd.DataFrame:
    """SPX whose next-week return falls as VIX rises, beside a rate input."""
    rng = np.random.default_rng(seed)
    vix = 18 + np.cumsum(rng.normal(0, 0.6, n)).clip(-10, 30)
    rate = 3 + np.cumsum(rng.normal(0, 0.03, n))
    # VIX is a clipped random walk here, not mean-reverting, so demean it over the
    # sample before using it to wiggle returns - otherwise a sustained drift in VIX
    # itself would leak into SPX's average return and could swamp the explicit
    # positive drift below, making the long-run equity premium ambiguous in sign.
    wiggle = -0.001 * (vix - vix.mean())
    daily = 0.0005 + np.r_[0.0, wiggle[:-1]] + rng.normal(0, 0.003, n)
    spx = 4000 * np.cumprod(1 + daily)
    return pd.DataFrame(
        {PRICE: spx, VIX: vix, RATE: rate}, index=pd.bdate_range("2023-01-02", periods=n)
    )


def _splitter() -> PurgedWalkForward:
    return PurgedWalkForward(train=200, test=20, embargo=HORIZON)


def _saved_function(db_path, *, degree: int = 2) -> tuple[pd.DataFrame, ActiveFunction]:
    """Fit a real ``DerivedPolynomial``, store it as the active equity model exactly
    as the Models page would, and read it back - round-tripping through the real
    DuckDB store rather than handing a hand-built ``ActiveFunction`` to the scenario."""
    frame = _frame()
    panel = align_and_lag(frame, list(TRANSFORMS), PRICE, horizon=HORIZON, transforms=TRANSFORMS)
    model = DerivedPolynomial(degree=degree)
    model.fit(panel, panel.frame.index[30:280])
    description = model.describe()
    inputs = function_inputs(description, Origin.DERIVED, list(TRANSFORMS))
    function = ActiveFunction(Origin.DERIVED, description, {s: TRANSFORMS[s] for s in inputs})

    set_active_model(
        EQUITY,
        "Polynomial (derived)",
        ModelRunResult(
            ic=0.05,
            oos_rank_ic=0.03,
            rmse=0.01,
            crash=CrashDiagnostics(recall=0.5, precision=0.5, f1=0.5, n_true_tail_days=10),
        ),
        high_risk=False,
        horizon=HORIZON,
        train_window=200,
        test_window=20,
        dataset_fingerprint="fp-scenario-flow",
        function=function,
        db_path=db_path,
    )
    reloaded = get_active_function(EQUITY, db_path=db_path)
    assert reloaded is not None
    return frame, reloaded


def _bond_research_frame(n: int = 400, seed: int = 11) -> pd.DataFrame:
    """A bond index whose next-week return rises with VIX's level, like ``_frame()``'s
    own SPX return does, so the lasso keeps VIX's term; its other 6 fixed carry
    inputs are plain, unrelated random walks. Includes SPX too, so the one frame
    can also feed ``_real_solve_inputs``'s equity signal."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    vix = 18 + np.cumsum(rng.normal(0, 0.6, n)).clip(-10, 30)
    spx = 4000 * np.cumprod(1 + rng.normal(0.0004, 0.003, n))
    daily = 0.0002 + 0.0004 * (vix - vix.mean()) / vix.std() + rng.normal(0, 0.0008, n)
    bond = 100 * np.cumprod(1 + daily)
    frame = pd.DataFrame({PRICE: spx, VIX: vix, BOND_PRICE: bond}, index=idx)
    for ticker in ("USGG10YR", "USGGBE10", "USYC2Y10", "LF98OAS", "LUACOAS", "JPMVXYGL"):
        frame[ticker] = 2 + np.cumsum(rng.normal(0, 0.03, n))
    return frame


def _saved_bond_function(db_path) -> tuple[pd.DataFrame, ActiveFunction]:
    """Fit a real ``CarryResearchPolynomial`` - the bond research model - and store
    it as the active bond model exactly as the Models page would (``Origin.DERIVED``
    under the hood, same as the equity research model), round-tripping through the
    real DuckDB store."""
    frame = _bond_research_frame()
    transforms = dict.fromkeys(CARRY_TICKERS, Transform.LEVEL)
    panel = align_and_lag(
        frame, list(CARRY_TICKERS), BOND_PRICE, horizon=HORIZON, transforms=transforms
    )
    model = CarryResearchPolynomial()
    # One row past 350: the fit drops its last training row, whose move is read past
    # every training label, so its regression uses rows 50-349, as before VERSION 2.
    model.fit(panel, panel.frame.index[50:351])
    description = model.describe()
    inputs = function_inputs(description, Origin.DERIVED, list(CARRY_TICKERS))
    function = ActiveFunction(Origin.DERIVED, description, {s: Transform.LEVEL for s in inputs})

    set_active_model(
        BOND,
        "Polynomial – fixed carry inputs (research)",
        ModelRunResult(
            ic=0.03,
            oos_rank_ic=0.025,
            rmse=0.008,
            crash=CrashDiagnostics(recall=0.4, precision=0.4, f1=0.4, n_true_tail_days=8),
        ),
        high_risk=False,
        horizon=HORIZON,
        train_window=None,
        test_window=20,
        dataset_fingerprint="fp-scenario-flow-bond",
        function=function,
        db_path=db_path,
    )
    reloaded = get_active_function(BOND, db_path=db_path)
    assert reloaded is not None
    return frame, reloaded


def _real_solve_inputs(frame: pd.DataFrame, *, trust: float = 1.0) -> tuple[pd.Series, dict, dict]:
    """A real ``SolveInputs`` for the frame's last date, built the way the backtest
    itself would (``solve_inputs_at``) rather than hand-constructed - with ``trust``
    forced so the test doesn't depend on there being 252 pairs of history, which
    ``forecast_trust`` estimation is already unit-tested on its own."""
    dates = frame.index
    rng = np.random.default_rng(7)
    bond_price = pd.Series(100 * np.cumprod(1 + rng.normal(0.0001, 0.003, len(dates))), index=dates)
    prices = {EQUITY: frame[PRICE], BOND: bond_price}
    signal_equity = -(frame[VIX] - frame[VIX].mean()) * 0.001
    signal_bond = pd.Series(0.0, index=dates)
    signals = {EQUITY: signal_equity, BOND: signal_bond}
    baselines = {EQUITY: pd.Series(0.0, index=dates), BOND: pd.Series(0.0, index=dates)}

    inputs = solve_inputs_at(
        dates[-1],
        signals,
        prices,
        baselines=baselines,
        horizon=HORIZON,
        train_window=200,
        embargo=HORIZON,
    )
    return replace(inputs, trust=trust), prices, signals


def _daily_returns(prices: dict[TargetRole, pd.Series]) -> pd.DataFrame:
    changes = {role: series.pct_change().dropna() for role, series in prices.items()}
    return pd.DataFrame(changes).dropna()


def test_a_zero_shock_through_a_saved_model_reproduces_its_own_prediction(tmp_path):
    db = tmp_path / "active.duckdb"
    frame, function = _saved_function(db)
    _, vix = last_observed(frame, VIX)

    result = run_scenario(function, frame, PRICE, {VIX: vix}, horizon=HORIZON, splitter=_splitter())

    assert result.shocked == pytest.approx(result.base)
    direct = evaluate(function, next_inputs(frame, PRICE, function.transforms, horizon=HORIZON))
    assert result.base == pytest.approx(direct)


def test_a_shock_flows_from_validation_through_evaluation_to_portfolio_impact(tmp_path):
    db = tmp_path / "active.duckdb"
    frame, function = _saved_function(db)

    levels, errors = validate([Shock(VIX, ShockKind.ABSOLUTE, "30")], frame, positive=(VIX,))
    assert errors == {}

    scenario = run_scenario(function, frame, PRICE, levels, horizon=HORIZON, splitter=_splitter())
    assert scenario.shocked != scenario.base

    solve_inputs, prices, _ = _real_solve_inputs(frame)
    breakdown = solve_inputs.solve()
    weights = {EQUITY: breakdown.equity, BOND: 1 - breakdown.equity}

    history = _daily_returns(prices)
    impact = portfolio_impact(
        history,
        {EQUITY: scenario.base, BOND: 0.0},
        {EQUITY: scenario.shocked, BOND: 0.0},
        weights,
        horizon=HORIZON,
        paths=2_000,
    )

    assert impact.weights.to_dict() == pytest.approx(weights)
    assert impact.return_change != 0.0


def test_the_held_weights_contract_survives_the_real_optimizer_path(tmp_path):
    db = tmp_path / "active.duckdb"
    frame, function = _saved_function(db)
    solve_inputs, prices, _ = _real_solve_inputs(frame)
    breakdown = solve_inputs.solve()
    weights = {EQUITY: breakdown.equity, BOND: 1 - breakdown.equity}
    history = _daily_returns(prices)

    big_shock = {EQUITY: -0.20, BOND: 0.0}  # large enough to tempt a re-solve
    impact = portfolio_impact(
        history, {EQUITY: 0.01, BOND: 0.0}, big_shock, weights, horizon=HORIZON, paths=2_000
    )

    assert impact.weights.to_dict() == pytest.approx(weights)


def test_reoptimising_under_a_negative_shock_moves_equity_below_the_held_weight(tmp_path):
    db = tmp_path / "active.duckdb"
    frame, function = _saved_function(db)
    solve_inputs, prices, _ = _real_solve_inputs(frame, trust=1.0)
    held_breakdown = solve_inputs.solve()
    weights = {EQUITY: held_breakdown.equity, BOND: 1 - held_breakdown.equity}
    history = _daily_returns(prices)

    impact = portfolio_impact(
        history,
        {EQUITY: 0.01, BOND: 0.0},
        {EQUITY: -0.03, BOND: 0.0},
        weights,
        horizon=HORIZON,
        paths=2_000,
        reoptimise=solve_inputs,
    )

    assert impact.reoptimised is not None
    assert impact.reoptimised.weights[EQUITY] != pytest.approx(held_breakdown.equity)
    assert impact.reoptimised.weights[EQUITY] < held_breakdown.equity


def test_an_extrapolating_shock_still_produces_a_usable_portfolio_impact(tmp_path):
    db = tmp_path / "active.duckdb"
    frame, function = _saved_function(db)
    _, vix = last_observed(frame, VIX)
    vix_history = frame[VIX]
    span = float(vix_history.max() - vix_history.min())
    # Well beyond the fitted fold's own training range, but inside MAX_RANGE_MULTIPLE
    # of the whole observed history, so validation accepts it as a stress, not a typo.
    extreme = vix + 4 * span
    assert abs(extreme - vix) < MAX_RANGE_MULTIPLE * span

    scenario = run_scenario(
        function, frame, PRICE, {VIX: extreme}, horizon=HORIZON, splitter=_splitter()
    )
    assert any(change.extrapolated for change in scenario.changes)

    solve_inputs, prices, _ = _real_solve_inputs(frame)
    breakdown = solve_inputs.solve()
    weights = {EQUITY: breakdown.equity, BOND: 1 - breakdown.equity}
    history = _daily_returns(prices)

    impact = portfolio_impact(
        history,
        {EQUITY: scenario.base, BOND: 0.0},
        {EQUITY: scenario.shocked, BOND: 0.0},
        weights,
        horizon=HORIZON,
        paths=2_000,
    )

    assert np.isfinite(impact.shocked.mean_return)
    assert np.isfinite(impact.shocked.volatility)


def test_a_shock_to_the_bond_research_models_own_signal_flows_through_to_portfolio_impact(
    tmp_path,
):
    """The bond research model is ``Origin.DERIVED`` under the hood, same as the
    equity research model, so it should shock and re-evaluate through the same
    generic scenario machinery the equity tests above already cover - with no
    train window, since it runs on an expanding window by design (FYP-32, FYP-33,
    the November 2026 bond research model)."""
    db = tmp_path / "active.duckdb"
    frame, function = _saved_bond_function(db)
    assert function.origin is Origin.DERIVED

    levels, errors = validate([Shock(VIX, ShockKind.ABSOLUTE, "30")], frame, positive=(VIX,))
    assert errors == {}

    scenario = run_scenario(
        function, frame, BOND_PRICE, levels, horizon=HORIZON, splitter=_splitter()
    )
    assert scenario.shocked != scenario.base

    solve_inputs, prices, _ = _real_solve_inputs(frame)
    breakdown = solve_inputs.solve()
    weights = {EQUITY: breakdown.equity, BOND: 1 - breakdown.equity}
    history = _daily_returns(prices)

    impact = portfolio_impact(
        history,
        {EQUITY: 0.0, BOND: scenario.base},
        {EQUITY: 0.0, BOND: scenario.shocked},
        weights,
        horizon=HORIZON,
        paths=2_000,
    )

    assert impact.weights.to_dict() == pytest.approx(weights)
    assert impact.return_change != 0.0


# --- a machine-learning model: its latest fold fitted again ---------------------------


def _ml_splitter() -> PurgedWalkForward:
    return PurgedWalkForward(train=60, test=10, embargo=HORIZON, tuning_rows=100)


def _saved_ml(db_path, role=EQUITY) -> tuple[pd.DataFrame, ModelRunResult, ActiveML]:
    """Run machine learning for real, set it active as the Models page does, and
    read back what its latest fold was fitted on."""
    frame = _frame()
    panel = align_and_lag(frame, list(TRANSFORMS), PRICE, horizon=HORIZON, transforms=TRANSFORMS)
    result, _description, tuning = run_boosted(
        panel, _ml_splitter(), n_trials=3, retune_trials=3
    )
    latest = tuning.latest
    set_active_model(
        role,
        "Machine Learning",
        result,
        high_risk=False,
        horizon=HORIZON,
        train_window=60,
        test_window=10,
        dataset_fingerprint="fp",
        ml=ActiveML(
            latest.library,
            latest.params,
            {s: TRANSFORMS[s] for s in latest.signals},
            latest.train_first,
            latest.train_last,
            latest.train_rows,
        ),
        db_path=db_path,
    )
    return frame, result, get_active_ml(role, db_path=db_path)


def test_an_ml_refit_of_the_latest_fold_reproduces_the_saved_latest_forecast(tmp_path):
    frame, result, ml = _saved_ml(tmp_path / "active.duckdb")
    assert ml is not None and ml.inputs

    panel = ml_panel(ml, frame, PRICE, horizon=HORIZON)
    fold = refit_latest_fold(ml, panel)
    saved = result.forecast.dropna()
    date = saved.index[-1]

    assert fold.predict(panel.frame.loc[date, list(ml.inputs)]) == saved.iloc[-1]


def test_a_zero_shock_to_an_ml_model_reproduces_its_base_case(tmp_path):
    frame, _result, ml = _saved_ml(tmp_path / "active.duckdb")
    signal = ml.inputs[0]
    _, level = last_observed(frame, signal)

    scenario = run_ml_scenario(ml, frame, PRICE, {signal: level}, horizon=HORIZON)

    assert scenario.shocked == scenario.base
    assert scenario.changes == ()


def test_a_shock_to_a_signal_an_ml_model_reads_moves_its_forecast_and_the_portfolio(tmp_path):
    frame, _result, ml = _saved_ml(tmp_path / "active.duckdb", role=BOND)
    assert VIX in ml.inputs
    low, high = frame[VIX].min(), frame[VIX].max()
    # Across the whole range VIX has covered, some level moves the trees' forecast.
    scenarios = [
        run_ml_scenario(ml, frame, PRICE, {VIX: level}, horizon=HORIZON)
        for level in np.linspace(low, high, 9)
    ]
    moved = next(s for s in scenarios if s.shocked != s.base)

    solve_inputs, prices, _ = _real_solve_inputs(frame)
    breakdown = solve_inputs.solve()
    weights = {EQUITY: breakdown.equity, BOND: 1 - breakdown.equity}
    impact = portfolio_impact(
        _daily_returns(prices),
        {EQUITY: 0.0, BOND: moved.base},
        {EQUITY: 0.0, BOND: moved.shocked},
        weights,
        horizon=HORIZON,
        paths=2_000,
    )

    assert impact.return_change != 0.0


def test_beyond_its_training_range_an_ml_model_forecasts_as_at_the_edge(tmp_path):
    frame, _result, ml = _saved_ml(tmp_path / "active.duckdb")
    panel = ml_panel(ml, frame, PRICE, horizon=HORIZON)
    fold = refit_latest_fold(ml, panel)
    signal = ml.inputs[0]
    low, high = fold.ranges[signal]
    base = next_inputs(frame, PRICE, ml.transforms, horizon=HORIZON)

    at_edge = fold.predict({**base, signal: high})
    beyond = fold.predict({**base, signal: high + 10 * (high - low)})

    assert beyond == at_edge


def test_an_ml_shock_beyond_the_training_range_is_reported_as_held_at_the_edge(tmp_path):
    frame, _result, ml = _saved_ml(tmp_path / "active.duckdb")
    assert VIX in ml.inputs
    _, last = last_observed(frame, VIX)

    scenario = run_ml_scenario(ml, frame, PRICE, {VIX: last + 200}, horizon=HORIZON)

    (change,) = [c for c in scenario.changes if c.signal == VIX]
    assert change.extrapolated
    assert change.clipped_to == change.trained_on[1]
