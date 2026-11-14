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
    next_inputs,
    run_scenario,
    validate,
)
from forecasting_engine.store.active_model import (
    ActiveFunction,
    get_active_function,
    set_active_model,
)
from forecasting_engine.validation.crash import CrashDiagnostics
from forecasting_engine.validation.splitters import PurgedWalkForward

EQUITY, BOND = TargetRole.EQUITY, TargetRole.BOND
PRICE, VIX, RATE = "SPX", "VIX", "USGG10YR"
TRANSFORMS = {VIX: Transform.LEVEL, RATE: Transform.DIFFERENCE}
HORIZON = 5


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


def test_reoptimising_under_a_real_shock_moves_the_schedule_the_way_fyp_246_promises(tmp_path):
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
