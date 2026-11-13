"""Portfolio runs: one row per distinct set of inputs, however often it is computed."""

from dataclasses import fields, replace
from datetime import datetime

import duckdb
import numpy as np
import pandas as pd
import pytest

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.portfolio.backtest import run_backtest
from forecasting_engine.reporting.model_metrics import ModelRunResult
from forecasting_engine.store.active_model import set_active_model
from forecasting_engine.store.portfolio_runs import (
    PortfolioInputs,
    get_saved_run,
    has_results,
    latest_simulation,
    list_portfolio_runs,
    portfolio_run_weights,
    portfolio_runs_table,
    record_portfolio_run,
    rename_portfolio_run,
    run_number,
    save_simulation,
    saved_paths,
)
from forecasting_engine.validation.crash import CrashDiagnostics

EQUITY, BOND = TargetRole.EQUITY, TargetRole.BOND

INPUTS = PortfolioInputs(
    dataset_sha256="a" * 64,
    equity_run_id="run-e",
    bond_run_id="run-b",
    equity_set_at=datetime(2026, 10, 20, 9),
    bond_set_at=datetime(2026, 10, 20, 9, 5),
    horizon=20,
    train_window=504,
    test_window=20,
    embargo=20,
    risk_level=3,
    bounds=(0.2, 0.8),
    costs_bps={"equity": 3.0, "bond": 5.0},
    code_version="abc1234",
)


@pytest.fixture
def db(tmp_path):
    return tmp_path / "data" / "test.duckdb"


def _backtest():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2024-01-31", periods=120)
    prices = {
        EQUITY: pd.Series(100 * np.cumprod(1 + rng.normal(0.001, 0.01, 120)), index=idx),
        BOND: pd.Series(100 * np.cumprod(1 + rng.normal(0.0002, 0.003, 120)), index=idx),
    }
    dates = idx[::20]
    schedule = pd.DataFrame(
        {EQUITY: np.linspace(0.4, 0.7, len(dates)), BOND: 1 - np.linspace(0.4, 0.7, len(dates))},
        index=dates,
    )
    return run_backtest(prices, schedule), schedule


RESULT, SCHEDULE = _backtest()


def _record(db, inputs=INPUTS, schedule=SCHEDULE):
    return record_portfolio_run(inputs, RESULT, schedule, db_path=db)


def _count(db, table):
    with duckdb.connect(str(db)) as conn:
        return conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


# --- the key ---------------------------------------------------------------------------


def test_the_same_inputs_always_give_the_same_key():
    assert INPUTS.run_key == replace(INPUTS).run_key
    assert len(INPUTS.run_key) == 64


def test_the_key_does_not_depend_on_the_order_costs_were_given_in():
    reordered = replace(INPUTS, costs_bps={"bond": 5.0, "equity": 3.0})

    assert reordered.run_key == INPUTS.run_key


@pytest.mark.parametrize(
    "change",
    [
        {"dataset_sha256": "b" * 64},
        {"equity_run_id": "another run"},
        {"bond_run_id": None},
        {"equity_set_at": datetime(2026, 10, 21, 9)},
        {"horizon": 5},
        {"train_window": 252},
        {"test_window": 10},
        {"embargo": 5},
        {"risk_level": 5},
        {"bounds": (0.3, 0.7)},
        {"costs_bps": {"equity": 3.0, "bond": 6.0}},
        {"code_version": "def5678"},
    ],
    ids=lambda change: next(iter(change)),
)
def test_any_one_input_changing_gives_another_key(change):
    assert replace(INPUTS, **change).run_key != INPUTS.run_key


# --- idempotent recording ----------------------------------------------------------------


def test_what_follows_from_the_data_and_models_is_not_an_input():
    inputs = {field.name for field in fields(PortfolioInputs)}

    assert "risk_level" in inputs
    assert not inputs & {"trust", "beta", "risk_aversion", "lambda", "premium", "tilt_strength"}


def test_the_first_record_of_a_key_is_new_and_every_repeat_is_not(db):
    assert _record(db) is True
    assert _record(db) is False
    assert _record(db) is False

    assert _count(db, "portfolio_runs") == 1


def test_a_repeat_writes_no_metrics_or_weights_either(db):
    _record(db)
    metrics, weights = _count(db, "portfolio_run_metrics"), _count(db, "portfolio_run_weights")

    _record(db)

    assert _count(db, "portfolio_run_metrics") == metrics
    assert _count(db, "portfolio_run_weights") == weights


def test_new_inputs_are_a_new_run(db):
    _record(db)

    assert _record(db, replace(INPUTS, risk_level=5)) is True
    assert _count(db, "portfolio_runs") == 2


def test_a_run_that_fails_part_way_leaves_nothing_so_a_retry_records_it(db):
    doubled = pd.concat([SCHEDULE, SCHEDULE.iloc[:1]])  # a rebalance date twice

    with pytest.raises(duckdb.ConstraintException):
        _record(db, schedule=doubled)

    assert _count(db, "portfolio_runs") == 0
    assert _record(db) is True  # the key isn't left behind blocking the retry


# --- what is kept --------------------------------------------------------------------------


def test_every_metric_of_both_portfolios_on_both_bases_is_kept(db):
    _record(db)

    with duckdb.connect(str(db)) as conn:
        kept = conn.execute(
            "SELECT portfolio, basis, metric, value FROM portfolio_run_metrics"
        ).df()
    sharpe = kept.query("portfolio == 'optimised' and basis == 'net' and metric == 'sharpe'")
    assert sharpe["value"].item() == pytest.approx(RESULT.metrics[("optimised", "net")].sharpe)
    assert set(kept["portfolio"]) == {"optimised", "baseline"}
    assert set(kept["basis"]) == {"gross", "net"}
    assert {"information_ratio", "tracking_error"} <= set(kept["metric"])


def test_the_weight_schedule_reads_back_as_it_was(db):
    _record(db)

    weights = portfolio_run_weights(INPUTS.run_key, db_path=db)

    assert list(weights.index) == list(SCHEDULE.index)
    assert weights.to_numpy() == pytest.approx(SCHEDULE[EQUITY].to_numpy())


def test_a_run_reads_back_with_its_inputs(db):
    _record(db)

    (run,) = list_portfolio_runs(db_path=db)

    assert run.run_key == INPUTS.run_key
    assert (run.equity_run_id, run.bond_run_id) == ("run-e", "run-b")
    assert run.inputs["bounds"] == [0.2, 0.8]
    assert run.inputs["equity_set_at"] == "2026-10-20T09:00:00"
    assert run.start_date == RESULT.start.date() and run.end_date == RESULT.end.date()


def test_the_table_pulls_the_inputs_that_vary_and_both_net_sharpes(db):
    _record(db)

    (row,) = portfolio_runs_table(db_path=db).itertuples()

    assert row.risk_level == 3 and row.min_weight == 0.2
    assert row.net_sharpe == pytest.approx(RESULT.metrics[("optimised", "net")].sharpe)
    assert row.baseline_net_sharpe == pytest.approx(RESULT.metrics[("baseline", "net")].sharpe)


# --- FYP-275: run numbers, names and saved results ---------------------------------


RESULTS = {"bases": {"net": {"comparison": [{"metric": "Sharpe", "optimised": "1.10"}]}}}


def _numbers(db):
    table = portfolio_runs_table(db_path=db)
    return dict(zip(table["run_key"], table["run_number"], strict=True))


def test_runs_are_numbered_in_the_order_they_were_first_recorded(db):
    second, third = replace(INPUTS, risk_level=5), replace(INPUTS, risk_level=1)
    for inputs in (INPUTS, second, third):
        _record(db, inputs)
    _record(db, INPUTS)  # shown again: still run 1, and no number used up

    assert _numbers(db) == {INPUTS.run_key: 1, second.run_key: 2, third.run_key: 3}
    assert list(portfolio_runs_table(db_path=db)["run_number"]) == [3, 2, 1]


def test_runs_recorded_before_there_were_numbers_get_them_oldest_first(db):
    _record(db)
    newer = replace(INPUTS, risk_level=5)
    _record(db, newer)
    with duckdb.connect(str(db)) as conn:
        conn.execute("DROP TABLE portfolio_run_labels")

    assert _numbers(db) == {INPUTS.run_key: 1, newer.run_key: 2}
    third = replace(INPUTS, risk_level=1)
    _record(db, third)
    assert _numbers(db)[third.run_key] == 3


def test_a_run_can_be_named_renamed_and_have_its_name_cleared(db):
    _record(db)
    assert portfolio_runs_table(db_path=db)["name"].isna().all()

    rename_portfolio_run(INPUTS.run_key, "  Before the rate cut  ", db_path=db)
    assert list(portfolio_runs_table(db_path=db)["name"]) == ["Before the rate cut"]

    rename_portfolio_run(INPUTS.run_key, "Second try", db_path=db)
    assert list(portfolio_runs_table(db_path=db)["name"]) == ["Second try"]

    rename_portfolio_run(INPUTS.run_key, "   ", db_path=db)
    assert portfolio_runs_table(db_path=db)["name"].isna().all()
    assert _numbers(db) == {INPUTS.run_key: 1}, "a name is not part of a run's identity"


def test_the_table_names_each_runs_models_even_after_others_are_set(db):
    forecast = pd.Series([0.01], index=pd.bdate_range("2024-01-01", periods=1))
    result = ModelRunResult(
        ic=0.1,
        oos_rank_ic=0.1,
        rmse=0.1,
        crash=CrashDiagnostics(recall=0.5, precision=0.5, f1=0.5, n_true_tail_days=1),
        forecast=forecast,
    )
    settings = {
        "high_risk": False,
        "horizon": 20,
        "train_window": 504,
        "test_window": 20,
        "dataset_fingerprint": "fp",
        "db_path": db,
    }
    set_active_model(EQUITY, "Polynomial", result, set_at=INPUTS.equity_set_at, **settings)
    set_active_model(BOND, "FF5 Benchmark", result, set_at=INPUTS.bond_set_at, **settings)
    _record(db)
    set_active_model(EQUITY, "Machine Learning", result, **settings)

    (row,) = portfolio_runs_table(db_path=db).itertuples()
    assert row.equity_model == "Polynomial"
    assert row.bond_model == "FF5 Benchmark"


def test_saved_results_come_back_whole_under_their_run(db):
    assert not has_results(INPUTS.run_key, db_path=db)
    record_portfolio_run(INPUTS, RESULT, SCHEDULE, results=RESULTS, db_path=db)

    assert has_results(INPUTS.run_key, db_path=db)
    run = get_saved_run(INPUTS.run_key, db_path=db)
    assert run.results == RESULTS
    assert (run.run_number, run.name, run.code_version) == (1, None, "abc1234")


def test_a_run_recorded_before_results_were_saved_cannot_be_opened_until_shown_again(db):
    _record(db)
    assert get_saved_run(INPUTS.run_key, db_path=db) is None
    assert not portfolio_runs_table(db_path=db)["has_results"].any()
    assert saved_paths(INPUTS.run_key, db_path=db) is None

    assert not record_portfolio_run(INPUTS, RESULT, SCHEDULE, results=RESULTS, db_path=db)

    assert get_saved_run(INPUTS.run_key, db_path=db).results == RESULTS
    assert portfolio_runs_table(db_path=db)["has_results"].all()


def test_a_runs_number_is_read_without_its_results(db):
    """What the Portfolio page asks on every rerun: is this run recorded with its
    results, and under which number. One row, not the whole saved run."""
    other = replace(INPUTS, risk_level=1)
    assert run_number(INPUTS.run_key, db_path=db) is None

    _record(db)  # recorded, but before results were kept
    record_portfolio_run(other, RESULT, SCHEDULE, results=RESULTS, db_path=db)

    assert run_number(INPUTS.run_key, db_path=db) == 1
    assert run_number(INPUTS.run_key, with_results=True, db_path=db) is None
    assert run_number(other.run_key, with_results=True, db_path=db) == 2
    assert run_number("no such run", db_path=db) is None

    record_portfolio_run(INPUTS, RESULT, SCHEDULE, results=RESULTS, db_path=db)
    assert run_number(INPUTS.run_key, with_results=True, db_path=db) == 1


def test_the_first_results_saved_for_a_run_are_kept(db):
    record_portfolio_run(INPUTS, RESULT, SCHEDULE, results=RESULTS, db_path=db)
    record_portfolio_run(INPUTS, RESULT, SCHEDULE, results={"other": 1}, db_path=db)

    assert get_saved_run(INPUTS.run_key, db_path=db).results == RESULTS
    assert _count(db, "portfolio_run_daily") == len(RESULT.optimised.gross)


def test_saved_paths_give_back_exactly_the_returns_that_were_recorded(db):
    record_portfolio_run(INPUTS, RESULT, SCHEDULE, results=RESULTS, db_path=db)

    saved = saved_paths(INPUTS.run_key, db_path=db)

    for basis in ("gross", "net"):
        for portfolio in ("optimised", "baseline"):
            mine, theirs = getattr(saved, portfolio), getattr(RESULT, portfolio)
            np.testing.assert_array_equal(getattr(mine, basis), getattr(theirs, basis))
    assert saved.optimised.net.index.equals(RESULT.optimised.net.index)
    for role in (EQUITY, BOND):
        np.testing.assert_array_equal(saved.index_returns[role], RESULT.index_returns[role])
    assert saved.start == RESULT.start and saved.end == RESULT.end
    np.testing.assert_allclose(saved.optimised.weights[EQUITY], SCHEDULE[EQUITY])


@pytest.mark.parametrize(
    "settings, rate",
    [({"risk_free_rate": 0.04}, 0.04), ({}, 0.0)],
    ids=["kept with the run", "saved before it was kept"],
)
def test_saved_paths_carry_the_risk_free_rate_the_run_was_shown_with(db, settings, rate):
    # The charts of an opened run are redrawn from its daily returns, and a rolling
    # Sharpe depends on the rate: a later change to the app's rate must not redraw it.
    results = {**RESULTS, "settings": settings}
    record_portfolio_run(INPUTS, RESULT, SCHEDULE, results=results, db_path=db)

    assert saved_paths(INPUTS.run_key, db_path=db).risk_free_rate == rate


def test_results_and_daily_returns_are_all_or_nothing(db):
    with pytest.raises(TypeError):
        record_portfolio_run(INPUTS, RESULT, SCHEDULE, results={"bad": object()}, db_path=db)

    assert _count(db, "portfolio_runs") == 0
    assert _count(db, "portfolio_run_labels") == 0
    assert _count(db, "portfolio_run_daily") == 0


def test_simulations_are_kept_per_method_and_paths_with_the_latest_returned(db):
    _record(db)
    assert latest_simulation(INPUTS.run_key, db_path=db) is None
    first = {"method": "bootstrap", "paths": 1000, "rows": ["a"]}
    second = {"method": "parametric", "paths": 1000, "rows": ["b"]}

    save_simulation(INPUTS.run_key, first, recorded_at=datetime(2026, 10, 21), db_path=db)
    save_simulation(INPUTS.run_key, second, recorded_at=datetime(2026, 10, 22), db_path=db)
    save_simulation(INPUTS.run_key, first, recorded_at=datetime(2026, 10, 20), db_path=db)

    assert latest_simulation(INPUTS.run_key, db_path=db) == second
    assert _count(db, "portfolio_run_simulations") == 2
