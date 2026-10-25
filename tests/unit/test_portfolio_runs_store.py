"""Portfolio runs: one row per distinct set of inputs, however often it is computed."""

from dataclasses import replace
from datetime import datetime

import duckdb
import numpy as np
import pandas as pd
import pytest

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.portfolio.backtest import run_backtest
from forecasting_engine.store.portfolio_runs import (
    PortfolioInputs,
    list_portfolio_runs,
    portfolio_run_weights,
    portfolio_runs_table,
    record_portfolio_run,
)

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
    risk_aversion=10.0,
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
        {"risk_aversion": 30.0},
        {"bounds": (0.3, 0.7)},
        {"costs_bps": {"equity": 3.0, "bond": 6.0}},
        {"code_version": "def5678"},
    ],
    ids=lambda change: next(iter(change)),
)
def test_any_one_input_changing_gives_another_key(change):
    assert replace(INPUTS, **change).run_key != INPUTS.run_key


# --- idempotent recording ----------------------------------------------------------------


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

    assert _record(db, replace(INPUTS, risk_aversion=30.0)) is True
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

    assert row.risk_aversion == 10.0 and row.min_weight == 0.2
    assert row.net_sharpe == pytest.approx(RESULT.metrics[("optimised", "net")].sharpe)
    assert row.baseline_net_sharpe == pytest.approx(RESULT.metrics[("baseline", "net")].sharpe)
