"""Several Streamlit sessions are threads of one process, all opening the same DuckDB
file. Opening it must not make them fail each other."""

import sys
import threading
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.portfolio.backtest import run_backtest
from forecasting_engine.store import _db
from forecasting_engine.store.portfolio_runs import (
    PortfolioInputs,
    list_portfolio_runs,
    portfolio_runs_table,
    record_portfolio_run,
)

SESSIONS = 8
EQUITY, BOND = TargetRole.EQUITY, TargetRole.BOND


def _inputs(risk_level: int) -> PortfolioInputs:
    return PortfolioInputs(
        dataset_sha256="a" * 64,
        equity_run_id="run-e",
        bond_run_id="run-b",
        equity_set_at=datetime(2026, 11, 1, 9),
        bond_set_at=datetime(2026, 11, 1, 9, 5),
        horizon=20,
        train_window=504,
        test_window=20,
        embargo=20,
        risk_level=risk_level,
        bounds=(0.2, 0.8),
        costs_bps={"equity": 3.0, "bond": 5.0},
        code_version="abc1234",
    )


def _backtest():
    idx = pd.bdate_range("2024-01-31", periods=60)
    rng = np.random.default_rng(0)
    prices = {
        role: pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, len(idx))), index=idx)
        for role in (EQUITY, BOND)
    }
    schedule = pd.DataFrame({EQUITY: 0.5, BOND: 0.5}, index=idx[::20])
    return run_backtest(prices, schedule), schedule


RESULT, SCHEDULE = _backtest()


@pytest.fixture
def db(tmp_path):
    return tmp_path / "data" / "test.duckdb"


def _all_at_once(work) -> list[BaseException]:
    """Run ``work(i)`` in ``SESSIONS`` threads released together; their exceptions."""
    start = threading.Barrier(SESSIONS)
    errors: list[BaseException] = []

    def session(i: int) -> None:
        start.wait()
        try:
            work(i)
        except BaseException as exc:  # noqa: BLE001 - collected for the assertion
            errors.append(exc)

    threads = [threading.Thread(target=session, args=(i,)) for i in range(SESSIONS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return errors


def test_sessions_reading_at_once_do_not_fail_each_other(db):
    list_portfolio_runs(db_path=db)  # the file exists, as it does once the app has run

    for _ in range(3):
        assert _all_at_once(lambda _: list_portfolio_runs(db_path=db)) == []


def test_sessions_recording_runs_at_once_each_get_their_own_number(db):
    list_portfolio_runs(db_path=db)
    inputs = [_inputs(i) for i in range(SESSIONS)]

    errors = _all_at_once(lambda i: record_portfolio_run(inputs[i], RESULT, SCHEDULE, db_path=db))

    assert errors == []
    numbers = portfolio_runs_table(db_path=db)["run_number"]
    assert sorted(numbers) == list(range(1, SESSIONS + 1))


def test_a_database_deleted_while_the_app_runs_has_its_tables_made_again(db):
    list_portfolio_runs(db_path=db)
    db.unlink()

    assert list_portfolio_runs(db_path=db) == []


# --- the app holds the database open between uses ---------------------------------------


@pytest.fixture
def kept_open():
    """``_db.keep_open()`` for one test, with nothing left held after it."""
    _db.keep_open()
    yield
    _db.release()


def _opens(monkeypatch) -> list[str]:
    """Every time the database file itself is opened from here on."""
    opened: list[str] = []
    real = _db.duckdb.connect

    def counting(path, *args, **kwargs):
        opened.append(str(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(_db.duckdb, "connect", counting)
    return opened


def test_a_script_opens_the_file_around_each_use_and_holds_nothing(db, monkeypatch):
    opened = _opens(monkeypatch)

    for _ in range(3):
        list_portfolio_runs(db_path=db)

    assert len(opened) == 3
    assert _db._held == {}


def test_the_app_opens_the_file_once_however_often_it_reads(db, monkeypatch, kept_open):
    opened = _opens(monkeypatch)

    for _ in range(5):
        list_portfolio_runs(db_path=db)

    assert len(opened) == 1


def test_a_run_recorded_while_the_file_is_held_is_read_back(db, kept_open):
    record_portfolio_run(_inputs(3), RESULT, SCHEDULE, db_path=db)

    assert [run.inputs["risk_level"] for run in list_portfolio_runs(db_path=db)] == [3]


def test_sessions_sharing_the_held_file_do_not_fail_each_other(db, kept_open):
    list_portfolio_runs(db_path=db)
    inputs = [_inputs(i) for i in range(SESSIONS)]

    errors = _all_at_once(lambda i: record_portfolio_run(inputs[i], RESULT, SCHEDULE, db_path=db))

    assert errors == []
    assert _all_at_once(lambda _: list_portfolio_runs(db_path=db)) == []
    assert sorted(portfolio_runs_table(db_path=db)["run_number"]) == list(range(1, SESSIONS + 1))


def test_releasing_writes_everything_back_and_lets_go_of_the_file(db):
    _db.keep_open()
    record_portfolio_run(_inputs(2), RESULT, SCHEDULE, db_path=db)

    _db.release()

    assert _db._held == {}
    # Nothing holds the file now, so it can be opened afresh and has the run in it.
    with _db.duckdb.connect(str(db), read_only=True) as conn:
        assert conn.execute("SELECT count(*) FROM portfolio_runs").fetchone() == (1,)


@pytest.mark.skipif(sys.platform == "win32", reason="Windows won't delete a file held open")
def test_a_held_database_deleted_while_the_app_runs_is_made_again(db, kept_open):
    record_portfolio_run(_inputs(1), RESULT, SCHEDULE, db_path=db)
    db.unlink()

    assert list_portfolio_runs(db_path=db) == []


def test_a_conflict_that_persists_is_raised_not_swallowed(monkeypatch):
    calls = []

    def always_conflicts():
        calls.append(1)
        raise _db.duckdb.TransactionException("write-write conflict")

    monkeypatch.setattr(_db, "_BACKOFF_SECONDS", 0.0)
    with pytest.raises(_db.duckdb.TransactionException):
        _db.retry_on_conflict(always_conflicts)
    assert len(calls) == _db.WRITE_ATTEMPTS
