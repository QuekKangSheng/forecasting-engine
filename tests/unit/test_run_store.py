"""The DuckDB run history: every fit as an event, with its metrics and forecasts."""

import math
from datetime import datetime, timedelta

import duckdb
import numpy as np
import pandas as pd
import pytest

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.ingest.upload import accept_upload, frame_csv_bytes, frame_sha256
from forecasting_engine.reporting.model_metrics import ModelRunResult
from forecasting_engine.store.runs import (
    DONE,
    FAILED,
    HEADLINE_METRICS,
    code_version,
    get_run,
    list_runs,
    metrics_of,
    record_run,
    run_forecasts,
    run_metrics,
    runs_table,
)
from forecasting_engine.validation.crash import CrashDiagnostics

EQUITY, BOND = TargetRole.EQUITY, TargetRole.BOND
T0 = datetime(2026, 10, 26, 9, 0, 0)
DATES = pd.bdate_range("2026-01-05", periods=5)


@pytest.fixture
def db(tmp_path):
    return tmp_path / "data" / "test.duckdb"


def _result(**overrides) -> ModelRunResult:
    fields = {
        "ic": 0.05,
        "oos_rank_ic": 0.04,
        "rmse": 0.012,
        "crash": CrashDiagnostics(recall=0.5, precision=0.25, f1=1 / 3, n_true_tail_days=8),
        "signal_rank_ic": 0.06,
        "signal_rank_ic_se": 0.02,
        "signal_rank_ic_se_test": 0.03,
        "forecast": pd.Series([0.01, 0.02, np.nan, -0.01, 0.0], index=DATES),
        "baseline": pd.Series(0.005, index=DATES),
        "realised": pd.Series([0.02, -0.01, 0.03, 0.0, 0.01], index=DATES),
    }
    return ModelRunResult(**{**fields, **overrides})


def _record(db, *, role=EQUITY, name="Polynomial (derived)", started=T0, **overrides):
    kwargs = {
        "role": role,
        "model_name": name,
        "started_at": started,
        "finished_at": started + timedelta(seconds=42),
        "dataset_sha256": "a" * 64,
        "config_key": "k" * 32,
        "settings": {"horizon": 20, "model_setting": ("x * y", (("x", "VIX"),))},
        "code_version": "abc1234",
        "db_path": db,
    }
    if "error" not in overrides:
        kwargs["result"] = _result()
    return record_run(**{**kwargs, **overrides})


# --- one run, written and read back ----------------------------------------------------


def test_a_finished_run_reads_back_as_it_was_recorded(db):
    written = _record(db)

    read = get_run(written.run_id, db_path=db)

    assert read == written
    assert read.status == DONE and read.error is None
    assert read.seconds == 42


def test_settings_round_trip_through_json_with_tuples_as_lists(db):
    run = _record(db)

    assert get_run(run.run_id, db_path=db).settings == {
        "horizon": 20,
        "model_setting": ["x * y", [["x", "VIX"]]],
    }


def test_every_metric_is_stored_one_row_each(db):
    run = _record(db)

    stored = run_metrics(run.run_id, db_path=db)

    assert stored == pytest.approx(metrics_of(_result()))
    assert stored["signal_t"] == pytest.approx(0.06 / 0.03)  # the wider standard error
    assert stored["crash_tail_days"] == 8


def test_a_metric_the_run_did_not_compute_is_absent_not_nan(db):
    run = _record(db, result=_result(rows_scored=None, oos_rank_ic_se=None))

    stored = run_metrics(run.run_id, db_path=db)

    assert "rows_scored" not in stored and "oos_rank_ic_se" not in stored


def test_forecasts_baselines_and_realised_returns_are_kept_per_date(db):
    run = _record(db)

    frame = run_forecasts(run.run_id, db_path=db)

    assert list(frame.index) == list(DATES)
    assert frame["predicted"].iloc[0] == 0.01
    assert math.isnan(frame["predicted"].iloc[2])  # stored as NULL, read as NaN
    assert (frame["baseline"] == 0.005).all()
    assert frame["realised"].tolist() == [0.02, -0.01, 0.03, 0.0, 0.01]


def test_nan_is_stored_as_sql_null(db):
    run = _record(db)

    with duckdb.connect(str(db)) as conn:
        nulls = conn.execute(
            "SELECT count(*) FROM run_forecasts WHERE run_id = ? AND predicted IS NULL",
            [run.run_id],
        ).fetchone()[0]

    assert nulls == 1


# --- failures are history too ----------------------------------------------------------


def test_a_failed_run_is_recorded_with_its_error_and_nothing_else(db):
    run = _record(db, error="not enough complete training rows")

    read = get_run(run.run_id, db_path=db)
    assert read.status == FAILED and read.error == "not enough complete training rows"
    assert run_metrics(run.run_id, db_path=db) == {}
    assert run_forecasts(run.run_id, db_path=db).empty


@pytest.mark.parametrize(
    ("error", "result"), [("boom", _result()), (None, None)], ids=["both", "neither"]
)
def test_a_run_has_a_result_or_an_error_never_both_or_neither(db, error, result):
    with pytest.raises(ValueError, match="either a result or an error"):
        _record(db, error=error, result=result)


def test_the_status_column_refuses_anything_but_done_or_failed(db):
    _record(db)  # creates the tables

    with duckdb.connect(str(db)) as conn, pytest.raises(duckdb.ConstraintException):
        conn.execute(
            "INSERT INTO runs VALUES ('x', 'equity', 'm', 'running', NULL, now(), now(), "
            "'s', 'k', '{}', NULL)"
        )


# --- every run is one transaction ------------------------------------------------------


def test_a_run_whose_forecasts_fail_to_write_leaves_nothing_behind(db):
    _record(db)  # one good run already there
    duplicated = pd.Series([0.01, 0.02], index=[DATES[0], DATES[0]])

    with pytest.raises(duckdb.ConstraintException):
        _record(db, run_id="broken", result=_result(forecast=duplicated, baseline=None,
                                                     realised=None))

    assert get_run("broken", db_path=db) is None
    assert run_metrics("broken", db_path=db) == {}
    assert len(list_runs(db_path=db)) == 1


def test_a_run_id_cannot_be_reused(db):
    _record(db, run_id="same")

    with pytest.raises(duckdb.ConstraintException):
        _record(db, run_id="same")


# --- reading the history ---------------------------------------------------------------


def test_runs_are_listed_newest_first_and_can_be_filtered_by_target(db):
    first = _record(db, started=T0)
    second = _record(db, role=BOND, started=T0 + timedelta(hours=1))
    third = _record(db, started=T0 + timedelta(hours=2))

    assert [r.run_id for r in list_runs(db_path=db)] == [
        third.run_id,
        second.run_id,
        first.run_id,
    ]
    assert [r.run_id for r in list_runs(role=EQUITY, db_path=db)] == [
        third.run_id,
        first.run_id,
    ]


def test_the_same_settings_run_twice_is_two_events(db):
    _record(db, started=T0)
    _record(db, started=T0 + timedelta(minutes=5))

    runs = list_runs(db_path=db)
    assert len(runs) == 2 and runs[0].config_key == runs[1].config_key


def test_the_runs_table_puts_headline_metrics_beside_each_run(db):
    done = _record(db, started=T0)
    failed = _record(db, started=T0 + timedelta(hours=1), error="boom")

    table = runs_table(db_path=db)

    assert list(table["run_id"]) == [failed.run_id, done.run_id]
    assert set(HEADLINE_METRICS) <= set(table.columns)
    row = table.set_index("run_id").loc[done.run_id]
    assert row["signal_rank_ic"] == pytest.approx(0.06)
    assert row["seconds"] == pytest.approx(42)
    assert table.set_index("run_id").loc[failed.run_id, ["ic", "rmse"]].isna().all()


def test_an_empty_history_reads_as_empty(db):
    assert list_runs(db_path=db) == []
    assert runs_table(db_path=db).empty


# --- lineage -----------------------------------------------------------------------------


def test_a_runs_dataset_hash_is_the_hash_the_committed_file_is_stored_under(tmp_path):
    frame = pd.DataFrame(
        {"Date": pd.bdate_range("2026-01-05", periods=3), "SPX": [1.0, 2.0, 3.0]}
    )

    accepted = accept_upload("merged.csv", frame_csv_bytes(frame), uploads_dir=tmp_path)

    assert frame_sha256(frame) == accepted.source.sha256
    assert accepted.source.path == tmp_path / f"{frame_sha256(frame)}.csv"


def test_one_corrected_value_gives_a_different_dataset_hash():
    frame = pd.DataFrame({"Date": pd.bdate_range("2026-01-05", periods=3), "SPX": [1.0, 2.0, 3.0]})
    corrected = frame.assign(SPX=[1.0, 2.5, 3.0])

    assert frame_sha256(frame) != frame_sha256(corrected)


def test_the_code_version_is_this_checkouts_commit():
    version = code_version()

    # A short commit (abc1234), or tag-relative once the repo is tagged
    # (v1.0-3-gabc1234), either with "-dirty" if the checkout had uncommitted changes.
    assert version is None or (version and " " not in version)
