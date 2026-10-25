"""Every SQL query in docs/run-history.md runs against a real run history: the doc's
examples break here, not in a reader's hands, when the schema changes."""

import re
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.ingest.upload import accept_upload, frame_csv_bytes, frame_sha256
from forecasting_engine.reporting.model_metrics import ModelRunResult
from forecasting_engine.store.runs import record_run
from forecasting_engine.store.uploads import record_upload
from forecasting_engine.validation.crash import CrashDiagnostics

DOC = Path(__file__).resolve().parents[2] / "docs" / "run-history.md"
QUERIES = re.findall(r"```sql\n(.*?)```", DOC.read_text(encoding="utf-8"), flags=re.S)
DATA = pd.DataFrame({"Date": pd.bdate_range("2026-01-05", periods=4), "SPX": [1.0, 2, 3, 4]})


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("history")
    db = tmp / "test.duckdb"
    record_upload(accept_upload("merged.csv", frame_csv_bytes(DATA), uploads_dir=tmp), db_path=db)
    dates = pd.bdate_range("2026-01-05", periods=4)
    result = ModelRunResult(
        ic=0.05,
        oos_rank_ic=0.04,
        rmse=0.01,
        crash=CrashDiagnostics(recall=0.5, precision=0.5, f1=0.5, n_true_tail_days=2),
        signal_rank_ic=0.06,
        forecast=pd.Series([0.01, 0.005, -0.01, 0.02], index=dates),
        baseline=pd.Series(0.005, index=dates),
        realised=pd.Series([0.02, 0.0, -0.03, -0.01], index=dates),
    )
    start = datetime(2026, 10, 26, 9)
    for minutes, version in ((0, "aaa1111"), (10, "bbb2222")):
        record_run(
            role=TargetRole.EQUITY,
            model_name="Polynomial (derived)",
            started_at=start + timedelta(minutes=minutes),
            finished_at=start + timedelta(minutes=minutes, seconds=11),
            dataset_sha256=frame_sha256(DATA),
            config_key="same settings",
            settings={"horizon": 20},
            result=result,
            code_version=version,
            db_path=db,
        )
    record_run(
        role=TargetRole.EQUITY,
        model_name="FF5 Benchmark",
        started_at=start + timedelta(minutes=20),
        finished_at=start + timedelta(minutes=20, seconds=1),
        dataset_sha256=frame_sha256(DATA),
        config_key="other",
        settings={},
        error="no route to host",
        db_path=db,
    )
    return db


def test_the_doc_has_its_queries():
    assert len(QUERIES) == 5


@pytest.mark.parametrize("query", QUERIES, ids=[f"query-{i + 1}" for i in range(len(QUERIES))])
def test_every_documented_query_runs_and_returns_rows(db, query):
    with duckdb.connect(str(db), read_only=True) as conn:
        frame = conn.execute(query).df()

    assert not frame.empty


def test_the_lineage_query_finds_the_file_each_run_was_fitted_on(db):
    with duckdb.connect(str(db), read_only=True) as conn:
        frame = conn.execute(QUERIES[1]).df()

    assert set(frame["filename"]) == {"merged.csv"}


def test_the_hit_rate_query_counts_only_dates_with_a_view(db):
    with duckdb.connect(str(db), read_only=True) as conn:
        frame = conn.execute(QUERIES[4]).df()

    # Of 4 dates, one forecast equals its baseline; of the other 3, two call it right.
    (row,) = frame.itertuples()
    assert row.dates_with_a_view == 6  # 3 dates × 2 runs
    assert row.direction_hit_rate == pytest.approx(round(2 / 3, 3))
