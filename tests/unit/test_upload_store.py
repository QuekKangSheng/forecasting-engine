"""The DuckDB upload event log."""

from datetime import datetime

import pytest

from forecasting_engine.ingest.upload import accept_upload
from forecasting_engine.store.uploads import find_upload, recent_uploads, record_upload

CSV = b"date,spx_close\n2026-01-02,4750.5\n2026-01-03,4762.1\n"
OTHER_CSV = b"date,spx_close\n2026-02-02,4800.0\n"


@pytest.fixture
def db(tmp_path):
    return tmp_path / "history" / "test.duckdb"


@pytest.fixture
def accepted(tmp_path):
    return accept_upload("signals.csv", CSV, uploads_dir=tmp_path / "uploads")


def test_reading_an_absent_database_creates_it_and_returns_nothing(db):
    assert recent_uploads(db_path=db) == []
    assert db.exists()


def test_an_upload_round_trips_every_field(db, accepted):
    at = datetime(2026, 8, 31, 9, 30, 0)
    record_upload(accepted, db_path=db, uploaded_at=at)

    (row,) = recent_uploads(db_path=db)
    assert row.filename == "signals.csv"
    assert row.sha256 == accepted.source.sha256
    assert row.size_bytes == len(CSV)
    assert row.row_count == 2
    assert row.uploaded_at == at


def test_two_uploads_produce_two_rows(db, accepted, tmp_path):
    other = accept_upload("other.csv", OTHER_CSV, uploads_dir=tmp_path / "uploads")
    record_upload(accepted, db_path=db)
    record_upload(other, db_path=db)

    assert len(recent_uploads(db_path=db)) == 2


def test_the_same_file_uploaded_twice_is_logged_twice(db, accepted):
    # The table is an event log of attempts, not a set of unique files.
    record_upload(accepted, db_path=db)
    record_upload(accepted, db_path=db)

    rows = recent_uploads(db_path=db)
    assert len(rows) == 2
    assert {row.sha256 for row in rows} == {accepted.source.sha256}


def test_the_newest_upload_comes_first(db, accepted, tmp_path):
    other = accept_upload("other.csv", OTHER_CSV, uploads_dir=tmp_path / "uploads")
    record_upload(accepted, db_path=db, uploaded_at=datetime(2026, 8, 30, 12, 0))
    record_upload(other, db_path=db, uploaded_at=datetime(2026, 8, 31, 12, 0))

    assert [row.filename for row in recent_uploads(db_path=db)] == ["other.csv", "signals.csv"]


def test_the_limit_caps_how_many_rows_come_back(db, accepted):
    for _ in range(5):
        record_upload(accepted, db_path=db)

    assert len(recent_uploads(limit=3, db_path=db)) == 3


def test_a_content_hash_finds_the_first_upload_of_that_content(db, accepted):
    record_upload(accepted, db_path=db, uploaded_at=datetime(2026, 8, 31, 12, 0))
    record_upload(accepted, db_path=db, uploaded_at=datetime(2026, 8, 30, 12, 0))

    found = find_upload(accepted.source.sha256, db_path=db)

    assert found.filename == "signals.csv"
    assert found.uploaded_at == datetime(2026, 8, 30, 12, 0)


def test_an_unknown_hash_finds_nothing(db, accepted):
    record_upload(accepted, db_path=db)

    assert find_upload("0" * 64, db_path=db) is None
