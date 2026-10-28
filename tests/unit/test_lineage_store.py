"""Which exports each committed dataset was merged from."""

from datetime import datetime

import duckdb
import pandas as pd
import pytest

from forecasting_engine.ingest.provenance import SourceFile
from forecasting_engine.ingest.upload import AcceptedUpload, keep_source
from forecasting_engine.store.lineage import (
    SIGNAL,
    TARGET,
    record_dataset,
    record_merge,
    sources_of,
)
from forecasting_engine.store.uploads import recent_uploads

WHEN = datetime(2026, 10, 26, 10, 0)


@pytest.fixture
def db(tmp_path):
    return tmp_path / "data" / "test.duckdb"


def _upload(name: str, data: bytes, rows: int = 3) -> AcceptedUpload:
    return AcceptedUpload(frame=pd.DataFrame({"x": range(rows)}), source=SourceFile.of(name, data))


MERGED = _upload("bloomberg_merged.csv", b"merged", rows=10)
SPX = _upload("spx.xlsx", b"spx bytes")
VIX = _upload("vix.xlsx", b"vix bytes")
LUACOAS = _upload("luacoas.xlsx", b"ig bytes")


def _merge(db, sources=((SPX, TARGET), (VIX, SIGNAL), (LUACOAS, SIGNAL))):
    record_merge(MERGED, sources, db_path=db, uploaded_at=WHEN)


def test_a_merge_logs_the_dataset_and_every_export_as_uploads(db):
    _merge(db)

    logged = {r.filename: r for r in recent_uploads(db_path=db)}
    assert set(logged) == {"bloomberg_merged.csv", "spx.xlsx", "vix.xlsx", "luacoas.xlsx"}
    assert logged["bloomberg_merged.csv"].row_count == 10
    assert all(r.uploaded_at == WHEN for r in logged.values())


def test_a_dataset_lists_the_exports_it_was_merged_from_targets_first(db):
    _merge(db)

    sources = sources_of(MERGED.source.sha256, db_path=db)

    assert [(s.role, s.filename) for s in sources] == [
        (TARGET, "spx.xlsx"),
        (SIGNAL, "luacoas.xlsx"),
        (SIGNAL, "vix.xlsx"),
    ]
    assert sources[0].sha256 == SPX.source.sha256 and sources[0].row_count == 3


def test_one_export_can_go_into_more_than_one_dataset(db):
    _merge(db)
    other = _upload("bloomberg_merged.csv", b"another merge")
    record_merge(other, [(SPX, TARGET)], db_path=db)

    assert {s.filename for s in sources_of(other.source.sha256, db_path=db)} == {"spx.xlsx"}
    assert len(sources_of(MERGED.source.sha256, db_path=db)) == 3


def test_linking_the_same_files_again_adds_nothing(db):
    _merge(db)
    pairs = [(SPX.source.sha256, TARGET), (VIX.source.sha256, SIGNAL)]

    record_dataset(MERGED.source.sha256, pairs, db_path=db)
    record_dataset(MERGED.source.sha256, pairs, db_path=db)

    assert len(sources_of(MERGED.source.sha256, db_path=db)) == 3


def test_a_merge_whose_links_fail_logs_none_of_its_uploads(db):
    _merge(db, sources=[(SPX, TARGET)])
    before = len(recent_uploads(db_path=db))

    with pytest.raises(duckdb.ConstraintException):  # "benchmark" is not a role
        record_merge(_upload("m2.csv", b"m2"), [(VIX, "benchmark")], db_path=db)

    assert len(recent_uploads(db_path=db)) == before


def test_an_export_missing_from_the_upload_log_is_still_listed_by_hash(db):
    record_dataset("d" * 64, [("e" * 64, SIGNAL)], db_path=db)

    (source,) = sources_of("d" * 64, db_path=db)
    assert source.sha256 == "e" * 64 and source.filename is None


def test_an_unknown_dataset_has_no_sources(db):
    assert sources_of("0" * 64, db_path=db) == []


# --- the raw bytes ---------------------------------------------------------------------


def test_an_export_is_kept_byte_for_byte_under_its_hash_and_extension(tmp_path):
    source = keep_source("SPX Index.XLSX", b"\x50\x4b raw bytes", uploads_dir=tmp_path)

    assert source.path == tmp_path / f"{source.sha256}.xlsx"
    assert source.path.read_bytes() == b"\x50\x4b raw bytes"
    assert source.name == "SPX Index.XLSX"


def test_keeping_the_same_bytes_again_does_not_rewrite_them(tmp_path):
    first = keep_source("a.csv", b"same", uploads_dir=tmp_path)
    stamp = first.path.stat().st_mtime_ns

    second = keep_source("renamed.csv", b"same", uploads_dir=tmp_path)

    assert second.path == first.path and first.path.stat().st_mtime_ns == stamp
