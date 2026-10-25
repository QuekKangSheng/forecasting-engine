"""An append-only log of upload attempts that succeeded.

Deliberately narrow. The architecture design puts full run history in
``store/runs.py`` in Sprint 6; this module records uploads and nothing else, so
that it does not pre-commit the run schema that sprint owns.

The table is an event log: the same file uploaded twice is two rows. What
happened and when is the point, not which distinct files exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from forecasting_engine.ingest.upload import AcceptedUpload
from forecasting_engine.store._db import DEFAULT_DB_PATH, connect

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS uploads (
    sha256      TEXT      NOT NULL,
    filename    TEXT      NOT NULL,
    size_bytes  BIGINT    NOT NULL,
    row_count   BIGINT    NOT NULL,
    uploaded_at TIMESTAMP NOT NULL
)
"""


@dataclass(frozen=True)
class UploadRecord:
    """One row of the log."""

    sha256: str
    filename: str
    size_bytes: int
    row_count: int
    uploaded_at: datetime


def record_upload(
    accepted: AcceptedUpload,
    *,
    db_path: Path = DEFAULT_DB_PATH,
    uploaded_at: datetime | None = None,
) -> UploadRecord:
    """Append ``accepted`` to the log and return the row that was written.

    ``uploaded_at`` defaults to now; pass it explicitly to keep tests
    deterministic.
    """
    record = UploadRecord(
        sha256=accepted.source.sha256,
        filename=accepted.source.name,
        size_bytes=accepted.source.size_bytes,
        row_count=accepted.row_count,
        uploaded_at=uploaded_at or datetime.now(),
    )
    with connect(db_path, _CREATE_TABLE) as conn:
        conn.execute(
            "INSERT INTO uploads VALUES (?, ?, ?, ?, ?)",
            [
                record.sha256,
                record.filename,
                record.size_bytes,
                record.row_count,
                record.uploaded_at,
            ],
        )
    return record


def recent_uploads(limit: int = 20, *, db_path: Path = DEFAULT_DB_PATH) -> list[UploadRecord]:
    """The most recent uploads, newest first."""
    with connect(db_path, _CREATE_TABLE) as conn:
        rows = conn.execute(
            "SELECT sha256, filename, size_bytes, row_count, uploaded_at "
            "FROM uploads ORDER BY uploaded_at DESC LIMIT ?",
            [limit],
        ).fetchall()
    return [UploadRecord(*row) for row in rows]



def find_upload(sha256: str, *, db_path: Path = DEFAULT_DB_PATH) -> UploadRecord | None:
    """The first time content with ``sha256`` was uploaded, or ``None``: how a run's
    dataset hash is traced back to the file it came from."""
    with connect(db_path, _CREATE_TABLE) as conn:
        row = conn.execute(
            "SELECT sha256, filename, size_bytes, row_count, uploaded_at "
            "FROM uploads WHERE sha256 = ? ORDER BY uploaded_at LIMIT 1",
            [sha256],
        ).fetchone()
    return None if row is None else UploadRecord(*row)
