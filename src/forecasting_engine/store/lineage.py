"""Which uploaded Bloomberg exports each committed dataset was merged from.

One dataset is merged from several exports, and one export can go into several
datasets (re-uploaded beside a different file, say). Neither side can hold the
other in a single column, so the relationship is its own table, one row per
pair: a link (or junction) table.

``dataset_sources`` holds only content hashes. A hash's filename, size and upload
time are in the upload log (``store/uploads``), and its bytes are in
``data/uploads/<sha256>.<ext>``. It can't reference the upload log with a foreign
key, because that log is an event log: the same file uploaded twice is two rows,
so a hash is not unique there, and a foreign key needs a unique parent.

``record_merge`` writes the merged dataset's upload, each export's upload and the
links between them in one transaction, so the log never holds a dataset without
its sources or sources without their dataset.

Recording a dataset is idempotent: committing the same files again adds nothing,
since the primary key is the pair and a pair already there is skipped
(``ON CONFLICT DO NOTHING``).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from forecasting_engine.ingest.upload import AcceptedUpload
from forecasting_engine.store._db import DEFAULT_DB_PATH, connect
from forecasting_engine.store.uploads import _CREATE_TABLE as _CREATE_UPLOADS

TARGET, SIGNAL = "target", "signal"

_CREATE_TABLES = (
    _CREATE_UPLOADS
    + """;
CREATE TABLE IF NOT EXISTS dataset_sources (
    dataset_sha256 TEXT NOT NULL,
    source_sha256  TEXT NOT NULL,
    role           TEXT NOT NULL CHECK (role IN ('target', 'signal')),
    PRIMARY KEY (dataset_sha256, source_sha256)
);
"""
)


@dataclass(frozen=True)
class Source:
    """One export a dataset was merged from, as the upload log knows it."""

    sha256: str
    role: str
    filename: str | None
    """``None`` if the export's hash is not in the upload log."""
    uploaded_at: datetime | None
    row_count: int | None


def record_merge(
    merged: AcceptedUpload,
    sources: Iterable[tuple[AcceptedUpload, str]],
    *,
    db_path: Path = DEFAULT_DB_PATH,
    uploaded_at: datetime | None = None,
) -> None:
    """Log the merged dataset and each ``(export, role)`` it was merged from as
    uploads, and link them, all in one transaction."""
    when = uploaded_at or datetime.now()
    sources = list(sources)
    uploads = [
        (a.source.sha256, a.source.name, a.source.size_bytes, a.row_count, when)
        for a in (merged, *(export for export, _ in sources))
    ]
    with connect(db_path, _CREATE_TABLES) as conn:
        conn.execute("BEGIN TRANSACTION")
        conn.executemany("INSERT INTO uploads VALUES (?, ?, ?, ?, ?)", uploads)
        _link(conn, merged.source.sha256, [(e.source.sha256, role) for e, role in sources])
        conn.execute("COMMIT")


def record_dataset(
    dataset_sha256: str,
    sources: Iterable[tuple[str, str]],
    *,
    db_path: Path = DEFAULT_DB_PATH,
) -> None:
    """Link ``dataset_sha256`` to each ``(source_sha256, role)`` it was merged from.
    A pair already linked is left as it is."""
    with connect(db_path, _CREATE_TABLES) as conn:
        _link(conn, dataset_sha256, sources)


def _link(conn, dataset_sha256: str, sources: Iterable[tuple[str, str]]) -> None:
    rows = [(dataset_sha256, sha, role) for sha, role in sources]
    if rows:
        conn.executemany(
            "INSERT INTO dataset_sources VALUES (?, ?, ?) ON CONFLICT DO NOTHING", rows
        )


def sources_of(dataset_sha256: str, *, db_path: Path = DEFAULT_DB_PATH) -> list[Source]:
    """The exports ``dataset_sha256`` was merged from, targets first, each with its
    first upload in the log."""
    with connect(db_path, _CREATE_TABLES) as conn:
        rows = conn.execute(
            """
            SELECT s.source_sha256, s.role, u.filename, u.uploaded_at, u.row_count
            FROM dataset_sources AS s
            LEFT JOIN (
                SELECT sha256, min(uploaded_at) AS uploaded_at,
                       arg_min(filename, uploaded_at) AS filename,
                       arg_min(row_count, uploaded_at) AS row_count
                FROM uploads
                GROUP BY sha256
            ) AS u ON u.sha256 = s.source_sha256
            WHERE s.dataset_sha256 = ?
            ORDER BY s.role DESC, u.filename
            """,
            [dataset_sha256],
        ).fetchall()
    return [Source(*row) for row in rows]
