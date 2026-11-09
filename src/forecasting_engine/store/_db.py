"""Shared DuckDB connection handling.

Both logs live in one database file and open it the same way, so the open,
create-if-absent and close sequence sits here rather than in each module.

Every Streamlit session is a thread of the same process, and they all open this one
file. ``CREATE TABLE IF NOT EXISTS`` changes DuckDB's catalog even when the table is
there, so two sessions running it at once fail one another with a write-write
conflict. So each module's tables are created once per process, under a lock, and
not on every open. Writes that could collide go through ``retry_on_conflict``: they
take turns, and one is tried again if a write that didn't take its turn collided.
(DuckDB lets only one process write to a file, so the sessions are all here.)
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TypeVar

import duckdb

#: Gitignored, because ``data/`` is.
DEFAULT_DB_PATH: Path = Path("data/forecasting.duckdb")

#: How many times a write that conflicts with another is tried in all.
WRITE_ATTEMPTS: int = 5
#: The wait before the n-th retry is n times this, so colliding writers spread out.
_BACKOFF_SECONDS: float = 0.05

#: (database, creation SQL) pairs already run in this process.
_created: set[tuple[str, str]] = set()
_created_lock = threading.Lock()
#: Sessions' conflicting writes take turns rather than fail each other.
_write_lock = threading.Lock()

T = TypeVar("T")


@contextlib.contextmanager
def connect(db_path: Path, create_sql: str) -> Iterator[duckdb.DuckDBPyConnection]:
    """Open ``db_path``, creating the file and the caller's tables if absent."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    key = (str(db_path.resolve()), create_sql)
    with _created_lock:
        if not db_path.exists():
            # A new file, or one deleted while the app ran: nothing in it exists yet.
            _created.difference_update({k for k in _created if k[0] == key[0]})
        conn = duckdb.connect(str(db_path))
        try:
            if key not in _created:
                conn.execute(create_sql)
                _created.add(key)
        except BaseException:
            conn.close()
            raise
    try:
        yield conn
    finally:
        conn.close()


def retry_on_conflict(write: Callable[[], T]) -> T:
    """``write()``, after any other session's that is under way, and tried again if a
    write that doesn't take turns here (another module's) conflicted with it.

    DuckDB fails the later of two transactions that change the same rows, or that
    both take the next run number, rather than making one wait. ``write`` must be a
    whole transaction, so a failed attempt has left nothing behind."""
    for attempt in range(1, WRITE_ATTEMPTS + 1):
        try:
            with _write_lock:
                return write()
        except (duckdb.TransactionException, duckdb.ConstraintException):
            if attempt == WRITE_ATTEMPTS:
                raise
            time.sleep(_BACKOFF_SECONDS * attempt)
    raise AssertionError("unreachable")  # pragma: no cover
