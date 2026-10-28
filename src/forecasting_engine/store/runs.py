"""An append-only history of model runs: what was fitted, on which data, under which
settings, by which version of the code, how long it took, and what came out.

This is the run history the architecture design gives ``store/runs.py``. The
page's result cache (``app/model_jobs``) keeps only the latest result per settings
and is overwritten by a re-run; this log keeps every fit as an event, failures
included, and can be queried.

Three tables, all keyed by ``run_id``:

- ``runs``: one row per fit. The columns a reader filters or sorts on are typed
  (role, model, status, times, dataset hash); the rest of the settings sit in one
  JSON column, so a new setting needs no schema change.
- ``run_metrics``: one row per run and metric (long format), so a new metric
  needs no schema change either.
- ``run_forecasts``: one row per run and date: the out-of-sample forecast, its
  fold's training-mean baseline, and the realised return.

**Lineage.** ``dataset_sha256`` is the SHA-256 of the committed dataset's
canonical CSV (``ingest.upload.frame_sha256``), the same hash the upload log
records for it and the name its bytes are stored under in ``data/uploads``. So any
run traces back to the exact data it saw, and unlike the shape-based
``dataset_fingerprint``, a single corrected value gives a different hash.

Every row of one run is written in a single transaction: a reader sees a run
with all its metrics and forecasts, or not at all.
"""

from __future__ import annotations

import json
import math
import subprocess
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from functools import cache
from pathlib import Path

import pandas as pd

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.reporting.model_metrics import ModelRunResult
from forecasting_engine.store._db import DEFAULT_DB_PATH, connect

DONE, FAILED = "done", "failed"

_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS runs (
    run_id         TEXT      PRIMARY KEY,
    role           TEXT      NOT NULL,
    model_name     TEXT      NOT NULL,
    status         TEXT      NOT NULL CHECK (status IN ('done', 'failed')),
    error          TEXT,
    started_at     TIMESTAMP NOT NULL,
    finished_at    TIMESTAMP NOT NULL,
    dataset_sha256 TEXT      NOT NULL,
    config_key     TEXT      NOT NULL,
    settings       JSON      NOT NULL,
    code_version   TEXT
);
CREATE TABLE IF NOT EXISTS run_metrics (
    run_id TEXT   NOT NULL REFERENCES runs (run_id),
    metric TEXT   NOT NULL,
    value  DOUBLE,
    PRIMARY KEY (run_id, metric)
);
CREATE TABLE IF NOT EXISTS run_forecasts (
    run_id    TEXT   NOT NULL REFERENCES runs (run_id),
    date      DATE   NOT NULL,
    predicted DOUBLE,
    baseline  DOUBLE,
    realised  DOUBLE,
    PRIMARY KEY (run_id, date)
);
"""

#: The headline metrics ``runs_table`` puts beside each run, in display order.
HEADLINE_METRICS: tuple[str, ...] = ("signal_rank_ic", "signal_t", "oos_rank_ic", "ic", "rmse")


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    role: TargetRole
    model_name: str
    status: str
    error: str | None
    started_at: datetime
    finished_at: datetime
    dataset_sha256: str
    config_key: str
    """The result cache's key for these settings: two runs with the same key were
    the same fit asked for twice."""
    settings: Mapping[str, object]
    code_version: str | None
    """The git commit the app ran from, suffixed ``-dirty`` if the checkout had
    uncommitted changes, or ``None`` where git isn't available."""

    @property
    def seconds(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()


def metrics_of(result: ModelRunResult) -> dict[str, float]:
    """Every scalar a run is judged by, flattened to ``name -> value``. ``None``
    (a figure the run didn't compute) is left out rather than stored as NaN, so
    "not computed" and "computed as NaN" stay distinguishable."""
    values = {
        "ic": result.ic,
        "oos_rank_ic": result.oos_rank_ic,
        "rmse": result.rmse,
        "signal_rank_ic": result.signal_rank_ic,
        "signal_t": result.signal_t,
        "signal_rank_ic_se": result.signal_rank_ic_se,
        "signal_rank_ic_se_test": result.signal_rank_ic_se_test,
        "oos_rank_ic_se": result.oos_rank_ic_se,
        "oos_rank_ic_se_test": result.oos_rank_ic_se_test,
        "rows_scored": result.rows_scored,
        "crash_recall": result.crash.recall,
        "crash_precision": result.crash.precision,
        "crash_f1": result.crash.f1,
        "crash_tail_days": result.crash.n_true_tail_days,
    }
    return {name: float(value) for name, value in values.items() if value is not None}


def record_run(
    *,
    role: TargetRole,
    model_name: str,
    started_at: datetime,
    finished_at: datetime,
    dataset_sha256: str,
    config_key: str,
    settings: Mapping[str, object],
    result: ModelRunResult | None = None,
    error: str | None = None,
    code_version: str | None = None,
    run_id: str | None = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> RunRecord:
    """Append one run: ``result`` for a fit that finished, ``error`` for one that
    failed (exactly one of the two)."""
    if (result is None) == (error is None):
        raise ValueError("a run has either a result or an error, not both or neither")
    record = RunRecord(
        run_id=run_id or uuid.uuid4().hex,
        role=role,
        model_name=model_name,
        status=DONE if error is None else FAILED,
        error=error,
        started_at=started_at,
        finished_at=finished_at,
        dataset_sha256=dataset_sha256,
        config_key=config_key,
        # Through JSON and back, so the record returned is what a later read gets:
        # JSON has no tuple, so a user formula and its bindings come back as lists.
        settings=json.loads(json.dumps(dict(settings), default=str)),
        code_version=code_version,
    )
    with connect(db_path, _CREATE_TABLES) as conn:
        conn.execute("BEGIN TRANSACTION")
        conn.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                record.run_id,
                role.value,
                model_name,
                record.status,
                error,
                started_at,
                finished_at,
                dataset_sha256,
                config_key,
                json.dumps(record.settings),
                code_version,
            ],
        )
        if result is not None:
            conn.executemany(
                "INSERT INTO run_metrics VALUES (?, ?, ?)",
                [(record.run_id, name, _finite(v)) for name, v in metrics_of(result).items()],
            )
            forecasts = _forecast_rows(result)
            if forecasts:
                conn.executemany(
                    "INSERT INTO run_forecasts VALUES (?, ?, ?, ?, ?)",
                    [(record.run_id, *row) for row in forecasts],
                )
        conn.execute("COMMIT")
    return record


def list_runs(
    *, role: TargetRole | None = None, limit: int = 200, db_path: Path = DEFAULT_DB_PATH
) -> list[RunRecord]:
    """The most recent runs, newest first, for ``role`` if given."""
    where, params = ("WHERE role = ?", [role.value]) if role is not None else ("", [])
    with connect(db_path, _CREATE_TABLES) as conn:
        rows = conn.execute(
            f"SELECT * FROM runs {where} ORDER BY started_at DESC, run_id LIMIT ?",
            [*params, limit],
        ).fetchall()
    return [_record(row) for row in rows]


def get_run(run_id: str, *, db_path: Path = DEFAULT_DB_PATH) -> RunRecord | None:
    with connect(db_path, _CREATE_TABLES) as conn:
        row = conn.execute("SELECT * FROM runs WHERE run_id = ?", [run_id]).fetchone()
    return None if row is None else _record(row)


def run_metrics(run_id: str, *, db_path: Path = DEFAULT_DB_PATH) -> dict[str, float]:
    with connect(db_path, _CREATE_TABLES) as conn:
        rows = conn.execute(
            "SELECT metric, value FROM run_metrics WHERE run_id = ? ORDER BY metric", [run_id]
        ).fetchall()
    return {metric: (math.nan if value is None else value) for metric, value in rows}


def run_forecasts(run_id: str, *, db_path: Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    """A run's forecasts, indexed by date, with ``predicted``, ``baseline`` and
    ``realised`` columns (empty for a failed run)."""
    with connect(db_path, _CREATE_TABLES) as conn:
        frame = conn.execute(
            "SELECT date, predicted, baseline, realised FROM run_forecasts "
            "WHERE run_id = ? ORDER BY date",
            [run_id],
        ).df()
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.set_index("date")


def runs_table(
    *, role: TargetRole | None = None, limit: int = 200, db_path: Path = DEFAULT_DB_PATH
) -> pd.DataFrame:
    """One row per run, newest first, with its headline metrics as columns.

    The metrics are stored long (a row per metric) and turned into columns here
    by conditional aggregation, one ``FILTER`` per metric, so the table needs no
    column per metric and this query is the only place the shape changes."""
    where, params = ("WHERE r.role = ?", [role.value]) if role is not None else ("", [])
    pivot = ", ".join(
        f"max(m.value) FILTER (WHERE m.metric = '{name}') AS {name}" for name in HEADLINE_METRICS
    )
    with connect(db_path, _CREATE_TABLES) as conn:
        return conn.execute(
            f"""
            SELECT r.run_id, r.started_at, r.role, r.model_name, r.status,
                   date_diff('millisecond', r.started_at, r.finished_at) / 1000.0 AS seconds,
                   r.dataset_sha256, r.config_key, r.code_version, {pivot}
            FROM runs r LEFT JOIN run_metrics m USING (run_id)
            {where}
            GROUP BY ALL
            ORDER BY r.started_at DESC, r.run_id
            LIMIT ?
            """,
            [*params, limit],
        ).df()


@cache
def code_version() -> str | None:
    """The short git commit the app is running from, with ``-dirty`` if the checkout
    has uncommitted changes (the commit alone would then claim code that didn't
    run), or ``None`` outside a git checkout. Read once per process."""
    try:
        out = subprocess.run(
            ["git", "describe", "--always", "--dirty", "--abbrev=7"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _forecast_rows(result: ModelRunResult) -> list[tuple]:
    if result.forecast is None:
        return []
    index = result.forecast.index
    baseline = (
        result.baseline.reindex(index) if result.baseline is not None else pd.Series(index=index)
    )
    realised = (
        result.realised.reindex(index) if result.realised is not None else pd.Series(index=index)
    )
    return [
        (pd.Timestamp(date).date(), _finite(p), _finite(b), _finite(r))
        for date, p, b, r in zip(
            index, result.forecast.to_numpy(), baseline.to_numpy(), realised.to_numpy(), strict=True
        )
    ]


def _finite(value) -> float | None:
    """NaN as SQL NULL: a NaN in a DOUBLE column compares unequal to itself and
    sorts oddly, where NULL is what every SQL reader expects for "no value"."""
    value = float(value)
    return value if math.isfinite(value) else None


def _record(row: tuple) -> RunRecord:
    (run_id, role, model, status, error, started, finished, sha, key, settings, version) = row
    return RunRecord(
        run_id=run_id,
        role=TargetRole(role),
        model_name=model,
        status=status,
        error=error,
        started_at=started,
        finished_at=finished,
        dataset_sha256=sha,
        config_key=key,
        settings=json.loads(settings),
        code_version=version,
    )
