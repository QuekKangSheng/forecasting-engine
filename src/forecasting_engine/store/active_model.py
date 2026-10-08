"""An append-only log of which model is active for each target role.

Setting a new active model never updates a row in place, it appends one, so
the active model for a role is simply its most recent row.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.reporting.model_metrics import ModelRunResult
from forecasting_engine.store._db import DEFAULT_DB_PATH, connect

_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS active_models (
    role        TEXT      NOT NULL,
    model_name  TEXT      NOT NULL,
    ic          DOUBLE    NOT NULL,
    oos_rank_ic DOUBLE    NOT NULL,
    rmse        DOUBLE    NOT NULL,
    pbo         DOUBLE,
    high_risk   BOOLEAN   NOT NULL,
    set_at      TIMESTAMP NOT NULL
);
ALTER TABLE active_models ADD COLUMN IF NOT EXISTS horizon INTEGER;
ALTER TABLE active_models ADD COLUMN IF NOT EXISTS train_window INTEGER;
ALTER TABLE active_models ADD COLUMN IF NOT EXISTS test_window INTEGER;
ALTER TABLE active_models ADD COLUMN IF NOT EXISTS dataset_fingerprint TEXT;
ALTER TABLE active_models ADD COLUMN IF NOT EXISTS signal_rank_ic DOUBLE;
CREATE TABLE IF NOT EXISTS active_model_forecasts (
    role       TEXT      NOT NULL,
    model_name TEXT      NOT NULL,
    set_at     TIMESTAMP NOT NULL,
    date       TIMESTAMP NOT NULL,
    predicted  DOUBLE
);
ALTER TABLE active_model_forecasts ADD COLUMN IF NOT EXISTS baseline DOUBLE;
"""


@dataclass(frozen=True)
class ActiveModelRecord:
    """One role's active model, as of ``set_at``, and the settings it was run under."""

    role: TargetRole
    model_name: str
    ic: float
    oos_rank_ic: float
    rmse: float
    pbo: float | None
    high_risk: bool
    set_at: datetime
    horizon: int
    train_window: int
    test_window: int
    dataset_fingerprint: str
    signal_rank_ic: float | None = None
    """``None`` for a model set active before the Signal Rank IC existed."""


def set_active_model(
    role: TargetRole,
    model_name: str,
    result: ModelRunResult,
    *,
    high_risk: bool,
    horizon: int,
    train_window: int,
    test_window: int,
    dataset_fingerprint: str,
    db_path: Path = DEFAULT_DB_PATH,
    set_at: datetime | None = None,
) -> ActiveModelRecord:
    """Append a new active-model row for ``role``, superseding any prior one.

    Also stores ``result.forecast`` (if present) as one row per date, with each
    date's ``result.baseline`` (its fold's training-mean target), so a later
    reader never has to re-run the model to get its predicted values or their
    signal back.
    """
    record = ActiveModelRecord(
        role=role,
        model_name=model_name,
        ic=result.ic,
        oos_rank_ic=result.oos_rank_ic,
        rmse=result.rmse,
        pbo=result.pbo,
        high_risk=high_risk,
        set_at=set_at or datetime.now(),
        horizon=horizon,
        train_window=train_window,
        test_window=test_window,
        dataset_fingerprint=dataset_fingerprint,
        signal_rank_ic=result.signal_rank_ic,
    )
    with connect(db_path, _CREATE_TABLES) as conn:
        conn.execute(
            "INSERT INTO active_models "
            "(role, model_name, ic, oos_rank_ic, rmse, pbo, high_risk, set_at, "
            "horizon, train_window, test_window, dataset_fingerprint, signal_rank_ic) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                record.role.value,
                record.model_name,
                record.ic,
                record.oos_rank_ic,
                record.rmse,
                record.pbo,
                record.high_risk,
                record.set_at,
                record.horizon,
                record.train_window,
                record.test_window,
                record.dataset_fingerprint,
                record.signal_rank_ic,
            ],
        )
        if result.forecast is not None:
            baseline = (
                result.baseline.reindex(result.forecast.index)
                if result.baseline is not None
                else pd.Series(float("nan"), index=result.forecast.index)
            )
            conn.executemany(
                "INSERT INTO active_model_forecasts "
                "(role, model_name, set_at, date, predicted, baseline) VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        record.role.value,
                        record.model_name,
                        record.set_at,
                        date,
                        float(value),
                        None if base != base else float(base),
                    )
                    for (date, value), base in zip(
                        result.forecast.items(), baseline.to_numpy(), strict=True
                    )
                ],
            )
    return record


def get_active_model(
    role: TargetRole, *, db_path: Path = DEFAULT_DB_PATH
) -> ActiveModelRecord | None:
    """The most recently set active model for ``role``, or ``None`` if never set."""
    with connect(db_path, _CREATE_TABLES) as conn:
        row = conn.execute(
            "SELECT role, model_name, ic, oos_rank_ic, rmse, pbo, high_risk, set_at, "
            "horizon, train_window, test_window, dataset_fingerprint, signal_rank_ic "
            "FROM active_models WHERE role = ? ORDER BY set_at DESC LIMIT 1",
            [role.value],
        ).fetchone()
    if row is None:
        return None
    return ActiveModelRecord(
        role=TargetRole(row[0]),
        model_name=row[1],
        ic=row[2],
        oos_rank_ic=row[3],
        rmse=row[4],
        pbo=row[5],
        high_risk=row[6],
        set_at=row[7],
        horizon=row[8],
        train_window=row[9],
        test_window=row[10],
        dataset_fingerprint=row[11],
        signal_rank_ic=row[12],
    )


def get_active_model_forecast(
    role: TargetRole, *, db_path: Path = DEFAULT_DB_PATH
) -> pd.Series | None:
    """The currently active model's saved forecast, indexed by date — or ``None``
    if there is no active model, or it was set before forecasts were captured."""
    current = get_active_model(role, db_path=db_path)
    if current is None:
        return None
    with connect(db_path, _CREATE_TABLES) as conn:
        rows = conn.execute(
            "SELECT date, predicted FROM active_model_forecasts "
            "WHERE role = ? AND set_at = ? ORDER BY date",
            [role.value, current.set_at],
        ).fetchall()
    if not rows:
        return None
    dates, values = zip(*rows, strict=True)
    return pd.Series(values, index=pd.DatetimeIndex(dates), dtype=float)


def get_active_model_signal(
    role: TargetRole, *, db_path: Path = DEFAULT_DB_PATH
) -> pd.Series | None:
    """The currently active model's signal — each saved forecast less its fold's
    training-mean target — indexed by date, or ``None`` if there is no active
    model or it was set before baselines were captured."""
    current = get_active_model(role, db_path=db_path)
    if current is None:
        return None
    with connect(db_path, _CREATE_TABLES) as conn:
        rows = conn.execute(
            "SELECT date, predicted - baseline FROM active_model_forecasts "
            "WHERE role = ? AND set_at = ? ORDER BY date",
            [role.value, current.set_at],
        ).fetchall()
        has_baseline = conn.execute(
            "SELECT count(baseline) FROM active_model_forecasts WHERE role = ? AND set_at = ?",
            [role.value, current.set_at],
        ).fetchone()[0]
    if not rows or not has_baseline:
        return None
    dates, values = zip(*rows, strict=True)
    return pd.Series(values, index=pd.DatetimeIndex(dates), dtype=float)


def settings_match(a: ActiveModelRecord, b: ActiveModelRecord) -> bool:
    """Whether ``a`` and ``b`` were produced under the same horizon, walk-forward
    windows and dataset — the check a two-asset run needs before combining them."""
    return (
        a.horizon == b.horizon
        and a.train_window == b.train_window
        and a.test_window == b.test_window
        and a.dataset_fingerprint == b.dataset_fingerprint
    )
