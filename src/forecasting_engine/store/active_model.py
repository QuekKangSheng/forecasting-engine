"""An append-only log of which model is active for each target role.

Setting a new active model never updates a row in place, it appends one, so
the active model for a role is simply its most recent row.

A polynomial set active also stores its fitted function (FYP-32): the most
recent fold's terms and coefficients, and per input its transform, clipping
bounds and standardisation. A scenario can then evaluate the function again on
shocked inputs without refitting. Every row of one setting is written in a
single transaction, keyed by role and ``set_at``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.ingest.align import Transform
from forecasting_engine.models.base import ModelDescription
from forecasting_engine.reporting.model_metrics import ModelRunResult
from forecasting_engine.reporting.polynomial_function import Origin
from forecasting_engine.store._db import DEFAULT_DB_PATH, connect

_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS active_models (
    role        TEXT      NOT NULL,
    model_name  TEXT      NOT NULL,
    ic          DOUBLE    NOT NULL,
    oos_rank_ic DOUBLE    NOT NULL,
    rmse        DOUBLE    NOT NULL,
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
CREATE TABLE IF NOT EXISTS active_model_functions (
    role      TEXT      NOT NULL,
    set_at    TIMESTAMP NOT NULL,
    origin    TEXT      NOT NULL,
    name      TEXT      NOT NULL,
    intercept DOUBLE
);
CREATE TABLE IF NOT EXISTS active_model_terms (
    role        TEXT      NOT NULL,
    set_at      TIMESTAMP NOT NULL,
    position    INTEGER   NOT NULL,
    term        TEXT      NOT NULL,
    coefficient DOUBLE    NOT NULL
);
CREATE TABLE IF NOT EXISTS active_model_inputs (
    role      TEXT      NOT NULL,
    set_at    TIMESTAMP NOT NULL,
    signal    TEXT      NOT NULL,
    transform TEXT      NOT NULL,
    clip_low  DOUBLE,
    clip_high DOUBLE,
    mean      DOUBLE,
    sd        DOUBLE
);
"""


@dataclass(frozen=True)
class ActiveFunction:
    """An active polynomial's fitted function, as its most recent fold fitted it."""

    origin: Origin
    description: ModelDescription
    """Terms, coefficients and intercept, with clipping bounds and standardisation
    for the inputs in ``transforms`` only: the ones the terms read."""
    transforms: Mapping[str, Transform]
    """Each input the function reads -> the transform it was fitted on. The
    research model reads its inputs as levels whatever the page's map says, so
    this is the run's own, not the page's."""

    @property
    def inputs(self) -> tuple[str, ...]:
        return tuple(self.transforms)


@dataclass(frozen=True)
class ActiveModelRecord:
    """One role's active model, as of ``set_at``, and the settings it was run under."""

    role: TargetRole
    model_name: str
    ic: float
    oos_rank_ic: float
    rmse: float
    high_risk: bool
    set_at: datetime
    horizon: int
    train_window: int | None
    """``None`` for a model set active while the equity derived polynomial still
    trained on every earlier date (an expanding window)."""
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
    train_window: int | None,
    test_window: int,
    dataset_fingerprint: str,
    function: ActiveFunction | None = None,
    db_path: Path = DEFAULT_DB_PATH,
    set_at: datetime | None = None,
) -> ActiveModelRecord:
    """Append a new active-model row for ``role``, superseding any prior one.

    Also stores ``result.forecast`` (if present) as one row per date, with each
    date's ``result.baseline`` (its fold's training-mean target), so a later
    reader never has to re-run the model to get its predicted values or their
    signal back; and ``function``, for a polynomial, so a scenario can evaluate
    it again. All of it is one transaction: a reader never sees a model active
    without its forecasts or function.
    """
    record = ActiveModelRecord(
        role=role,
        model_name=model_name,
        ic=result.ic,
        oos_rank_ic=result.oos_rank_ic,
        rmse=result.rmse,
        high_risk=high_risk,
        set_at=set_at or datetime.now(),
        horizon=horizon,
        train_window=train_window,
        test_window=test_window,
        dataset_fingerprint=dataset_fingerprint,
        signal_rank_ic=result.signal_rank_ic,
    )
    with connect(db_path, _CREATE_TABLES) as conn:
        conn.execute("BEGIN TRANSACTION")
        conn.execute(
            "INSERT INTO active_models "
            "(role, model_name, ic, oos_rank_ic, rmse, high_risk, set_at, "
            "horizon, train_window, test_window, dataset_fingerprint, signal_rank_ic) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                record.role.value,
                record.model_name,
                record.ic,
                record.oos_rank_ic,
                record.rmse,
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
        if function is not None:
            _insert_function(conn, record, function)
        conn.execute("COMMIT")
    return record


def _insert_function(conn, record: ActiveModelRecord, function: ActiveFunction) -> None:
    description = function.description
    key = (record.role.value, record.set_at)
    conn.execute(
        "INSERT INTO active_model_functions (role, set_at, origin, name, intercept) "
        "VALUES (?, ?, ?, ?, ?)",
        [*key, function.origin.value, description.name, description.intercept],
    )
    conn.executemany(
        "INSERT INTO active_model_terms (role, set_at, position, term, coefficient) "
        "VALUES (?, ?, ?, ?, ?)",
        [
            (*key, position, term, float(coefficient))
            for position, (term, coefficient) in enumerate(
                zip(description.terms, description.coefficients, strict=True)
            )
        ],
    )
    bounds = description.input_bounds or {}
    centre = description.standardisation or {}
    conn.executemany(
        "INSERT INTO active_model_inputs "
        "(role, set_at, signal, transform, clip_low, clip_high, mean, sd) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (*key, signal, transform.value, *bounds.get(signal, (None, None)),
             *centre.get(signal, (None, None)))
            for signal, transform in function.transforms.items()
        ],
    )


def get_active_function(
    role: TargetRole, *, db_path: Path = DEFAULT_DB_PATH
) -> ActiveFunction | None:
    """The currently active model's fitted function, or ``None`` if there is no
    active model, it is not a polynomial, or it was set before functions were
    stored."""
    current = get_active_model(role, db_path=db_path)
    if current is None:
        return None
    key = [role.value, current.set_at]
    with connect(db_path, _CREATE_TABLES) as conn:
        head = conn.execute(
            "SELECT origin, name, intercept FROM active_model_functions "
            "WHERE role = ? AND set_at = ?",
            key,
        ).fetchone()
        if head is None:
            return None
        terms = conn.execute(
            "SELECT term, coefficient FROM active_model_terms "
            "WHERE role = ? AND set_at = ? ORDER BY position",
            key,
        ).fetchall()
        inputs = conn.execute(
            "SELECT signal, transform, clip_low, clip_high, mean, sd "
            "FROM active_model_inputs WHERE role = ? AND set_at = ? ORDER BY rowid",
            key,
        ).fetchall()
    origin, name, intercept = head
    bounds = {s: (low, high) for s, _, low, high, _, _ in inputs if low is not None}
    centre = {s: (mean, sd) for s, _, _, _, mean, sd in inputs if mean is not None}
    return ActiveFunction(
        origin=Origin(origin),
        description=ModelDescription(
            name=name,
            terms=tuple(t for t, _ in terms),
            coefficients=tuple(c for _, c in terms),
            intercept=intercept,
            input_bounds=bounds or None,
            standardisation=centre or None,
        ),
        transforms={s: Transform(t) for s, t, *_ in inputs},
    )


def get_active_model(
    role: TargetRole, *, db_path: Path = DEFAULT_DB_PATH
) -> ActiveModelRecord | None:
    """The most recently set active model for ``role``, or ``None`` if never set."""
    with connect(db_path, _CREATE_TABLES) as conn:
        row = conn.execute(
            "SELECT role, model_name, ic, oos_rank_ic, rmse, high_risk, set_at, "
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
        high_risk=row[5],
        set_at=row[6],
        horizon=row[7],
        train_window=row[8],
        test_window=row[9],
        dataset_fingerprint=row[10],
        signal_rank_ic=row[11],
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
    """Whether ``a`` and ``b`` were produced under the same horizon, test windows and
    dataset — the check a two-asset run needs before combining them. Their train
    windows may differ (a record set before every model moved to one rolling window
    can have none): what matters is that both forecast the same days over the same
    horizon."""
    return (
        a.horizon == b.horizon
        and a.test_window == b.test_window
        and a.dataset_fingerprint == b.dataset_fingerprint
    )
