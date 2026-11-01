"""Every distinct optimiser-and-backtest result the Portfolio page has shown, once each.

The page recomputes the optimiser and the backtest on every view: a slider moved
and moved back, a tab switched, a reload. Logging each computation would log the
same result many times over. So a portfolio run is identified by a **deterministic
key**, the SHA-256 of everything its result depends on (``run_key``), and recorded
with ``INSERT … ON CONFLICT DO NOTHING``: writing the same key again changes
nothing. That makes recording **idempotent**, so it is safe on every page view.

The key covers every input: the data's content hash, both active models (their
run ids and when they were set), the windows, the risk aversion and bounds, the
trading costs, and the code version. Leaving the code version out would let a
change in how results are computed hide behind the first row stored for those
inputs.

``DO NOTHING`` keeps the first write and never updates it, so the table stays
append-only. An upsert (``DO UPDATE SET last_seen = …``) would record when a
result was last viewed, but at the price of rewriting rows.

Three tables, keyed by ``run_key``:

- ``portfolio_runs``: one row per distinct input set, with the inputs as JSON.
- ``portfolio_run_metrics``: per run, portfolio and basis, each performance metric
  (long format).
- ``portfolio_run_weights``: the weight schedule, one row per rebalance.

The daily return paths are not stored: they follow from the inputs and the stored
data, and would be about 8,000 rows a run. Storing what is costly or impossible
to recompute, and recomputing the rest, keeps the log small.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.portfolio.backtest import BacktestResult
from forecasting_engine.store._db import DEFAULT_DB_PATH, connect

_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS portfolio_runs (
    run_key        TEXT      PRIMARY KEY,
    recorded_at    TIMESTAMP NOT NULL,
    dataset_sha256 TEXT      NOT NULL,
    equity_run_id  TEXT,
    bond_run_id    TEXT,
    start_date     DATE      NOT NULL,
    end_date       DATE      NOT NULL,
    inputs         JSON      NOT NULL,
    code_version   TEXT
);
CREATE TABLE IF NOT EXISTS portfolio_run_metrics (
    run_key   TEXT   NOT NULL REFERENCES portfolio_runs (run_key),
    portfolio TEXT   NOT NULL,
    basis     TEXT   NOT NULL,
    metric    TEXT   NOT NULL,
    value     DOUBLE,
    PRIMARY KEY (run_key, portfolio, basis, metric)
);
CREATE TABLE IF NOT EXISTS portfolio_run_weights (
    run_key       TEXT   NOT NULL REFERENCES portfolio_runs (run_key),
    date          DATE   NOT NULL,
    equity_weight DOUBLE NOT NULL,
    PRIMARY KEY (run_key, date)
);
"""


@dataclass(frozen=True)
class PortfolioInputs:
    """Everything a portfolio run's result depends on."""

    dataset_sha256: str
    equity_run_id: str | None
    bond_run_id: str | None
    equity_set_at: datetime
    bond_set_at: datetime
    """When each active model was set: identifies it even without a run id."""
    horizon: int
    train_window: int
    test_window: int
    embargo: int
    risk_aversion: float
    bounds: tuple[float, float]
    costs_bps: Mapping[str, float]
    code_version: str | None

    @property
    def run_key(self) -> str:
        """The SHA-256 of the inputs as canonical JSON: sorted keys, no spaces,
        datetimes as ISO text. Equal inputs give an equal key on any machine."""
        canonical = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), default=_iso
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PortfolioRun:
    run_key: str
    recorded_at: datetime
    dataset_sha256: str
    equity_run_id: str | None
    bond_run_id: str | None
    start_date: datetime
    end_date: datetime
    inputs: Mapping[str, object]
    code_version: str | None


def record_portfolio_run(
    inputs: PortfolioInputs,
    result: BacktestResult,
    weights: pd.DataFrame,
    *,
    recorded_at: datetime | None = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> bool:
    """Record ``result`` under ``inputs``' key unless that key is already there.

    Returns whether it was new. One transaction: the parent row is inserted with
    ``ON CONFLICT DO NOTHING … RETURNING``, and only if it came back (the key was
    new) are its metrics and weights written, so a repeat writes nothing at all."""
    key = inputs.run_key
    equity = weights.columns[0]
    with connect(db_path, _CREATE_TABLES) as conn:
        conn.execute("BEGIN TRANSACTION")
        inserted = conn.execute(
            "INSERT INTO portfolio_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT DO NOTHING RETURNING run_key",
            [
                key,
                recorded_at or datetime.now(),
                inputs.dataset_sha256,
                inputs.equity_run_id,
                inputs.bond_run_id,
                result.start.date(),
                result.end.date(),
                json.dumps(asdict(inputs), sort_keys=True, default=_iso),
                inputs.code_version,
            ],
        ).fetchall()
        if inserted:
            conn.executemany(
                "INSERT INTO portfolio_run_metrics VALUES (?, ?, ?, ?, ?)",
                [
                    (key, portfolio, basis, name, _finite(value))
                    for (portfolio, basis), metrics in result.metrics.items()
                    for name, value in asdict(metrics).items()
                ]
                + [
                    (key, "optimised", basis, name, _finite(value))
                    for basis, relative in result.relative.items()
                    for name, value in asdict(relative).items()
                ],
            )
            conn.executemany(
                "INSERT INTO portfolio_run_weights VALUES (?, ?, ?)",
                [(key, date.date(), float(w)) for date, w in weights[equity].items()],
            )
        conn.execute("COMMIT")
    return bool(inserted)


def list_portfolio_runs(
    *, limit: int = 200, db_path: Path = DEFAULT_DB_PATH
) -> list[PortfolioRun]:
    """The recorded portfolio runs, newest first."""
    with connect(db_path, _CREATE_TABLES) as conn:
        rows = conn.execute(
            "SELECT * FROM portfolio_runs ORDER BY recorded_at DESC, run_key LIMIT ?", [limit]
        ).fetchall()
    return [
        PortfolioRun(*row[:7], inputs=json.loads(row[7]), code_version=row[8]) for row in rows
    ]


def portfolio_runs_table(*, limit: int = 200, db_path: Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    """One row per portfolio run, newest first, with its inputs that change from
    run to run and both portfolios' net Sharpe ratios as columns."""
    with connect(db_path, _CREATE_TABLES) as conn:
        return conn.execute(
            """
            SELECT p.run_key, p.recorded_at, p.start_date, p.end_date,
                   CAST(p.inputs ->> 'risk_aversion' AS DOUBLE) AS risk_aversion,
                   CAST(p.inputs -> 'bounds' ->> 0 AS DOUBLE) AS min_weight,
                   p.equity_run_id, p.bond_run_id, p.code_version,
                   max(m.value) FILTER (WHERE m.portfolio = 'optimised' AND m.basis = 'net'
                                        AND m.metric = 'sharpe') AS net_sharpe,
                   max(m.value) FILTER (WHERE m.portfolio = 'baseline' AND m.basis = 'net'
                                        AND m.metric = 'sharpe') AS baseline_net_sharpe,
                   max(m.value) FILTER (WHERE m.portfolio = 'optimised' AND m.basis = 'net'
                                        AND m.metric = 'information_ratio') AS information_ratio
            FROM portfolio_runs AS p
            LEFT JOIN portfolio_run_metrics AS m USING (run_key)
            GROUP BY ALL
            ORDER BY p.recorded_at DESC, p.run_key
            LIMIT ?
            """,
            [limit],
        ).df()


def portfolio_run_weights(run_key: str, *, db_path: Path = DEFAULT_DB_PATH) -> pd.Series:
    with connect(db_path, _CREATE_TABLES) as conn:
        frame = conn.execute(
            "SELECT date, equity_weight FROM portfolio_run_weights WHERE run_key = ? "
            "ORDER BY date",
            [run_key],
        ).df()
    return pd.Series(
        frame["equity_weight"].to_numpy(), index=pd.to_datetime(frame["date"]), dtype=float
    )


def _iso(value: object) -> str:
    if isinstance(value, datetime | pd.Timestamp):
        return value.isoformat()
    if isinstance(value, TargetRole):
        return value.value
    raise TypeError(f"{value!r} has no canonical JSON form")


def _finite(value) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None
