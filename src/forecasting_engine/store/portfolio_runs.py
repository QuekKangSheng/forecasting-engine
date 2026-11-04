"""Every distinct optimiser-and-backtest result the Portfolio page has shown, once each.

The page recomputes the optimiser and the backtest on every view: a slider moved
and moved back, a tab switched, a reload. Logging each computation would log the
same result many times over. So a portfolio run is identified by a **deterministic
key**, the SHA-256 of everything its result depends on (``run_key``), and recorded
with ``INSERT … ON CONFLICT DO NOTHING``: writing the same key again changes
nothing. That makes recording **idempotent**, so it is safe on every page view.

The key covers every input: the data's content hash, both active models (their
run ids and when they were set), the windows, the risk level and bounds, the
trading costs, and the code version. Leaving the code version out would let a
change in how results are computed hide behind the first row stored for those
inputs. The forecast trust and each level's risk aversion are not inputs: they
follow from the data and the models already in the key.

``DO NOTHING`` keeps the first write and never updates it, so the table stays
append-only. An upsert (``DO UPDATE SET last_seen = …``) would record when a
result was last viewed, but at the price of rewriting rows.

Keyed by ``run_key``:

- ``portfolio_runs``: one row per distinct input set, with the inputs as JSON.
- ``portfolio_run_metrics``: per run, portfolio and basis, each performance metric
  (long format).
- ``portfolio_run_weights``: the weight schedule, one row per rebalance.
- ``portfolio_run_labels``: the run's number and its user-given name. Apart from the
  name, the only thing a user can change, every table here is append-only, so the
  name sits in a table of its own rather than making ``portfolio_runs`` mutable.
- ``portfolio_run_results``: what the Portfolio page showed, as one JSON document
  (``reporting/run_results``), so a run can be opened without solving anything again.
  It is only ever read whole, so nothing is gained by splitting it into columns.
- ``portfolio_run_daily``: the daily returns behind the run's charts, one wide row per
  day. Recomputing them from the stored data would need that file to still exist, and
  would give today's code's answer rather than the one the run showed.
- ``portfolio_run_simulations``: each Monte Carlo simulation run on it, by method and
  paths. A simulation is seeded, so the same two give the same figures.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.portfolio.backtest import BacktestResult, PortfolioPath
from forecasting_engine.store._db import DEFAULT_DB_PATH, connect
from forecasting_engine.store.active_model import _CREATE_TABLES as _CREATE_ACTIVE_MODELS

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
CREATE TABLE IF NOT EXISTS portfolio_run_labels (
    run_key    TEXT    PRIMARY KEY REFERENCES portfolio_runs (run_key),
    run_number INTEGER NOT NULL UNIQUE,
    name       TEXT
);
CREATE TABLE IF NOT EXISTS portfolio_run_results (
    run_key TEXT PRIMARY KEY REFERENCES portfolio_runs (run_key),
    results JSON NOT NULL
);
CREATE TABLE IF NOT EXISTS portfolio_run_daily (
    run_key         TEXT   NOT NULL REFERENCES portfolio_runs (run_key),
    date            DATE   NOT NULL,
    optimised_gross DOUBLE NOT NULL,
    optimised_net   DOUBLE NOT NULL,
    baseline_gross  DOUBLE NOT NULL,
    baseline_net    DOUBLE NOT NULL,
    equity          DOUBLE NOT NULL,
    bond            DOUBLE NOT NULL,
    PRIMARY KEY (run_key, date)
);
CREATE TABLE IF NOT EXISTS portfolio_run_simulations (
    run_key     TEXT      NOT NULL REFERENCES portfolio_runs (run_key),
    method      TEXT      NOT NULL,
    paths       INTEGER   NOT NULL,
    recorded_at TIMESTAMP NOT NULL,
    results     JSON      NOT NULL,
    PRIMARY KEY (run_key, method, paths)
);
"""

#: Numbers every run without one, oldest first, after the highest number so far. Run
#: whenever the tables are opened too, which numbers runs recorded before there were
#: numbers, once.
_NUMBER_NEW_RUNS = """
INSERT INTO portfolio_run_labels (run_key, run_number)
SELECT run_key,
       (SELECT coalesce(max(run_number), 0) FROM portfolio_run_labels)
           + row_number() OVER (ORDER BY recorded_at, run_key)
FROM portfolio_runs
WHERE run_key NOT IN (SELECT run_key FROM portfolio_run_labels);
"""

_OPEN_TABLES = _CREATE_TABLES + _NUMBER_NEW_RUNS


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
    risk_level: int
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


@dataclass(frozen=True)
class SavedRun:
    """A portfolio run as it can be opened later."""

    run_key: str
    run_number: int
    name: str | None
    recorded_at: datetime
    dataset_sha256: str
    code_version: str | None
    results: Mapping[str, object] = field(default_factory=dict)
    """``reporting/run_results.run_results``, as the Portfolio page showed it."""


def record_portfolio_run(
    inputs: PortfolioInputs,
    result: BacktestResult,
    weights: pd.DataFrame,
    *,
    results: Mapping[str, object] | None = None,
    recorded_at: datetime | None = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> bool:
    """Record ``result`` under ``inputs``' key unless that key is already there.

    Returns whether it was new. One transaction: the parent row is inserted with
    ``ON CONFLICT DO NOTHING … RETURNING``, and only if it came back (the key was
    new) are its number, metrics and weights written, so a repeat writes nothing at
    all. ``results`` and the daily returns are written with their own ``DO NOTHING``,
    so a run recorded before results were saved gets them the next time it is shown.
    The caller can leave ``results`` out once ``has_results`` says they are there."""
    key = inputs.run_key
    equity = weights.columns[0]
    with connect(db_path, _OPEN_TABLES) as conn:
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
            conn.execute(_NUMBER_NEW_RUNS)
        if results is not None:
            saved = conn.execute(
                "INSERT INTO portfolio_run_results VALUES (?, ?) "
                "ON CONFLICT DO NOTHING RETURNING run_key",
                [key, json.dumps(results, sort_keys=True, default=_iso)],
            ).fetchall()
            if saved:
                _insert_daily(conn, key, result)
        conn.execute("COMMIT")
    return bool(inserted)


def _insert_daily(conn, key: str, result: BacktestResult) -> None:
    # A frame rather than executemany: thousands of rows a run, written while the page
    # waits, and DuckDB reads a frame in one columnar insert.
    daily = pd.DataFrame(
        {
            "run_key": key,
            "date": result.optimised.gross.index,
            "optimised_gross": result.optimised.gross.to_numpy(),
            "optimised_net": result.optimised.net.to_numpy(),
            "baseline_gross": result.baseline.gross.to_numpy(),
            "baseline_net": result.baseline.net.to_numpy(),
            "equity": result.index_returns[TargetRole.EQUITY].to_numpy(),
            "bond": result.index_returns[TargetRole.BOND].to_numpy(),
        }
    )
    conn.register("new_daily", daily)
    conn.execute("INSERT INTO portfolio_run_daily SELECT * FROM new_daily")
    conn.unregister("new_daily")


def has_results(run_key: str, *, db_path: Path = DEFAULT_DB_PATH) -> bool:
    """Whether the run has saved results, so a caller needn't build them again."""
    with connect(db_path, _OPEN_TABLES) as conn:
        return (
            conn.execute(
                "SELECT 1 FROM portfolio_run_results WHERE run_key = ?", [run_key]
            ).fetchone()
            is not None
        )


def rename_portfolio_run(
    run_key: str, name: str | None, *, db_path: Path = DEFAULT_DB_PATH
) -> None:
    """Set a run's name, or clear it with nothing but whitespace."""
    name = (name or "").strip() or None
    with connect(db_path, _OPEN_TABLES) as conn:
        conn.execute("UPDATE portfolio_run_labels SET name = ? WHERE run_key = ?", [name, run_key])


def get_saved_run(run_key: str, *, db_path: Path = DEFAULT_DB_PATH) -> SavedRun | None:
    """A run with its saved results, or ``None`` if it has none: recorded before
    results were saved, and not shown since."""
    with connect(db_path, _OPEN_TABLES) as conn:
        row = conn.execute(
            """
            SELECT p.run_key, l.run_number, l.name, p.recorded_at, p.dataset_sha256,
                   p.code_version, r.results
            FROM portfolio_runs AS p
            JOIN portfolio_run_labels AS l USING (run_key)
            JOIN portfolio_run_results AS r USING (run_key)
            WHERE p.run_key = ?
            """,
            [run_key],
        ).fetchone()
    return None if row is None else SavedRun(*row[:6], results=json.loads(row[6]))


def saved_paths(run_key: str, *, db_path: Path = DEFAULT_DB_PATH) -> BacktestResult | None:
    """The run's daily returns and weights in a ``BacktestResult``, so the charts are
    drawn by the same functions as the live page's. Paths only: its metrics, turnover
    and costs are empty, since those are in the saved results. ``None`` without any."""
    with connect(db_path, _OPEN_TABLES) as conn:
        daily = conn.execute(
            "SELECT * EXCLUDE (run_key) FROM portfolio_run_daily WHERE run_key = ? ORDER BY date",
            [run_key],
        ).df()
    if daily.empty:
        return None
    daily.index = pd.to_datetime(daily.pop("date"))
    weights = portfolio_run_weights(run_key, db_path=db_path)
    schedule = pd.DataFrame({TargetRole.EQUITY: weights, TargetRole.BOND: 1 - weights})

    def path(name: str, schedule: pd.DataFrame) -> PortfolioPath:
        empty = pd.Series(dtype=float)
        return PortfolioPath(daily[f"{name}_gross"], daily[f"{name}_net"], schedule, empty, empty)

    return BacktestResult(
        start=weights.index[0],
        end=daily.index[-1],
        optimised=path("optimised", schedule),
        baseline=path("baseline", pd.DataFrame()),
        index_returns=daily[["equity", "bond"]].rename(
            columns={"equity": TargetRole.EQUITY, "bond": TargetRole.BOND}
        ),
        metrics={},
        relative={},
        costs_bps={},
    )


def save_simulation(
    run_key: str,
    results: Mapping[str, object],
    *,
    recorded_at: datetime | None = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> None:
    """Attach a Monte Carlo simulation to its run. The same method and paths again
    gives the same seeded figures, so a repeat keeps the first."""
    with connect(db_path, _OPEN_TABLES) as conn:
        conn.execute(
            "INSERT INTO portfolio_run_simulations VALUES (?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
            [
                run_key,
                results["method"],
                results["paths"],
                recorded_at or datetime.now(),
                json.dumps(results, sort_keys=True, default=_iso),
            ],
        )


def latest_simulation(run_key: str, *, db_path: Path = DEFAULT_DB_PATH) -> Mapping | None:
    """The run's most recent simulation, or ``None`` if none was run."""
    with connect(db_path, _OPEN_TABLES) as conn:
        row = conn.execute(
            "SELECT results FROM portfolio_run_simulations WHERE run_key = ? "
            "ORDER BY recorded_at DESC LIMIT 1",
            [run_key],
        ).fetchone()
    return None if row is None else json.loads(row[0])


def list_portfolio_runs(
    *, limit: int = 200, db_path: Path = DEFAULT_DB_PATH
) -> list[PortfolioRun]:
    """The recorded portfolio runs, newest first."""
    with connect(db_path, _OPEN_TABLES) as conn:
        rows = conn.execute(
            "SELECT * FROM portfolio_runs ORDER BY recorded_at DESC, run_key LIMIT ?", [limit]
        ).fetchall()
    return [
        PortfolioRun(*row[:7], inputs=json.loads(row[7]), code_version=row[8]) for row in rows
    ]


def portfolio_runs_table(*, limit: int = 200, db_path: Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    """One row per portfolio run, newest first, with its number and name, the models
    active for it, its inputs that change from run to run and both portfolios' net
    Sharpe ratios as columns. The models come from the active-model log, which is
    append-only: a run's ``set_at`` times still find them after others are set."""
    with connect(db_path, _CREATE_ACTIVE_MODELS + _OPEN_TABLES) as conn:
        return conn.execute(
            """
            SELECT p.run_key, l.run_number, l.name, p.recorded_at, p.start_date, p.end_date,
                   equity_model.model_name AS equity_model,
                   bond_model.model_name AS bond_model,
                   r.run_key IS NOT NULL AS has_results,
                   CAST(p.inputs ->> 'risk_level' AS INTEGER) AS risk_level,
                   CAST(p.inputs -> 'bounds' ->> 0 AS DOUBLE) AS min_weight,
                   p.equity_run_id, p.bond_run_id, p.code_version,
                   max(m.value) FILTER (WHERE m.portfolio = 'optimised' AND m.basis = 'net'
                                        AND m.metric = 'sharpe') AS net_sharpe,
                   max(m.value) FILTER (WHERE m.portfolio = 'baseline' AND m.basis = 'net'
                                        AND m.metric = 'sharpe') AS baseline_net_sharpe,
                   max(m.value) FILTER (WHERE m.portfolio = 'optimised' AND m.basis = 'net'
                                        AND m.metric = 'information_ratio') AS information_ratio
            FROM portfolio_runs AS p
            JOIN portfolio_run_labels AS l USING (run_key)
            LEFT JOIN portfolio_run_results AS r USING (run_key)
            LEFT JOIN portfolio_run_metrics AS m USING (run_key)
            LEFT JOIN active_models AS equity_model
                ON equity_model.role = 'equity'
                AND equity_model.set_at = CAST(p.inputs ->> 'equity_set_at' AS TIMESTAMP)
            LEFT JOIN active_models AS bond_model
                ON bond_model.role = 'bond'
                AND bond_model.set_at = CAST(p.inputs ->> 'bond_set_at' AS TIMESTAMP)
            GROUP BY ALL
            ORDER BY l.run_number DESC
            LIMIT ?
            """,
            [limit],
        ).df()


def portfolio_run_weights(run_key: str, *, db_path: Path = DEFAULT_DB_PATH) -> pd.Series:
    with connect(db_path, _OPEN_TABLES) as conn:
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
