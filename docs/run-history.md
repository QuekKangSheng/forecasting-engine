# Run history

Every model fit the app makes, and every distinct portfolio result, is recorded in DuckDB (`data/forecasting.duckdb`) by
`src/forecasting_engine/store/runs.py`: what was fitted, on which data, under which
settings and code version, how long it took, and what came out. Failures are recorded
too. The **Run history** page lists the runs and opens any one of them.

The Models page also caches results in `data/model_runs/*.pkl`, but that cache keeps
only the latest result for each setting and a re-run overwrites it. The history keeps
every fit, can be queried with SQL, and doesn't depend on Python's pickle format.

## Tables

```
uploads (sha256, filename, size_bytes, row_count, uploaded_at)       ← store/uploads.py
   ▲  sha256 = source_sha256                ▲  sha256 = dataset_sha256
dataset_sources (dataset_sha256, source_sha256, role)                ← store/lineage.py
                 PK (dataset_sha256, source_sha256)
                         ▲  dataset_sha256
runs (run_id PK, role, model_name, status, error, started_at, finished_at,
      dataset_sha256, config_key, settings JSON, code_version)
   ├── run_metrics   (run_id → runs, metric, value)        PK (run_id, metric)
   ├── run_forecasts (run_id → runs, date, predicted, baseline, realised)
   │                                                        PK (run_id, date)
   ▲  run_id
active_models (…, run_id)                                            ← store/active_model.py
```

| Table | One row per | Why it's shaped this way |
|---|---|---|
| `runs` | fit actually executed | An **event log**: append-only, never updated. The same settings run twice are two rows, so a change in results between them (new code, new data) stays visible. Columns that are filtered or sorted on are typed; the remaining settings are one JSON column, so a new setting needs no schema change. |
| `run_metrics` | run × metric | **Long format**: a new metric is a new row, not a new column. `runs_table` turns the headline metrics into columns with one `FILTER` aggregate each. |
| `run_forecasts` | run × date | The out-of-sample forecast, its fold's training-mean baseline and the realised return, so a past run can be inspected or rescored without refitting. |
| `dataset_sources` | dataset × export | A **link table**: one dataset is merged from several exports and one export can go into several datasets, so neither side can hold the other in a column. Re-committing the same files adds nothing (`ON CONFLICT DO NOTHING`). |

## Guarantees

- **One run, one transaction.** A run's `runs` row, metrics and forecasts are written
  between `BEGIN` and `COMMIT`. If any insert fails, none of it is kept; a test forces
  a failure partway through to prove it.
- **Constraints in the database, not just the code.** `run_id` is a primary key,
  `(run_id, metric)` and `(run_id, date)` can't repeat, `status` must be `done` or
  `failed`, and metrics and forecasts must reference an existing run.
- **A missing value is `NULL`, not `NaN`.** A missing forecast or an undefined metric is
  stored as SQL `NULL`, which every SQL reader understands. A metric a run never
  computed has no row at all, which keeps "not computed" apart from "computed, but
  undefined".
- **Recording never breaks a fit.** The job worker calls the recorder after each fit
  ends; if the write fails, the error is printed and the fit's result still stands.
  Loading a cached result is not a fit, so it isn't recorded.
- **Settings round-trip.** Settings are stored as JSON, so tuples come back as lists.
  `record_run` returns the record as it will read back, never a version that only
  existed in memory.

## Portfolio runs

The Portfolio page recomputes the optimiser and the backtest on every view, so a
"run" there isn't a click: logging each computation would log the same result many
times. `store/portfolio_runs.py` records each **distinct** result once.

- **A deterministic key.** `run_key` is the SHA-256 of everything the result depends
  on, as canonical JSON (sorted keys, no spaces, datetimes as ISO text): the data's
  content hash, both active models (their run ids and when they were set), the
  windows, the risk aversion and limits, the trading costs and the code version.
  The same inputs give the same key on any machine. Leave out the code version, and a
  change in how results are computed would hide behind the first row stored for
  those inputs.
- **Idempotent writes.** The run is inserted with `ON CONFLICT DO NOTHING RETURNING
  run_key`. A new key comes back and its metrics and weight schedule are written in
  the same transaction; a repeat comes back empty and writes nothing. So recording on
  every page view is safe: a hundred views of the same settings are one row, a slider
  moved and moved back adds one.
- **Insert, not upsert.** `DO UPDATE SET last_seen = now()` would record when a result
  was last viewed, but by rewriting rows; `DO NOTHING` keeps the log append-only.
- **Store what can't be recomputed.** Metrics and the weight schedule are kept; the
  daily return paths (about 8,000 rows a run) follow from the inputs and the stored
  data, so they aren't.

| Table | One row per | Key |
|---|---|---|
| `portfolio_runs` | distinct set of inputs | `run_key` |
| `portfolio_run_metrics` | run × portfolio × basis × metric | `(run_key, portfolio, basis, metric)` |
| `portfolio_run_weights` | run × rebalance | `(run_key, date)` |

## Lineage

Every figure on the Portfolio page can be traced back to the Bloomberg files behind it:

```
active model ──run_id──▶ run ──dataset_sha256──▶ dataset ──dataset_sources──▶ exports
                                                    │                            │
                                     data/uploads/<sha256>.csv     data/uploads/<sha256>.xlsx
```

- **Exports.** The Data page keeps every uploaded export's bytes as they arrived,
  under their hash and extension, and logs it in `uploads`. When the merged dataset
  is logged, its exports and the links between them are written in the same
  transaction, once per set of files.
- **Active model to run.** Setting a model active records the newest successful
  run of the same settings on the same data (`runs.latest_run`). `run_id` was added
  to `active_models` as an **additive migration** (`ALTER TABLE … ADD COLUMN IF NOT
  EXISTS`): existing databases gain the column, and older rows read `NULL`.
- **No foreign key to `uploads`.** `uploads` is an event log, so a hash appears once
  per upload, and a foreign key needs a unique parent. The links hold hashes, and a
  join picks a hash's first upload.
- **Joins, not recursion.** The chain always has the same four steps, so plain joins
  follow it. A recursive query (`WITH RECURSIVE`) is for chains of unknown length,
  such as datasets built from other datasets, which this app doesn't have.

`dataset_sha256` is the SHA-256 of the committed dataset's canonical CSV
(`ingest.upload.frame_sha256`: no index, ISO dates, `\n` line endings, UTF-8). The
Data page stores the merged dataset under the same hash in `data/uploads/<sha256>.csv`
and logs it in `uploads`. So a run traces back to the exact bytes it was fitted on.

This replaces the app's `dataset_fingerprint` (row count, first and last date, column
names) for this purpose: a fingerprint can't tell two datasets of the same shape
apart, so a single corrected value would pass unnoticed. A content hash changes
with any byte.

`code_version` is `git describe --always --dirty`: the commit the app ran from, with
`-dirty` when the checkout had uncommitted changes, since the commit alone would then
claim code that didn't run. `config_key` is the result cache's key for the settings, so
runs sharing it were the same fit asked for again.

## Queries

Open the database read-only while the app is running, e.g. with the DuckDB CLI:
`duckdb -readonly data/forecasting.duckdb`. Every query below is run by
`tests/unit/test_run_history_doc.py`, so they stay valid as the schema changes.

The latest successful run of each model, per target:

```sql
SELECT role, model_name, max(started_at) AS last_run, arg_max(run_id, started_at) AS run_id
FROM runs
WHERE status = 'done'
GROUP BY role, model_name
ORDER BY role, model_name;
```

Which file each run was fitted on:

```sql
SELECT r.model_name, r.started_at, u.filename, u.uploaded_at, u.row_count
FROM runs AS r
LEFT JOIN (
    SELECT sha256, min(uploaded_at) AS uploaded_at,
           arg_min(filename, uploaded_at) AS filename, any_value(row_count) AS row_count
    FROM uploads
    GROUP BY sha256
) AS u ON u.sha256 = r.dataset_sha256
ORDER BY r.started_at DESC;
```

Whether re-running the same settings changed the result, for example after a code
change:

```sql
SELECT r.config_key, r.started_at, r.code_version, m.value AS signal_rank_ic
FROM runs AS r
JOIN run_metrics AS m USING (run_id)
WHERE m.metric = 'signal_rank_ic'
  AND r.config_key IN (SELECT config_key FROM runs GROUP BY config_key HAVING count(*) > 1)
ORDER BY r.config_key, r.started_at;
```

From each target's current active model back to the exact Bloomberg files. The active
model log is append-only, so the current model is the newest row per target: a window
function numbers each target's rows newest first, and `QUALIFY` keeps row 1. Naming
that step (`WITH current_active AS …`, a CTE) keeps it apart from the joins:

```sql
WITH current_active AS (
    SELECT *
    FROM active_models
    QUALIFY row_number() OVER (PARTITION BY role ORDER BY set_at DESC) = 1
)
SELECT a.role, a.model_name AS active_model, r.run_id, r.finished_at AS fitted_at,
       r.code_version, s.role AS source_role, u.filename AS source_file, s.source_sha256
FROM current_active AS a
JOIN runs AS r ON r.run_id = a.run_id
JOIN dataset_sources AS s ON s.dataset_sha256 = r.dataset_sha256
LEFT JOIN (
    SELECT sha256, arg_min(filename, uploaded_at) AS filename
    FROM uploads
    GROUP BY sha256
) AS u ON u.sha256 = s.source_sha256
ORDER BY a.role, s.role DESC, u.filename;
```

Which portfolio settings beat the 50/50 benchmark after costs, best first:

```sql
SELECT p.inputs ->> 'risk_aversion' AS risk_aversion,
       p.inputs -> 'bounds' ->> 0 AS min_weight,
       max(m.value) FILTER (WHERE m.portfolio = 'optimised') AS net_sharpe,
       max(m.value) FILTER (WHERE m.portfolio = 'baseline') AS baseline_net_sharpe
FROM portfolio_runs AS p
JOIN portfolio_run_metrics AS m USING (run_key)
WHERE m.basis = 'net' AND m.metric = 'sharpe'
GROUP BY ALL
HAVING net_sharpe > baseline_net_sharpe
ORDER BY net_sharpe DESC;
```

How often each model fails, and how long it takes:

```sql
SELECT model_name,
       count(*) AS runs,
       count(*) FILTER (WHERE status = 'failed') AS failed,
       round(avg(date_diff('millisecond', started_at, finished_at)) / 1000, 1) AS avg_seconds
FROM runs
GROUP BY model_name
ORDER BY runs DESC;
```

How often each model called the direction of the return against its baseline right,
from the stored forecasts, without refitting. Only dates where the model took a view
count: where the forecast equals the baseline there is no direction to be right about.

```sql
SELECT r.model_name,
       count(*) AS dates_with_a_view,
       round(avg(CASE WHEN sign(f.predicted - f.baseline) = sign(f.realised - f.baseline)
                      THEN 1 ELSE 0 END), 3) AS direction_hit_rate
FROM runs AS r
JOIN run_forecasts AS f USING (run_id)
WHERE f.predicted IS NOT NULL AND f.realised IS NOT NULL AND f.predicted <> f.baseline
GROUP BY r.model_name
ORDER BY direction_hit_rate DESC;
```

## Not covered yet

- **Schema versions.** Tables change with `CREATE … IF NOT EXISTS` and `ALTER TABLE …
  ADD COLUMN IF NOT EXISTS`, which is enough for added columns but can't rename or
  reshape one. A migrations table, applying numbered changes in order, would.
- **Older records.** Datasets committed before exports were logged have no
  `dataset_sources` rows, and models set active before `run_id` existed have none;
  both stay as they are rather than being guessed at.
