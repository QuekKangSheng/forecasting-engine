# Run history

Every model fit the app makes is recorded in DuckDB (`data/forecasting.duckdb`) by
`src/forecasting_engine/store/runs.py`: what was fitted, on which data, under which
settings and code version, how long it took, and what came out. Failures are recorded
too. The **Run history** page lists the runs and opens any one of them.

The Models page also caches results in `data/model_runs/*.pkl`, but that cache keeps
only the latest result for each setting and a re-run overwrites it. The history keeps
every fit, can be queried with SQL, and doesn't depend on Python's pickle format.

## Tables

```
uploads (sha256, filename, size_bytes, row_count, uploaded_at)       ← store/uploads.py
   ▲  sha256 = dataset_sha256
runs (run_id PK, role, model_name, status, error, started_at, finished_at,
      dataset_sha256, config_key, settings JSON, code_version)
   ├── run_metrics   (run_id → runs, metric, value)        PK (run_id, metric)
   └── run_forecasts (run_id → runs, date, predicted, baseline, realised)
                                                            PK (run_id, date)
```

| Table | One row per | Why it's shaped this way |
|---|---|---|
| `runs` | fit actually executed | An **event log**: append-only, never updated. The same settings run twice are two rows, so a change in results between them (new code, new data) stays visible. Columns that are filtered or sorted on are typed; the remaining settings are one JSON column, so a new setting needs no schema change. |
| `run_metrics` | run × metric | **Long format**: a new metric is a new row, not a new column. `runs_table` turns the headline metrics into columns with one `FILTER` aggregate each. |
| `run_forecasts` | run × date | The out-of-sample forecast, its fold's training-mean baseline and the realised return, so a past run can be inspected or rescored without refitting. |

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

## Lineage

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

- **Portfolio runs.** The optimiser and backtest recompute on every page view rather
  than on a button, so recording them needs a deduplication key first, or every
  slider move would be a "run".
- **Raw exports.** Lineage reaches the merged dataset, which is stored and logged; the
  individual Bloomberg exports it was merged from are not logged one by one.
- **Which run an active model came from.** An active model stores its own forecasts
  and function; linking it to its `run_id` would tie the two logs together.
