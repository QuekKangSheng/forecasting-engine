# Forecasting Engine

Forecasts short-horizon returns for liquid equity and bond indices, validates
those forecasts against overfitting, feeds them into portfolio construction, and
reports tail risk. Built for Alpha Norm by Finlytics (IS484).

## Getting started

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). On macOS:
`brew install uv`.

```bash
uv sync --all-extras
```

Then run the dashboard:

```bash
uv run streamlit run app/Home.py
```

It opens at <http://localhost:8501>. Go to the **Data** page and upload the
terminal's original Bloomberg CSV history exports together. The page accepts
their fields as exported, joins them on date, reports data-quality findings and
offers the merged data for download. Each input file may be up to 25 MB.

## Checks

```bash
uv run pytest
```

```bash
uv run ruff check .
```

Both run in CI on every pull request.

## Deployment

The dashboard is hosted on Streamlit Community Cloud, connected to this
repo's `main` branch. Merging a pull request into `main` (which requires
CI to pass) automatically redeploys the live app — no manual push or
hosting step is needed.

## What works today

The left edge of the pipeline: getting data in, and checking it.

- **Bloomberg upload and merge** — any number of original Bloomberg CSV exports,
  with arbitrary fields, joined on date without requiring manual edits.
- **Fama-French factors** — Ken French's daily five-factor file, downloaded on
  request and kept under its content hash for the FF5 benchmark.
- **Generic validation** — checks dates and numeric Bloomberg fields without
  imposing the temporary signal-CSV contract.
- **Robust outlier detection** — uses median absolute deviation on day-over-day
  changes across every numeric Bloomberg column; values are reported, not altered.
- **Gap review** — identifies rows with missing values and lets the user include
  or exclude them from downloads.
- **Data quality report** — on the dashboard's front page: date range,
  per-column completeness, and every flagged observation, expandable by column.

Forecasting, portfolio construction and risk analysis are not built yet.

## Interface

The dashboard uses a restrained GitHub Primer-style light/dark palette. Semantic
green, amber and red are reserved for status. Both themes are defined in
`.streamlit/config.toml`.

Status appears as a lozenge — a short uppercase badge — rather than a coloured
word or a symbol, because it reads at a glance in a list. `app/ui.py` holds that
and the shared presentation helpers.

No emoji anywhere: an internal analytical tool should read as a tool.

## Layout

```
src/forecasting_engine/     core library, never imports Streamlit
  extraction/               active generic Bloomberg merge and validation
  ingest/                   upload checks, Fama-French factors, lag-safe alignment
  store/                    DuckDB history
app/                        Streamlit dashboard, no maths
  ui.py                     lozenges, status rows, shared presentation
docs/                       data specification, design, decisions
tests/                      unit, integration, functional
```

## Documentation

- [Bloomberg exports](docs/bloomberg-exports.md) — accepted export shape and merge
- [Data specification](docs/data-specification.md) — archived temporary signal-CSV contract
- [Quality report contract](docs/quality-report-contract.md) — cross-ticket
  design decisions for the shared report model
- [Outlier detection](docs/outlier-detection.md) — the method, its calibration
  against real data, and the evidence for each choice
- [Market calendars](docs/market-calendars.md) — the calendar source, the
  per-signal mapping, and how gaps are reconciled against it
- [Ingestion consolidation](docs/ingestion-consolidation.md) — historical decision record;
  superseded for the active dashboard by the generic Bloomberg-only workflow
- [Architecture design](docs/superpowers/specs/2026-07-29-forecasting-engine-architecture-design.md)
- [Phase 0 plan](docs/superpowers/plans/2026-07-29-phase-0-scaffold-and-ingestion.md)
