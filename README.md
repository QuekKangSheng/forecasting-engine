# Forecasting Engine

Forecasts short-horizon returns for liquid equity and bond indices, validates
those forecasts against overfitting, and feeds them into an equity/bond portfolio
that is backtested against a 50/50 benchmark. Built for Alpha Norm by Finlytics
(IS484).

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

It opens at <http://localhost:8501>. The pages follow the order a portfolio
manager works in:

1. **Data** — upload the terminal's original Bloomberg history exports (CSV or
   XLSX), with the equity and bond target indices in one box and every other
   signal in another. The page accepts their fields as exported, joins them on
   date, reports data-quality findings and offers the merged data for download. Each input file may be up to 25 MB. Click **Use
   Updated Data** to hand the data to the other pages.
2. **Models** — one tab per target index. **Run** fits every model family under
   the same settings and compares them in one table, with pass/fail promotion
   gates. Set the model to carry forward as the target's **active model**, and
   check whether its forecast direction would have paid.
3. **Portfolio Optimizer** — once an equity and a bond model are active, splits
   between the two indices at each rebalance and backtests that allocation
   against an equal-weight benchmark.

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

**Data in, and checked**

- **Bloomberg upload and merge** — any number of original Bloomberg CSV or XLSX
  exports, with arbitrary fields, joined on date without requiring manual edits.
- **Fama-French factors** — Ken French's daily five-factor file, downloaded on
  request and kept under its content hash for the FF5 benchmark.
- **Generic validation** — checks dates and numeric Bloomberg fields.
- **Robust outlier detection** — uses median absolute deviation on day-over-day
  changes across every numeric Bloomberg column; values are reported, not altered.
- **Gap review** — identifies rows with missing values and lets the user include
  or exclude them from downloads.
- **Data quality report** — on the dashboard's front page: date range,
  per-column completeness, and every flagged observation, expandable by column.
- **Lag-safe alignment** — signals are read as of each target date and lagged
  one row, so a model never sees a value before it was published.

**Forecasting and validation**

- **Model families**, each forecasting a 1- or 5-day forward return:
  - a naive training-mean baseline, the bar every model has to beat;
  - the Fama-French five-factor benchmark (equity only);
  - a polynomial, one at a time: derived automatically from the signals, or the
    user's own shape (placeholders pointed at signals) with only a scale and
    intercept fitted;
  - machine learning: XGBoost and LightGBM, tuned with Optuna on a rolling
    schedule, with SHAP feature attribution.
- **Walk-forward validation** — every model is trained on a past window and
  scored on the days after it, with an embargo between the two, under one shared
  set of settings that the results table states.
- **Overfitting checks** — pooled out-of-sample Rank IC, PBO (probability of
  backtest overfitting), crash-day diagnostics, and promotion gates on Rank IC
  and PBO.
- **Active model** — the model chosen to carry forward for each index, saved in
  DuckDB, with a confirmation step for a model that failed both gates.
- **Directional P&L** — for one model and one index, what holding the index only
  when the forecast says it will rise would have earned against holding it
  throughout, over the whole out-of-sample period, with hit rate and share of
  days invested.

**Portfolio**

- **Mean-variance optimiser** — combines the active equity and bond forecasts
  with recent risk into long-only weights at each rebalance.
- **Backtest against 50/50** — runs the optimised weights and a monthly-reset
  50/50 benchmark through the same days, before and after trading costs, and
  compares Sharpe, Sortino, Calmar and maximum drawdown side by side.

Tail-risk reporting (VaR and CVaR) and significance checks on the backtest are
not built yet.

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
  features/                 signal screening
  models/                   naive, Fama-French, polynomial and boosted forecasters
  validation/               walk-forward splitter, metrics, PBO, crash checks, gates
  portfolio/                optimiser, backtest, performance, directional P&L
  reporting/                tables and labels the dashboard shows
  store/                    DuckDB upload and active-model history
app/                        Streamlit dashboard, no maths
  app_pages/                Home, Data, Models, Portfolio Optimizer
  ui.py                     lozenges, status rows, shared presentation
  glossary.py               the plain-language explanations shown on hover
docs/                       data specification, design, decisions
tests/                      unit, integration, functional
```

## Documentation

- [Methodology](docs/methodology.md) — every step from export to score, and every
  parameter, checked against the code by the test suite
- [Validation review](docs/validation-review.md) — decisions behind the
  validation metrics and gates
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
