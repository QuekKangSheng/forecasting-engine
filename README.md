# Forecasting Engine

Forecasts short-horizon returns for liquid equity and bond indices, validates
those forecasts against overfitting, and feeds them into an equity/bond portfolio
that is backtested against a 50/50 benchmark. Built for Alpha Norm by Finlytics
(IS484).

## Getting started

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). On macOS:
`brew install uv`. `.python-version` pins 3.13, the version CI runs on; uv uses
it, and downloads it if it isn't installed.

```bash
uv sync --all-extras
```

Then run the dashboard:

```bash
uv run streamlit run app/Home.py
```

It opens at <http://localhost:8501>. Two things about how it runs, both there to keep
pages quick to open and to respond:

- It does not watch the source files for edits. While developing, start it with
  `uv run streamlit run app/Home.py --server.fileWatcherType auto` and it reruns
  whenever a file is saved.
- It holds `data/forecasting.duckdb` open for as long as it runs, so no other program
  can open that file until the app is stopped.

A Run spreads independent work over up to
4 processes; on a machine with more cores, set `FORECASTING_WORKERS` (for
example `FORECASTING_WORKERS=8`) before starting it. The results are the same
either way. `app/Home.py` is the entry file: it builds the sidebar navigation
over the six pages in `app/app_pages/`, which follow the order a portfolio
manager works in:

1. **Home** — where the pipeline stands (data, models, portfolio) and what to do
   next. It only reads what the other pages left and the active models; it never
   fits a model or runs a backtest.
2. **Data** — upload the terminal's original Bloomberg history exports (CSV or
   XLSX), with the equity and bond target indices in one box and every other
   signal in another. The page accepts their fields as exported, joins them on
   date, reports data-quality findings and offers the merged data for download. Each input file may be up to 25 MB. Click **Use
   Updated Data** to hand the data to the other pages.
3. **Models** — one tab per target index. **Run** fits the ticked model families
   (all of them by default) in the background, under the same settings, and
   compares them in one table, with pass/fail promotion gates; **Run all
   targets** does both tabs at once. Set the model to carry forward as the
   target's **active model**, and check whether its forecast direction would
   have paid.
4. **Portfolio Optimizer** — once an equity and a bond model are active, splits
   between the two indices at each rebalance and backtests that allocation
   against an equal-weight benchmark.
5. **Scenarios** — shock signals the active polynomials read and see each index's
   forecast, and the optimised portfolio, beside the base case, without refitting.
6. **Run history** — every model fit and every distinct Portfolio result recorded
   in DuckDB, newest first; open one to see its settings, results and the data it
   was fitted on.

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
- **Data quality report** — on the Data page: date range, per-column
  completeness, and every flagged observation, expandable by column. Home shows
  how many observations it flagged.
- **Lag-safe alignment** — signals are read as of each target date and lagged
  one row, so a model never sees a value before it was published.

**Forecasting and validation**

- **Model families**, each forecasting a 1-, 5-, 10- or 20-day forward return:
  - a naive training-mean baseline, the bar every model has to beat;
  - the Fama-French five-factor benchmark (equity only);
  - a polynomial, one at a time: derived automatically from the signals (a grid of
    lasso fits at degrees 1 to 3 and a shrinkage fit, chosen per run),
    or the user's own shape (placeholders pointed at signals) with only a scale and
    intercept fitted;
  - each tab's research model, run as the model research ran it, with a caption
    saying its score is optimistic: for equity a polynomial on fixed economic
    inputs (HY OAS, IG OAS, VIX and the 2s10s slope), and for bond one on fixed
    carry inputs (the 10-year yield and breakeven, the 2s10s slope, HY and IG OAS,
    VIX and FX implied volatility);
  - on the Bond tab, a second research model on the calendar of each forecast
    window (how many month-ends, month starts and jobs-report Fridays it holds),
    since bond returns are higher on a month's last trading day; it carries
    information at the 1- and 5-day horizons only;
  - machine learning: XGBoost and LightGBM, tuned with Optuna on a rolling
    schedule, with SHAP feature attribution.
- **Walk-forward validation** — every model on both targets is trained on the
  latest 504 trading days and scored on the next 20, with a 20-day embargo
  between the two, under one shared set of settings that the results table states.
  The two fixed-input research models alone train on every earlier day, on the
  same test windows.
- **Overfitting checks** — Signal Rank IC (what a model adds to its own average
  forecast, out of sample) with its t-statistic, crash-day diagnostics, and a
  promotion gate on both: Signal Rank IC above 0.02 and t above 2.
- **Active model** — the model chosen to carry forward for each index, saved in
  DuckDB, with a confirmation step for a model that failed both gates.
- **Run history** — every model fit recorded in DuckDB, failures included: when it
  ran, the SHA-256 of the exact data, its settings and code version, how long it
  took, and its metrics and forecasts. A page lists the runs and traces each one to
  the uploaded file it was fitted on. Every distinct Portfolio result is recorded
  too, once, under a key hashed from everything it depends on. See [docs/run-history.md](docs/run-history.md)
  for the tables, guarantees and example queries.
- **Directional P&L** — for one model and one index, what holding the index only
  when the model's signal (its forecast less its own average forecast) is above
  zero would have earned against holding it throughout, over the whole
  out-of-sample period, with hit rate and share of days invested.

**Portfolio**

- **Optimiser** — a two-asset Markowitz allocation. Each 1–5 risk level targets a
  volatility evenly spaced between the lowest-risk and highest-risk mixes the
  weight bounds allow, and tilts from there towards whichever index the active
  models' signals favour, by as much as their record has earned and only once that
  record is statistically significant. The long-run premium it starts from blends
  the data's own with a published one (6.6% a year, worth 30 years). It rebalances
  every forecast horizon, trading straight back to each target.
- **Backtest against 50/50** — runs the optimised weights and a monthly-reset
  50/50 benchmark through the same days, before and after trading costs, and
  compares Sharpe, Sortino, Calmar and maximum drawdown side by side.
- **Significance of the outperformance** — a paired, Newey-West-adjusted
  t-statistic on the daily difference between the two, with a plain-language
  verdict at 95% confidence, so luck isn't mistaken for skill. A second one tests
  the portfolio against the same risk level with no forecasts, so the forecasts are
  the only difference.
- **Tail risk** — historical one-day VaR and CVaR at 95% and 99% beside maximum
  drawdown, for both portfolios; the Monte Carlo methods are listed as "Not run"
  until computed.
- **Monte Carlo simulation** — simulates the next forecast horizon from the latest
  weights, centred on the active models' forecasts, by historical bootstrap or a
  fitted normal. It fills the Monte Carlo rows of the tail-risk table and adds VaR,
  CVaR and maximum drawdown over the whole horizon. GARCH is not built yet.

**Scenarios**

- **Signal shocks** — choose signals the active polynomials read and give each a
  new value or a % change; the shocks are validated (a number, above zero where
  the series always is, within 10 times its observed range) and then both indices'
  active functions are evaluated again on the shocked inputs, without refitting,
  beside the base case. A shocked input outside the model's training range is
  flagged as an extrapolation. The active polynomial's fitted function is saved in
  DuckDB with the active model for this.
- **Portfolio impact** — the shocked forecasts carried through to the optimised
  portfolio at its latest weights, held as they are (the Portfolio page's last
  result on the committed data and the models active now): return, volatility and
  max drawdown over the next horizon beside the base case, from the same Monte
  Carlo paths, so the difference is the shock's alone.
- **Re-optimise under shock** — an optional switch that solves the latest
  allocation again from the shocked forecasts, with the same risk level, limits,
  covariance and forecast trust, and shows the re-solved weights and their
  simulated results in a separate, clearly labelled column. The held weights and
  the Portfolio page are left as they are.

### Results on the full history

On the Bloomberg data from 31 Aug 2006 to Nov 2026 (16 Nov 2026 rerun):

- The derived polynomial passes the gate for equity at 1, 5 and 10 days, and
  misses at the default 20 days (t 1.98).
- No bond model passes at any horizon.
- The optimiser's risk levels behave as designed, but the forecasts add no
  value after costs at any horizon. With the team's default pair (derived
  equity, machine-learning bond) they cost significantly at the 5-day horizon.
  With the trust gate ignored, no level did significantly better than without
  forecasts (t above 2; best +1.56).

See [methodology §20](docs/methodology.md) and
[the evidence](docs/evidence/2026-11-16-full-history.md).

## Interface

The dashboard is styled as an operations console: one dark theme, designed as
such rather than flipped from a light one, with one blue for whatever can be
acted on. The theme is in `.streamlit/config.toml`; every colour is also in
`app/tokens.py`, which the stylesheet and the charts draw from, and
`tests/unit/test_tokens.py` fails if the two drift apart. Both typefaces, IBM Plex
Sans and IBM Plex Mono, are served by the app from `app/static`.

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
  models/                   naive, Fama-French, polynomial (lasso and shrinkage
                            fits, user-supplied, the three research models) and
                            boosted forecasters
  validation/               walk-forward splitter, metrics, crash checks, gates
  portfolio/                optimiser, backtest, performance, directional P&L
  risk/                     historical VaR and CVaR, Monte Carlo simulation
  scenario/                 signal shocks and their portfolio impact
  reporting/                tables and labels the dashboard shows, the run report
  store/                    DuckDB run history, uploads and lineage, active
                            models, portfolio runs
app/                        Streamlit dashboard, no maths
  Home.py                   entry file: the sidebar navigation
  app_pages/                Home, Data, Models, Portfolio Optimizer, Scenarios,
                            Run history
  model_jobs.py             the background queue a Run is fitted on
  ui.py                     lozenges, status rows, shared presentation
  glossary.py               the plain-language explanations shown on hover
docs/                       data specification, design, decisions
tests/                      unit, integration, functional
```

## Documentation

- [Methodology](docs/methodology.md) — every step from export to score, and every
  parameter, checked against the code by the test suite
- [Validation review](docs/validation-review.md) — the 18 Sep 2026 review of the
  validation pipeline, kept as a record; a note at the top lists what has since
  been superseded
- [Run history](docs/run-history.md) — the DuckDB tables, their guarantees and
  example queries
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
  and its [team guide](docs/architecture.html) — the July 2026 design, before the
  build; a note at the top of each lists where the built app differs
- [Phase 0 plan](docs/superpowers/plans/2026-07-29-phase-0-scaffold-and-ingestion.md)
