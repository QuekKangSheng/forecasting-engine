# Forecasting Engine — Architecture Design

**Project:** IS484 Forecasting Engine for Strategic Asset Allocation
**Team:** Finlytics · **Sponsor:** Alpha Norm (Dr. Catalin Burlacu)
**Date:** 2026-07-29
**Status:** Approved — pending answers to the open questions in §11

> **Note added 16 Nov 2026: the original design, not the built app.** This is the
> design drafted on 29 Jul 2026, before the build, and is kept as that record. The
> app as built differs from it in these main ways:
>
> - **Pages.** `app/Home.py` builds the navigation over six pages in
>   `app/app_pages/`: Home, Data, Models, Portfolio Optimizer, Scenarios and Run
>   history. There is no `app/pages/` folder.
> - **Data.** The Data page takes Bloomberg's original CSV and XLSX exports and
>   merges them on date (`extraction/`). The authored CSV contract was retired for
>   an open schema.
> - **How a Run runs.** There is no `RunConfig` or `pipeline.py`. The Models page
>   queues a Run for one background worker thread in the Streamlit server
>   (`app/model_jobs.py`), which fits the ticked models in turn and spreads
>   independent parts of a fit over up to four processes by default
>   (`parallel.py`). Each result is cached under `data/model_runs/`, keyed by a
>   hash of the data and settings it depends on.
> - **Store.** One DuckDB file, `data/forecasting.duckdb`, holds an append-only log
>   of model runs, each with a random run id rather than a config hash, and also the
>   uploads and their lineage, each target's active model and every distinct
>   Portfolio result ([run history](../../run-history.md)).
> - **Model families.** A naive training-mean baseline, the Fama-French
>   five-factor benchmark (equity only), the polynomial (user-supplied, or derived
>   from lasso fits at degrees 1 to 3 and a shrinkage fit), a research model on
>   each target (a polynomial on fixed economic inputs for equity, one on fixed
>   carry inputs for bond), and XGBoost and LightGBM. There is no PySR.
> - **Validation.** No PBO, DSR or CPCV: a model is promoted on its Signal Rank IC
>   and its t-statistic. The windows are 504 training days, 20 test days and a
>   20-day embargo, not 756, 63 and 5 (the two research models alone train on
>   every earlier day), and the default horizon is 20 days, not 5.
> - **Portfolio.** The optimiser is a two-asset Markowitz rule in
>   `portfolio/optimize.py`, not PyPortfolioOpt.
>
> For the app as it is now, read [the methodology](../../methodology.md) and
> [the README](../../../README.md).

---

## 1. Purpose

Alpha Norm needs an internal analytical tool that forecasts short-horizon returns for
liquid equity and bond indices, validates those forecasts against overfitting, feeds
them into portfolio construction, and reports tail risk under both history and
simulation — all in one place.

This document specifies how the system is built. It does not restate the project
proposal; it records the architectural decisions that the proposal leaves open.

## 2. Constraints that shaped the design

| Constraint | Consequence |
|---|---|
| Runs locally; AWS deferred | No infra work in scope. Compute boundary kept clean so heavy jobs *could* move later. |
| Two active developers — Raphael (data), Joash (models) | One seam, not a four-way contract lattice. Optimise for two people not blocking each other. |
| No data in hand at design time | Schema-first. The CSV contract is authored, not inferred, and synthetic fixtures stand in until real exports arrive. |
| 12 weeks, 6 sprints, 3 UAT rounds | Nothing speculative gets built. Framework-building is explicitly rejected. |
| Sponsor wants "function terms and coefficients" | Interpretable polynomial is the production path; tree models are comparators only. |

## 3. Decision: layered core library with a thin Streamlit shell

All computation lives in `src/forecasting_engine/`, an installable package that never
imports Streamlit. `app/` renders and holds no maths.

**Alternatives rejected.**

*Notebook-first, harden later.* Fastest to a first forecast and conventional for quant
work, but the deliverables include a deployment-ready app, a traceable test matrix and
reproducibility guarantees. Notebook-first projects reliably fail to converge on those,
and here the last two sprints are already committed to portfolio work, scenario analysis
and two UAT rounds. Notebooks are retained for exploration *inside* this architecture.

*Config-driven DAG runner with content-addressed caching.* Reproducibility becomes
structural and caching directly mitigates the runtime risk, but building it would consume
Sprints 1–2. Rejected as framework-building.

**One idea retained from the DAG approach:** a frozen `RunConfig` fully determines a run,
and its content hash is the run id. Reproducibility and cache keys fall out of that
without an orchestration engine.

## 4. Pipeline and ownership

Ten stages, matching the To-Be workflow in the proposal.

| # | Stage | Produces | Owner |
|---|---|---|---|
| 1 | Load & validate | `RawFrame` | Raphael |
| 2 | Quality checks | `QualityReport` | Raphael |
| 3 | Align & lag | **`FeaturePanel`** | Raphael |
| 4 | Screen signals | selected features | Raphael |
| 5 | Fit forecasters | fitted `Forecaster` | Joash |
| 6 | Walk-forward validation | OOS predictions | Joash |
| 7 | Score & gate | `ModelComparison` | Joash |
| 8 | Optimise portfolio | `Allocation` | shared |
| 9 | Risk analysis | `RiskReport` | shared |
| 10 | Scenario shocks | `ScenarioResult` | shared |

**The seam is stage 3 → 4.** Raphael owns everything producing a `FeaturePanel`; Joash
owns everything consuming one. Because the type is frozen and synthetic fixtures exist on
both sides, neither developer waits on the other.

## 5. The three contracts

### 5.1 `FeaturePanel` — a provably lag-safe dataset

```python
@dataclass(frozen=True)
class FeaturePanel:
    frame:      pd.DataFrame      # DatetimeIndex; lagged signals + forward targets
    signals:    tuple[str, ...]   # feature column names
    targets:    tuple[str, ...]   # forward-return column names
    lag_days:   int               # observation lag actually applied (>= 1)
    provenance: Provenance        # source files, content hashes, build timestamp
```

Constructible only by `align_and_lag()`. `__post_init__` asserts `lag_days >= 1` and that
no target column appears in `signals`. No fitting function anywhere accepts a bare
DataFrame, so there is no type-legal way to train on unlagged data.

`panel.matrix(idx)` returns `(X, y)` for a given index — the single accessor models use.

### 5.2 `Forecaster` — the model protocol

```python
class Forecaster(Protocol):
    name: str

    def fit(self, panel: FeaturePanel, train: pd.DatetimeIndex) -> None: ...
    def predict(self, panel: FeaturePanel, idx: pd.DatetimeIndex) -> pd.Series: ...
    def describe(self) -> ModelDescription: ...
```

`describe()` returns terms and coefficients — the sponsor's stated output format, and the
reason the polynomial remains the production baseline.

Implementations: `UserPolynomial` (applies a supplied function, no fitting),
`DerivedPolynomial` (PolynomialFeatures degree ≤ 5 + LASSO/ElasticNet), `FamaFrench5`
(statsmodels OLS benchmark), `Boosted` (XGBoost/LightGBM + Optuna), `Symbolic` (PySR,
stretch only).

### 5.3 `RunConfig` — everything needed to reproduce a run

```python
@dataclass(frozen=True)
class RunConfig:
    data:      DataSpec        # csv path + content hash, date range
    features:  FeatureSpec     # screening rule, top-k
    model:     ModelSpec       # type, hyperparams, or user-supplied polynomial
    split:     SplitSpec       # train length, test length, embargo days
    portfolio: PortfolioSpec
    seed:      int
```

Hashes to the run id persisted in DuckDB. Same config in, same numbers out. Cache keys
come free, which is the practical mitigation for the sponsor's warning that degree-5
searches "might take a lot of time to run".

## 6. Leak prevention, structurally

Success Criterion 1 requires look-ahead bias to be *structurally prevented*. Four rules,
enforced in code rather than by convention:

1. Every signal shifts by at least one day inside `align_and_lag()`. Nothing else shifts anything.
2. Slower series join with `merge_asof(direction="backward")`. Forward-fill only, with a documented cap — `bfill` and `interpolate` are banned by a lint rule.
3. Revised macro series carry their release date, not their reference date (FRED/ALFRED vintages).
4. Walk-forward splits purge overlapping label windows and add an embargo gap.

`PurgedWalkForward` is the only splitter in the codebase:

```
[———— TRAIN 756d ————][EMB 5d][— TEST 63d —]   → rolls forward, refits, repeats
```

Overlapping forward returns leak across a naive split; the embargo is the gap that stops it.

Risk 2's own mitigation — re-run with one extra day of lag and check whether predictive
power collapses — becomes an automated test. The same applies to Risk 1: a PBO above
threshold fails the build rather than being noticed in review.

## 7. Repository layout

```
forecasting-engine/
├── pyproject.toml                 # uv-managed, single source of dependencies
├── docs/
│   ├── data-specification.md      # THE CSV contract — a named deliverable
│   ├── architecture.md
│   └── superpowers/specs/
├── src/forecasting_engine/
│   ├── config.py                  # RunConfig, DataSpec, ModelSpec, SplitSpec
│   ├── fixtures.py                # synthetic data generator (CLI entry point)
│   ├── ingest/                    # ── Raphael ──
│   │   ├── schema.py              # column contract (pandera)
│   │   ├── loader.py              # CSV → RawFrame
│   │   ├── quality.py             # → QualityReport
│   │   └── align.py               # → FeaturePanel     ★ the gate
│   ├── features/screening.py      # univariate IC screening
│   ├── models/                    # ── Joash ──
│   │   ├── base.py                # Forecaster protocol, ModelDescription
│   │   ├── polynomial.py          # user-supplied + derived
│   │   ├── famafrench.py          # statsmodels OLS benchmark
│   │   ├── boosted.py             # XGBoost / LightGBM + Optuna
│   │   └── symbolic.py            # PySR — stretch only
│   ├── validation/
│   │   ├── splitters.py           # PurgedWalkForward, CPCV
│   │   ├── metrics.py             # IC, RankIC, RMSE, crash recall
│   │   └── overfitting.py         # PBO (CSCV), DSR
│   ├── portfolio/
│   │   ├── optimise.py            # PyPortfolioOpt mean-variance
│   │   └── performance.py         # Sharpe/Sortino/Calmar/MaxDD + Newey-West t-stat
│   ├── risk/{tail,montecarlo}.py  # VaR, CVaR, MaxDD, Monte Carlo
│   ├── scenario/shock.py          # re-predict under shocked inputs, no refit
│   ├── store/runs.py              # DuckDB persistence
│   └── pipeline.py                # RunConfig → RunResult
├── app/                           # Streamlit — no maths
│   ├── Home.py
│   └── pages/1_Data.py … 5_Scenario.py
├── notebooks/                     # research; outputs stripped on commit
├── tests/{unit,integration,fixtures}/
└── .github/workflows/ci.yml       # pytest + ruff on every push
```

Module boundary rule: a file that grows past roughly 300 lines is doing too much and gets
split. Each module answers three questions — what does it do, how do you call it, what
does it depend on.

## 8. Research workflow

Notebooks are clients of the library, not rivals to it. The dependency runs one way:
notebooks import from `src/`, and nothing imports from a notebook.

```python
%load_ext autoreload
%autoreload 2                       # edit src/, next cell picks it up

from forecasting_engine.ingest import load, align_and_lag
from forecasting_engine.validation import PurgedWalkForward, rank_ic

panel = align_and_lag(load("data/raw/signals.csv"))
folds = PurgedWalkForward(train=756, test=63, embargo=5).split(panel)
```

Three lines produce a leak-free, embargoed dataset; everything after is ordinary ad-hoc
pandas. An editable install plus `autoreload` is what makes this feel native — without it
the pattern gets abandoned.

**Promotion rule.** A cell that produced a *finding* stays in the notebook. A cell that
produced a *capability* — anything worth running twice — moves into `src/` with a test.

**Why one shared evaluation harness matters.** Gap #1 in the proposal is that there is
"no structured process to derive, compare, and validate alternative specifications against
a consistent benchmark". A single `Forecaster` protocol behind a single harness *is* that
process. Adding a model means writing `fit`, `predict` and `describe`; it is then scored
automatically on the same folds, the same metrics and the same PBO/DSR as every other
model. Evaluating models in separate notebooks would rebuild the exact gap the project
exists to close.

## 9. Testing

| Layer | What it proves | Cadence |
|---|---|---|
| Unit | Schema validators, lag arithmetic, metric maths against hand-computed values on small frames | every push |
| Property | Leak audits: extra-lag re-run, embargo boundary, no target reachable as a feature | every push |
| Integration | Full `RunConfig` → `RunResult` over fixtures; polynomial path bit-identical across runs | every push |
| Functional | Dashboard journeys traced to user stories — this is the test matrix deliverable | pre-UAT |
| Performance | Wall-clock for degree-5 search and Monte Carlo, recorded so Risk 8 is measured | weekly |

Determinism is testable and therefore tested. The polynomial path must return identical
numbers for identical inputs. ML comparators are seeded and asserted within tolerance,
consistent with the non-determinism the sponsor has already accepted.

Fixtures are synthetic CSVs matching `docs/data-specification.md`, generated with
deliberate gaps, duplicates and a synthetic crash window so the quality checks and crash
recall metric have something to detect.

## 10. Roadmap

| Phase | Dates | Ships | Lead |
|---|---|---|---|
| Phase 0 | now – 23 Aug | Repo, pyproject, CI, data specification, fixture generator, `FeaturePanel` frozen | Raphael |
| Sprint 1 | 24 Aug – 6 Sep | Ingest, quality report, align & lag with full tests. Fama-French baseline stub | Raphael |
| Sprint 2 | 7 – 20 Sep | Signal screening (Raphael). Polynomial fit + user-supplied path, `PurgedWalkForward` (Joash) | both |
| Sprint 3 | 21 Sep – 4 Oct | IC/RankIC/RMSE, PBO, DSR, crash recall. Dashboard shell. **UAT 1** | both |
| Sprint 4 | 5 – 18 Oct | XGBoost/LightGBM + Optuna. Model comparison view. **UAT 2** | Joash |
| Sprint 5 | 19 Oct – 1 Nov | Mean-variance optimisation, performance metrics, Newey-West t-stat, VaR/CVaR | both |
| Sprint 6 | 2 – 15 Nov | Scenario shocks, Monte Carlo, DuckDB run history, handover docs. **UAT 3** | both |

**Deviation from the proposal's Gantt:** portfolio optimisation moves from Sprints 3–4 to
Sprint 5, keeping the forecasting engine ahead of UAT 2 where sponsor feedback is most
expensive to receive late. This changes a document the sponsor has already seen and should
be raised with Dr. Burlacu rather than absorbed silently.

## 11. Open questions

These block specific work and need answers from the sponsor, ideally during Phase 0.

**Q1 — Which bond index?** The proposal defers this to Sprint 1 with Bloomberg Global
Aggregate as the working assumption. It determines columns in the data specification, so
it wants answering before Phase 0 closes.

**Q2 — What is the forecast horizon?** "Short-horizon" is never pinned down. One day and
five days imply different target construction and different embargo lengths. *This is the
blocking question* — the data specification cannot be finalised without it.

**Q3 — How is a crash defined?** Crash recall is a first-class success metric, so the
label needs an explicit threshold: a drawdown depth, a return quantile, or dated crisis
windows. The metric cannot be computed without one.

**Q4 — Is PySR in scope?** It appears in the tools table but in no sprint, and it carries
a Julia dependency. Recommend treating it as stretch and cutting it if Sprint 4 is tight.

### Working assumptions until answered

So Phase 0 is not blocked, implementation proceeds on these, all of which are single
config values rather than structural commitments:

- **Q1:** Bloomberg Global Aggregate, with the bond column named generically so a swap is a data change.
- **Q2:** 5-day forward return as the primary target, with horizon a `FeatureSpec` field so 1-day is a config change. Embargo defaults to horizon + 1 trading day.
- **Q3:** Crash = 5-day forward return below the 5th percentile of the training window, with GFC/COVID/2022 windows also available as dated labels.
- **Q4:** Out of scope for planning; revisited only if Sprint 4 finishes early.
