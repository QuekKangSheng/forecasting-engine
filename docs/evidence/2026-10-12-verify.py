"""Run once, as scratch/verify.py on the Toby branch at commit cbd859b, to produce
docs/evidence/2026-10-12-design-verification.md, with the derived grid set to
degrees 1-3 x (lasso, elastic net) + shrinkage. Kept unchanged as the record of
that run: it imports validation.pbo, which later commits removed, and reads the
grid from CANDIDATE_CONFIGS, so it reproduces that file only at cbd859b.

Measurement run for the design-lock verification brief (Tasks 1-3).

Run from the repo root:  uv run python scratch/verify.py
Writes the tables to scratch/verify_output.md and caches every model run's folds
under scratch/cache/ so a rerun only recomputes what is missing.

Everything reuses the package. New logic, written here because the package has
no equivalent:
  * CappedExpanding    - EXP5: every row before the embargo, at most the latest
                         1,260, first test window at row 524.
  * TunerSplit         - Table 1E W504 workaround: boosted's Optuna tuner reads
                         ``splitter.train`` (252) while the main fits use 504 rows.
  * ml_candidates      - run_boosted's body, stopping before it discards each
                         library's folds (Tables 2A/2B need them).
  * band_schedule      - Table 3B's no-trade band.
  * the reporting helpers (test years, exclusions, switches, cost drag).

The custom splitters live in this module (imported as ``verify``) rather than in
``__main__`` because the package's worker processes start with a blank
``__main__`` and must unpickle them by module name.
"""

from __future__ import annotations

import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("FORECASTING_WORKERS", "12")

from forecasting_engine.extraction.targets import TargetRole  # noqa: E402
from forecasting_engine.ingest.align import (  # noqa: E402
    align_and_lag,
    select_signals,
    transform_for,
)
from forecasting_engine.models import boosted  # noqa: E402
from forecasting_engine.models.boosted import run_boosted  # noqa: E402
from forecasting_engine.models.naive import run_naive  # noqa: E402
from forecasting_engine.models.polynomial import (  # noqa: E402
    CANDIDATE_CONFIGS,
    SHRINKAGE,
    _evaluate_candidate,
    run_derived_polynomial,
)
from forecasting_engine.models.shrinkage import ShrinkagePolynomial  # noqa: E402,F401
from forecasting_engine.parallel import map_processes  # noqa: E402
from forecasting_engine.portfolio import backtest, optimize  # noqa: E402
from forecasting_engine.portfolio.directional import directional_pnl  # noqa: E402
from forecasting_engine.validation import metrics  # noqa: E402
from forecasting_engine.validation.harness import (  # noqa: E402
    pooled,
    select_best_candidate,
    signal_forecasts,
    summarize,
)
from forecasting_engine.validation.pbo import compute_pbo  # noqa: E402
from forecasting_engine.validation.splitters import (  # noqa: E402
    TUNING_ROWS,
    PurgedWalkForward,
    expanding,
)

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
OUT = HERE / "results.md"

EQ, BOND = TargetRole.EQUITY, TargetRole.BOND
NAMES = {EQ: "equity (SPX)", BOND: "bond (LBUSTRUU)"}
H, TEST, EMB = 20, 20, 20
HORIZONS = (1, 5, 10, 20)
CAP_ROWS = 1260
BOUNDS = (0.2, 0.8)
BAND_COST = 0.0008
EXCLUDE = (pd.Timestamp("2025-03-01"), pd.Timestamp("2025-06-30"))
TEST_YEARS = range(2019, 2027)
PBO_BLOCKS = (8, 10, 12, 16)

#: The grid as it will be once degrees 4 and 5 are dropped: degrees 1-3 x (lasso,
#: elastic net), plus shrinkage (7 setups). Passed in here; src/ is unchanged.
DERIVED_GRID = tuple(c for c in CANDIDATE_CONFIGS if c.degree <= 3)
CONFIGS = [(c.degree, c.regularizer, c.max_terms) for c in DERIVED_GRID] + [
    (1, SHRINKAGE, None)
]
CONFIG_NAMES = [SHRINKAGE if r == SHRINKAGE else f"degree{d}_{r}" for d, r, _ in CONFIGS]


# --------------------------------------------------------------------------- splitters (new)


class CappedExpanding(PurgedWalkForward):
    """EXP5: expanding, capped at the latest ``train`` rows before the embargo, with
    the first test window where the 504-row tuning period puts it (row 524) rather
    than at ``train + embargo``."""

    def split(self, panel):
        index = panel.frame.index
        test_start = self.tuning_rows + self.embargo
        while test_start + self.test <= len(index):
            yield (
                self.window_before(panel, test_start, self.train),
                index[test_start : test_start + self.test],
            )
            test_start += self.test


class TunerSplit(PurgedWalkForward):
    """Workaround for Table 1E W504: ``train`` is what boosted's tuner builds its
    mini walk-forward from (252); the main folds fit on ``fit_train`` (504)."""

    def __init__(self, tune_train, fit_train, test, embargo, tuning_rows):
        super().__init__(tune_train, test, embargo, tuning_rows)
        self.fit_train = fit_train

    def split(self, panel):
        yield from PurgedWalkForward(
            self.fit_train, self.test, self.embargo, self.tuning_rows
        ).split(panel)


def base(train):
    return PurgedWalkForward(train, TEST, EMB, tuning_rows=TUNING_ROWS)


WINDOWS = {
    "W252": lambda: base(252),
    "W504": lambda: base(504),
    "EXP5": lambda: CappedExpanding(CAP_ROWS, TEST, EMB, tuning_rows=TUNING_ROWS),
    "EXP": lambda: expanding(base(252)),
}


# --------------------------------------------------------------------------- data


def load():
    commit = pickle.load(open(ROOT / "data" / "last_commit.pkl", "rb"))
    merged = commit["extraction_committed"]
    targets = commit["extraction_committed_targets"]
    sources = commit["extraction_committed_sources"]
    numeric = [c for c in merged.columns if c != "Date" and merged[c].dtype.kind in "fi"]
    signals = select_signals(numeric, list(targets.values()), sources)
    transforms = {c: transform_for(sources.get(c)) for c in signals}
    return merged.set_index("Date"), targets, signals, transforms


DATA = None


def data():
    global DATA
    if DATA is None:
        DATA = load()
    return DATA


_PANELS = {}


def panel(role, h):
    if (role, h) not in _PANELS:
        indexed, targets, signals, transforms = data()
        _PANELS[role, h] = align_and_lag(
            indexed, signals, targets[role], horizon=h, transforms=transforms
        )
    return _PANELS[role, h]


def prices():
    indexed, targets, _, _ = data()
    return {role: indexed[targets[role]].dropna() for role in (EQ, BOND)}


# --------------------------------------------------------------------------- model runs


TIMINGS: dict[str, float] = {}


def cached(key, fn):
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{key}.pkl"
    if path.exists():
        return pickle.load(open(path, "rb"))
    start = time.perf_counter()
    value = fn()
    TIMINGS[key] = time.perf_counter() - start
    print(f"  {key}: {TIMINGS[key]:.0f}s", flush=True)
    # Atomic, so a parallel prefetch process never reads a half-written file.
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(value, fh)
    os.replace(tmp, path)
    return value


def prefetch(specs):
    """Fill the cache for ``kind:role:window[:h]`` specs, so independent runs can be
    spread over several processes. ``main`` then reads them from the cache."""
    roles = {"equity": EQ, "bond": BOND}
    for spec in specs:
        if spec == "xderived":
            crosscheck_derived()
            continue
        if spec == "xml":
            crosscheck_ml()
            continue
        kind, role, window, *rest = spec.split(":")
        h = int(rest[0]) if rest else H
        {"derived": derived, "ml": ml}[kind](roles[role], window, h) if kind != "naive" else naive(
            roles[role], window
        )


def crosscheck_derived():
    """The package's own run_derived_polynomial on bond W252, 7-setup grid. Same cache
    key and value as main's cross-check, so main reads it instead of refitting."""

    def run():
        res, _ = run_derived_polynomial(panel(BOND, H), base(252), candidates=DERIVED_GRID)
        return res.signal_rank_ic

    return cached("crosscheck7_run_derived_bond_W252", run)


def crosscheck_ml():
    """The package's own run_boosted on equity W252, cached under main's key."""

    def run():
        res, _desc, log = run_boosted(panel(EQ, H), base(252))
        return res.signal_rank_ic, log.library

    return cached("crosscheck_run_boosted_equity_W252", run)


def derived_candidates(pnl, splitter):
    """run_derived_polynomial's body up to selection: every setup's folds."""
    fits = map_processes(
        _evaluate_candidate, CONFIGS, [pnl] * len(CONFIGS), [splitter] * len(CONFIGS)
    )
    return dict(zip(CONFIG_NAMES, fits, strict=True))


def ml_candidates(pnl, splitter):
    """run_boosted's body up to selection: each library's folds."""
    folds = list(splitter.split(pnl))
    index = pnl.frame.index
    starts = [index.get_loc(test[0]) for _train, test in folds]
    triggers, fold_tunes = boosted._schedule(starts, boosted.RETUNE_EVERY)
    windows = [splitter.window_before(pnl, starts[i], splitter.tuning_rows) for i in triggers]
    mini = PurgedWalkForward(splitter.train, splitter.test, splitter.embargo)
    for window in windows:
        boosted._require_tuning_folds(boosted._restrict(pnl, window), mini)
    libraries = list(boosted._LIBRARIES)
    outcomes = map_processes(
        boosted._tune_and_evaluate,
        libraries,
        *(
            [argument] * len(libraries)
            for argument in (
                pnl, splitter, windows, folds, fold_tunes, boosted.N_TRIALS, boosted.RETUNE_TRIALS
            )
        ),
    )
    return {lib: fits for lib, (_chosen, fits) in zip(libraries, outcomes, strict=True)}


def derived(role, window, h=H):
    return cached(
        f"derived7_{role.value}_{window}_h{h}",
        lambda: derived_candidates(panel(role, h), WINDOWS[window]()),
    )


def ml(role, window, h=H):
    split = base(252) if window == "W252" else TunerSplit(252, 504, TEST, EMB, TUNING_ROWS)
    return cached(f"ml_{role.value}_{window}_h{h}", lambda: ml_candidates(panel(role, h), split))


def naive(role, window):
    return cached(f"naive_{role.value}_{window}", lambda: run_naive(panel(role, H), WINDOWS[window]()))


_BEST = {}


def best(per, key):
    """select_best_candidate's winner (PBO at the default 16 blocks comes with it)."""
    if key not in _BEST:
        _BEST[key] = cached(f"best_{key}", lambda: select_best_candidate(per))
    return _BEST[key]


# --------------------------------------------------------------------------- scoring helpers


def score(folds):
    res, _ = summarize(folds)
    wider = max(res.signal_rank_ic_se, res.signal_rank_ic_se_test)
    return dict(
        sric=res.signal_rank_ic,
        se1=res.signal_rank_ic_se,
        se2=res.signal_rank_ic_se_test,
        t=res.signal_rank_ic / wider if wider else float("nan"),
        oos=res.oos_rank_ic,
        rows=res.rows_scored,
        res=res,
    )


def signal_and_realised(folds):
    return signal_forecasts(folds), pooled(folds)[1]


def test_year(dates):
    return dates.year + (dates.month >= 10).astype(int)


def by_test_year(folds):
    s, r = signal_and_realised(folds)
    ty = pd.Series(test_year(s.index), index=s.index)
    return {y: metrics.rank_ic(s[ty == y], r[ty == y]) for y in TEST_YEARS}


def excluding(folds, h=H):
    s, r = signal_and_realised(folds)
    keep = ~((s.index >= EXCLUDE[0]) & (s.index <= EXCLUDE[1]))
    s, r = s[keep], r[keep]
    lags = max(len(f.test) for f in folds)
    se = max(metrics.rank_ic_se(s, r, lags=h - 1), metrics.rank_ic_se(s, r, lags=lags))
    return metrics.rank_ic(s, r), se


def pbo_inputs(per):
    return {name: signal_and_realised(folds) for name, folds in per.items()}


def single_linkage(names, corr, threshold=0.95):
    parent = {n: n for n in names}

    def find(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if corr.loc[a, b] > threshold:
                parent[find(a)] = find(b)
    groups = {}
    for n in names:
        groups.setdefault(find(n), []).append(n)
    return list(groups.values())


# --------------------------------------------------------------------------- portfolio helpers


def decisions(signals, px, h, lam):
    """weight_schedule's loop, keeping each rebalance's breakdown and D."""
    dates = optimize.common_rebalance_dates(
        px, horizon=h, train_window=252, test_window=TEST, embargo=EMB
    )
    expected = optimize.expected_returns(dates, signals)
    out = []
    for date in dates:
        row = expected.loc[date]
        if row.isna().any():
            continue
        try:
            cov = optimize.covariance_at_rebalance(
                date, px, horizon=h, train_window=252, embargo=EMB
            )
        except optimize.OptimizeDataError:
            continue
        b = optimize.weight_breakdown(row, cov, risk_aversion=lam, bounds=BOUNDS)
        d = cov.loc[EQ, EQ] - 2 * cov.loc[EQ, BOND] + cov.loc[BOND, BOND]
        out.append((date, b, float(d)))
    return out


def schedule_of(decs):
    return pd.DataFrame(
        {EQ: [b.equity for _, b, _ in decs], BOND: [1 - b.equity for _, b, _ in decs]},
        index=pd.DatetimeIndex([d for d, _, _ in decs]),
    )


def at_bound(w):
    return any(math.isclose(w, x, abs_tol=1e-12) for x in BOUNDS)


def band_schedule(decs, px, lam):
    """Table 3B's no-trade band, drifting held weights exactly as backtest.chain does."""
    full = schedule_of(decs)
    returns = backtest.asset_returns(px, full)
    by_date = {d: (b, dd) for d, b, dd in decs}
    held, rows, skipped = None, {}, 0
    for i, (day, row) in enumerate(returns[[EQ, BOND]].iterrows()):
        if i > 0 and held is not None:
            daily = row.to_numpy(dtype=float)
            held = held * (1 + daily) / (1 + float(held @ daily))
        if day not in by_date:
            continue
        b, d = by_date[day]
        target = b.equity
        if held is None:
            new = target
        else:
            current = float(held[0])
            band = BAND_COST / (lam * d) if d > 0 else 0.0
            if abs(target - current) <= band:
                skipped += 1
                continue
            new = min(max(target - math.copysign(band, target - current), BOUNDS[0]), BOUNDS[1])
        rows[day] = {EQ: new, BOND: 1 - new}
        held = np.array([new, 1 - new])
    return pd.DataFrame(rows).T.sort_index(), skipped


def backtest_row(bt, at_bound_share, skipped_share=None):
    opt = bt.optimised
    years = (bt.end - bt.start).days / 365.25
    after_open = opt.turnover.iloc[1:]
    g, n = bt.metrics["optimised", "gross"], bt.metrics["optimised", "net"]
    base_net = bt.metrics["baseline", "net"]
    row = {
        "rebalances/yr": len(opt.turnover) / years,
        "mean turnover per rebalance": after_open.mean(),
        "annual turnover": after_open.sum() / years,
        "annual cost drag (bp)": (g.annual_return - n.annual_return) * 1e4,
        "ann. return gross / net": f"{g.annual_return:.2%} / {n.annual_return:.2%}",
        "Sharpe gross / net": f"{g.sharpe:.2f} / {n.sharpe:.2f}",
        "50/50 ann. return net": f"{base_net.annual_return:.2%}",
        "50/50 Sharpe net": base_net.sharpe,
        "% rebalances at a bound": at_bound_share * 100,
    }
    if skipped_share is not None:
        row["% rebalances skipped"] = skipped_share * 100
    return row


# --------------------------------------------------------------------------- markdown


def fmt(v, dp=2):
    if v is None:
        return "n/a"
    if isinstance(v, str):
        return v
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if v != v:
        return "n/a"
    return f"{v:.{dp}f}"


def table(title, header, rows):
    lines = [f"### {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- main


ORDER = ("1A", "1B", "1C", "1D", "1E", "2A", "2B", "2C", "3A", "3B", "3C", "3D")
HEAD_1A = ["target", "window", "model", "chosen setup", "Signal Rank IC", "s.e. h−1",
           "s.e. test-window", "t (wider)", "plain OOS Rank IC", "rows scored"]
HEAD_3 = ["h", "rebalances/yr", "mean turnover per rebalance", "annual turnover",
          "annual cost drag (bp)", "optimised ann. return gross / net",
          "optimised Sharpe gross / net", "50/50 ann. return net", "50/50 Sharpe net",
          "% rebalances at a bound"]


class Results:
    """scratch/results.md, rewritten whenever a table is finished, tables in brief order."""

    def __init__(self):
        self.tables = {k: f"### {k}\n\n_pending_\n" for k in ORDER}
        self.notes: list[str] = []
        self.footer = ""

    def put(self, key, text):
        self.tables[key] = text
        self.write()
        print(f"TABLE {key} written", flush=True)

    def write(self):
        body = "\n".join(self.tables[k] for k in ORDER)
        body += "\n## Notes (raw)\n\n" + "\n".join(f"- {n}" for n in self.notes)
        OUT.write_text(body + "\n\n" + self.footer + "\n", encoding="utf-8")


def derived_best(role, window, h=H):
    per = derived(role, window, h)
    name, _ = best(per, f"derived7_{role.value}_{window}_h{h}")
    return per, name


def ml_best(role, window, h=H):
    per = ml(role, window, h)
    name, _ = best(per, f"ml_{role.value}_{window}_h{h}")
    return per, name


def score_row(role, window, model, setup, folds):
    s = score(folds)
    return [NAMES[role], window, model, setup, fmt(s["sric"], 3), fmt(s["se1"], 3),
            fmt(s["se2"], 3), fmt(s["t"]), fmt(s["oos"], 3), fmt(s["rows"])]


def check_test_dates():
    for role in (EQ, BOND):
        pnl = panel(role, H)
        tests = {w: [t for _, t in WINDOWS[w]().split(pnl)] for w in WINDOWS}
        ref = tests["W504"]
        for w, t in tests.items():
            if len(t) != len(ref) or any(not a.equals(b) for a, b in zip(t, ref, strict=False)):
                sys.exit(f"{role.value} {w} yields different test dates; stopping as briefed.")
        print(f"{role.value}: {len(pnl.frame)} rows {pnl.frame.index[0]:%Y-%m-%d}.."
              f"{pnl.frame.index[-1]:%Y-%m-%d}; {len(ref)} folds, first test "
              f"{ref[0][0]:%Y-%m-%d}", flush=True)


def table_1e(out):
    rows = []
    for role in (EQ, BOND):
        for w in ("W252", "W504"):
            per, name = ml_best(role, w)
            rows.append(score_row(role, w, "ML", name, per[name]))
    out.put("1E", table("1E", HEAD_1A, rows))


def table_1d(out):
    per, name = derived_best(EQ, "EXP")
    rows = []
    for model, folds in (("derived", per[name]), ("shrinkage", per[SHRINKAGE])):
        for label, keep in (("< 5 years (< 1,260 rows)", lambda f: len(f.train) < CAP_ROWS),
                            ("≥ 5 years (≥ 1,260 rows)", lambda f: len(f.train) >= CAP_ROWS)):
            sub = tuple(f for f in folds if keep(f))
            s = score(sub)
            rows.append([model, label, fmt(s["sric"], 3), fmt(max(s["se1"], s["se2"]), 3),
                         fmt(len(sub))])
    out.put("1D", table("1D", ["model", "training window", "Signal Rank IC", "wider s.e.",
                               "folds"], rows))


def tables_1abc(out, t1):
    rows_a = []
    for role in (EQ, BOND):
        for w in WINDOWS:
            per, name = t1[role, w]
            rows_a.append(score_row(role, w, "derived", name, per[name]))
            rows_a.append(score_row(role, w, "shrinkage", SHRINKAGE, per[SHRINKAGE]))
            res, _ = naive(role, w)
            rows_a.append([NAMES[role], w, "naive", "training mean", "n/a", "n/a", "n/a", "n/a",
                           fmt(res.oos_rank_ic, 3), fmt(res.rows_scored)])
    out.put("1A", table("1A", HEAD_1A, rows_a))

    years = list(TEST_YEARS)
    ty, rows_b, wins = {}, [], []
    for role in (EQ, BOND):
        for w in WINDOWS:
            per, name = t1[role, w]
            for model, folds in (("derived", per[name]), ("shrinkage", per[SHRINKAGE])):
                ty[role, w, model] = by_test_year(folds)
                rows_b.append([NAMES[role], w, model]
                              + [fmt(ty[role, w, model][y], 3) for y in years])
    for role in (EQ, BOND):
        for model in ("derived", "shrinkage"):
            ref = ty[role, "W504", model]
            counts = [f"{w} {sum(1 for y in years if ty[role, w, model][y] > ref[y])}/{len(years)}"
                      for w in ("W252", "EXP5", "EXP")]
            wins.append(f"- {NAMES[role]}, {model}: test years beating W504 — " + ", ".join(counts))
    out.put("1B", table("1B", ["target", "window", "model"] + [f"TY{y}" for y in years], rows_b)
            + "\n" + "\n".join(wins) + "\n")

    rows_c = []
    for role in (EQ, BOND):
        for w in WINDOWS:
            per, name = t1[role, w]
            for model, folds in (("derived", per[name]), ("shrinkage", per[SHRINKAGE])):
                ic, se = excluding(folds)
                rows_c.append([NAMES[role], w, model, fmt(ic, 3), fmt(se, 3), fmt(ic / se)])
    out.put("1C", table("1C", ["target", "window", "model",
                               "Signal Rank IC (excl. 1 Mar–30 Jun 2025)", "wider s.e.", "t"],
                        rows_c))


def tables_2(out):
    current = {
        (EQ, "derived (EXP)"): derived_best(EQ, "EXP")[0],
        (BOND, "derived (W252)"): derived_best(BOND, "W252")[0],
        (EQ, "ML (W252)"): ml_best(EQ, "W252")[0],
        (BOND, "ML (W252)"): ml_best(BOND, "W252")[0],
    }
    rows = []
    for (role, family), per in current.items():
        inputs = pbo_inputs(per)
        vals = []
        for n in PBO_BLOCKS:
            start = time.perf_counter()
            vals.append(compute_pbo(inputs, n_blocks=n).pbo)
            took = time.perf_counter() - start
            if took > 600:
                out.notes.append(f"PBO {NAMES[role]} {family} @{n} took {took / 60:.1f} min.")
        rows.append([NAMES[role], family, len(per)] + [fmt(v) for v in vals])
    out.put("2A", table("2A", ["target", "family", "setups"]
                        + [f"PBO @{n}" for n in PBO_BLOCKS], rows))

    short = {n: n.replace("degree", "d").replace("_elasticnet", "_en") for n in CONFIG_NAMES}
    blocks, rows_c = [], []
    for role, family in ((EQ, "derived (EXP)"), (BOND, "derived (W252)")):
        per = current[role, family]
        frame = pd.concat({n: signal_forecasts(f) for n, f in per.items()}, axis=1).dropna()
        corr = frame.corr()
        blocks.append(table(f"2B — {NAMES[role]} {family}",
                            ["setup"] + [short[n] for n in corr.columns],
                            [[short[a]] + [f"{corr.loc[a, b]:.2f}" for b in corr.columns]
                             for a in corr.index]))
        clusters = single_linkage(list(corr.columns), corr)
        scores = {n: score(per[n])["sric"] for n in per}
        kept = [max(c, key=lambda n: scores[n] if scores[n] == scores[n] else -1)
                for c in clusters]
        merged = [c for c in clusters if len(c) > 1]
        desc = "; ".join("{" + ", ".join(short[n] for n in c) + "}" for c in merged) or "none merged"
        pbo16 = (compute_pbo({n: signal_and_realised(per[n]) for n in kept}, n_blocks=16).pbo
                 if len(kept) >= 2 else float("nan"))
        rows_c.append([NAMES[role], f"{len(clusters)} ({desc})", len(kept), fmt(pbo16)])
    out.put("2B", "### 2B\n\n" + "\n".join(blocks))
    out.put("2C", table("2C", ["target", "clusters (members)", "setups kept",
                               "PBO @16 after merging"], rows_c))


def task3_signals(h):
    eq_per, eq_name = derived_best(EQ, "EXP", h)
    bd_per, bd_name = ml_best(BOND, "W252", h)
    return {EQ: (eq_per[eq_name], eq_name), BOND: (bd_per[bd_name], bd_name)}


def table_3c(out, px):
    run = task3_signals(20)
    sig = {role: signal_forecasts(run[role][0]) for role in (EQ, BOND)}
    rows = []
    for level in sorted(optimize.RISK_AVERSION_SCALE):
        lam = optimize.risk_aversion_for(level)
        decs = decisions(sig, px, 20, lam)
        bt = backtest.run_backtest(px, schedule_of(decs))
        n = bt.metrics["optimised", "net"]
        rows.append([level, fmt(lam, 0), f"{n.annual_return:.2%}", fmt(n.sharpe),
                     f"{n.max_drawdown:.2%}",
                     fmt(sum(b.capped for _, b, _ in decs) / len(decs) * 100)])
    out.put("3C", table("3C", ["level", "λ", "net ann. return", "net Sharpe", "max drawdown",
                               "% rebalances at a bound"], rows))


def tables_3abd(out, px):
    runs = {h: task3_signals(h) for h in HORIZONS}
    lam3 = optimize.risk_aversion_for(optimize.DEFAULT_RISK_LEVEL)
    rows_a, rows_b = [], []
    for h, run in runs.items():
        sig = {role: signal_forecasts(run[role][0]) for role in (EQ, BOND)}
        decs = decisions(sig, px, h, lam3)
        sched = optimize.weight_schedule(sig, px, horizon=h, train_window=252, test_window=TEST,
                                         embargo=EMB, risk_aversion=lam3, bounds=BOUNDS)
        assert np.allclose(sched[EQ].to_numpy(), schedule_of(decs)[EQ].to_numpy())
        bt = backtest.run_backtest(px, sched)
        share = sum(b.capped for _, b, _ in decs) / len(decs)
        rows_a.append([h] + [fmt(v) for v in backtest_row(bt, share).values()])
        banded, skipped = band_schedule(decs, px, lam3)
        bt_b = backtest.run_backtest(px, banded)
        traded_bound = sum(at_bound(w) for w in banded[EQ]) / len(banded)
        rows_b.append([h] + [fmt(v) for v in
                             backtest_row(bt_b, traded_bound, skipped / len(decs)).values()])
    out.put("3A", table("3A", HEAD_3, rows_a))
    out.put("3B", table("3B", HEAD_3 + ["% rebalances skipped"], rows_b))

    rows_d = []
    for role, bp in ((EQ, 3.0), (BOND, 5.0)):
        for h, run in runs.items():
            res, _ = summarize(run[role][0])
            d = directional_pnl(res.forecast, res.realised, horizon=h)
            end = panel(role, h).label_end.loc[d.end]
            years = (end - d.start).days / 365.25
            switches = int(d.invested.astype(int).diff().abs().sum())
            strat = float(np.prod(1 + d.strategy.to_numpy())) ** (1 / years) - 1
            hold = float(np.prod(1 + d.buy_and_hold.to_numpy())) ** (1 / years) - 1
            rows_d.append([NAMES[role], h, d.calls, fmt(switches / years),
                           fmt(switches / years * bp), f"{strat:.2%}", f"{hold:.2%}"])
    out.put("3D", table("3D", ["index", "h", "steps", "in/out switches per year",
                               "annual cost (bp) at 3 bp equity / 5 bp bond per switch",
                               "gross strategy ann. return", "buy-and-hold ann. return"], rows_d))
    out.notes.append("Task 3 setups by h: " + "; ".join(
        f"h{h} equity={r[EQ][1]}, bond={r[BOND][1]}" for h, r in runs.items()))


def main():
    t0 = time.perf_counter()
    out = Results()
    out.write()
    _, _, signals, _ = data()
    print(f"signals ({len(signals)}): {signals}", flush=True)
    check_test_dates()
    out.notes.append("All four windows give identical test dates per target (checked before "
                     "any fit, and again on every derived run's folds).")
    px = prices()

    # Inputs already cached: 1E.
    table_1e(out)

    # Equity EXP unlocks 1D and, with the cached bond ML, 3C.
    derived_best(EQ, "EXP")
    table_1d(out)
    table_3c(out, px)

    # Bond W252 unlocks Task 2.
    derived_best(BOND, "W252")
    tables_2(out)

    # The rest of Task 1.
    t1 = {}
    for role in (EQ, BOND):
        for w in WINDOWS:
            per, name = derived_best(role, w)
            ref = [t for _, t in WINDOWS["W504"]().split(panel(role, H))]
            assert all(f.test.equals(t) for f, t in zip(per[name], ref, strict=True))
            t1[role, w] = per, name
    tables_1abc(out, t1)

    # Task 3 at h = 1, 5, 10.
    tables_3abd(out, px)

    # Cross-checks: the decomposed calls against the package's own runners.
    def check_derived():
        res, _ = run_derived_polynomial(panel(BOND, H), base(252), candidates=DERIVED_GRID)
        return res.signal_rank_ic

    got = cached("crosscheck7_run_derived_bond_W252", check_derived)
    per, name = t1[BOND, "W252"]
    mine = score(per[name])["sric"]
    out.notes.append(f"run_derived_polynomial(candidates=7-setup grid) cross-check, bond W252: "
                     f"{got:.6f} vs decomposed {mine:.6f} "
                     f"({'match' if math.isclose(got, mine, abs_tol=1e-12) else 'MISMATCH'}).")

    def check_ml():
        res, _desc, log = run_boosted(panel(EQ, H), base(252))
        return res.signal_rank_ic, log.library

    got = cached("crosscheck_run_boosted_equity_W252", check_ml)
    per, name = ml_best(EQ, "W252")
    mine = score(per[name])["sric"]
    out.notes.append(f"run_boosted cross-check, equity W252: {got[1]} {got[0]:.6f} vs decomposed "
                     f"{name} {mine:.6f} ({'match' if math.isclose(got[0], mine, abs_tol=1e-12) and got[1] == name else 'MISMATCH'}).")

    for role in (EQ, BOND):
        pnl = panel(role, H)
        per, name = t1[role, "W252"]
        folds = per[name]
        out.notes.append(f"{NAMES[role]}: {len(pnl.frame)} rows {pnl.frame.index[0]:%d %b %Y}–"
                         f"{pnl.frame.index[-1]:%d %b %Y}, {len(folds)} folds, test dates "
                         f"{folds[0].test[0]:%d %b %Y}–{folds[-1].test[-1]:%d %b %Y}.")
    out.footer = (f"Runtime this invocation: {(time.perf_counter() - t0) / 60:.1f} min. "
                  "Fresh model runs: " + ", ".join(f"{k} {v:.0f}s" for k, v in TIMINGS.items()))
    out.write()
    print("DONE", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    import verify

    if len(sys.argv) > 1 and sys.argv[1] == "prefetch":
        verify.prefetch(sys.argv[2:])
    else:
        verify.main()
