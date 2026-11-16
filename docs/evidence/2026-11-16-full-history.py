"""The final rerun on the Bloomberg history from 31 Aug 2006, after Phases 1–4.

1. Every model's Signal Rank IC and t, for each target and horizon, and whether it
   passes the gates; the research models' pre-2016 hold-out.
2. The portfolio per risk level against 50/50 and against the same level with no
   forecasts, at each horizon, for two model pairs: the team's Sprint 4 default
   (derived polynomial for equity, machine learning for bond), the main result, and
   the derived polynomial for both. Both were run before the Sprint 4 pair was made
   the main result. Then the same backtests with ``β̂`` used whatever its t (still
   zero before 252 pairs), as a diagnostic.
3. Descriptive only, changing nothing: why the 20-day trust gate opens 3 times here
   against 26 in the 1 Nov file. Two model pairs (1 Nov's: derived polynomial for
   equity, machine learning for bond; and the derived polynomial for both) on two
   datasets (the Sep 2026 upload and the data from 2006): gate openings and t
   against no forecasts in each of the four cells. At h = 5 and 20 for each cell:
   the trust regression's slope and t for each calendar year on its own, and the
   gate's openings and forecast value with its pairs starting on 1 Jan 2010.

Models are fitted through the functions the Models page's runners call, at the page's
defaults, in parallel processes. The Sep 2026 upload is the Data page's last commit,
``data/last_commit.pkl``, relative to the working directory.

Run: uv run --project <repo> python -I <this file> <folder holding the 11 .xlsx files>
"""

from __future__ import annotations

import argparse
import importlib.util
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.ingest import fama_french
from forecasting_engine.ingest.align import align_and_lag
from forecasting_engine.ingest.upload import frame_sha256
from forecasting_engine.models.boosted import run_boosted
from forecasting_engine.models.carry_research import carry_inputs, run_carry_research_polynomial
from forecasting_engine.models.famafrench import FACTOR_COLUMNS, merge_factors, run_famafrench
from forecasting_engine.models.naive import run_naive
from forecasting_engine.models.polynomial import (
    CANDIDATE_CONFIGS,
    DerivedPolynomial,
    run_derived_polynomial,
)
from forecasting_engine.models.sign_ruled import (
    HOLDOUT_BEFORE,
    economic_signs,
    run_sign_ruled_polynomial,
)
from forecasting_engine.portfolio import optimize
from forecasting_engine.validation import metrics
from forecasting_engine.validation.gates import evaluate_candidate

EQUITY, BOND = TargetRole.EQUITY, TargetRole.BOND
HORIZONS = (1, 5, 10, 20)
BREAKDOWN_HORIZONS = (5, 20)
PAIRS_FROM = pd.Timestamp("2010-01-01")
TRAIN_WINDOW, TEST_WINDOW, EMBARGO_DAYS, MAX_TERMS = 504, 20, 20, 10
MODELS = ("naive", "derived", "research", "ml", "ff5")
NAMES = {
    "naive": "Naive (training mean)",
    "derived": "Polynomial (derived)",
    "research": "Polynomial, research",
    "ml": "Machine learning",
    "ff5": "Fama-French 5",
}
OLD, NEW = "Sep 2026 upload", "from 31 Aug 2006"
PAIRINGS = {
    "Sprint 4 default (derived equity, ML bond)": {EQUITY: "derived", BOND: "ml"},
    "derived for both": {EQUITY: "derived", BOND: "derived"},
}

_HERE = Path(__file__).resolve().parent


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


history = _load("history_2006", "2026-11-15-history-2006.py")
fixes = _load("optimiser_fixes", "2026-11-16-optimiser-fixes.py")
table, day = history.table, history.day

_BUNDLES: dict[str, dict] = {}


def _init(bundles: dict[str, dict]) -> None:
    _BUNDLES.update(bundles)


def fit(job: tuple[str, TargetRole, str, int]):
    """One model on one dataset at one horizon, as the Models page runs it."""
    label, role, model, horizon = job
    merged, indexed, targets, sources, signals, transforms = history.page_inputs(_BUNDLES[label])
    price = targets[role]
    panel = align_and_lag(indexed, signals, price, horizon=horizon, transforms=transforms)
    split = history.splitter()
    started = time.perf_counter()
    if model == "naive":
        result = run_naive(panel, split)[0]
    elif model == "derived":
        candidates = tuple(
            DerivedPolynomial(degree=c.degree, regularizer=c.regularizer, max_terms=MAX_TERMS)
            for c in CANDIDATE_CONFIGS
        )
        result = run_derived_polynomial(panel, split, candidates=candidates)[0]
    elif model == "research" and role == EQUITY:
        signs = economic_signs(signals, sources, role)
        result = run_sign_ruled_polynomial(indexed, signs, price, horizon, split)[0]
    elif model == "research":
        inputs = carry_inputs(signals, sources, role)
        result = run_carry_research_polynomial(indexed, inputs, price, horizon, split)[0]
    elif model == "ml":
        result = run_boosted(panel, split)[0]
    else:
        factors = merge_factors(merged, fama_french.resolve().file.frame).set_index("Date")
        ff_panel = align_and_lag(
            factors, list(FACTOR_COLUMNS), price, horizon=horizon, exact=FACTOR_COLUMNS
        )
        result = run_famafrench(ff_panel, split)[0]
    return job, result, time.perf_counter() - started


def jobs() -> list[tuple[str, TargetRole, str, int]]:
    every = [
        (NEW, role, model, h)
        for h in HORIZONS
        for role in (EQUITY, BOND)
        for model in MODELS
        if not (model == "ff5" and role == BOND)
    ]
    every += [
        (OLD, role, model, h)
        for h in BREAKDOWN_HORIZONS
        for role, model in ((EQUITY, "derived"), (BOND, "derived"), (BOND, "ml"))
    ]
    # The slowest first, so the pool isn't left waiting on one at the end.
    return sorted(every, key=lambda j: j[2] not in ("ml", "derived"))


# ── 1. Models ────────────────────────────────────────────────────────────────


def model_section(results, seconds) -> None:
    print("## 1. Every model, each target and horizon\n")
    print(
        "Signal Rank IC and t (the Signal Rank IC over the larger of its two standard "
        "errors), at the page's defaults. The gate: Signal Rank IC above 0.02 and t above 2; "
        "the naive baseline is not gated. Runtime is wall-clock seconds per fit, measured "
        "locally with several fits running at once, so each is slower than alone.\n"
    )
    for role in (EQUITY, BOND):
        rows = []
        for model in MODELS:
            for h in HORIZONS:
                key = (NEW, role, model, h)
                if key not in results:
                    continue
                r = results[key]
                if model == "naive":
                    verdict = "not gated"
                else:
                    failed = evaluate_candidate(r.gated_rank_ic, r.signal_t).failed_gates
                    verdict = "passes" if not failed else "fails " + " and ".join(failed)
                rows.append(
                    [
                        NAMES[model],
                        h,
                        f"{r.gated_rank_ic:+.4f}",
                        f"{r.signal_t:.2f}",
                        f"{r.rows_scored:,}",
                        verdict,
                        f"{seconds[key]:.0f} s",
                    ]
                )
        print(f"### {role.value.capitalize()}\n")
        table(["model", "h", "Signal Rank IC", "t", "rows", "gate", "runtime"], rows)

    print(f"### Research models' pre-2016 hold-out (windows ending before {day(HOLDOUT_BEFORE)})\n")
    rows = []
    for role in (EQUITY, BOND):
        for h in HORIZONS:
            r = results[(NEW, role, "research", h)]
            held = r.holdout
            rows.append(
                [
                    role.value,
                    h,
                    f"{r.gated_rank_ic:+.4f}",
                    f"{r.signal_t:.2f}",
                    "—" if held is None else f"{held.signal_rank_ic:+.4f}",
                    "—" if held is None else f"{held.t:.2f}",
                    "—" if held is None else held.windows,
                ]
            )
    table(
        [
            "target",
            "h",
            "full Signal Rank IC",
            "full t",
            "hold-out Signal Rank IC",
            "hold-out t",
            "hold-out windows",
        ],
        rows,
    )


# ── 2. Portfolio ─────────────────────────────────────────────────────────────


def prices_of(bundle: dict) -> dict[TargetRole, pd.Series]:
    _, indexed, targets, *_ = history.page_inputs(bundle)
    return {role: indexed[targets[role]].dropna() for role in (EQUITY, BOND)}


def schedule(results, label, pairing, prices, horizon, *, gate=2.0):
    signals = {
        role: results[(label, role, model, horizon)].signal.dropna()
        for role, model in pairing.items()
    }
    baselines = {
        role: results[(label, role, model, horizon)].baseline.dropna()
        for role, model in pairing.items()
    }
    inputs = optimize.schedule_inputs(
        signals,
        prices,
        baselines=baselines,
        horizon=horizon,
        train_window=TRAIN_WINDOW,
        test_window=TEST_WINDOW,
        embargo=EMBARGO_DAYS,
        trust_gate=gate,
    )
    pairs = optimize.trust_pairs(signals, baselines, prices, horizon=horizon)
    return inputs, pairs


def opened(inputs) -> str:
    rebalances = inputs.rebalances.values()
    return f"{sum(r.trust > 0 for r in rebalances)} of {len(rebalances)}"


def no_forecast_t(runs) -> str:
    ts = [row[-1] for row in fixes.level_rows(runs, "") if row[1] != 5]
    if "—" in ts:
        return "identical paths" if set(ts) == {"—"} else ", ".join(ts)
    ts = [float(t) for t in ts]
    return f"{min(ts):+.2f} to {max(ts):+.2f}"


def portfolio_section(results, prices) -> None:
    print("## 2. Portfolio per risk level\n")
    print(
        "Each pair's levels, then 50/50. Net of 3 bp equity and 5 bp bond on what each "
        "rebalance trades; turnover and costs a year leave out the first allocation. The "
        "first pair is the main result.\n"
    )
    for h in HORIZONS:
        rows, notes, dates = [], [], []
        for name, pairing in PAIRINGS.items():
            inputs, _ = schedule(results, NEW, pairing, prices, h)
            dates = list(inputs.rebalances)
            runs = fixes.run_levels(inputs, prices, h)
            rows += fixes.level_rows(runs, name)
            notes.append(f"{name}: trust gate open at {opened(inputs)}")
        rows.append(fixes.benchmark_row(runs[3][0][1]))
        print(f"### h = {h}: {len(dates):,} rebalances, {day(dates[0])} to {day(dates[-1])}\n")
        print("; ".join(notes) + ".\n")
        table(fixes.LEVEL_HEADER, rows)

    print("## 2b. Diagnostic: the trust gate ignored\n")
    print(
        "The same backtests with β̂ used whatever its t: the slope clipped to [0, 1] as "
        "soon as 252 pairs are known. The product never does this; it shows what the gate "
        "is holding back. t vs no forecasts compares each level with itself at β̂ = 0.\n"
    )
    for h in HORIZONS:
        rows, notes = [], []
        for name, pairing in PAIRINGS.items():
            inputs, _ = schedule(results, NEW, pairing, prices, h, gate=-np.inf)
            runs = fixes.run_levels(inputs, prices, h)
            rows += fixes.level_rows(runs, name)
            notes.append(f"{name}: β̂ above zero at {opened(inputs)}")
        print(f"### h = {h}, ungated\n")
        print("; ".join(notes) + ".\n")
        table(fixes.LEVEL_HEADER, rows)


# ── 3. The gate breakdown ────────────────────────────────────────────────────


def from_2010(inputs, pairs, horizon):
    """``inputs`` with each rebalance's trust worked out again from the pairs dated
    from ``PAIRS_FROM`` only; every other rule unchanged."""
    later = pairs[pairs.index >= PAIRS_FROM]
    rebalances = {}
    for date, r in inputs.rebalances.items():
        trust, shut = optimize._trust_and_gate(date, later, horizon=horizon, gate=2.0)
        rebalances[date] = replace(r, trust=trust, gate_shut=shut)
    return replace(inputs, rebalances=rebalances)


def by_year(pairs, horizon) -> list[list[object]]:
    rows = []
    for year, group in pairs.groupby(pairs.index.year):
        x, y = group["x"].to_numpy(dtype=float), group["y"].to_numpy(dtype=float)
        partial = group.index[0].month > 1 or group.index[-1].month < 12
        if len(group) < 3 or np.ptp(x) == 0:
            rows.append([f"{year}{' (part)' if partial else ''}", len(group), "—", "—"])
            continue
        fit_ = metrics.overlap_fit(y, x, lags=horizon - 1)
        rows.append(
            [
                f"{year}{' (part)' if partial else ''}",
                len(group),
                f"{fit_.params[1]:+.2f}",
                f"{fit_.tvalues[1]:+.2f}",
            ]
        )
    return rows


def breakdown_section(results, prices_by_label) -> None:
    print("## 3. Why the 20-day gate opens 3 times here against 26 on 1 Nov\n")
    print(
        "Descriptive only; nothing was changed because of it. Every cell runs today's "
        "optimiser (no band, π̄ with its prior); the 1 Nov file ran the band and the "
        "plain expanding premium, and its machine-learning fit was the app's on that day.\n"
    )
    for h in BREAKDOWN_HORIZONS:
        print(f"### h = {h}: the four cells, pairs from the first out-of-sample date\n")
        rows, cells = [], {}
        for name, pairing in PAIRINGS.items():
            for label in (OLD, NEW):
                prices = prices_by_label[label]
                inputs, pairs = schedule(results, label, pairing, prices, h)
                cells[(name, label)] = (inputs, pairs, prices)
                runs = fixes.run_levels(inputs, prices, h)
                rows.append(
                    [
                        name,
                        label,
                        f"{len(pairs):,} from {day(pairs.index[0])}",
                        opened(inputs),
                        no_forecast_t(runs),
                    ]
                )
        table(
            ["models", "data", "trust pairs", "gate open at", "t vs no forecasts, levels 1–4"], rows
        )

        print(f"### h = {h}: pairs from {day(PAIRS_FROM)} instead\n")
        rows = []
        for (name, label), (inputs, pairs, prices) in cells.items():
            if pairs.index[0] >= PAIRS_FROM:
                rows.append(
                    [name, label, f"pairs start {day(pairs.index[0])}: unchanged", "—", "—"]
                )
                continue
            later = from_2010(inputs, pairs, h)
            runs = fixes.run_levels(later, prices, h)
            rows.append(
                [
                    name,
                    label,
                    f"{(pairs.index >= PAIRS_FROM).sum():,}",
                    opened(later),
                    no_forecast_t(runs),
                ]
            )
        table(
            ["models", "data", "trust pairs", "gate open at", "t vs no forecasts, levels 1–4"], rows
        )

        print(f"### h = {h}: the trust slope for each calendar year on its own\n")
        print(
            f"OLS of y on x with an intercept over the year's pairs (by forecast date), "
            f"standard errors over {h - 1} lags counted in full, as the gate's. One year "
            f"holds about {252 // h} independent {h}-day returns, so each t is noisy.\n"
        )
        for (name, label), (_, pairs, _) in cells.items():
            print(f"**{name}, {label}**\n")
            table(["year", "pairs", "slope", "t"], by_year(pairs, h))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    bundles = {
        NEW: history.read_folder(args.folder),
        OLD: pickle.loads(Path("data/last_commit.pkl").read_bytes()),
    }
    print(
        f"Sep 2026 upload (`data/last_commit.pkl`): content hash "
        f"`{frame_sha256(bundles[OLD]['extraction_committed'])}`.\n"
    )
    fama_french.resolve()  # fetched once here, not by every worker
    started = time.perf_counter()
    results, seconds = {}, {}
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(bundles,)) as pool:
        for job, result, took in pool.map(fit, jobs()):
            results[job], seconds[job] = result, took
    print(f"All {len(results)} fits: {time.perf_counter() - started:.0f} s wall-clock.\n")
    prices = {label: prices_of(bundle) for label, bundle in bundles.items()}
    model_section(results, seconds)
    portfolio_section(results, prices[NEW])
    breakdown_section(results, prices)


if __name__ == "__main__":
    main()
