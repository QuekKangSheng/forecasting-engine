"""The Bloomberg history from 31 Aug 2006: coverage, where each part starts, how the files
compare with the Sep 2026 upload, both research models by period (every window, the
pre-2016 hold-out, the windows from Aug 2016), and runtime against the Sep 2026 upload.

Reads the 11 Bloomberg exports through the same functions the Data page calls
(``bloomberg_xlsx.read_export``, the per-file schema check, ``bloomberg_csv.merge``,
``drop_empty_columns``), and fits each model through the functions the Models page's
runners call, at the page's defaults. The Sep 2026 upload it compares against is the
Data page's last commit, ``data/last_commit.pkl``, relative to the working directory.

Only the research models' scores are printed. Every other model's figures on this data
belong to the single final rerun after the optimiser fixes; here only their runtime is.

Run: uv run --project <repo> python -I <this file> <folder holding the 11 .xlsx files>
     [--models naive,derived,research,ml,ff5]
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
import time
from pathlib import Path

import pandas as pd

from forecasting_engine.extraction import bloomberg_csv, bloomberg_xlsx, validation
from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.ingest import fama_french
from forecasting_engine.ingest.align import (
    Transform,
    align_and_lag,
    select_signals,
    transform_for,
)
from forecasting_engine.ingest.upload import frame_sha256
from forecasting_engine.models.boosted import run_boosted
from forecasting_engine.models.carry_research import (
    CARRY_INPUTS,
    CarryResearchPolynomial,
    carry_inputs,
    run_carry_research_polynomial,
)
from forecasting_engine.models.famafrench import FACTOR_COLUMNS, merge_factors, run_famafrench
from forecasting_engine.models.naive import run_naive
from forecasting_engine.models.polynomial import (
    CANDIDATE_CONFIGS,
    DerivedPolynomial,
    run_derived_polynomial,
)
from forecasting_engine.models.sign_ruled import (
    ECONOMIC_INPUTS,
    HOLDOUT_BEFORE,
    LEVEL_ROWS,
    SignRuledPolynomial,
    economic_signs,
    run_sign_ruled_polynomial,
)
from forecasting_engine.portfolio.optimize import MIN_TRUST_HISTORY, common_rebalance_dates
from forecasting_engine.validation.harness import evaluate, holdout, summarize
from forecasting_engine.validation.splitters import TUNING_ROWS, PurgedWalkForward, expanding

EQUITY, BOND = TargetRole.EQUITY, TargetRole.BOND
DATE = bloomberg_csv.DATE_COLUMN

# The Models page's defaults (app/model_settings.py), hardcoded so this script has no
# import-path dependency on the app/ package.
HORIZON = 20
TRAIN_WINDOW = 504
TEST_WINDOW = 20
EMBARGO_DAYS = 20
MAX_TERMS = 10
TARGET_FIELD = "TOT_RETURN_INDEX_GROSS_DVDS"
TARGET_SECURITIES = {EQUITY: "SPX Index", BOND: "LBUSTRUU Index"}
DATA_START = pd.Timestamp("2006-08-31")
MAX_MISSING = 0.05


def table(header: list[str], rows: list[list[object]]) -> None:
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in rows:
        print("| " + " | ".join(str(v) for v in row) + " |")
    print()


def day(value: pd.Timestamp) -> str:
    return "—" if pd.isna(value) else f"{value.day} {value:%b %Y}"


# ── The upload ───────────────────────────────────────────────────────────────


def read_folder(folder: Path) -> dict:
    """Every export in ``folder`` read, checked and merged as the Data page does, in the
    shape the Data page commits (``data/last_commit.pkl``)."""
    exports, rows = [], []
    for path in sorted(folder.glob("*.xlsx")):
        data = path.read_bytes()
        export = bloomberg_xlsx.read_export(path.name, data)
        errors = validation.schema_errors(export.frame.sort_values(DATE))
        if errors:
            raise SystemExit(f"{path.name} fails the schema: {'; '.join(errors)}")
        dates = export.frame[DATE]
        rows.append(
            [
                path.name,
                f"`{hashlib.sha256(data).hexdigest()}`",
                f"{len(export.frame):,}",
                day(dates.min()),
                day(dates.max()),
            ]
        )
        exports.append(export)
    print("## The files\n")
    table(["file", "SHA-256", "rows", "first", "last"], rows)
    # Targets first, as the Data page passes them: earlier exports win a shared label.
    exports.sort(key=lambda e: e.security not in TARGET_SECURITIES.values())
    merged, dropped = bloomberg_csv.drop_empty_columns(bloomberg_csv.merge(exports))
    columns = [c for c in merged.columns if c != DATE]
    sources = bloomberg_csv.column_sources(exports, columns)
    targets = {
        role: next(
            c
            for c, s in sources.items()
            if s.security == security and s.field.startswith(TARGET_FIELD)
        )
        for role, security in TARGET_SECURITIES.items()
    }
    print(
        f"Merged: {len(merged):,} rows, {day(merged[DATE].min())} to {day(merged[DATE].max())}, "
        f"{len(columns)} columns, content hash (`ingest.upload.frame_sha256`) "
        f"`{frame_sha256(merged)}`. Empty columns dropped: {dropped or 'none'}.\n"
    )
    return {
        "extraction_committed": merged,
        "extraction_committed_targets": targets,
        "extraction_committed_sources": sources,
    }


def coverage(bundle: dict) -> None:
    """Per column, its first and last valid date and the share of the shared target
    calendar's days in each year it has no value for, with the plan's stop rule."""
    frame = bundle["extraction_committed"].set_index(DATE)
    sources = bundle["extraction_committed_sources"]
    targets = bundle["extraction_committed_targets"]
    calendar = frame[list(targets.values())].dropna().index
    watched = {
        *targets.values(),
        *(c for c, s in sources.items() if s.ticker in ECONOMIC_INPUTS[EQUITY]),
        *(c for c, s in sources.items() if s.ticker in CARRY_INPUTS[BOND]),
    }
    years = sorted(set(calendar.year))
    rows, stops = [], []
    for column in frame.columns:
        values = frame[column]
        present = values.dropna().index
        shares = []
        for year in years:
            days = calendar[(calendar.year == year) & (calendar >= DATA_START)]
            share = float(values.reindex(days).isna().mean()) if len(days) else float("nan")
            shares.append(share)
            if column in watched and share > MAX_MISSING:
                stops.append(f"{column}: {share:.1%} missing in {year}")
        if column in watched and present.min() > DATA_START:
            stops.append(f"{column} starts {day(present.min())}")
        rows.append(
            [
                column + (" *" if column in watched else ""),
                day(present.min()),
                day(present.max()),
                *(f"{s:.1%}" for s in shares),
            ]
        )
    print("## Coverage\n")
    print(
        "Share of the shared target calendar's days (both target indices priced) with no "
        f"value, per calendar year; 2006 from {day(DATA_START)}. * = a target or a research "
        f"model input, where the stop rule applies (starts after {day(DATA_START)}, or a "
        f"year more than {MAX_MISSING:.0%} missing).\n"
    )
    table(["column", "first", "last", *map(str, years)], rows)
    print(f"**Stop rule:** {'; '.join(stops) if stops else 'not triggered.'}\n")


# ── Where each part starts ───────────────────────────────────────────────────


def splitter() -> PurgedWalkForward:
    return PurgedWalkForward(
        train=TRAIN_WINDOW, test=TEST_WINDOW, embargo=EMBARGO_DAYS, tuning_rows=TUNING_ROWS
    )


def page_inputs(bundle: dict):
    merged = bundle["extraction_committed"]
    targets = bundle["extraction_committed_targets"]
    sources = bundle["extraction_committed_sources"]
    numeric = [c for c in merged.columns if c != DATE and merged[c].dtype.kind in "fi"]
    signals = select_signals(numeric, list(targets.values()), sources)
    transforms = {c: transform_for(sources.get(c)) for c in signals}
    return merged, merged.set_index(DATE), targets, sources, signals, transforms


def start_dates(bundle: dict) -> None:
    _, indexed, targets, _, signals, transforms = page_inputs(bundle)
    panels = {
        role: align_and_lag(indexed, signals, col, horizon=HORIZON, transforms=transforms)
        for role, col in targets.items()
    }
    rows = []
    tests = {}
    for role, panel in panels.items():
        index = panel.frame.index
        folds = list(splitter().split(panel))
        expanding_folds = list(expanding(splitter()).split(panel))
        tests[role] = pd.DatetimeIndex(sorted({d for _, test in folds for d in test}))
        rows += [
            [role.value, "rows on its calendar", f"{len(index):,}", day(index[0]), day(index[-1])],
            [
                role.value,
                f"tuning period (first {TUNING_ROWS} rows)",
                TUNING_ROWS,
                day(index[0]),
                day(index[TUNING_ROWS - 1]),
            ],
            [
                role.value,
                "first test window (after tuning + embargo)",
                TEST_WINDOW,
                day(folds[0][1][0]),
                day(folds[0][1][-1]),
            ],
            [
                role.value,
                f"first rolling training window ({TRAIN_WINDOW} rows, purged)",
                len(folds[0][0]),
                day(folds[0][0][0]),
                day(folds[0][0][-1]),
            ],
            [
                role.value,
                "first expanding (research) training window",
                len(expanding_folds[0][0]),
                day(expanding_folds[0][0][0]),
                day(expanding_folds[0][0][-1]),
            ],
            [role.value, "test windows", len(folds), day(folds[0][1][0]), day(folds[-1][1][-1])],
        ]
        if role == EQUITY:
            first = expanding_folds[0][0][:LEVEL_ROWS]
            rows.append(
                [
                    role.value,
                    f"equity research level (first {LEVEL_ROWS} training labels)",
                    len(first),
                    day(first[0]),
                    day(panel.label_end.loc[first[-1]]),
                ]
            )
    print("## Where each part starts\n")
    print(
        f"Page defaults: {HORIZON}-day horizon, {TRAIN_WINDOW}-row train window, "
        f"{TEST_WINDOW}-row test window, {EMBARGO_DAYS}-row embargo.\n"
    )
    table(["target", "part", "rows", "from", "to"], rows)
    shared = tests[EQUITY].intersection(tests[BOND])
    label_end = pd.concat([panels[r].label_end.reindex(shared) for r in panels], axis=1).max(axis=1)
    known = label_end.sort_values()
    print(
        f"Forecast pairs (both targets forecast the date): {len(shared):,}, from "
        f"{day(shared[0])}. Pair number {MIN_TRUST_HISTORY}'s outcome is known on "
        f"{day(known.iloc[MIN_TRUST_HISTORY - 1])}: the earliest the forecast trust can "
        "open, if its t is above 2 then.\n"
    )
    rebalances = common_rebalance_dates(
        {role: indexed[col] for role, col in targets.items()},
        horizon=HORIZON,
        train_window=TRAIN_WINDOW,
        test_window=TEST_WINDOW,
        embargo=EMBARGO_DAYS,
    )
    print(
        f"Rebalances (every {HORIZON}th out-of-sample date both targets share): "
        f"{len(rebalances)}, {day(rebalances[0])} to {day(rebalances[-1])}.\n"
    )


# ── The models ───────────────────────────────────────────────────────────────


def runners(bundle: dict, role: TargetRole):
    merged, indexed, targets, sources, signals, transforms = page_inputs(bundle)
    price = targets[role]
    panel = align_and_lag(indexed, signals, price, horizon=HORIZON, transforms=transforms)
    split = splitter()

    def derived():
        candidates = tuple(
            DerivedPolynomial(degree=c.degree, regularizer=c.regularizer, max_terms=MAX_TERMS)
            for c in CANDIDATE_CONFIGS
        )
        return run_derived_polynomial(panel, split, candidates=candidates)[0]

    def research():
        if role == EQUITY:
            signs = economic_signs(signals, sources, role)
            return run_sign_ruled_polynomial(indexed, signs, price, HORIZON, split)[0]
        inputs = carry_inputs(signals, sources, role)
        return run_carry_research_polynomial(indexed, inputs, price, HORIZON, split)[0]

    def ff5():
        resolved = fama_french.resolve()
        factors = merge_factors(merged, resolved.file.frame).set_index(DATE)
        ff_panel = align_and_lag(
            factors, list(FACTOR_COLUMNS), price, horizon=HORIZON, exact=FACTOR_COLUMNS
        )
        return run_famafrench(ff_panel, split)[0]

    every = {
        "naive": lambda: run_naive(panel, split)[0],
        "derived": derived,
        "research": research,
        "ml": lambda: run_boosted(panel, split)[0],
    }
    if role == EQUITY:
        every["ff5"] = ff5
    return every


def run_models(bundles: dict[str, dict], models: list[str]) -> None:
    timing, research = [], []
    for role in (EQUITY, BOND):
        for name in models:
            row = [role.value, name]
            for label, bundle in bundles.items():
                every = runners(bundle, role)
                if name not in every:
                    row.append("—")
                    continue
                started = time.perf_counter()
                try:
                    result = every[name]()
                except Exception as exc:  # noqa: BLE001 - reported, not hidden
                    row.append(f"failed: {exc}")
                    continue
                row.append(f"{time.perf_counter() - started:.0f} s")
                if name == "research":
                    research.append([role.value, label, result])
            timing.append(row)
            print(f"  ran {role.value} {name}", flush=True)
    print("\n## Runtime\n")
    print(
        "Wall-clock seconds for one fit of each model at the page's defaults, measured "
        "locally (not on the hosted app), one model at a time.\n"
    )
    table(["target", "model", *bundles], timing)
    print("## Research models: full period and pre-2016 hold-out\n")
    rows = []
    for role, label, result in research:
        held = result.holdout
        rows.append(
            [
                role,
                label,
                f"{result.gated_rank_ic:+.4f}",
                f"{result.signal_t:.2f}",
                f"{result.rows_scored:,}",
                "—" if held is None else f"{held.signal_rank_ic:+.4f}",
                "—" if held is None else f"{held.t:.2f}",
                "—" if held is None else held.windows,
                "—" if held is None else f"{day(held.first)} – {day(held.last_label_end)}",
            ]
        )
    print(f"Hold-out: the test windows whose every label ends before {day(HOLDOUT_BEFORE)}.\n")
    table(
        [
            "target",
            "data",
            "Signal Rank IC",
            "t",
            "rows",
            "hold-out Signal Rank IC",
            "hold-out t",
            "hold-out windows",
            "hold-out span",
        ],
        rows,
    )


def compare_uploads(before: dict, after: dict) -> None:
    """Per column, over the dates both uploads hold: how often the two differ, by how
    much at most, and by how much their daily changes differ at most."""
    old = before["extraction_committed"].set_index(DATE)
    new = after["extraction_committed"].set_index(DATE)
    shared = old.index.intersection(new.index)
    rows = []
    for column in old.columns:
        if column not in new.columns:
            rows.append([column, "not in the 2006 files", "", "", "", "", ""])
            continue
        x, y = new.loc[shared, column], old.loc[shared, column]
        both = x.notna() & y.notna()
        gap = (x - y)[both].abs()
        differ = gap > 0
        ratio = (x / y)[both & differ]
        change = (x.pct_change(fill_method=None) - y.pct_change(fill_method=None)).abs()
        rows.append(
            [
                column,
                f"{int(both.sum()):,}",
                f"{int((x.notna() != y.notna()).sum()):,}",
                f"{differ.mean():.1%}",
                f"{gap.max():.6g}",
                "—" if ratio.empty else f"{ratio.min():.6f} – {ratio.max():.6f}",
                f"{change.max():.2g}",
            ]
        )
    print("## The 2006 files against the Sep 2026 upload\n")
    print(
        f"Over the {len(shared):,} dates both hold, {day(shared[0])} to {day(shared[-1])}. "
        f"The Sep 2026 upload has {len(old.index.difference(new.index)):,} dates the 2006 "
        "files do not. A difference is any at all; the last column is the largest gap "
        "between the two series' daily percentage changes, which is what a log-return "
        "transform reads.\n"
    )
    table(
        [
            "column",
            "dates with a value in both",
            "dates with a value in only one",
            "share differing",
            "largest difference",
            "2006 ÷ Sep 2026 where they differ",
            "largest gap in daily change",
        ],
        rows,
    )


def research_folds(bundle: dict, role: TargetRole):
    """A research model's panel and folds, fitted exactly as its run function fits
    them, so a score can be taken over any subset of its test windows."""
    _, indexed, targets, sources, signals, _ = page_inputs(bundle)
    price = targets[role]
    if role == EQUITY:
        signs = economic_signs(signals, sources, role)
        inputs, factory = list(signs), lambda: SignRuledPolynomial(dict(signs))
    else:
        inputs, factory = list(carry_inputs(signals, sources, role)), CarryResearchPolynomial
    panel = align_and_lag(
        indexed,
        inputs,
        price,
        horizon=HORIZON,
        transforms=dict.fromkeys(inputs, Transform.LEVEL),
    )
    return panel, evaluate(factory, panel, expanding(splitter()))


def research_by_period(bundle: dict) -> None:
    """Each research model on the data from 31 Aug 2006 over every test window, over
    its pre-2016 hold-out, and over the windows starting on or after the cutoff."""
    rows = []
    for role in (EQUITY, BOND):
        panel, folds = research_folds(bundle, role)
        later = tuple(f for f in folds if f.test[0] >= HOLDOUT_BEFORE)
        held = holdout(panel, folds, HOLDOUT_BEFORE)
        for name, result, windows, span in (
            ("every test window", summarize(folds)[0], len(folds), (folds[0], folds[-1])),
            ("pre-2016 hold-out", held, held.windows, None),
            (
                f"windows starting on or after {day(HOLDOUT_BEFORE)}",
                summarize(later)[0],
                len(later),
                (later[0], later[-1]),
            ),
        ):
            if span is None:
                ic, t, rows_scored = result.signal_rank_ic, result.t, result.rows_scored
                dates = f"{day(result.first)} – labels to {day(result.last_label_end)}"
            else:
                ic, t, rows_scored = result.gated_rank_ic, result.signal_t, result.rows_scored
                dates = f"{day(span[0].test[0])} – {day(span[1].test[-1])}"
            rows.append(
                [role.value, name, f"{ic:+.4f}", f"{t:.2f}", windows, f"{rows_scored:,}", dates]
            )
    print("## Research models on the data from 31 Aug 2006, by period\n")
    print(
        "Each model's folds are fitted once, as its run fits them, and scored over each "
        "set of test windows. The hold-out keeps a window only if its every label ends "
        f"before {day(HOLDOUT_BEFORE)}; the later set keeps every window whose first test "
        "date is on or after it. A window whose test dates start before the cutoff but "
        "whose labels end after it is in neither.\n"
    )
    table(["target", "test windows", "Signal Rank IC", "t", "windows", "rows", "span"], rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--models", default="naive,derived,research,ml,ff5")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    after = read_folder(args.folder)
    coverage(after)
    start_dates(after)
    before = pickle.loads(Path("data/last_commit.pkl").read_bytes())
    print(
        f"Sep 2026 upload (`data/last_commit.pkl`): content hash "
        f"`{frame_sha256(before['extraction_committed'])}`.\n"
    )
    compare_uploads(before, after)
    research_by_period(after)
    run_models({"Sep 2026 upload": before, "from 31 Aug 2006": after}, args.models.split(","))


if __name__ == "__main__":
    main()
