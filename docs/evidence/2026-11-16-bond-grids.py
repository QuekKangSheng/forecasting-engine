"""Bond grids for the derived polynomial, 16 Nov 2026
(docs/evidence/2026-11-16-bond-grids.md).

Every run is the Models page's own walk-forward (test 20 rows, embargo 20, first 504
rows a tuning period, labels purged) and the page's own score (Signal Rank IC against
the model's own level, t on the larger of the Hansen-Hodrick and test-window standard
errors). What varies is the grid:

  library   which columns the fit is offered
            page          the page's panel: gauges as levels, rates as 1-row changes,
                          prices as 1-row log returns
            page+lvl      + each rate's level beside its change
            page+lvl+mom  + every series' change over the last h rows
            lvl           the gauges and the rates as levels only
            lvl+mom       levels and h-row changes, no 1-row changes
  fit       lasso1/2/3    the page's lasso at degree 1, 2, 3, screened per fold
            lasso1_ns     degree 1 with no screening
            shrink        the page's shrinkage candidate, screened per fold
            carry         the bond research model's fit (levels given their own
                          label-window innovations), on the lvl library only
  window    roll504       the page's: latest 504 rows
            cap1260       every earlier row, at most the latest 1,260 (five years)
            expanding     every earlier row

Each fit is causal, so one pass over the whole file scores both periods: "dev" is the
test windows whose every label ends before 1 Oct 2024, "held" the windows starting on
or after it.

Run: uv run --project <repo> python -I <this file> <folder of exports>
        [--horizons 20] [--windows roll504,cap1260,expanding] [--libraries ...]
        [--fits ...] [--workers 3] [--only <regex on library/fit/window>]
        [--out results.jsonl] [--keep <folder for each run's series>]
The folder holds the Bloomberg exports the Data page takes, .csv or .xlsx. A window is
``roll<rows>`` (the latest rows), ``cap<rows>`` (every earlier row, at most that many)
or ``expanding``. Results go to ``--out`` one JSON line per run, and a rerun skips runs
already there.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting_engine.extraction import bloomberg_csv, bloomberg_xlsx
from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.ingest.align import (
    FeaturePanel,
    Transform,
    align_and_lag,
    select_signals,
    transform_for,
)
from forecasting_engine.models.carry_research import CarryResearchPolynomial
from forecasting_engine.models.polynomial import DerivedPolynomial
from forecasting_engine.models.shrinkage import ShrinkagePolynomial
from forecasting_engine.validation.harness import evaluate, signal_forecasts, summarize
from forecasting_engine.validation.splitters import TUNING_ROWS, PurgedWalkForward, expanding

DATE = bloomberg_csv.DATE_COLUMN
TARGET_FIELD = "TOT_RETURN_INDEX"
TARGET_SECURITIES = {TargetRole.EQUITY: "SPX Index", TargetRole.BOND: "LBUSTRUU Index"}
TEST_WINDOW, EMBARGO, MAX_TERMS = 20, 20, 10
HELD_FROM = pd.Timestamp("2024-10-01")

LIBRARIES = ("page", "page+lvl", "page+lvl+mom", "lvl", "lvl+mom")
WINDOWS = ("roll504", "cap1260", "expanding")
FITS = ("lasso1", "lasso2", "lasso3", "lasso1_ns", "shrink", "carry")
#: Degree 3 on twenty-odd columns is two thousand terms a fold; left to the smaller libraries.
SMALL_LIBRARIES = ("page", "page+lvl", "lvl")


class CappedExpanding(PurgedWalkForward):
    """The page's test windows, each trained on every earlier row up to ``train``."""

    def split(self, panel: FeaturePanel):
        index = panel.frame.index
        test_start = self.tuning_rows + self.embargo
        while test_start + self.test <= len(index):
            yield (
                self.window_before(panel, test_start, self.train),
                index[test_start : test_start + self.test],
            )
            test_start += self.test


def splitter(window: str) -> PurgedWalkForward:
    page = PurgedWalkForward(train=504, test=TEST_WINDOW, embargo=EMBARGO, tuning_rows=TUNING_ROWS)
    if window == "expanding":
        return expanding(page)
    digits = window.lstrip("rolcap")
    kind, rows = window[: len(window) - len(digits)], int(digits)
    # Whatever the train window, the first test window opens where the page opens it.
    if kind == "roll" and rows <= TUNING_ROWS:
        return PurgedWalkForward(
            train=rows, test=TEST_WINDOW, embargo=EMBARGO, tuning_rows=TUNING_ROWS
        )
    if kind in ("roll", "cap"):
        return CappedExpanding(
            train=rows, test=TEST_WINDOW, embargo=EMBARGO, tuning_rows=TUNING_ROWS
        )
    raise ValueError(window)


def read_folder(folder: Path) -> dict:
    """The exports read and merged as the Data page merges them."""
    exports = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() == ".csv":
            exports.append(bloomberg_csv.read_export(path.name, path.read_bytes()))
        elif path.suffix.lower() == ".xlsx":
            exports.append(bloomberg_xlsx.read_export(path.name, path.read_bytes()))
    if not exports:
        raise SystemExit(f"no .csv or .xlsx export in {folder}")
    exports.sort(key=lambda e: e.security not in TARGET_SECURITIES.values())
    merged, _ = bloomberg_csv.drop_empty_columns(bloomberg_csv.merge(exports))
    columns = [c for c in merged.columns if c != DATE]
    sources = bloomberg_csv.column_sources(exports, columns)
    targets = {}
    for role, security in TARGET_SECURITIES.items():
        own = sorted(c for c, s in sources.items() if s.security == security)
        total = [c for c in own if sources[c].field.startswith(TARGET_FIELD)]
        last = [c for c in own if sources[c].field == "PX_LAST"]
        targets[role] = (total or last or own)[0]
    return {"merged": merged, "sources": sources, "targets": targets}


def build_panel(bundle: dict, library: str, horizon: int) -> FeaturePanel:
    merged, sources, targets = bundle["merged"], bundle["sources"], bundle["targets"]
    indexed = merged.set_index(DATE)
    price = targets[TargetRole.BOND]
    numeric = [c for c in merged.columns if c != DATE and merged[c].dtype.kind in "fi"]
    signals = select_signals(numeric, list(targets.values()), sources)
    transforms = {c: transform_for(sources.get(c)) for c in signals}
    page = align_and_lag(indexed, signals, price, horizon=horizon, transforms=transforms)
    if library == "page":
        return page
    # Every series' level as of each date, lagged a row like any signal.
    levels = align_and_lag(
        indexed, signals, price, horizon=horizon, transforms=dict.fromkeys(signals, Transform.LEVEL)
    ).frame[signals]
    rates = [c for c in signals if transforms[c] is Transform.DIFFERENCE]
    gauges = [c for c in signals if transforms[c] is Transform.LEVEL]
    prices = [c for c in signals if transforms[c] is Transform.LOG_RETURN]
    rate_levels = {f"{c}_LEVEL": levels[c] for c in rates}
    momentum = {f"{c}_CHG{horizon}": levels[c].diff(horizon) for c in [*gauges, *rates]}
    momentum.update({f"{c}_RET{horizon}": np.log(levels[c]).diff(horizon) for c in prices})
    extra, chosen = {}, []
    if library == "page+lvl":
        extra, chosen = rate_levels, [*signals, *rate_levels]
    elif library == "page+lvl+mom":
        extra = {**rate_levels, **momentum}
        chosen = [*signals, *rate_levels, *momentum]
    elif library == "lvl":
        extra, chosen = rate_levels, [*gauges, *rate_levels]
    elif library == "lvl+mom":
        extra = {**rate_levels, **momentum}
        chosen = [*gauges, *rate_levels, *momentum]
    else:
        raise ValueError(library)
    target = page.targets[0]
    frame = page.frame.assign(**extra)[[*chosen, price, target]]
    return replace(page, frame=frame, signals=tuple(chosen))


def forecaster(fit: str):
    """(factory, screened per fold?)"""
    if fit.startswith("lasso"):
        degree = int(fit[5])
        return (lambda: DerivedPolynomial(degree, "lasso", MAX_TERMS)), not fit.endswith("_ns")
    if fit == "shrink":
        return ShrinkagePolynomial, True
    if fit == "carry":
        return CarryResearchPolynomial, False
    raise ValueError(fit)


def _score(folds) -> dict:
    if not folds:
        return {"ic": None, "t": None, "windows": 0}
    result, _ = summarize(folds)
    return {
        "ic": _num(result.gated_rank_ic),
        "t": _num(result.signal_t),
        "windows": len(folds),
        "rows": result.rows_scored,
    }


def _num(value: float) -> float | None:
    return None if value != value else round(float(value), 4)


def _terms(folds) -> list[dict]:
    """Per term: the share of folds that kept it and, of those, the share positive."""
    kept: dict[str, list[float]] = {}
    for fold in folds:
        for term, coefficient in zip(
            fold.description.terms, fold.description.coefficients, strict=True
        ):
            kept.setdefault(term, []).append(coefficient)
    rows = [
        {
            "term": term,
            "kept": round(len(c) / len(folds), 3),
            "positive": round(sum(x > 0 for x in c) / len(c), 3),
        }
        for term, c in kept.items()
    ]
    return sorted(rows, key=lambda r: -r["kept"])[:8]


_BUNDLE: dict = {}
_KEEP: Path | None = None


def _init(bundle: dict, keep: str | None) -> None:
    global _KEEP
    _BUNDLE.update(bundle)
    _KEEP = Path(keep) if keep else None


def run(job: tuple[str, str, str, int]) -> dict:
    library, fit, window, horizon = job
    started = time.perf_counter()
    panel = build_panel(_BUNDLE, library, horizon)
    factory, screen = forecaster(fit)
    folds = evaluate(factory, panel, splitter(window), screen=screen)
    ends = {f.fold: panel.label_end.reindex(f.test).max() for f in folds}
    dev = tuple(f for f in folds if pd.notna(ends[f.fold]) and ends[f.fold] < HELD_FROM)
    held = tuple(f for f in folds if f.test[0] >= HELD_FROM)
    result, latest = summarize(folds)
    constant = result.constant_forecasts
    if _KEEP is not None:
        name = re.sub(r"[^\w]+", "_", f"{library}__{fit}__{window}__h{horizon}")
        with open(_KEEP / f"{name}.pkl", "wb") as handle:
            pickle.dump(
                {
                    "forecast": result.forecast,
                    "signal": signal_forecasts(folds),
                    "realised": result.realised,
                },
                handle,
            )
    return {
        "library": library,
        "fit": fit,
        "window": window,
        "horizon": horizon,
        "columns": len(panel.signals),
        "all": _score(folds),
        "dev": _score(dev),
        "held": _score(held),
        "constant_folds": constant.constant,
        "folds": constant.folds,
        "latest_constant": bool(folds[-1].predicted.dropna().nunique() <= 1),
        "latest_terms": list(latest.terms),
        "latest_coefficients": [round(float(c), 6) for c in latest.coefficients],
        "latest_intercept": round(float(latest.intercept), 6),
        "terms": _terms(folds),
        "first_test": str(folds[0].test[0].date()),
        "last_test": str(folds[-1].test[-1].date()),
        "seconds": round(time.perf_counter() - started, 1),
    }


def jobs(
    horizons: list[int],
    windows: tuple[str, ...] = WINDOWS,
    libraries: tuple[str, ...] = LIBRARIES,
    fits: tuple[str, ...] = FITS,
) -> list[tuple[str, str, str, int]]:
    every = []
    for horizon in horizons:
        for window in windows:
            for library in libraries:
                for fit in fits:
                    if fit == "carry" and library != "lvl":
                        continue
                    if fit == "lasso3" and library not in SMALL_LIBRARIES:
                        continue
                    every.append((library, fit, window, horizon))
    return every


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--horizons", default="20")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--only", default="")
    parser.add_argument("--windows", default=",".join(WINDOWS))
    parser.add_argument("--libraries", default=",".join(LIBRARIES))
    parser.add_argument("--fits", default=",".join(FITS))
    parser.add_argument("--out", type=Path, default=Path("results.jsonl"))
    parser.add_argument("--keep", type=Path, default=None, help="folder for each run's series")
    args = parser.parse_args()
    bundle = read_folder(args.folder)
    merged = bundle["merged"]
    print(
        f"{len(merged):,} rows, {merged[DATE].min():%d %b %Y} to {merged[DATE].max():%d %b %Y}; "
        f"bond target {bundle['targets'][TargetRole.BOND]}",
        flush=True,
    )
    todo = jobs(
        [int(h) for h in args.horizons.split(",")],
        tuple(args.windows.split(",")),
        tuple(args.libraries.split(",")),
        tuple(args.fits.split(",")),
    )
    if args.only:
        todo = [j for j in todo if re.search(args.only, "/".join(map(str, j[:3])))]
    done = set()
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done.add((row["library"], row["fit"], row["window"], row["horizon"]))
    todo = [j for j in todo if j not in done]
    if args.keep:
        args.keep.mkdir(parents=True, exist_ok=True)
    print(f"{len(todo)} runs to do, {len(done)} already in {args.out}", flush=True)
    keep = str(args.keep) if args.keep else None
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(bundle, keep)) as pool:
        pending = {pool.submit(run, job): job for job in todo}
        for future in as_completed(pending):
            job = pending[future]
            try:
                row = future.result()
            except Exception as exc:  # noqa: BLE001 - a failed run is a result too
                row = {
                    "library": job[0],
                    "fit": job[1],
                    "window": job[2],
                    "horizon": job[3],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            with args.out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            shown = row.get("error") or (
                f"all {row['all']['ic']} t {row['all']['t']} | dev {row['dev']['ic']} | "
                f"held {row['held']['ic']} | {row['seconds']} s"
            )
            print(f"{'/'.join(map(str, job))}: {shown}", flush=True)


if __name__ == "__main__":
    main()
