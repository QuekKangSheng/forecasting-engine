"""The Bond tab's derived polynomial and machine learning, chosen per window from earlier
windows only, against the whole-period pick, on the data from 31 Aug 2006.

For each horizon, the functions the Models page's runners call are run once for the bond
target, with the page's defaults: ``run_derived_polynomial(..., role=BOND)`` and
``run_boosted(..., role=BOND)``. Both choose online (``harness.summarize_online``). The
candidates' walk-forwards they evaluated are captured on the way, so the whole-period pick
the page made before (``harness.select_best_candidate``) is scored on the same fits, with
nothing refitted. Each is scored over the whole period, "fresh" (test windows whose every
label ends before 1 Aug 2016) and "later" (test windows starting 31 Aug 2016 - 30 Sep
2026), as in the research (s15-bond-grids-honest, ``select15.py``), whose G0 rows the
derived polynomial's figures are checked against.

Heavy: machine learning tunes 70-odd Optuna studies per library per horizon. Run it with
nothing else heavy on the machine; ``FORECASTING_WORKERS`` (default 4) caps its processes.

Run: uv run --project <repo> python -I <this file> <folder holding the 11 .xlsx files>
     [--models derived,ml] [--horizons 1,5,10,20]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import pandas as pd

from forecasting_engine.extraction import bloomberg_csv, bloomberg_xlsx
from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.ingest.align import align_and_lag, select_signals, transform_for
from forecasting_engine.ingest.upload import frame_sha256
from forecasting_engine.models import boosted, polynomial
from forecasting_engine.models.polynomial import CANDIDATE_CONFIGS, DerivedPolynomial
from forecasting_engine.validation import harness
from forecasting_engine.validation.splitters import TUNING_ROWS, PurgedWalkForward

DATE = bloomberg_csv.DATE_COLUMN
BOND = TargetRole.BOND
TARGET_FIELD = "TOT_RETURN_INDEX_GROSS_DVDS"
TARGET_SECURITIES = {TargetRole.EQUITY: "SPX Index", BOND: "LBUSTRUU Index"}
HORIZONS = (1, 5, 10, 20)

# The Models page's defaults (app/model_settings.py).
TRAIN_WINDOW, TEST_WINDOW, EMBARGO_DAYS, MAX_TERMS = 504, 20, 20, 10

FRESH_BEFORE = pd.Timestamp("2016-08-01")
LATER_FROM = pd.Timestamp("2016-08-31")
NEW_FROM = pd.Timestamp("2026-10-01")

#: s15-bond-grids-honest/part_b.txt, G0 (today's grid): (whole, fresh, later) as
#: (Signal Rank IC, t), online then the page's whole-period pick, and the three
#: candidates used in the most windows online, most first.
RESEARCH = {
    1: (
        ((0.029, 1.84), (0.015, 0.63), (0.034, 1.60)),
        ((0.029, 1.84), (0.015, 0.63), (0.034, 1.60)),
        "shrinkage 228",
    ),
    5: (
        ((0.032, 1.19), (-0.008, -0.20), (0.038, 1.07)),
        ((0.030, 1.00), (-0.009, -0.24), (0.043, 1.04)),
        "shrinkage 151, degree1_lasso 42, degree3_lasso 23",
    ),
    10: (
        ((-0.022, -0.53), (-0.113, -1.98), (0.053, 0.98)),
        ((0.021, 0.48), (-0.049, -0.80), (0.056, 0.95)),
        "degree1_lasso 133, degree2_lasso 61, shrinkage 25",
    ),
    20: (
        ((-0.054, -1.02), (-0.111, -1.44), (-0.007, -0.09)),
        ((0.030, 0.49), (-0.088, -1.03), (0.092, 1.14)),
        "degree1_lasso 149, degree2_lasso 39, degree3_lasso 27",
    ),
}


def table(header: list[str], rows: list[list[object]]) -> None:
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in rows:
        print("| " + " | ".join(str(v) for v in row) + " |")
    print()


def day(value: pd.Timestamp) -> str:
    return "—" if pd.isna(value) else f"{value.day} {value:%b %Y}"


def read_folder(folder: Path):
    """The exports merged as the Data page merges them, and the page's bond panel inputs."""
    exports, rows = [], []
    for path in sorted(folder.glob("*.xlsx")):
        data = path.read_bytes()
        export = bloomberg_xlsx.read_export(path.name, data)
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
    if not exports:
        raise SystemExit(f"no .xlsx export in {folder}")
    print("## The files\n")
    table(["file", "SHA-256", "rows", "first", "last"], rows)
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
    numeric = [c for c in merged.columns if c != DATE and merged[c].dtype.kind in "fi"]
    signals = select_signals(numeric, list(targets.values()), sources)
    transforms = {c: transform_for(sources.get(c)) for c in signals}
    print(
        f"Merged: {len(merged):,} rows, {day(merged[DATE].min())} to {day(merged[DATE].max())}, "
        f"content hash (`ingest.upload.frame_sha256`) `{frame_sha256(merged)}`. Empty columns "
        f"dropped: {dropped or 'none'}. Bond target `{targets[BOND]}`; {len(signals)} "
        f"signals.\n"
    )
    return merged.set_index(DATE), targets[BOND], signals, transforms


def splitter() -> PurgedWalkForward:
    return PurgedWalkForward(
        train=TRAIN_WINDOW, test=TEST_WINDOW, embargo=EMBARGO_DAYS, tuning_rows=TUNING_ROWS
    )


# ── Capturing the candidates the app evaluated ───────────────────────────────

_CAPTURED: dict = {}
_REAL = harness.summarize_online


def _capture(per_candidate, panel, default):
    _CAPTURED["per_candidate"], _CAPTURED["default"] = dict(per_candidate), default
    return _REAL(per_candidate, panel, default)


polynomial.summarize_online = _capture
boosted.summarize_online = _capture


# ── Scores ──────────────────────────────────────────────────────────────────


def _period(test: pd.DatetimeIndex, label_end: pd.Series) -> str | None:
    end = label_end.reindex(test).max()
    if pd.notna(end) and end < FRESH_BEFORE:
        return "fresh"
    if test[0] >= NEW_FROM:
        return "new"
    if test[0] >= LATER_FROM:
        return "later"
    return None


def _scores(folds, label_end) -> dict:
    out = {}
    for name in ("whole", "fresh", "later"):
        part = folds if name == "whole" else tuple(
            f for f in folds if _period(f.test, label_end) == name
        )
        result, _ = harness.summarize(part)
        out[name] = (result.gated_rank_ic, result.signal_t)
    return out


def both_ways(result, panel) -> dict:
    """The app's online result's folds and the whole-period pick's, on the same fits."""
    per_candidate, default = _CAPTURED["per_candidate"], _CAPTURED["default"]
    picks = harness.select_online(per_candidate, panel.label_end, default)
    online = tuple(per_candidate[p][k] for k, p in enumerate(picks))
    assert harness.summarize(online)[0].signal_rank_ic == result.signal_rank_ic
    best = harness.select_best_candidate(per_candidate)
    return {
        "online": _scores(online, panel.label_end),
        "biased": _scores(per_candidate[best], panel.label_end),
        "pick": best,
        "used": ", ".join(f"{n} {c}" for n, c in result.selection.windows if c),
        # As the research prints it: the three most used, most first.
        "top": ", ".join(
            f"{n} {c}" for n, c in sorted(result.selection.windows, key=lambda nc: -nc[1])[:3]
            if c
        ),
        "latest": result.selection.latest,
    }


def fit(model: str, horizon: int, indexed, bond, signals, transforms) -> dict:
    panel = align_and_lag(indexed, signals, bond, horizon=horizon, transforms=transforms)
    started = time.perf_counter()
    if model == "derived":
        candidates = tuple(
            DerivedPolynomial(degree=c.degree, regularizer=c.regularizer, max_terms=MAX_TERMS)
            for c in CANDIDATE_CONFIGS
        )
        result, _ = polynomial.run_derived_polynomial(
            panel, splitter(), candidates=candidates, role=BOND
        )
    else:
        result, _, _ = boosted.run_boosted(panel, splitter(), role=BOND)
    out = both_ways(result, panel)
    out["seconds"] = time.perf_counter() - started
    return out


def _cell(score: tuple[float, float]) -> str:
    return f"{score[0]:+.3f}, t {score[1]:.2f}"


def report(model: str, rows: dict[int, dict]) -> None:
    title = "Derived polynomial" if model == "derived" else "Machine learning"
    print(f"## {title}, bond\n")
    out = []
    for h, r in rows.items():
        for way in ("online", "biased"):
            s = r[way]
            out.append(
                [
                    h,
                    "per window, earlier windows only" if way == "online" else
                    f"whole-period pick ({r['pick']})",
                    _cell(s["whole"]),
                    _cell(s["fresh"]),
                    _cell(s["later"]),
                    r["used"] if way == "online" else "",
                    f"{r['seconds']:.0f} s" if way == "online" else "",
                ]
            )
    table(["h", "selection", "whole", "fresh", "later", "windows used", "runtime"], out)
    if model != "derived":
        return
    print("**Against the research** (`part_b.txt`, G0), to 3 decimals:\n")
    out, worst = [], 0.0
    for h, r in rows.items():
        theirs_online, theirs_biased, used = RESEARCH[h]
        for way, theirs in (("online", theirs_online), ("biased", theirs_biased)):
            for period, (ic, t) in zip(("whole", "fresh", "later"), theirs, strict=True):
                mine = r[way][period]
                gap = max(abs(round(mine[0], 3) - ic), abs(round(mine[1], 2) - t))
                worst = max(worst, abs(round(mine[0], 3) - ic))
                out.append([h, way, period, f"{ic:+.3f}, t {t:.2f}", _cell(mine),
                            "same" if gap < 1e-9 else f"differs by {gap:.3f}"])
        out.append([h, "windows used (top 3)", "", used, r["top"],
                    "same" if used == r["top"] else "differs"])
    table(["h", "selection", "period", "research", "app", ""], out)
    print(f"Largest Signal Rank IC difference at 3 decimals: {worst:.3f}.\n")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--models", default="derived,ml")
    parser.add_argument("--horizons", default=",".join(map(str, HORIZONS)))
    args = parser.parse_args()
    indexed, bond, signals, transforms = read_folder(args.folder)
    print(
        f"Online selection: `MIN_SELECTION_HISTORY` = {harness.MIN_SELECTION_HISTORY}; "
        f"defaults: derived `{polynomial.ONLINE_SELECTION_DEFAULT[BOND]}`, machine learning "
        f"`{boosted.ONLINE_SELECTION_DEFAULT[BOND]}`.\n"
    )
    horizons = [int(h) for h in args.horizons.split(",")]
    for model in args.models.split(","):
        rows = {}
        for h in horizons:
            rows[h] = fit(model, h, indexed, bond, signals, transforms)
            print(f"<!-- {model} h={h} done in {rows[h]['seconds']:.0f} s -->", flush=True)
        report(model, rows)


if __name__ == "__main__":
    main()
