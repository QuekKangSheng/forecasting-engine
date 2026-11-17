"""The bond calendar model (``models/calendar_research``) on the data from 31 Aug 2006.

1. The files read, as the Data page reads and merges them.
2. The calendar the coming trading days are taken from: the bond index's own dates
   against each candidate calendar, and the months whose last row in the data is not
   the bond market's last trading day.
3. The model's columns against the November 2026 research's (``s14-bond-short/grid14.py``,
   ``calendar``, copied below as ``research_calendar``), row for row.
4. The model at h = 1, 5, 10 and 20, through the function the Models page's runner
   calls, at the page's defaults: Signal Rank IC and t over the whole scored period,
   "fresh" (test windows whose every label ends before 1 Aug 2016, the hold-out the
   page shows) and "later" (test windows starting 31 Aug 2016 - 30 Sep 2026), constant
   folds, the latest fold's equation and how often each column was kept and with
   which sign; and the research's own figures beside them.

Self-contained: it imports only the forecasting_engine package and hardcodes the Models
page's defaults (app/model_settings.py), so it has no import-path dependency on app/.

Run: uv run --project <repo> python -I <this file> <folder holding the 11 .xlsx files>
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

from forecasting_engine.extraction import bloomberg_csv, bloomberg_xlsx
from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.ingest.align import align_and_lag
from forecasting_engine.ingest.upload import frame_sha256
from forecasting_engine.models.calendar_research import (
    HOLIDAY_CALENDAR,
    VERSION,
    label_window_calendar,
    run_calendar_research_polynomial,
)
from forecasting_engine.models.shrinkage import ShrinkagePolynomial
from forecasting_engine.models.sign_ruled import HOLDOUT_BEFORE
from forecasting_engine.validation.gates import evaluate_candidate
from forecasting_engine.validation.harness import evaluate, summarize
from forecasting_engine.validation.splitters import TUNING_ROWS, PurgedWalkForward

DATE = bloomberg_csv.DATE_COLUMN
BOND = TargetRole.BOND
TARGET_FIELD = "TOT_RETURN_INDEX_GROSS_DVDS"
TARGET_SECURITIES = {TargetRole.EQUITY: "SPX Index", BOND: "LBUSTRUU Index"}
HORIZONS = (1, 5, 10, 20)

# The Models page's defaults (app/model_settings.py).
TRAIN_WINDOW, TEST_WINDOW, EMBARGO_DAYS = 504, 20, 20

# The research's periods (s13-bond-2006/grid13.py).
LATER_FROM = pd.Timestamp("2016-08-31")
NEW_FROM = pd.Timestamp("2026-10-01")

#: The research's figures for ``cal / shrink / roll504`` (s14-bond-short/results_lin.jsonl),
#: run on main at 1bc88e0: (whole IC, t), (fresh IC, t), (later IC, t), constant folds.
RESEARCH = {
    1: ((0.0510, 3.2487), (0.0637, 2.5164), (0.0443, 2.1094), 0),
    5: ((0.0834, 3.2448), (0.0550, 1.4080), (0.0936, 2.7164), 0),
}


def table(header: list[str], rows: list[list[object]]) -> None:
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in rows:
        print("| " + " | ".join(str(v) for v in row) + " |")
    print()


def day(value: pd.Timestamp) -> str:
    return "—" if pd.isna(value) else f"{value.day} {value:%b %Y}"


# ── The data ─────────────────────────────────────────────────────────────────


def read_folder(folder: Path) -> tuple[pd.DataFrame, str]:
    """The exports merged as the Data page merges them, and the bond target's column."""
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
    # Targets first, as the Data page passes them: earlier exports win a shared label.
    exports.sort(key=lambda e: e.security not in TARGET_SECURITIES.values())
    merged, dropped = bloomberg_csv.drop_empty_columns(bloomberg_csv.merge(exports))
    columns = [c for c in merged.columns if c != DATE]
    sources = bloomberg_csv.column_sources(exports, columns)
    bond = next(
        c
        for c, s in sources.items()
        if s.security == TARGET_SECURITIES[BOND] and s.field.startswith(TARGET_FIELD)
    )
    print(
        f"Merged: {len(merged):,} rows, {day(merged[DATE].min())} to {day(merged[DATE].max())}, "
        f"{len(columns)} columns, content hash (`ingest.upload.frame_sha256`) "
        f"`{frame_sha256(merged)}`. Empty columns dropped: {dropped or 'none'}. Bond target: "
        f"`{bond}`.\n"
    )
    return merged.set_index(DATE), bond


def splitter() -> PurgedWalkForward:
    return PurgedWalkForward(
        train=TRAIN_WINDOW, test=TEST_WINDOW, embargo=EMBARGO_DAYS, tuning_rows=TUNING_ROWS
    )


# ── The holiday calendar ─────────────────────────────────────────────────────


def calendars(dates: pd.DatetimeIndex) -> None:
    lo, hi = dates.min(), dates.max()
    candidates = {
        name: pd.DatetimeIndex(mcal.get_calendar(name).valid_days(lo, hi).tz_localize(None))
        for name in ("SIFMA_US", "NYSE")
    }
    federal = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    candidates["US federal holidays (pandas)"] = pd.date_range(lo, hi, freq=federal)
    candidates["every weekday"] = pd.bdate_range(lo, hi)
    rows = []
    for name, sessions in candidates.items():
        extra, missing = dates.difference(sessions), sessions.difference(dates)
        total = len(extra) + len(missing)
        rows.append([name, f"{len(extra):,}", f"{len(missing):,}", f"{total:,}"])
    print(
        "## The calendar for the coming trading days\n\n"
        f"`HOLIDAY_CALENDAR` = `{HOLIDAY_CALENDAR}`."
    )
    print(
        f"The bond index's {len(dates):,} dates, {day(lo)} to {day(hi)}, against each "
        "calendar's trading days over the same span.\n"
    )
    table(["calendar", "index dates it calls closed", "its days the index lacks", "total"], rows)
    sifma = candidates["SIFMA_US"]
    extra = dates.difference(sifma)
    recent = int((extra >= "2023-11-01").sum())
    print(
        f"Of the {len(extra)} index dates SIFMA_US calls closed, {recent} "
        "fall from Nov 2023 on, when the exports began to carry a row on every bond-market "
        "holiday with the price unchanged: "
        + ", ".join(f"{d:%d %b %Y}" for d in extra[extra >= "2023-11-01"][:6])
        + ", … The rest: "
        + ", ".join(f"{d:%d %b %Y}" for d in extra[extra < "2023-11-01"])
        + ".\n"
    )
    by_month = pd.Series(dates, index=dates).groupby([dates.year, dates.month]).max()
    sifma_month = pd.Series(sifma, index=sifma).groupby([sifma.year, sifma.month]).max()
    both = by_month.index.intersection(sifma_month.index)
    differ = [
        f"{by_month[k]:%d %b %Y} (SIFMA: {sifma_month[k]:%d %b})"
        for k in both
        if by_month[k] != sifma_month[k] and k != (hi.year, hi.month)
    ]
    print(
        f"Months whose last row in the data is not SIFMA's last trading day ({len(differ)} of "
        f"{len(both) - 1} complete months): {', '.join(differ) or 'none'}. The model reads the "
        "data's own dates wherever they run, as the research did, so there the month-end flag "
        "sits on the data's last row.\n"
    )


# ── The columns against the research's ───────────────────────────────────────


def research_calendar(index: pd.DatetimeIndex, horizon: int) -> pd.DataFrame:
    """``s14-bond-short/grid14.py``'s ``calendar``, as it ran: the data's own dates only,
    so its last ``h`` rows have no value, and its last row counts as a month-end."""
    idx = pd.DatetimeIndex(index)
    month = idx.year * 12 + idx.month
    nxt = np.r_[month[1:], -1]
    prev = np.r_[-1, month[:-1]]
    day_flags = {
        "CAL_MONTH_END": (month != nxt).astype(float),
        "CAL_MONTH_START": (month != prev).astype(float),
        "CAL_JOBS_FRIDAY": ((idx.dayofweek == 4) & (idx.day <= 7)).astype(float),
    }
    if horizon == 1:
        for d, name in enumerate(("MON", "TUE", "WED", "THU")):
            day_flags[f"CAL_{name}"] = (idx.dayofweek == d).astype(float)
    return pd.DataFrame(day_flags, index=idx).rolling(horizon).sum().shift(-horizon)


def compare_columns(indexed: pd.DataFrame, bond: str) -> None:
    index = align_and_lag(indexed, [], bond, horizon=1).frame.index
    print("## The columns against the research's\n")
    rows = []
    for h in HORIZONS:
        ours, theirs = label_window_calendar(index, h), research_calendar(index, h)
        assert list(ours.columns) == list(theirs.columns)
        defined = theirs.notna().all(axis=1)
        unequal = (ours[defined] != theirs[defined]).any(axis=1)
        # The research's last row is a month-end whatever its date; the last label that
        # reads it is the row h before it.
        tail = ours.index[~defined]
        named = ", ".join(f"{d:%d %b %Y}" for d in ours.index[defined][unequal]) or "—"
        rows.append(
            [
                h,
                ", ".join(ours.columns),
                f"{int(defined.sum()):,}",
                f"{int(unequal.sum())}: {named}",
                f"{len(tail)} ({day(tail[0])} – {day(tail[-1])}), research none",
            ]
        )
    table(
        [
            "h",
            "columns",
            "rows where both have a value",
            "rows that differ",
            "rows only the app fills",
        ],
        rows,
    )
    tail = label_window_calendar(index, 5).tail(6)
    print(
        f"The last six rows at h = 5, their windows filled past {day(index[-1])} from "
        f"{HOLIDAY_CALENDAR}:\n"
    )
    table(
        ["date", *tail.columns],
        [[f"{d:%a %d %b %Y}", *(f"{v:g}" for v in row)] for d, row in tail.iterrows()],
    )


# ── The model ────────────────────────────────────────────────────────────────


_DATA: dict = {}


def _init(indexed: pd.DataFrame, bond: str) -> None:
    _DATA.update(indexed=indexed, bond=bond)


def _period(test: pd.DatetimeIndex, label_end: pd.Series) -> str | None:
    end = label_end.reindex(test).max()
    if pd.notna(end) and end < HOLDOUT_BEFORE:
        return "fresh"
    if test[0] >= NEW_FROM:
        return "new"
    if test[0] >= LATER_FROM:
        return "later"
    return None  # the windows straddling Aug 2016


def _by_period(folds, panel) -> dict[str, tuple[float, float, int]]:
    scores = {}
    for name in ("fresh", "later"):
        part = tuple(f for f in folds if _period(f.test, panel.label_end) == name)
        r, _ = summarize(part)
        scores[name] = (r.gated_rank_ic, r.signal_t, len(part))
    return scores


def research_fit(horizon: int):
    """The page's harness on the research's own columns, tail and all."""
    base = align_and_lag(_DATA["indexed"], [], _DATA["bond"], horizon=horizon)
    columns = research_calendar(base.frame.index, horizon)
    panel = replace(
        base, frame=pd.concat([columns, base.frame], axis=1), signals=tuple(columns.columns)
    )
    folds = evaluate(ShrinkagePolynomial, panel, splitter(), screen=True)
    result, _ = summarize(folds)
    return horizon, result, _by_period(folds, panel)


def fit(horizon: int):
    started = time.perf_counter()
    result, description, panel = run_calendar_research_polynomial(
        _DATA["indexed"], _DATA["bond"], horizon, splitter()
    )
    seconds = time.perf_counter() - started
    # The same folds again, scored by period. The fit is deterministic (SEED), so these
    # are the run's own folds, as the assert below checks.
    folds = evaluate(ShrinkagePolynomial, panel, splitter(), screen=True)
    scores = _by_period(folds, panel)
    kept: dict[str, list[float]] = {}
    for f in folds:
        for term, c in zip(f.description.terms, f.description.coefficients, strict=True):
            kept.setdefault(term, []).append(c)
    terms = {
        t: (len(c) / len(folds), sum(x > 0 for x in c) / len(c)) for t, c in kept.items()
    }
    same = summarize(folds)[0]
    assert same.signal_rank_ic == result.signal_rank_ic, "the rescoring is not the run's folds"
    tested = (folds[0].test[0], folds[-1].test[-1])
    return horizon, result, description, scores, terms, tested, seconds


def _cell(ic: float, t: float) -> str:
    return f"{ic:+.3f}, t {t:.2f}"


def report(fits: dict, theirs_here: dict) -> None:
    print(f"## The model, `VERSION` {VERSION}, at the page's defaults\n")
    print(
        f"Train {TRAIN_WINDOW} rows (rolling), test {TEST_WINDOW}, embargo {EMBARGO_DAYS}, the "
        f"first {TUNING_ROWS} rows a tuning period; columns screened per fold and fitted by "
        "`ShrinkagePolynomial`. Signal Rank IC and t as on the Models page; the gate is Signal "
        "Rank IC above 0.02 and t above 2. Fresh: test windows whose every label ends before "
        f"{day(HOLDOUT_BEFORE)} (the page's pre-2016 figure). Later: test windows starting "
        f"{day(LATER_FROM)} – 30 Sep 2026.\n"
    )
    rows = []
    for h in HORIZONS:
        _, result, _, scores, _, tested, seconds = fits[h]
        outcome = evaluate_candidate(result.gated_rank_ic, result.signal_t)
        gate = "passes" if outcome.promoted else "fails " + " and ".join(outcome.failed_gates)
        held = result.holdout
        assert held is not None and abs(held.signal_rank_ic - scores["fresh"][0]) < 1e-12
        rows.append(
            [
                h,
                _cell(result.gated_rank_ic, result.signal_t),
                gate,
                f"{_cell(*scores['fresh'][:2])} ({scores['fresh'][2]})",
                f"{_cell(*scores['later'][:2])} ({scores['later'][2]})",
                f"{result.constant_forecasts.constant} of {result.constant_forecasts.folds}",
                f"{result.rows_scored:,}",
                f"{tested[0]:%d %b %Y} – {tested[1]:%d %b %Y}",
                f"{seconds:.0f} s",
            ]
        )
    table(
        [
            "h",
            "whole period",
            "gate",
            "fresh (windows)",
            "later (windows)",
            "constant folds",
            "rows",
            "test dates",
            "runtime",
        ],
        rows,
    )
    print(
        "**Against the research** (`cal / shrink / roll504`, main at `1bc88e0`), and this "
        "checkout's harness run on the research's own columns (`research_calendar`):\n"
    )
    rows = []
    for h, (whole, fresh, later, constant) in RESEARCH.items():
        _, result, _, scores, _, _, _ = fits[h]
        here, here_scores = theirs_here[h]
        ours = [(result.gated_rank_ic, result.signal_t), scores["fresh"][:2], scores["later"][:2]]
        again = [
            (here.gated_rank_ic, here.signal_t),
            here_scores["fresh"][:2],
            here_scores["later"][:2],
        ]
        for name, theirs, mine, rerun in zip(
            ("whole", "fresh", "later"), (whole, fresh, later), ours, again, strict=True
        ):
            rows.append(
                [
                    h,
                    name,
                    f"{theirs[0]:+.4f}, t {theirs[1]:.4f}",
                    f"{rerun[0]:+.4f}, t {rerun[1]:.4f}",
                    f"{mine[0]:+.4f}, t {mine[1]:.4f}",
                    f"{mine[0] - theirs[0]:+.4f}, {mine[1] - theirs[1]:+.4f}",
                ]
            )
        rows.append(
            [
                h,
                "constant folds",
                constant,
                here.constant_forecasts.constant,
                result.constant_forecasts.constant,
                "",
            ]
        )
        moved = (result.forecast - here.forecast).abs() > 0
        named = ", ".join(f"{d:%d %b %Y}" for d in moved.index[moved]) or "none"
        rows.append([h, "forecasts that differ", "", "", f"{int(moved.sum())}: {named}", ""])
    table(
        [
            "h",
            "period",
            "research (its log)",
            "research's columns, this checkout",
            "the app",
            "app − research",
        ],
        rows,
    )
    print("**Terms**: the share of folds keeping each column, and of those the share positive.\n")
    rows = []
    for h in HORIZONS:
        _, _, description, _, terms, _, _ = fits[h]
        listed = sorted(terms.items(), key=lambda kv: -kv[1][0])
        pairs = zip(description.terms, description.coefficients, strict=True)
        latest = ", ".join(f"{t} {c:+.6f}" for t, c in pairs)
        rows.append(
            [
                h,
                "; ".join(f"{t} {k:.0%} ({p:.0%} +)" for t, (k, p) in listed) or "none",
                f"{latest or 'no term'}; intercept {description.intercept:+.6f}",
            ]
        )
    table(["h", "kept", "latest fold"], rows)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    indexed, bond = read_folder(args.folder)
    dates = pd.DatetimeIndex(indexed[bond].dropna().index)
    calendars(dates)
    compare_columns(indexed, bond)
    started = time.perf_counter()
    with ProcessPoolExecutor(
        len(HORIZONS) + len(RESEARCH), initializer=_init, initargs=(indexed, bond)
    ) as pool:
        theirs = pool.map(research_fit, RESEARCH)
        fits = {row[0]: row for row in pool.map(fit, HORIZONS)}
        theirs_here = {h: (result, scores) for h, result, scores in theirs}
    print(f"All {len(fits) + len(theirs_here)} fits: {time.perf_counter() - started:.0f} s.\n")
    report(fits, theirs_here)


if __name__ == "__main__":
    main()
