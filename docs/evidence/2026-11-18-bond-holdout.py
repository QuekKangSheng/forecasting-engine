"""The bond derived polynomial on a hold-out nobody had scored, 18 Nov 2026
(docs/evidence/2026-11-18-bond-holdout.md).

Two commands.

  run <folder of exports> [--horizons 1,5,10,20] [--split DATE] [--workers N]
      Runs the app's own bond derived polynomial (models/hybrid: the mean of 14 fits) on
      the Bloomberg exports in the folder, read and merged as the Data page merges them,
      at the Models page's defaults: train window 504, test window 20, embargo 20, the
      first 504 dates a tuning period, a term cap of 10. Prints, per horizon, the
      whole-period Signal Rank IC and t, and the same over the test windows whose every
      label ends before 1 Aug 2016 and over the windows starting from 31 Aug 2016 to
      30 Sep 2026: the two decades of the research's rule. With --split DATE the two
      parts are instead the windows whose labels end before DATE and the windows
      starting on or after it (the public hold-out's halves were 1999-01-01).

  build-public <out folder> [--dither N] [--refresh]
      Builds the public hold-out for 2 Jan 1990 - 30 Aug 2006: downloads public series
      (Yahoo Finance, FRED) into <out>/raw, as downloaded, and writes them in the
      upload's file format, with the upload's tickers, into <out>/upload_format, so the
      app's own pipeline reads them unchanged. --dither N also writes N copies whose
      bond target is built from the fund's NAV plus an independent uniform draw within
      half a cent either side (<out>/upload_format_dither1 ...), the rounding re-test.
      Nothing here computes a predictive statistic. Do not commit the data.

Run: uv run --project <repo> python -I <this file> run <folder> [--horizons 1,5,10,20]
     uv run --project <repo> python -I <this file> build-public <out> --dither 5
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting_engine.extraction import bloomberg_csv, bloomberg_xlsx
from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.ingest.align import select_signals, transform_for
from forecasting_engine.models.hybrid import bond_derived_folds, bond_derived_panel
from forecasting_engine.parallel import WORKERS_ENV
from forecasting_engine.validation.harness import summarize
from forecasting_engine.validation.splitters import TUNING_ROWS, PurgedWalkForward

sys.stdout.reconfigure(encoding="utf-8")

DATE = bloomberg_csv.DATE_COLUMN
TARGET_FIELD = "TOT_RETURN_INDEX"
TARGET_SECURITIES = {TargetRole.EQUITY: "SPX Index", TargetRole.BOND: "LBUSTRUU Index"}
TRAIN_WINDOW, TEST_WINDOW, EMBARGO, MAX_TERMS = 504, 20, 20, 10
FRESH_BEFORE = pd.Timestamp("2016-08-01")  # models/sign_ruled.HOLDOUT_BEFORE
LATER_FROM, LATER_BEFORE = pd.Timestamp("2016-08-31"), pd.Timestamp("2026-10-01")


# --- run -------------------------------------------------------------------------------


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
        if own:
            targets[role] = (total or last or own)[0]
    if TargetRole.BOND not in targets:
        raise SystemExit(f"no {TARGET_SECURITIES[TargetRole.BOND]} export in {folder}")
    return {"merged": merged, "sources": sources, "targets": targets}


def _score(folds) -> dict:
    if not folds:
        return {"ic": None, "t": None, "windows": 0}
    result, _ = summarize(tuple(folds))
    number = lambda v: None if v != v else round(float(v), 4)
    return {
        "ic": number(result.gated_rank_ic),
        "t": number(result.signal_t),
        "windows": len(folds),
        "rows": result.rows_scored,
    }


def _parts(folds, label_end: pd.Series, split: pd.Timestamp | None) -> dict[str, list]:
    """The windows of each part: by default the research's two decades, labels ending
    before 1 Aug 2016 and windows starting 31 Aug 2016 - 30 Sep 2026; with ``split``,
    the windows whose labels end before it and those starting on or after it."""
    first_name, second_name = ("before", "after") if split else ("fresh", "later")
    first, second = [], []
    for fold in folds:
        end = label_end.reindex(fold.test).max()
        start = fold.test[0]
        before = split if split is not None else FRESH_BEFORE
        if pd.notna(end) and end < before:
            first.append(fold)
        elif split is not None and start >= split:
            second.append(fold)
        elif split is None and LATER_FROM <= start < LATER_BEFORE:
            second.append(fold)
    return {first_name: first, second_name: second}


def run(folder: Path, horizons: list[int], split: pd.Timestamp | None) -> None:
    bundle = read_folder(folder)
    merged, sources, targets = bundle["merged"], bundle["sources"], bundle["targets"]
    indexed = merged.set_index(DATE)
    numeric = [c for c in merged.columns if c != DATE and merged[c].dtype.kind in "fi"]
    signals = select_signals(numeric, list(targets.values()), sources)
    transforms = {c: transform_for(sources.get(c)) for c in signals}
    price = targets[TargetRole.BOND]
    splitter = PurgedWalkForward(
        train=TRAIN_WINDOW, test=TEST_WINDOW, embargo=EMBARGO, tuning_rows=TUNING_ROWS
    )
    print(
        f"{len(merged):,} rows, {merged[DATE].min():%d %b %Y} to {merged[DATE].max():%d %b %Y}; "
        f"bond target {price}; signals {', '.join(f'{s} ({transforms[s]})' for s in signals)}",
        flush=True,
    )
    for horizon in horizons:
        started = time.perf_counter()
        panel = bond_derived_panel(indexed, signals, transforms, price, horizon)
        folds = bond_derived_folds(panel, splitter, MAX_TERMS)
        if not folds:
            raise SystemExit(splitter.too_short(panel))
        row = {"horizon": horizon, "all": _score(folds)}
        row |= {name: _score(part) for name, part in _parts(folds, panel.label_end, split).items()}
        row["first_test"] = str(folds[0].test[0].date())
        row["last_test"] = str(folds[-1].test[-1].date())
        row["seconds"] = round(time.perf_counter() - started, 1)
        shown = " | ".join(
            f"{name} {part['ic']} (t {part['t']}, {part['windows']} windows)"
            for name, part in row.items()
            if isinstance(part, dict)
        )
        print(f"h = {horizon}: {shown}", flush=True)
        print(json.dumps(row), flush=True)


# --- build-public ------------------------------------------------------------------

START, END = pd.Timestamp("1990-01-02"), pd.Timestamp("2006-08-30")
FETCH_FROM, FETCH_TO = "1962-01-01", "2006-08-31"
AGENT = "Mozilla/5.0 (research script)"
YAHOO = {
    "VBMFX": "VBMFX",
    "VWEHX": "VWEHX",
    "SP500TR": "^SP500TR",
    "GSPC": "^GSPC",
    "DXY": "DX-Y.NYB",
    "VIX": "^VIX",
}
FRED = ("DGS10", "T10Y2Y", "DBAA")
DITHER_SEED = 20261118
HALF_CENT = 0.005


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def download(raw: Path, refresh: bool) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    p1 = int(pd.Timestamp(FETCH_FROM, tz="UTC").timestamp())
    p2 = int(pd.Timestamp(FETCH_TO, tz="UTC").timestamp())
    for name, symbol in YAHOO.items():
        path = raw / f"yahoo_{name}.json"
        if refresh or not path.exists():
            quoted = urllib.request.quote(symbol)
            path.write_bytes(
                _get(
                    f"https://query1.finance.yahoo.com/v8/finance/chart/{quoted}"
                    f"?period1={p1}&period2={p2}&interval=1d&events=div%7CcapitalGains"
                )
            )
    for series in FRED:
        path = raw / f"fred_{series}.csv"
        if refresh or not path.exists():
            path.write_bytes(
                _get(
                    f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
                    f"&cosd={FETCH_FROM}&coed={FETCH_TO}"
                )
            )


def yahoo(raw: Path, name: str) -> tuple[pd.Series, pd.Series]:
    """(close, distributions), each by date. Yahoo files every distribution of these
    funds under dividends; ``split_distributions`` tells income from capital gains."""
    result = json.loads((raw / f"yahoo_{name}.json").read_text())["chart"]["result"][0]
    dates = pd.to_datetime(result["timestamp"], unit="s").normalize()
    close = pd.Series(result["indicators"]["quote"][0]["close"], index=dates, dtype=float).dropna()
    close = close[~close.index.duplicated(keep="last")].sort_index()
    rows = result.get("events", {}).get("dividends", {})
    events = pd.Series(
        {
            pd.to_datetime(v["date"], unit="s").normalize(): float(v["amount"])
            for v in rows.values()
        },
        dtype=float,
    ).sort_index()
    return close, events


def split_distributions(nav: pd.Series, events: pd.Series) -> tuple[pd.Series, pd.Series]:
    """(income, capital gains) from a fund's distributions, by two rules about the
    distributions themselves (amount and date), not about returns:

    - an amount over twice the median of the 13 distributions around it is that median
      of income plus a capital gain (the rest);
    - in a month with two distributions, one that is under half that median and not on
      the month's last trading day is a capital gain. (A small one on the last trading
      day is the year-end income top-up of the years the fund paid mid-month.)

    Income is declared daily and never sits in the NAV; a capital gain leaves the NAV on
    its ex-date, so the two must be added back differently."""
    median = events.rolling(13, center=True, min_periods=5).median()

    def month_of(d):
        return (d.year, d.month)

    last_day = nav.groupby([nav.index.year, nav.index.month]).apply(lambda s: s.index.max())
    per_month = events.groupby([events.index.year, events.index.month]).size()
    income, gains = {}, {}
    for d, amount in events.items():
        if amount > 2 * median[d]:
            income[d], gains[d] = float(median[d]), float(amount - median[d])
        elif (
            per_month[month_of(d)] > 1
            and amount < 0.5 * median[d]
            and d != last_day.get(month_of(d))
        ):
            gains[d] = float(amount)
        else:
            income[d] = float(amount)
    return pd.Series(income, dtype=float), pd.Series(gains, dtype=float)


def fred(raw: Path, series: str) -> pd.Series:
    frame = pd.read_csv(io.BytesIO((raw / f"fred_{series}.csv").read_bytes()))
    frame.columns = ["date", "value"]
    values = pd.to_numeric(frame["value"], errors="coerce")
    return pd.Series(values.to_numpy(), index=pd.to_datetime(frame["date"])).dropna()


def falls_on_ex_date(nav: pd.Series, income: pd.Series) -> tuple[float, int]:
    """The construction check: mean NAV change on income ex-dates, as a share of the mean
    distribution. About -1: the NAV falls by it (income declared at the ex-date). About
    0: it does not (income declared daily, the NAV never held it)."""
    change = nav.diff()
    on = [d for d in income.index if d in change.index and change[d] == change[d]]
    return float(change[on].mean() / income[on].mean()), len(on)


def total_return(nav: pd.Series, income: pd.Series, gains: pd.Series) -> tuple[pd.Series, str]:
    """A total-return index from a fund's NAV and distributions."""
    ratio, _ = falls_on_ex_date(nav, income)
    added = pd.Series(0.0, index=nav.index)
    if ratio < -0.5:
        how = "income added on its ex-date"
        for d, amount in income.items():
            if d in added.index:
                added[d] += amount
    else:
        how = "income spread equally over the trading days since the previous distribution"
        previous = nav.index[0] - pd.Timedelta(days=1)
        for d, amount in income.items():
            days = nav.index[(nav.index > previous) & (nav.index <= d)]
            if len(days):
                added[days] += amount / len(days)
            previous = d
    for d, amount in gains.items():
        if d in added.index:
            added[d] += amount
    growth = (nav + added) / nav.shift(1)
    growth.iloc[0] = 1.0
    return 100.0 * growth.cumprod(), how


def write(
    folder: Path, name: str, security: str, columns: dict[str, pd.Series], currency: bool = False
) -> None:
    """One file in the upload's format: its metadata lines, then dates newest first."""
    frame = pd.DataFrame(columns).dropna(how="all")
    frame = frame[(frame.index >= START) & (frame.index <= END)].sort_index(ascending=False)
    lines = [
        f"Security,{security}",
        f"Start Date,{START.month}/{START.day}/{START.year}",
        f"End Date,{END.month}/{END.day}/{END.year}",
        "Period,D",
    ]
    if currency:
        lines.append("Currency,USD")
    lines += ["", "Date," + ",".join(frame.columns)]
    for d, row in frame.iterrows():
        cells = ["" if v != v else f"{v:.9g}" for v in row]
        lines.append(f"{d.month}/{d.day}/{d.year}," + ",".join(cells))
    (folder / f"{name}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"  {name}.csv  {len(frame):,} rows  "
        f"{frame.index.min():%d %b %Y} to {frame.index.max():%d %b %Y}"
    )


def build_public(out: Path, dither: int, refresh: bool) -> None:
    raw, upload = out / "raw", out / "upload_format"
    download(raw, refresh)
    upload.mkdir(parents=True, exist_ok=True)

    print("Construction check: mean NAV change on income ex-dates / mean distribution")
    funds = {}
    for name in ("VBMFX", "VWEHX"):
        nav, events = yahoo(raw, name)
        income, gains = split_distributions(nav, events)
        ratio, n = falls_on_ex_date(nav, income)
        index, how = total_return(nav, income, gains)
        funds[name] = (nav, index)
        print(f"  {name}: ratio {ratio:+.2f} over {n} income distributions -> {how}")
        inside = gains[(gains.index >= START) & (gains.index <= END)]
        added = {str(d.date()): round(v, 3) for d, v in inside.items()}
        print(f"         capital gains added on their ex-date, {START.year}-{END.year}: {added}")
        index.to_csv(raw / f"built_{name}_total_return.csv", header=["total_return_index"])
        nav.to_csv(raw / f"built_{name}_nav.csv", header=["nav"])

    spx_tr, _ = yahoo(raw, "SP500TR")
    spx, _ = yahoo(raw, "GSPC")
    dxy, _ = yahoo(raw, "DXY")
    vix, _ = yahoo(raw, "VIX")
    dgs10, t10y2y, dbaa = fred(raw, "DGS10"), fred(raw, "T10Y2Y"), fred(raw, "DBAA")
    baa_spread = (dbaa - dgs10).dropna()

    print("Upload-format files:")
    bond = funds["VBMFX"][1]
    write(
        upload,
        "LBUSTRUU_Index_public_VBMFX",
        "LBUSTRUU Index",
        {"TOT_RETURN_INDEX_GROSS_DVDS": bond, "PX_LAST": bond},
        currency=True,
    )
    write(
        upload,
        "SPX_Index_public",
        "SPX Index",
        {"TOT_RETURN_INDEX_GROSS_DVDS": spx_tr, "PX_LAST": spx},
        currency=True,
    )
    write(
        upload,
        "LF98TRUU_Index_public_VWEHX",
        "LF98TRUU Index",
        {"TOT_RETURN_INDEX_GROSS_DVDS": funds["VWEHX"][1]},
    )
    write(upload, "DXY_Curncy_public", "DXY Curncy", {"PX_LAST": dxy})
    write(upload, "VIX_Index_public", "VIX Index", {"PX_LAST": vix}, currency=True)
    write(upload, "USGG10YR_Index_public_DGS10", "USGG10YR Index", {"PX_LAST": dgs10})
    write(upload, "USYC2Y10_Index_public_T10Y2Y", "USYC2Y10 Index", {"PX_LAST": 100.0 * t10y2y})
    write(upload, "LUACOAS_Index_public_BAA_less_10y", "LUACOAS Index", {"PX_LAST": baa_spread})
    assert not np.isnan(bond[(bond.index >= START) & (bond.index <= END)]).any()
    dithered(out, raw, upload, dither)


def dithered(out: Path, raw: Path, upload: Path, copies: int) -> None:
    """Copies of the hold-out whose target is built from the NAV plus an independent
    uniform draw within half a cent either side (copy k from seed DITHER_SEED + k), so
    days the rounded NAV calls equal are no longer tied. Every other file is the one in
    upload_format."""
    nav, events = yahoo(raw, "VBMFX")
    income, gains = split_distributions(nav, events)
    for k in range(1, copies + 1):
        folder = out / f"upload_format_dither{k}"
        folder.mkdir(exist_ok=True)
        for path in upload.glob("*.csv"):
            if not path.name.startswith("LBUSTRUU"):
                shutil.copy(path, folder / path.name)
        draw = np.random.default_rng(DITHER_SEED + k).uniform(-HALF_CENT, HALF_CENT, len(nav))
        index, how = total_return(nav + draw, income, gains)
        assert how.startswith("income spread"), how
        print(f"Copy {k} (seed {DITHER_SEED + k}):")
        write(
            folder,
            "LBUSTRUU_Index_public_VBMFX_dither",
            "LBUSTRUU Index",
            {"TOT_RETURN_INDEX_GROSS_DVDS": index, "PX_LAST": index},
            currency=True,
        )


# --- the command line -------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    scored = commands.add_parser("run", help="run the app's bond derived polynomial")
    scored.add_argument("folder", type=Path)
    scored.add_argument("--horizons", default="1,5,10,20")
    scored.add_argument("--split", type=pd.Timestamp, default=None)
    scored.add_argument("--workers", type=int, default=None, help="processes a run may use")
    public = commands.add_parser("build-public", help="build the public hold-out's files")
    public.add_argument("out", type=Path)
    public.add_argument("--dither", type=int, default=0)
    public.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    if args.command == "run":
        if args.workers is not None:
            os.environ[WORKERS_ENV] = str(args.workers)
        run(args.folder, [int(h) for h in args.horizons.split(",")], args.split)
    else:
        build_public(args.out, args.dither, args.refresh)


if __name__ == "__main__":
    main()
