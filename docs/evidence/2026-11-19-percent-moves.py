"""Evidence for scoring prices on percentage moves in the extreme-move check.

Reads a folder of Bloomberg exports with the app's own readers and counts the flagged
moves two ways: every column on its change in points (the check before 19 Nov 2026),
and prices and total-return indices on their percentage change (the check now).

    uv run python docs/evidence/2026-11-19-percent-moves.py <folder of exports>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting_engine.extraction import bloomberg_csv, bloomberg_xlsx, validation
from forecasting_engine.ingest.align import price_columns

THRESHOLDS = (3, 4, 5, 6, 8, 10, 12, 15)
ERAS = ((2006, 2012), (2013, 2019), (2020, 2026))


def read(folder: Path) -> tuple[pd.DataFrame, frozenset[str]]:
    exports = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in (".csv", ".xlsx"):
            continue
        reader = bloomberg_xlsx.read_export if path.suffix.lower() == ".xlsx" else bloomberg_csv.read_export
        exports.append(validation.drop_exact_repeats(reader(path.name, path.read_bytes())))
    frame = bloomberg_csv.merge(exports)
    return frame, price_columns(bloomberg_csv.column_sources(exports, frame.columns))


def flagged(frame: pd.DataFrame, ratio: frozenset[str], threshold: float) -> pd.DataFrame:
    saved = validation.MAD_THRESHOLD
    validation.MAD_THRESHOLD = threshold
    try:
        return validation._big_moves(frame, ratio)
    finally:
        validation.MAD_THRESHOLD = saved


def eras(moves: pd.DataFrame) -> str:
    years = pd.to_datetime(moves["date"]).dt.year
    return " / ".join(str(int(((years >= a) & (years <= b)).sum())) for a, b in ERAS)


def main() -> None:
    frame, prices = read(Path(sys.argv[1]))
    dates = frame[validation.DATE_COLUMN]
    print(f"{len(frame):,} rows, {dates.min().date()} to {dates.max().date()}; "
          f"{len(frame.columns) - 1} columns, {len(prices)} of them prices: {sorted(prices)}\n")

    print("| threshold | all columns in points | prices on percentage moves |")
    print("|---|---|---|")
    for threshold in THRESHOLDS:
        before = len(flagged(frame, frozenset(), threshold))
        after = len(flagged(frame, prices, threshold))
        print(f"| {threshold} | {before:,} | {after:,} |")

    before, after = flagged(frame, frozenset(), 8.0), flagged(frame, prices, 8.0)
    print("\n| column | in points | 2006-12 / 2013-19 / 2020-26 | on percentage | 2006-12 / 2013-19 / 2020-26 |")
    print("|---|---|---|---|---|")
    for column in frame.columns:
        if column not in prices:
            continue
        b, a = before[before["column"] == column], after[after["column"] == column]
        print(f"| `{column}` | {len(b)} | {eras(b)} | {len(a)} | {eras(a)} |")

    column = next(c for c in sorted(prices) if c.startswith("SPX") and "TOT_RETURN" in c)
    values = pd.to_numeric(frame[column], errors="coerce")
    points, logs = values.diff(), np.log(values).diff()
    z_points = validation._robust_z(points.dropna()).abs()
    z_logs = validation._robust_z(logs.dropna()).abs()
    print(f"\n`{column}`: its ten largest percentage moves\n")
    print("| date | move | in points | score on points | score on percentage |")
    print("|---|---|---|---|---|")
    for idx in logs.abs().sort_values(ascending=False).head(10).index:
        print(f"| {dates[idx].date()} | {np.expm1(logs[idx]):+.2%} | {points[idx]:+,.1f} | "
              f"{z_points[idx]:.1f} | {z_logs[idx]:.1f} |")


if __name__ == "__main__":
    main()
