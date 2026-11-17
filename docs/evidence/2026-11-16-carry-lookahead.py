"""The bond research model before and after its look-ahead purge (carry VERSION 1 → 2).

Phase 6 found that at h = 20 the model's fit read the first test-window row: a training
row's innovation is the move in each input's level over its label window, and the level
dated at the label's end sits on the next row, the first of the test window. VERSION 2
drops every training row whose move is read from a row past the latest training
label's end.

Both versions are fitted here on the data from 31 Aug 2006, at every horizon, through
the function the Models page's runner calls, at the page's defaults. VERSION 1 is
VERSION 2 with that purge switched off, which is the whole of the change.

Run: uv run --project <repo> python -I <this file> <folder holding the 11 .xlsx files>
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from forecasting_engine.models import carry_research
from forecasting_engine.models.sign_ruled import HOLDOUT_BEFORE

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("full_history", _HERE / "2026-11-16-full-history.py")
full = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(full)
history, table, day = full.history, full.table, full.day

HORIZONS = full.HORIZONS
VERSIONS = (1, 2)


_PURGE = carry_research.moves_within_labels


def _no_purge(panel, train):
    return pd.Series(True, index=train)


def _init(bundles: dict[str, dict]) -> None:
    full._init(bundles)


def fit(job: tuple[int, int]):
    version, horizon = job
    # A worker process runs several jobs, so the version is set on every one.
    carry_research.moves_within_labels = _no_purge if version == 1 else _PURGE
    started = time.perf_counter()
    _, result, _ = full.fit((full.NEW, full.BOND, "research", horizon))
    return job, result, time.perf_counter() - started


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    bundles = {full.NEW: history.read_folder(args.folder)}
    jobs = [(v, h) for v in VERSIONS for h in HORIZONS]
    started = time.perf_counter()
    with ProcessPoolExecutor(len(jobs), initializer=_init, initargs=(bundles,)) as pool:
        results = {job: (r, s) for job, r, s in pool.map(fit, jobs)}
    print(f"All {len(results)} fits: {time.perf_counter() - started:.0f} s wall-clock.\n")

    print(
        "## Bond research model, VERSION 1 against VERSION 2\n\n"
        "Signal Rank IC and t as on the Models page; the gate is Signal Rank IC above 0.02 "
        f"and t above 2. Hold-out: test windows whose labels end before {day(HOLDOUT_BEFORE)}."
        "\n"
    )
    rows = []
    for h in HORIZONS:
        for v in VERSIONS:
            r, seconds = results[(v, h)]
            held = r.holdout
            failed = full.evaluate_candidate(r.gated_rank_ic, r.signal_t).failed_gates
            rows.append(
                [
                    h,
                    v,
                    f"{r.gated_rank_ic:+.4f}",
                    f"{r.signal_t:.2f}",
                    f"{r.rows_scored:,}",
                    "passes" if not failed else "fails " + " and ".join(failed),
                    "—" if held is None else f"{held.signal_rank_ic:+.4f}",
                    "—" if held is None else f"{held.t:.2f}",
                    "—" if held is None else held.windows,
                    f"{seconds:.0f} s",
                ]
            )
    table(
        [
            "h",
            "version",
            "Signal Rank IC",
            "t",
            "rows",
            "gate",
            "hold-out Signal Rank IC",
            "hold-out t",
            "hold-out windows",
            "runtime",
        ],
        rows,
    )


if __name__ == "__main__":
    main()
