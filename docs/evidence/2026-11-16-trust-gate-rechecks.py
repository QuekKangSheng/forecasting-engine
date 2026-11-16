"""How often the forecast-trust gate opens when the forecasts have no skill, given that
it is checked again at every rebalance.

At one rebalance the gate opens when the trust slope's t-statistic is above 2, which
with no skill should happen about 2.3% of the time (one-sided). It is checked again
at every rebalance on an expanding sample, so the chance that it opens at some
point over a backtest is higher. This estimates that chance on data like ours.

Design, fixed before running:

- Data: the Bloomberg history from 31 Aug 2006. Signals: the derived polynomial for
  both equity and bond, the active models of the optimiser evidence
  (2026-11-16-optimiser-fixes.md), at h = 1, 5 and 20.
- No skill: the signal difference ``Δs`` is shifted in a circle against its dates
  by a random number of pair dates, at least ``MIN_SHIFT`` from either end. The
  signal keeps its own persistence and the returns are the real ones, but the two
  no longer line up.
- ``DRAWS`` shifts per horizon from ``SEED``. At every rebalance the product's own
  gate (``optimize._trust_and_gate``) is run on the shifted pairs it would have
  known by then.
- Reported per horizon: the share of draws in which the gate opens at least once;
  the mean share of rebalances it is open; the share open at the last rebalance
  alone, which should be near the single-check rate; and, for reference, the same
  on the unshifted signals.

Shifts of the same series overlap, so the draws are not independent; the standard
errors given treat them as if they were and are only a guide.

Run: uv run --project <repo> python -I <this file> <folder holding the 11 .xlsx files>
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.portfolio import optimize

EQUITY, BOND = TargetRole.EQUITY, TargetRole.BOND
HORIZONS = (1, 5, 20)
DRAWS = 500
SEED = 20261116
MIN_SHIFT = 252
TRAIN_WINDOW = 504
TEST_WINDOW = 20
EMBARGO_DAYS = 20

_HERE = Path(__file__).resolve().parent


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gate_path(pairs: pd.DataFrame, dates: list[pd.Timestamp], horizon: int) -> np.ndarray:
    """Whether the gate is open at each of ``dates``."""
    return np.array(
        [optimize._trust_and_gate(d, pairs, horizon=horizon, gate=2.0)[0] > 0 for d in dates]
    )


def _shifted_path(args) -> np.ndarray:
    pairs, dates, horizon, shift = args
    shifted = pairs.assign(x=np.roll(pairs["x"].to_numpy(), shift))
    return gate_path(shifted, dates, horizon)


def horizon_section(history, fixes, bundle, prices, horizon: int, pool) -> None:
    print(f"## h = {horizon}\n")
    started = time.perf_counter()
    signals, baselines = {}, {}
    for role in (EQUITY, BOND):
        signals[role], baselines[role], _ = fixes.derived(bundle, role, horizon)
    fitted = time.perf_counter()
    rebalances = optimize.common_rebalance_dates(
        prices,
        horizon=horizon,
        train_window=TRAIN_WINDOW,
        test_window=TEST_WINDOW,
        embargo=EMBARGO_DAYS,
    )
    dates = list(optimize.expected_returns(rebalances, signals).dropna().index)
    pairs = optimize.trust_pairs(signals, baselines, prices, horizon=horizon)
    checked = [d for d in dates if len(optimize.pairs_known_by(d, pairs)) >= 252]

    rng = np.random.default_rng(SEED + horizon)
    shifts = rng.integers(MIN_SHIFT, len(pairs) - MIN_SHIFT, size=DRAWS)
    paths = np.array(
        list(pool.map(_shifted_path, [(pairs, dates, horizon, int(s)) for s in shifts]))
    )
    actual = gate_path(pairs, dates, horizon)
    simulated = time.perf_counter()

    ever = paths.any(axis=1)
    share_open = paths.mean(axis=1)
    last = paths[:, -1]

    def rate(hits: np.ndarray) -> str:
        p = hits.mean()
        return f"{p:.1%} (± {math.sqrt(p * (1 - p) / len(hits)):.1%})"

    print(
        f"{len(pairs):,} pairs from {history.day(pairs.index[0])}; {len(dates):,} rebalances, "
        f"{len(checked):,} of them with at least 252 pairs known, from "
        f"{history.day(checked[0])} to {history.day(dates[-1])}.\n"
    )
    history.table(
        ["", "opens at least once", "mean share of rebalances open", "open at the last rebalance"],
        [
            [
                f"no skill ({DRAWS} shifted draws)",
                rate(ever),
                f"{share_open.mean():.1%}",
                rate(last),
            ],
            [
                "actual signals (reference)",
                "yes" if actual.any() else "no",
                f"{actual.mean():.1%} ({int(actual.sum())} of {len(actual)})",
                "yes" if actual[-1] else "no",
            ],
        ],
    )
    first = [int(np.argmax(p)) for p in paths if p.any()]
    if first:
        print(
            f"In draws where it opens, it first opens at rebalance {int(np.median(first)) + 1} "
            f"of {len(dates)} (median), and is open at {share_open[ever].mean():.1%} of "
            "rebalances on average.\n"
        )
    print(
        f"Runtime, measured locally: models {fitted - started:.0f} s, simulation "
        f"{simulated - fitted:.0f} s.\n"
    )


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--horizons", default=",".join(map(str, HORIZONS)))
    args = parser.parse_args()
    history = _load("history_2006", "2026-11-15-history-2006.py")
    fixes = _load("optimiser_fixes", "2026-11-16-optimiser-fixes.py")
    bundle = history.read_folder(args.folder)
    _, indexed, targets, *_ = history.page_inputs(bundle)
    prices = {role: indexed[targets[role]].dropna() for role in (EQUITY, BOND)}
    print(
        f"No-skill draws: {DRAWS} per horizon, seed {SEED} + h, shifts of at least "
        f"{MIN_SHIFT} pair dates. Gate: t above 2 on the trust slope, at least 252 pairs.\n"
    )
    with ProcessPoolExecutor() as pool:
        for horizon in map(int, args.horizons.split(",")):
            horizon_section(history, fixes, bundle, prices, horizon, pool)


if __name__ == "__main__":
    main()
