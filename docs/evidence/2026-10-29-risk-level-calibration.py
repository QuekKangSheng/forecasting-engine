"""Calibration of the new starting-weight risk levels.

Fits equity (derived polynomial) and bond (ML) once, at the Models page's
current defaults, on the real committed dataset cached in data/last_commit.pkl.
For each of the 5 RISK_LEVELS, backtests the optimiser's
weight schedule at the fixed TILT_STRENGTH and records net annual return, net
Sharpe, max drawdown, and the share of rebalances a bound actually capped.

Run from the repo root: uv run python docs/evidence/2026-10-29-risk-level-calibration.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.ingest.align import align_and_lag, select_signals, transform_for
from forecasting_engine.models.boosted import run_boosted
from forecasting_engine.models.polynomial import (
    CANDIDATE_CONFIGS,
    DerivedPolynomial,
    run_derived_polynomial,
)
from forecasting_engine.portfolio import backtest, optimize
from forecasting_engine.validation.splitters import TUNING_ROWS, PurgedWalkForward

ROOT = Path(__file__).resolve().parents[2]
EQUITY, BOND = TargetRole.EQUITY, TargetRole.BOND

# Current Models/Portfolio page defaults (app/model_settings.py), hardcoded here so
# this evidence script has no import-path dependency on the app/ package.
HORIZON = 20
TRAIN_WINDOW = 504
TEST_WINDOW = 20
EMBARGO_DAYS = 20
MAX_TERMS = 10
BOUNDS = (0.2, 0.8)


def load_committed() -> tuple[pd.DataFrame, dict[TargetRole, str], dict]:
    bundle = pickle.loads((ROOT / "data" / "last_commit.pkl").read_bytes())
    return (
        bundle["extraction_committed"],
        bundle["extraction_committed_targets"],
        bundle["extraction_committed_sources"],
    )


def fit_signal(
    role: TargetRole, indexed: pd.DataFrame, price_col: str, targets: dict, sources: dict
) -> pd.Series:
    numeric_cols = [c for c in indexed.columns]
    signal_cols = select_signals(numeric_cols, list(targets.values()), sources)
    transforms = {c: transform_for(sources.get(c)) for c in signal_cols}
    panel = align_and_lag(indexed, signal_cols, price_col, horizon=HORIZON, transforms=transforms)
    splitter = PurgedWalkForward(
        train=TRAIN_WINDOW, test=TEST_WINDOW, embargo=EMBARGO_DAYS, tuning_rows=TUNING_ROWS
    )
    if role == EQUITY:
        candidates = tuple(
            DerivedPolynomial(degree=c.degree, regularizer=c.regularizer, max_terms=MAX_TERMS)
            for c in CANDIDATE_CONFIGS
        )
        result, _ = run_derived_polynomial(panel, splitter, candidates=candidates)
    else:
        result, _, _ = run_boosted(panel, splitter)
    signal = result.signal
    if signal is None:
        raise RuntimeError(f"{role}: no forecast/baseline on the fitted result.")
    return signal


def main() -> None:
    merged, targets, sources = load_committed()
    indexed = merged.set_index("Date")
    prices = {role: indexed[targets[role]].dropna() for role in (EQUITY, BOND)}

    signals = {
        EQUITY: fit_signal(EQUITY, indexed, targets[EQUITY], targets, sources),
        BOND: fit_signal(BOND, indexed, targets[BOND], targets, sources),
    }

    rows = []
    for level in optimize.RISK_LEVELS:
        bounds = optimize.bounds_for_level(level, BOUNDS)
        start = optimize.starting_weight_for(level, BOUNDS)
        breakdowns = optimize.schedule_breakdowns(
            signals,
            prices,
            horizon=HORIZON,
            train_window=TRAIN_WINDOW,
            test_window=TEST_WINDOW,
            embargo=EMBARGO_DAYS,
            starting_weight=start,
            tilt_strength=optimize.TILT_STRENGTH,
            bounds=bounds,
        )
        bounds_text = f"{bounds[0]:.0%}-{bounds[1]:.0%}"
        if not breakdowns:
            rows.append([level, f"{start:.0%}", bounds_text, "no rebalances", "", "", ""])
            continue
        schedule = optimize.schedule_weights(breakdowns)
        bt = backtest.run_backtest(prices, schedule)
        net = bt.metrics[("optimised", "net")]
        at_bound = sum(b.capped for b in breakdowns.values()) / len(breakdowns)
        rows.append(
            [
                level,
                f"{start:.0%}",
                bounds_text,
                f"{net.annual_return:.2%}",
                f"{net.sharpe:.2f}",
                f"{net.max_drawdown:.2%}",
                f"{at_bound:.0%}",
            ]
        )

    header = [
        "level",
        "starting weight",
        "bounds",
        "net annual return",
        "net Sharpe",
        "max drawdown",
        "% rebalances at a bound",
    ]
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    for row in [header, *rows]:
        print(" | ".join(str(v).ljust(w) for v, w in zip(row, widths, strict=True)))


if __name__ == "__main__":
    main()
