"""The optimiser after its two fixes: no no-trade band, and a long-run premium anchored
to a published prior. Rerun on the Bloomberg history from 31 Aug 2006 at h = 1, 5
and 20.

The active model is the derived polynomial for both equity and bond, decided before
any result was seen. Neither research model is used: both scored about zero on their
pre-2016 hold-out (docs/evidence/2026-11-15-history-2006.md).

For each horizon it reports, per risk level, net annual return, net Sharpe, max
drawdown, turnover and costs a year, and t against 50/50 and against the same level
with no forecasts; per rebalance, β̂, Δs, π̄ and k; whether a riskier level ever holds
less equity than a safer one; and π̄ itself. The old no-trade band is rerun beside it as
a sensitivity, from this script's own copy of the removed ``_banded``, and the per-level
table again with the prior at 3.6% and 9.6% a year, for disclosure only.

The files are read and merged as the Data page does, by the 15 Nov script's
``read_folder``, which prints each file's SHA-256.

Run: uv run --project <repo> python -I <this file> <folder holding the 11 .xlsx files>
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.ingest.align import align_and_lag
from forecasting_engine.models.polynomial import (
    CANDIDATE_CONFIGS,
    DerivedPolynomial,
    run_derived_polynomial,
)
from forecasting_engine.portfolio import backtest, optimize
from forecasting_engine.portfolio.optimize import WeightBreakdown
from forecasting_engine.portfolio.performance import TRADING_DAYS_PER_YEAR, relative
from forecasting_engine.validation.splitters import TUNING_ROWS, PurgedWalkForward

EQUITY, BOND = TargetRole.EQUITY, TargetRole.BOND
HORIZONS = (1, 5, 20)
TRAIN_WINDOW = 504
TEST_WINDOW = 20
EMBARGO_DAYS = 20
MAX_TERMS = 10
BOUNDS = (0.2, 0.8)
PRIORS = (0.036, 0.096)
PREMIUM_DATES = {
    "end-2008": pd.Timestamp("2008-12-31"),
    "9 Mar 2009 (the S&P 500 low)": pd.Timestamp("2009-03-09"),
    "end-Mar 2009": pd.Timestamp("2009-03-31"),
    "end-May 2020": pd.Timestamp("2020-05-29"),
}

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("history_2006", _HERE / "2026-11-15-history-2006.py")
history = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(history)
table, day, page_inputs = history.table, history.day, history.page_inputs


# ── The active models ────────────────────────────────────────────────────────


def derived(bundle: dict, role: TargetRole, horizon: int):
    """The derived polynomial at ``horizon``, through the Models page's runner, at its
    defaults: its signal and fold levels, as Set Active would save them."""
    _, indexed, targets, _, signals, transforms = page_inputs(bundle)
    panel = align_and_lag(indexed, signals, targets[role], horizon=horizon, transforms=transforms)
    split = PurgedWalkForward(
        train=TRAIN_WINDOW, test=TEST_WINDOW, embargo=EMBARGO_DAYS, tuning_rows=TUNING_ROWS
    )
    candidates = tuple(
        DerivedPolynomial(degree=c.degree, regularizer=c.regularizer, max_terms=MAX_TERMS)
        for c in CANDIDATE_CONFIGS
    )
    result = run_derived_polynomial(panel, split, candidates=candidates)[0]
    return result.signal.dropna(), result.baseline.dropna(), result


# ── The removed no-trade band, kept here for the sensitivity ─────────────────


def _banded(target: float, drifted: float, band: float, bounds) -> float:
    """The removed ``optimize._banded``'s equity weight: trade to the nearest bound if
    the portfolio has drifted outside ``bounds``; otherwise hold at ``drifted`` if the
    target is within ``band`` of it, or move to the band's near edge."""
    lower, upper = bounds
    if not lower <= drifted <= upper:
        return min(max(drifted, lower), upper)
    gap = target - drifted
    if abs(gap) <= band:
        return drifted
    return min(max(target - math.copysign(band, gap), lower), upper)


def banded_schedule(inputs, breakdowns, prices, level) -> pd.DataFrame:
    """The removed ``breakdowns_from`` loop: each target banded against the weight the
    last allocation drifted to, the band ``c·(w*_L − w_mv)/π̄``."""
    cost = sum(backtest.DEFAULT_COSTS_BPS[role] for role in backtest.ASSETS) / 1e4
    daily = backtest.asset_returns(prices, pd.DataFrame(index=pd.DatetimeIndex(list(breakdowns))))
    returns = daily[list(backtest.ASSETS)].to_numpy(dtype=float)
    held, weights = None, {}
    for i, date in enumerate(daily.index):
        if i > 0 and held is not None:
            held = held * (1 + returns[i]) / (1 + float(held @ returns[i]))
        if date not in breakdowns:
            continue
        target = breakdowns[date].equity
        if held is not None:
            rebalance = inputs.rebalances[date]
            scale = optimize._tilt_scale(
                level, breakdowns[date].level_weight, rebalance.covariance, rebalance.premium
            )
            target = _banded(target, float(held[0]), cost * scale, BOUNDS)
        weights[date] = target
        held = np.array([target, 1 - target])
    equity = pd.Series(weights)
    return pd.DataFrame({EQUITY: equity, BOND: 1 - equity})


# ── One horizon ──────────────────────────────────────────────────────────────


def solve(inputs, level, *, trust=None) -> dict[pd.Timestamp, WeightBreakdown]:
    return optimize.breakdowns_from(inputs, risk_level=level, bounds=BOUNDS, trust=trust)


def with_prior(inputs, prices, horizon, annual):
    """``inputs`` with every rebalance's π̄ worked out again from a different prior."""
    saved = optimize.PREMIUM_PRIOR_ANNUAL
    optimize.PREMIUM_PRIOR_ANNUAL = annual
    try:
        premiums = optimize.long_run_premium(prices, horizon=horizon)
    finally:
        optimize.PREMIUM_PRIOR_ANNUAL = saved
    return replace(
        inputs,
        rebalances={
            d: replace(r, premium=float(premiums[d])) for d, r in inputs.rebalances.items()
        },
    )


def run_levels(inputs, prices, horizon, *, band=False):
    """Per level: the optimised backtest and the same level's with no forecasts, both
    under the band if ``band``."""
    runs = {}
    for level in optimize.RISK_LEVELS:
        pair = []
        for trust in (None, 0.0):
            breakdowns = solve(inputs, level, trust=trust)
            weights = (
                banded_schedule(inputs, breakdowns, prices, level)
                if band
                else optimize.schedule_weights(breakdowns)
            )
            pair.append((weights, backtest.run_backtest(prices, weights)))
        runs[level] = pair
    return runs


def years(result) -> float:
    return len(result.optimised.net) / TRADING_DAYS_PER_YEAR


def level_rows(runs, label: str) -> list[list[object]]:
    """Turnover and costs a year leave out the first allocation, bought from cash."""
    rows = []
    for level, ((_, result), (_, no_forecast)) in runs.items():
        net = result.metrics[("optimised", "net")]
        path = result.optimised
        rows.append(
            [
                label,
                level,
                f"{net.annual_return:.2%}",
                f"{net.sharpe:.2f}",
                f"{net.max_drawdown:.1%}",
                f"{path.turnover.iloc[1:].sum() / years(result):.0%}",
                f"{path.costs.iloc[1:].sum() / years(result):.3%}",
                f"{result.relative['net'].spread_t:.2f}",
                t_text(relative(path.net, no_forecast.optimised.net).spread_t),
            ]
        )
    return rows


LEVEL_HEADER = [
    "",
    "level",
    "net annual return",
    "net Sharpe",
    "max drawdown",
    "turnover a year",
    "costs a year",
    "t vs 50/50",
    "t vs no forecasts",
]


def t_text(value: float) -> str:
    return "—" if value != value else f"{value:.2f}"


def benchmark_row(result) -> list[object]:
    net = result.metrics[("baseline", "net")]
    path = result.baseline
    return [
        "50/50 benchmark",
        "—",
        f"{net.annual_return:.2%}",
        f"{net.sharpe:.2f}",
        f"{net.max_drawdown:.1%}",
        f"{path.turnover.iloc[1:].sum() / years(result):.0%}",
        f"{path.costs.iloc[1:].sum() / years(result):.3%}",
        "—",
        "—",
    ]


def inversions(runs) -> tuple[int, int]:
    """Rebalances where a riskier level holds less equity than the next safer one, and
    the rebalances checked."""
    weights = pd.DataFrame({level: pair[0][0][EQUITY] for level, pair in runs.items()})
    riskier_less = (weights.diff(axis=1).iloc[:, 1:] > 1e-12).any(axis=1)
    return int(riskier_less.sum()), len(weights)


def per_rebalance(inputs, horizon) -> pd.DataFrame:
    """β̂, Δs, π̄ and ``k = β̂·Δs/π̄``, so that ``w = w*_L + k·(w*_L − w_mv)`` before the
    bounds. ``k`` is blank where π̄ ≤ 0, which gives no tilt."""
    rows = {}
    for date, r in inputs.rebalances.items():
        signal_diff = float(r.signal[EQUITY] - r.signal[BOND])
        k = r.trust * signal_diff / r.premium if r.premium > 0 else float("nan")
        rows[date] = {
            "beta": r.trust,
            "gate_shut": r.gate_shut,
            "ds": signal_diff,
            "premium": r.premium,
            "k": k,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def premium_report(prices, inputs, horizon) -> None:
    premium = optimize.long_run_premium(prices, horizon=horizon)
    annual = premium * TRADING_DAYS_PER_YEAR / horizon
    at_rebalances = annual.reindex(list(inputs.rebalances))
    low = annual.idxmin()
    print(
        f"π̄ in annual terms: {annual.min():.2%} to {annual.max():.2%} over the calendar, "
        f"lowest on {day(low)}; {at_rebalances.min():.2%} to {at_rebalances.max():.2%} at "
        f"the rebalances. At or below zero on {(premium <= 0).sum()} days.\n"
    )
    rows = []
    for name, date in PREMIUM_DATES.items():
        on = annual.index[annual.index <= date][-1]
        rows.append([name, day(on), f"{annual[on]:.2%}", f"{premium[on]:.3%}"])
    table(["date", "trading day", "π̄ a year", f"π̄ over {horizon} days"], rows)


def horizon_section(bundle, prices, horizon) -> None:
    print(f"## h = {horizon}\n")
    started = time.perf_counter()
    signals, baselines, models = {}, {}, {}
    for role in (EQUITY, BOND):
        signals[role], baselines[role], models[role] = derived(bundle, role, horizon)
    fitted = time.perf_counter()
    for role, result in models.items():
        print(
            f"Derived polynomial, {role.value}: Signal Rank IC {result.gated_rank_ic:+.3f}, "
            f"t {result.signal_t:.2f}, {result.rows_scored:,} rows scored."
        )
    inputs = optimize.schedule_inputs(
        signals,
        prices,
        baselines=baselines,
        horizon=horizon,
        train_window=TRAIN_WINDOW,
        test_window=TEST_WINDOW,
        embargo=EMBARGO_DAYS,
    )
    dates = list(inputs.rebalances)
    print(f"\n{len(dates):,} rebalances, {day(dates[0])} to {day(dates[-1])}.\n")

    runs = run_levels(inputs, prices, horizon)
    banded = run_levels(inputs, prices, horizon, band=True)
    print("### Per level, after costs\n")
    rows = level_rows(runs, "no band") + level_rows(banded, "old band (sensitivity)")
    rows.append(benchmark_row(runs[3][0][1]))
    table(LEVEL_HEADER, rows)

    print("### Riskier levels holding less equity than safer ones\n")
    table(
        ["", "rebalances where a riskier level holds less equity", "rebalances"],
        [["no band", *inversions(runs)], ["old band (sensitivity)", *inversions(banded)]],
    )

    print("### π̄\n")
    premium_report(prices, inputs, horizon)

    print("### Per rebalance\n")
    frame = per_rebalance(inputs, horizon)
    open_ = frame[frame["beta"] > 0]
    shut = int(frame["gate_shut"].sum())
    print(
        f"Gate open (β̂ > 0) at {len(open_):,} of {len(frame):,} rebalances, shut at {shut:,}, "
        f"too little history at {len(frame) - len(open_) - shut:,}. "
        f"|k| above 1, where the tilt is larger than w*_L − w_mv itself: "
        f"{int((frame['k'].abs() > 1).sum()):,}.\n"
    )
    shown = frame if horizon == HORIZONS[-1] else open_
    if horizon != HORIZONS[-1]:
        print("Only the rebalances with the gate open, where β̂ is not zero.\n")
    table(
        ["rebalance", "β̂", "Δs", "π̄", "k"],
        [
            [
                day(date),
                f"{row.beta:.3f}",
                f"{row.ds:+.4%}",
                f"{row.premium:.4%}",
                "—" if row.k != row.k else f"{row.k:+.3f}",
            ]
            for date, row in shown.iterrows()
        ],
    )

    print("### Prior sensitivity, no band (disclosure only)\n")
    rows = []
    for annual in PRIORS:
        rows += level_rows(
            run_levels(with_prior(inputs, prices, horizon, annual), prices, horizon),
            f"prior {annual:.1%}",
        )
    table(LEVEL_HEADER, rows)
    done = time.perf_counter()
    print(
        f"Runtime here: models {fitted - started:.0f} s, optimiser and backtests "
        f"{done - fitted:.0f} s.\n"
    )


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--horizons", default=",".join(map(str, HORIZONS)))
    args = parser.parse_args()
    bundle = history.read_folder(args.folder)
    _, indexed, targets, *_ = page_inputs(bundle)
    prices = {role: indexed[targets[role]].dropna() for role in (EQUITY, BOND)}
    print(
        f"Prior: {optimize.PREMIUM_PRIOR_ANNUAL:.1%} a year, worth "
        f"{optimize.PREMIUM_PRIOR_YEARS} years. Costs: "
        f"{backtest.DEFAULT_COSTS_BPS[EQUITY]:g} bp equity, "
        f"{backtest.DEFAULT_COSTS_BPS[BOND]:g} bp bond. Bounds {BOUNDS[0]:.0%}–{BOUNDS[1]:.0%}.\n"
    )
    for horizon in map(int, args.horizons.split(",")):
        horizon_section(bundle, prices, horizon)


if __name__ == "__main__":
    main()
