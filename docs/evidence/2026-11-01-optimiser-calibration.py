"""Calibration of the volatility-targeted risk levels and the earned forecast trust.

Reads what the app itself produced, so the figures are the Portfolio page's: the
dataset committed on the Data page (data/last_commit.pkl) and the active models set on
the Models page (data/forecasting.duckdb), both relative to the working directory. For
each of the 5 RISK_LEVELS, runs the optimiser and the backtest with the t-statistic
gate, without it, and with no forecasts, and reports the weights, bound shares, risk
aversion, performance, the forecasts' P&L and the gate's timeline.

Run from a folder holding that data/: uv run --project <repo> python <this file>
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.portfolio import backtest, optimize
from forecasting_engine.portfolio.performance import TRADING_DAYS_PER_YEAR, relative
from forecasting_engine.store import active_model

EQUITY, BOND = TargetRole.EQUITY, TargetRole.BOND

# Current Models/Portfolio page defaults (app/model_settings.py), hardcoded here so
# this evidence script has no import-path dependency on the app/ package.
HORIZON = 20
TRAIN_WINDOW = 504
TEST_WINDOW = 20
EMBARGO_DAYS = 20
BOUNDS = (0.2, 0.8)
SETTINGS = {
    "horizon": HORIZON,
    "train_window": TRAIN_WINDOW,
    "test_window": TEST_WINDOW,
    "embargo": EMBARGO_DAYS,
    "bounds": BOUNDS,
}
POST_2023 = pd.Timestamp("2023-01-01")
UNGATED = float("-inf")


def table(header: list[str], rows: list[list[object]]) -> None:
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in rows:
        print("| " + " | ".join(str(v) for v in row) + " |")
    print()


def t_text(value: float) -> str:
    return "—" if value != value else f"{value:.2f}"


def load():
    bundle = pickle.loads(Path("data/last_commit.pkl").read_bytes())
    indexed = bundle["extraction_committed"].set_index("Date")
    targets = bundle["extraction_committed_targets"]
    prices = {role: indexed[targets[role]].dropna() for role in (EQUITY, BOND)}
    signals = {role: active_model.get_active_model_signal(role) for role in (EQUITY, BOND)}
    baselines = {role: active_model.get_active_model_baseline(role) for role in (EQUITY, BOND)}
    print(f"Equity model: {active_model.get_active_model(EQUITY).model_name}")
    print(f"Bond model: {active_model.get_active_model(BOND).model_name}")
    print(f"Data: {indexed.index.min():%d %b %Y} to {indexed.index.max():%d %b %Y}\n")
    return prices, signals, baselines


def run(prices, signals, baselines, level, **kwargs):
    breakdowns = optimize.schedule_breakdowns(
        signals, prices, baselines=baselines, risk_level=level, **SETTINGS, **kwargs
    )
    return breakdowns, backtest.run_backtest(prices, optimize.schedule_weights(breakdowns))


def performance_row(name, result, portfolio="optimised") -> list[object]:
    """Turnover is a year's, leaving out the first allocation, which buys from cash."""
    net = result.metrics[(portfolio, "net")]
    path = getattr(result, portfolio)
    annual = float(path.turnover.iloc[1:].mean()) * (
        TRADING_DAYS_PER_YEAR / HORIZON if portfolio == "optimised" else 12
    )
    spread = result.relative["net"].spread_t if portfolio == "optimised" else float("nan")
    return [
        name,
        f"{net.annual_return:.2%}",
        f"{net.sharpe:.2f}",
        f"{net.max_drawdown:.1%}",
        f"{annual:.0%}",
        t_text(spread),
    ]


def weights_table(runs) -> None:
    rows = []
    for level, (breakdowns, _) in runs.items():
        values = list(breakdowns.values())
        target = np.array([b.level_weight for b in values])
        final = np.array([b.equity for b in values])
        at_bound = np.isclose(final, BOUNDS[0]) | np.isclose(final, BOUNDS[1])
        target_at = np.isclose(target, BOUNDS[0]) | np.isclose(target, BOUNDS[1])
        rows.append(
            [
                level,
                f"{target.mean():.1%}",
                f"{target.min():.1%}–{target.max():.1%}",
                f"{final.mean():.1%}",
                f"{final.min():.1%}–{final.max():.1%}",
                f"{target_at.mean():.0%}",
                f"{np.mean([b.capped for b in values]):.0%}",
                f"{at_bound.mean():.0%}",
                f"{sum(b.held for b in values[1:]) / (len(values) - 1):.0%}",
            ]
        )
    table(
        [
            "level",
            "avg w*_L",
            "range w*_L",
            "avg final",
            "range final",
            "w*_L at a bound",
            "tilted target capped",
            "final at a bound",
            "held by band",
        ],
        rows,
    )


def risk_aversion_table(prices, trusts) -> None:
    premiums = optimize.long_run_premium(prices, horizon=HORIZON)
    rows = []
    for level in optimize.RISK_LEVELS:
        lambdas, ks = [], []
        for date, trust in trusts.items():
            covariance = optimize.covariance_at_rebalance(
                date, prices, horizon=HORIZON, train_window=TRAIN_WINDOW, embargo=EMBARGO_DAYS
            )
            weight = optimize.level_weight(level, covariance, BOUNDS)
            scale = optimize._tilt_scale(level, weight, covariance, float(premiums.get(date, 0.0)))
            if scale > 0:
                lambdas.append(1 / (scale * optimize._var_diff(covariance)))
                if trust > 0:
                    ks.append(lambdas[-1] / trust)
        if not lambdas:
            rows.append([level, "infinite or none", "—", "—", "40"])
            continue
        rows.append(
            [
                level,
                f"{np.median(lambdas):.1f}",
                f"{min(lambdas):.1f}–{max(lambdas):.1f}",
                f"{np.median(ks):.1f}" if ks else "—",
                "40",
            ]
        )
    table(["level", "median λ_L", "range λ_L", "median k = λ_L/β̂, gate open", "old k"], rows)


def trust_status(date, pairs) -> tuple[str, float, float, int]:
    known = len(optimize.pairs_known_by(date, pairs))
    fit = optimize._trust_fit(date, pairs, horizon=HORIZON)
    if fit is None:
        return "too little history", float("nan"), float("nan"), known
    slope, t = float(fit.params[1]), float(fit.tvalues[1])
    return ("open" if optimize.forecast_trust(date, pairs, horizon=HORIZON) > 0 else "shut"), slope, t, known


def gate_timeline(dates, pairs) -> dict[pd.Timestamp, float]:
    """Each stretch of rebalances with the same gate state, and each rebalance's trust."""
    statuses = [(date, *trust_status(date, pairs)) for date in dates]
    rows, start = [], 0
    for i in range(1, len(statuses) + 1):
        if i == len(statuses) or statuses[i][1] != statuses[start][1]:
            stretch = statuses[start:i]
            ts = [s[3] for s in stretch if s[3] == s[3]]
            slopes = [s[2] for s in stretch if s[2] == s[2]]
            rows.append(
                [
                    stretch[0][1],
                    f"{stretch[0][0]:%d %b %Y}",
                    f"{stretch[-1][0]:%d %b %Y}",
                    len(stretch),
                    f"{min(slopes):.2f} to {max(slopes):.2f}" if slopes else "—",
                    f"{min(ts):.2f} to {max(ts):.2f}" if ts else "—",
                ]
            )
            start = i
    table(["gate", "from", "to", "rebalances", "raw slope", "t over h−1 lags"], rows)
    return {s[0]: (optimize.forecast_trust(s[0], pairs, horizon=HORIZON)) for s in statuses}


def ungated_trust(dates, pairs) -> dict[pd.Timestamp, float]:
    return {d: optimize.forecast_trust(d, pairs, horizon=HORIZON, gate=UNGATED) for d in dates}


def post_2023_fit(pairs, last) -> None:
    """The slope on pairs dated from 2023 only, set beside every pair's. For reading the
    full-sample figure: the rule never uses a sub-period."""
    known = optimize.pairs_known_by(last, pairs)
    rows = []
    for name, part in (("every pair", known), ("dated 2023 onwards", known[known.index >= POST_2023])):
        fit = sm.OLS(part["y"].to_numpy(), sm.add_constant(part["x"].to_numpy())).fit(
            cov_type="HAC", cov_kwds={"maxlags": HORIZON - 1}
        )
        r2 = float(fit.rsquared)
        rows.append(
            [
                name,
                f"{len(part):,}",
                f"{len(part) / HORIZON:.0f}",
                f"{fit.params[1]:.3f}",
                f"{fit.tvalues[1]:.2f}",
                f"{r2:.3f}",
            ]
        )
    table(["pairs", "count", "independent, about", "slope", "t over h−1 lags", "R²"], rows)


def forecast_pnl(runs, no_forecast, label) -> None:
    """What the forecasts changed: the optimised path less the same level's no-forecast
    path, before and after costs, and how many decisions that rests on."""
    rows = []
    for level, (breakdowns, result) in runs.items():
        baseline_breakdowns, baseline = no_forecast[level]
        gross = result.optimised.gross - baseline.optimised.gross
        net = result.optimised.net - baseline.optimised.net
        spread = relative(result.optimised.net, baseline.optimised.net)
        tilted = [d for d, b in breakdowns.items() if b.forecast_tilt != 0]
        moved = [
            d
            for d in tilted
            if not np.isclose(breakdowns[d].equity, baseline_breakdowns[d].equity)
        ]
        episodes = 0
        dates = list(breakdowns)
        for i, date in enumerate(dates):
            if date in tilted and (i == 0 or dates[i - 1] not in tilted):
                episodes += 1
        rows.append(
            [
                level,
                f"{gross.mean() * TRADING_DAYS_PER_YEAR:+.2%}",
                f"{gross.sum():+.1%}",
                f"{net.mean() * TRADING_DAYS_PER_YEAR:+.2%}",
                f"{(result.optimised.costs.sum() - baseline.optimised.costs.sum()):+.2%}",
                len(tilted),
                len(moved),
                episodes,
                f"{(net.abs() > 1e-12).sum():,} of {len(net):,}",
                t_text(spread.spread_t),
            ]
        )
    print(f"**{label}**\n")
    table(
        [
            "level",
            "gross P&L, a year",
            "gross P&L, summed",
            "net P&L, a year",
            "extra costs, summed",
            "non-zero tilts",
            "tilts with a different weight",
            "separate runs of tilts",
            "days the paths differ",
            "t vs no forecasts",
        ],
        rows,
    )


def main() -> None:
    prices, signals, baselines = load()
    pairs = optimize.trust_pairs(signals, baselines, prices, horizon=HORIZON)
    dates = optimize.common_rebalance_dates(
        prices, horizon=HORIZON, train_window=TRAIN_WINDOW, test_window=TEST_WINDOW, embargo=EMBARGO_DAYS
    )

    gated = {level: run(prices, signals, baselines, level) for level in optimize.RISK_LEVELS}
    ungated = {
        level: run(prices, signals, baselines, level, trust_gate=UNGATED)
        for level in optimize.RISK_LEVELS
    }
    no_forecast = {
        level: run(prices, signals, baselines, level, trust=0.0) for level in optimize.RISK_LEVELS
    }

    print(f"{len(pairs):,} pairs, {len(dates)} rebalances from {dates[0]:%d %b %Y} to {dates[-1]:%d %b %Y}\n")

    print("## Gate timeline\n")
    trusts = gate_timeline(dates, pairs)
    plain = ungated_trust(dates, pairs)
    open_values = [v for v in trusts.values() if v > 0]
    first_open = next((d for d, v in trusts.items() if v > 0), None)
    first_plain = next((d for d, v in plain.items() if v > 0), None)
    after = [v for d, v in plain.items() if first_plain is not None and d >= first_plain]
    print(f"Gated: non-zero at {len(open_values)} of {len(dates)}, first {first_open}, average when open {np.mean(open_values) if open_values else float('nan'):.3f}")
    print(f"Ungated: first non-zero {first_plain}, average after that {np.mean(after):.3f}, latest {list(plain.values())[-1]:.3f}, clipped at 1 at {sum(v == 1 for v in plain.values())} rebalances")
    premium = optimize.long_run_premium(prices, horizon=HORIZON).reindex(dates)
    print(f"Premium per {HORIZON} days: {premium.min():.2%} to {premium.max():.2%}, at or below zero at {(premium <= 0).sum()} of {len(dates)} rebalances\n")

    print("## Slope on all pairs and on pairs from 2023\n")
    post_2023_fit(pairs, dates[-1])

    print("## Weights, gated\n")
    weights_table(gated)
    print("## Weights, ungated\n")
    weights_table(ungated)
    print("## Risk aversion\n")
    risk_aversion_table(prices, trusts)

    print("## Performance, after costs\n")
    rows = []
    for level in optimize.RISK_LEVELS:
        rows.append(performance_row(f"Level {level}, gated", gated[level][1]))
        rows.append(performance_row(f"Level {level}, ungated", ungated[level][1]))
        rows.append(performance_row(f"Level {level}, no forecasts", no_forecast[level][1]))
    rows.append(performance_row("50/50 benchmark", gated[3][1], "baseline"))
    table(["portfolio", "net annual return", "net Sharpe", "max drawdown", "turnover a year", "t vs 50/50"], rows)

    print("## What the forecasts changed\n")
    forecast_pnl(gated, no_forecast, "With the gate")
    forecast_pnl(ungated, no_forecast, "Without the gate")

    print("## Other minimums, gated, level by level\n")
    rows = []
    for floor in (0.1,):
        for level in optimize.RISK_LEVELS:
            settings = {**SETTINGS, "bounds": (floor, 1 - floor)}
            b = optimize.schedule_breakdowns(signals, prices, baselines=baselines, risk_level=level, **settings)
            n = optimize.schedule_breakdowns(signals, prices, baselines=baselines, risk_level=level, trust=0.0, **settings)
            bt = backtest.run_backtest(prices, optimize.schedule_weights(b))
            nf = backtest.run_backtest(prices, optimize.schedule_weights(n))
            net = bt.metrics[("optimised", "net")]
            rows.append(
                [
                    f"{floor:.0%}",
                    level,
                    f"{net.annual_return:.2%}",
                    f"{net.sharpe:.2f}",
                    t_text(bt.relative["net"].spread_t),
                    t_text(relative(bt.optimised.net, nf.optimised.net).spread_t),
                ]
            )
    table(["minimum", "level", "net annual return", "net Sharpe", "t vs 50/50", "t vs no forecasts"], rows)


if __name__ == "__main__":
    main()
