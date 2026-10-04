"""FYP-19: the optimised portfolio against the equal-weight benchmark, as rows a
page can show side by side, and as cumulative paths a chart can draw."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from forecasting_engine.portfolio.backtest import BASES, BacktestResult
from forecasting_engine.portfolio.directional import cumulative
from forecasting_engine.portfolio.performance import PerformanceMetrics

PORTFOLIO_LABELS: dict[str, str] = {
    "optimised": "Optimised",
    "baseline": "Equal-weight benchmark",
}

#: Metric heading -> (field on ``PerformanceMetrics``, shown as a percentage?).
METRICS: dict[str, tuple[str, bool]] = {
    "Annual return": ("annual_return", True),
    "Sharpe": ("sharpe", False),
    "Sortino": ("sortino", False),
    "Calmar": ("calmar", False),
    "Max drawdown": ("max_drawdown", True),
}


@dataclass(frozen=True)
class ComparisonRow:
    metric: str
    optimised: str
    baseline: str
    difference: str
    """Optimised minus benchmark. Every metric here is better when higher —
    drawdown included, since it is negative — so a positive difference always
    favours the optimised portfolio."""


def comparison_rows(result: BacktestResult, basis: str) -> list[ComparisonRow]:
    """One row per metric, optimised beside the benchmark, on ``basis`` ("gross"
    or "net" of trading costs)."""
    if basis not in BASES:
        raise ValueError(f"basis must be one of {BASES}, got {basis!r}")
    optimised = result.metrics[("optimised", basis)]
    baseline = result.metrics[("baseline", basis)]
    rows = []
    for name, (field, percent) in METRICS.items():
        a, b = _value(optimised, field), _value(baseline, field)
        rows.append(
            ComparisonRow(
                metric=name,
                optimised=_fmt(a, percent),
                baseline=_fmt(b, percent),
                difference=_fmt(a - b, percent, signed=True, points=percent),
            )
        )
    return rows


def cumulative_paths(result: BacktestResult, basis: str) -> pd.DataFrame:
    """Each portfolio's compounded return to date on ``basis``, a column each."""
    if basis not in BASES:
        raise ValueError(f"basis must be one of {BASES}, got {basis!r}")
    return pd.DataFrame(
        {
            PORTFOLIO_LABELS[name]: cumulative(getattr(getattr(result, name), basis))
            for name in PORTFOLIO_LABELS
        }
    )


def _value(metrics: PerformanceMetrics, field: str) -> float:
    return float(getattr(metrics, field))


def _fmt(value: float, percent: bool, *, signed: bool = False, points: bool = False) -> str:
    if value != value:  # NaN: an undefined ratio (no variation, no loss, no drawdown)
        return "—"
    sign = "+" if signed else ""
    if percent:
        text = f"{value * 100:{sign}.2f}"
        return f"{text} pts" if points else f"{text}%"
    return f"{value:{sign}.2f}"
