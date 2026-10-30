"""FYP-19: the optimised portfolio against the equal-weight benchmark, as rows a
page can show side by side, and as cumulative paths a chart can draw. FYP-56:
the same two portfolios' historical tail risk, as rows beside max drawdown.
FYP-38: those rows for every VaR/CVaR method, with "Not run" for any not computed.
FYP-20: whether the optimised portfolio's outperformance is statistically
significant, as a figure and a plain-language label. FYP-254: each index's and
the optimised portfolio's rolling volatility, as paths a chart can draw. FYP-258:
the same three series' drawdown from their running peak, likewise."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.portfolio.backtest import ASSETS, BASES, BacktestResult
from forecasting_engine.portfolio.directional import cumulative
from forecasting_engine.portfolio.performance import (
    ROLLING_VOL_WINDOW,
    SIGNIFICANCE_CONFIDENCE,
    PerformanceMetrics,
    Significance,
    drawdown,
    rolling_volatility,
)
from forecasting_engine.risk.tail import (
    HISTORICAL,
    TAIL_METHODS,
    VAR_CONFIDENCES,
    TailRisk,
)

NOT_RUN: str = "Not run"

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


def rolling_volatility_paths(
    result: BacktestResult,
    basis: str,
    labels: Mapping[TargetRole, str],
    *,
    window: int = ROLLING_VOL_WINDOW,
) -> pd.DataFrame:
    """Rolling annualised volatility of each index, named by ``labels``, and of the
    optimised portfolio on ``basis``, a column each. An index pays no trading costs,
    so only the portfolio follows ``basis``. Rows start once ``window`` days have
    passed, so a backtest shorter than that has none."""
    returns = _index_and_portfolio_returns(result, basis, labels)
    return pd.DataFrame(
        {name: rolling_volatility(r, window) for name, r in returns.items()}
    ).dropna()


def drawdown_paths(
    result: BacktestResult, basis: str, labels: Mapping[TargetRole, str]
) -> pd.DataFrame:
    """How far each index, named by ``labels``, and the optimised portfolio on
    ``basis`` stand below their running peak on each backtest day, a column each.
    As with rolling volatility, only the portfolio follows ``basis``."""
    returns = _index_and_portfolio_returns(result, basis, labels)
    return pd.DataFrame({name: drawdown(r) for name, r in returns.items()})


def _index_and_portfolio_returns(
    result: BacktestResult, basis: str, labels: Mapping[TargetRole, str]
) -> dict[str, pd.Series]:
    """Each index's daily returns, named by ``labels``, then the optimised
    portfolio's on ``basis``: the three lines the risk charts draw."""
    if basis not in BASES:
        raise ValueError(f"basis must be one of {BASES}, got {basis!r}")
    returns = {labels[role]: result.index_returns[role] for role in ASSETS}
    returns[PORTFOLIO_LABELS["optimised"]] = getattr(result.optimised, basis)
    return returns


@dataclass(frozen=True)
class TailRow:
    metric: str
    optimised: str
    baseline: str
    method: str = HISTORICAL


def tail_rows(
    result: BacktestResult,
    basis: str,
    *,
    simulated: Mapping[str, Mapping[tuple[str, str], TailRisk]] | None = None,
) -> list[TailRow]:
    """VaR and CVaR at each confidence for every method in ``TAIL_METHODS``, for
    both portfolios on ``basis``. Historical's rows come first and are followed by
    max drawdown and each VaR's breach rate against what it should be, which only
    the historical path has, so every Historical row sits together. Losses are
    positive percentages; lower is better on every row.

    ``simulated`` maps a Monte Carlo method to ``{(portfolio, basis): TailRisk}``,
    keyed like ``result.tail_risk``; a method without both portfolios on ``basis``
    shows ``NOT_RUN``. Historical figures always come from the backtest."""
    if basis not in BASES:
        raise ValueError(f"basis must be one of {BASES}, got {basis!r}")
    simulated = simulated or {}
    for method, risks in simulated.items():
        if method == HISTORICAL or method not in TAIL_METHODS:
            raise ValueError(
                f"simulated methods must be one of {TAIL_METHODS[1:]}, got {method!r}"
            )
        for key, risk in risks.items():
            if risk.method != method:
                raise ValueError(f"{key} under {method!r} is tagged {risk.method!r}")
    optimised = result.tail_risk[("optimised", basis)]
    baseline = result.tail_risk[("baseline", basis)]
    historical_only = [
        TailRow(
            "Max drawdown",
            _fmt(optimised.max_drawdown, True),
            _fmt(baseline.max_drawdown, True),
        )
    ]
    for mine, theirs in zip(optimised.levels, baseline.levels, strict=True):
        historical_only.append(
            TailRow(
                f"VaR {mine.confidence:.0%} breach rate (expected "
                f"{mine.expected_breach_rate:.0%})",
                _fmt(mine.breach_rate, True),
                _fmt(theirs.breach_rate, True),
            )
        )
    rows = []
    for method in TAIL_METHODS:
        risks = result.tail_risk if method == HISTORICAL else simulated.get(method, {})
        mine_risk = risks.get(("optimised", basis))
        theirs_risk = risks.get(("baseline", basis))
        if mine_risk is None or theirs_risk is None:
            for confidence in VAR_CONFIDENCES:
                level = f"{confidence:.0%}"
                rows.append(TailRow(f"1-day VaR {level}", NOT_RUN, NOT_RUN, method))
                rows.append(TailRow(f"1-day CVaR {level}", NOT_RUN, NOT_RUN, method))
            continue
        for mine, theirs in zip(mine_risk.levels, theirs_risk.levels, strict=True):
            level = f"{mine.confidence:.0%}"
            rows.append(
                TailRow(f"1-day VaR {level}", _fmt(mine.var, True), _fmt(theirs.var, True), method)
            )
            rows.append(
                TailRow(
                    f"1-day CVaR {level}", _fmt(mine.cvar, True), _fmt(theirs.cvar, True), method
                )
            )
        if method == HISTORICAL:
            rows.extend(historical_only)
    return rows


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


@dataclass(frozen=True)
class SignificanceSummary:
    """The spread t-statistic as a page shows it."""

    t: str
    """The raw t-statistic, or "—" when it can't be computed."""
    label: str
    """Plain-language verdict, e.g. "Statistically significant at 95% confidence"."""
    tone: str
    """A ``ui.lozenge`` tone: success, danger or neutral."""
    detail: str
    """What the test was run on, for a caption."""


def significance_summary(result: BacktestResult, basis: str) -> SignificanceSummary:
    """Is the optimised portfolio's outperformance of the benchmark on ``basis``
    more than noise? Read from the backtest's own paired, Newey-West t-statistic."""
    if basis not in BASES:
        raise ValueError(f"basis must be one of {BASES}, got {basis!r}")
    relative = result.relative[basis]
    level = f"{SIGNIFICANCE_CONFIDENCE:.0%}"
    verdict = relative.significance
    label, tone = {
        Significance.OUTPERFORMS: (f"Statistically significant at {level} confidence", "success"),
        Significance.UNDERPERFORMS: (
            f"Significantly worse than the benchmark at {level} confidence",
            "danger",
        ),
        Significance.NOT_SIGNIFICANT: ("Not statistically significant", "neutral"),
        Significance.UNDEFINED: ("Cannot be tested", "neutral"),
    }[verdict]
    if verdict == Significance.UNDEFINED:
        detail = (
            "The two portfolios' daily returns never differ, or the backtest has fewer "
            "than three days, so there is no difference to test."
        )
    else:
        detail = (
            f"Paired test on {relative.days:,} daily return differences (optimised minus "
            f"benchmark), Newey-West adjusted for {relative.spread_t_lags} days of "
            "neighbouring correlation."
        )
    t = "—" if relative.spread_t != relative.spread_t else f"{relative.spread_t:.2f}"
    return SignificanceSummary(t=t, label=label, tone=tone, detail=detail)
