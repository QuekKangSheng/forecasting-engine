"""Turn the active equity and bond models' signals into a weight schedule — the
input ``portfolio.backtest.run_backtest`` chains through history.

The rule is the two-asset Markowitz solution ``w = w_mv + μ/(λ·D)``, where ``w_mv``
is the minimum-variance equity weight, ``D`` the variance of the difference between
the two indices' returns, ``μ`` the expected equity-minus-bond return over the
horizon and ``λ`` the risk aversion (Markowitz 1952; Merton 1972 for the
minimum-variance decomposition).

A risk level fixes a target volatility, not ``λ``: the levels are evenly spaced in
volatility between the lowest-risk mix the bounds allow (level 5) and the highest
(level 1), and ``w*_L`` is the weight on the efficient side that has that volatility.
Each level's ``λ_L`` is whatever makes ``w*_L`` optimal with no forecast information,
``μ = π̄``, so ``λ_L = π̄ / (D·(w*_L − w_mv))``: reverse optimisation (Sharpe 1974;
Black & Litterman 1992). ``π̄`` is the long-run equity-minus-bond premium, the mean
over every day to date.

The expected return used is ``μ = π̄ + β̂·Δs``: the premium plus the part of the
signal difference the models have earned trust for (Grinold 1994; Black & Litterman
1992). Substituting gives ``w = w*_L + β̂·Δs·(w*_L − w_mv)/π̄``. Level 5 is infinite
risk aversion, so it ignores forecasts. Level 1 is the riskiest mix the bounds allow,
so with the default bounds it sits on the upper bound and can only tilt down: that
follows from the bounds, not a separate rule. Level 3 is not pinned to 50/50. The
50/50 benchmark used for comparison (``backtest.BASELINE_WEIGHTS``) is separate and
never moves.

A model's signal is its forecast less its own level for that fold, which for most
models is the fold's training-mean return (``validation.harness.signal_forecasts``).
The levels are left out on purpose: equity's trailing mean is far above bond's in
most windows and swings with recent performance, so fed to the optimiser it pins
the allocation to a bound whatever the signals say.

``β̂`` is estimated on an expanding window, from each past date's signal difference
and the return that followed it — a Mincer-Zarnowitz regression of realised on
forecast (Mincer & Zarnowitz 1969) — using only returns fully known before the
rebalance (``forecast_trust``). Until ``MIN_TRUST_HISTORY`` such pairs exist, and whenever the
slope's t-statistic is not above ``validation.gates.SIGNIFICANCE_T_GATE``,
it is zero, so the forecasts move nothing.

A rebalance happens once every ``horizon`` trading days over the out-of-sample
period, so each allocation is held for exactly as long as the forecast it was
set from looks ahead.

Between rebalances the portfolio drifts with the two indices' returns, and a
rebalance trades only when the new target is worth its cost: within a no-trade band
of ``c / (λ_L·D) = c·(w*_L − w_mv)/π̄`` around the drifted weight, where ``c`` is the
cost of moving the equity weight by 1 (equity's plus bond's cost), the portfolio
holds; outside it, it trades only as far as the band's near edge. A portfolio that
has drifted outside the weight bounds is traded back to the nearest bound, whatever
the band. The first rebalance buys the target from cash.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.ingest.align import FeaturePanel, align_and_lag
from forecasting_engine.portfolio.backtest import (
    ASSETS,
    DEFAULT_COSTS_BPS,
    asset_returns,
)
from forecasting_engine.validation import metrics
from forecasting_engine.validation.gates import SIGNIFICANCE_T_GATE
from forecasting_engine.validation.splitters import TUNING_ROWS, PurgedWalkForward


class OptimizeDataError(ValueError):
    """The message is written for a portfolio manager, like the other models' errors."""


RISK_LEVELS: tuple[int, ...] = (1, 2, 3, 4, 5)
"""The sponsor's 1-5 risk scale, 1 the riskiest. Each level targets a volatility
evenly spaced between the lowest-risk and highest-risk mixes the bounds allow."""

DEFAULT_RISK_LEVEL: int = 3
"""Working default, not sponsor-confirmed: the middle of the scale."""

DEFAULT_WEIGHT_BOUNDS: tuple[float, float] = (0.2, 0.8)
"""Working default, not sponsor-confirmed: stops either asset being pushed to a
corner by a small, noisy difference between the two forecasts. Revisit once
Alpha Norm states how far the allocation may tilt from 50/50."""

MIN_TRUST_HISTORY: int = 252
"""A convention, not a derived value: about a year of daily forecast-and-outcome pairs
before the forecasts move any weight. With a 20-day horizon those pairs overlap, so
they hold about 13 independent observations and the first estimates are noisy."""

_PRICE = "price"


def _common_price_calendar(prices: Mapping[TargetRole, pd.Series]) -> pd.DatetimeIndex:
    calendars = [prices[role].dropna().index for role in ASSETS]
    common = calendars[0]
    for other in calendars[1:]:
        common = common.intersection(other)
    return common.sort_values()


def common_rebalance_dates(
    prices: Mapping[TargetRole, pd.Series],
    *,
    horizon: int,
    train_window: int,
    test_window: int,
    embargo: int,
) -> pd.DatetimeIndex:
    """Every ``horizon``-th out-of-sample date on the calendar both targets share,
    from the first walk-forward test window's start.

    An allocation is held until the next rebalance, so holding it for the
    forecast's own horizon is what the forecast speaks to: a 5-day forecast held
    20 days says nothing about the last 15 of them.

    Equity's and bond's own active models were each walk-forward split over
    their *own* calendar, so splitting one shared calendar keeps both targets'
    out-of-sample period, and so the rebalance dates, in step.
    """
    calendar = _common_price_calendar(prices)
    panel = FeaturePanel(
        frame=pd.DataFrame(index=calendar), signals=(), targets=(), horizon=horizon
    )
    splitter = PurgedWalkForward(
        train=train_window, test=test_window, embargo=embargo, tuning_rows=TUNING_ROWS
    )
    tested = [test for _, test in splitter.split(panel) if len(test)]
    if not tested:
        return pd.DatetimeIndex([])
    return tested[0].append(tested[1:])[::horizon]


def expected_returns(
    rebalance_dates: pd.DatetimeIndex, forecasts: Mapping[TargetRole, pd.Series]
) -> pd.DataFrame:
    """Each target's own forecast as of each rebalance date — the mean-variance
    solver's expected-return input, one row per rebalance, a column per asset.

    Reads the most recent forecast at or before the rebalance date rather than
    requiring an exact match: equity's and bond's own saved forecasts can fall
    on slightly different calendars, and a model's most recent view as of that
    date is what it would actually be using if asked that day — never a date
    after it, so this stays as leakage-free as an exact match would be.
    """
    return pd.DataFrame(
        {
            role: forecasts[role].sort_index().reindex(rebalance_dates, method="ffill")
            for role in ASSETS
        }
    )


def training_window(
    rebalance_date: pd.Timestamp,
    calendar: pd.DatetimeIndex,
    *,
    horizon: int,
    train_window: int,
    embargo: int,
) -> pd.DatetimeIndex:
    """The ``train_window`` dates on ``calendar`` ending ``embargo`` days before
    ``rebalance_date`` — the same purge gap a forecasting model itself trains
    under, so covariance never reaches into days the model couldn't see either.
    """
    if rebalance_date not in calendar:
        raise OptimizeDataError(
            f"{rebalance_date:%d %b %Y} is not a trading day on the shared equity/bond calendar."
        )
    panel = FeaturePanel(
        frame=pd.DataFrame(index=calendar), signals=(), targets=(), horizon=horizon
    )
    splitter = PurgedWalkForward(train=train_window, test=0, embargo=embargo)
    position = calendar.get_loc(rebalance_date)
    return splitter.window_before(panel, position, train_window)


def covariance_at_rebalance(
    rebalance_date: pd.Timestamp,
    prices: Mapping[TargetRole, pd.Series],
    *,
    horizon: int,
    train_window: int,
    embargo: int,
) -> pd.DataFrame:
    """Equity/bond daily-return covariance over the training window ending before
    ``rebalance_date``, scaled from daily to ``horizon``-day terms."""
    calendar = _common_price_calendar(prices)
    train_dates = training_window(
        rebalance_date, calendar, horizon=horizon, train_window=train_window, embargo=embargo
    )
    returns = pd.DataFrame(
        {role: prices[role].reindex(train_dates).pct_change() for role in ASSETS}
    ).dropna()
    if len(returns) < 2:
        raise OptimizeDataError(
            f"Not enough trading days before {rebalance_date:%d %b %Y} to estimate a "
            f"covariance (need at least 2 daily returns, got {len(returns)})."
        )
    return returns.cov() * horizon


def long_run_premium(prices: Mapping[TargetRole, pd.Series], *, horizon: int) -> pd.Series:
    """``π̄``: the mean daily equity-minus-bond return over every day to date,
    scaled to ``horizon``-day terms, one value per day of the shared calendar.

    A rebalance trades at its day's close, so that day's own return is already known."""
    calendar = _common_price_calendar(prices)
    returns = pd.DataFrame(
        {role: prices[role].reindex(calendar).pct_change() for role in ASSETS}
    ).dropna()
    equity, bond = ASSETS
    return (returns[equity] - returns[bond]).expanding().mean() * horizon


def trust_pairs(
    signals: Mapping[TargetRole, pd.Series],
    baselines: Mapping[TargetRole, pd.Series],
    prices: Mapping[TargetRole, pd.Series],
    *,
    horizon: int,
) -> pd.DataFrame:
    """One row per date with a stored forecast for both models: ``x``, the signal
    difference; ``y``, the realised equity-minus-bond return over the horizon starting
    that date, less the same difference in levels taken off the forecasts; and
    ``label_end``, the date the later of the two returns is known.

    The returns are the labels the models were trained and scored on
    (``ingest.align.align_and_lag``)."""
    equity, bond = ASSETS
    panels = {
        role: align_and_lag(prices[role].to_frame(_PRICE), (), _PRICE, horizon) for role in ASSETS
    }
    labels = {role: panel.frame[panel.targets[0]] for role, panel in panels.items()}
    return pd.DataFrame(
        {
            "x": signals[equity] - signals[bond],
            "y": labels[equity] - labels[bond] - (baselines[equity] - baselines[bond]),
            "label_end": pd.concat([panels[role].label_end for role in ASSETS], axis=1).max(axis=1),
        }
    ).dropna()


def pairs_known_by(rebalance_date: pd.Timestamp, pairs: pd.DataFrame) -> pd.DataFrame:
    """The ``trust_pairs`` rows whose horizon fully ended before ``rebalance_date``."""
    return pairs[pairs["label_end"] < rebalance_date]


def _trust_fit(rebalance_date: pd.Timestamp, pairs: pd.DataFrame, *, horizon: int):
    """The OLS of ``y`` on ``x`` with an intercept over every pair known by
    ``rebalance_date``, or ``None`` with fewer than ``MIN_TRUST_HISTORY`` pairs or where
    ``x`` never varies and has no slope. Its standard errors count h - 1 lags in full,
    for the overlap between neighbouring h-day labels, as the model metrics' do
    (``validation.metrics.overlap_fit``)."""
    known = pairs_known_by(rebalance_date, pairs)
    x, y = known["x"].to_numpy(dtype=float), known["y"].to_numpy(dtype=float)
    if len(known) < MIN_TRUST_HISTORY or np.ptp(x) == 0:
        return None
    return metrics.overlap_fit(y, x, lags=horizon - 1)


def forecast_trust(
    rebalance_date: pd.Timestamp,
    pairs: pd.DataFrame,
    *,
    horizon: int,
    gate: float = SIGNIFICANCE_T_GATE,
) -> float:
    """``β̂``: how much of the signal difference the models have earned trust for. The
    slope of ``_trust_fit``, clipped to ``[0, 1]`` (Grinold 1994). Zero where there is no
    fit, or where the slope's t-statistic is not above ``gate``."""
    fit = _trust_fit(rebalance_date, pairs, horizon=horizon)
    if fit is None or not fit.tvalues[1] > gate:
        return 0.0
    return min(max(float(fit.params[1]), 0.0), 1.0)


def forecast_gate_shut(
    rebalance_date: pd.Timestamp,
    pairs: pd.DataFrame,
    *,
    horizon: int,
    gate: float = SIGNIFICANCE_T_GATE,
) -> bool:
    """Whether there is a fit but its slope's t-statistic is not above ``gate``."""
    fit = _trust_fit(rebalance_date, pairs, horizon=horizon)
    return fit is not None and not fit.tvalues[1] > gate


@dataclass(frozen=True)
class WeightBreakdown:
    """How one rebalance's equity weight was reached, for a reader to follow."""

    level_weight: float
    """``w*_L``: equity's weight at this risk level's target volatility, where the
    allocation sits with no forecast."""
    forecast_tilt: float
    """What the forecasts add to equity, before any bound."""
    equity: float
    """Equity's final weight, after the bounds and any no-trade band."""
    target: float | None = None
    """Equity's weight after the bounds, before any no-trade band. ``None`` where
    no band applied, so it is ``equity``."""
    band: float = 0.0
    """The no-trade band's half-width around ``drifted``: 0 where none applied."""
    drifted: float | None = None
    """Equity's weight as the last allocation drifted to by this rebalance, or
    ``None`` at the first, which buys from cash."""
    held: bool = False
    """Whether this rebalance traded nothing: the target was within the band."""

    @property
    def unconstrained(self) -> float:
        return self.level_weight + self.forecast_tilt

    @property
    def bounded(self) -> float:
        return self.equity if self.target is None else self.target

    @property
    def capped(self) -> bool:
        return self.bounded != self.unconstrained


def level_weight(
    level: int, covariance: pd.DataFrame, bounds: tuple[float, float] = DEFAULT_WEIGHT_BOUNDS
) -> float:
    """``w*_L``: the equity weight whose volatility is level ``level``'s share of the
    way from the lowest-risk mix the bounds allow up to the highest, on the
    ``w ≥ w_mv`` branch of ``σ²(w) = σ_mv² + D·(w − w_mv)²``.

    With the minimum-variance weight at or above the upper bound there is no room to
    take more risk, so every level holds that bound."""
    if level not in RISK_LEVELS:
        raise ValueError(f"risk level must be 1 to 5, got {level}")
    upper = bounds[1]
    minimum = _min_variance_weight(covariance)
    lowest = _lowest_risk_weight(minimum, bounds)
    if minimum is None:
        return lowest
    if minimum >= upper or level == RISK_LEVELS[0]:
        return upper
    if level == RISK_LEVELS[-1]:
        return lowest
    low_vol = math.sqrt(_variance(lowest, covariance))
    high_vol = math.sqrt(_variance(upper, covariance))
    target_vol = low_vol + (RISK_LEVELS[-1] - level) / 4 * (high_vol - low_vol)
    excess = (target_vol**2 - _variance(minimum, covariance)) / _var_diff(covariance)
    return min(max(minimum + math.sqrt(max(excess, 0.0)), lowest), upper)


def weight_breakdown(
    expected_signal: pd.Series,
    covariance: pd.DataFrame,
    *,
    premium: float,
    trust: float = 0.0,
    risk_level: int = DEFAULT_RISK_LEVEL,
    bounds: tuple[float, float] = DEFAULT_WEIGHT_BOUNDS,
) -> WeightBreakdown:
    """``solve_weights``'s equity weight, split into the level's no-forecast weight
    and the forecasts' tilt from it: ``w = w*_L + β̂·Δs·(w*_L − w_mv)/π̄``, then
    clipped to ``bounds``."""
    equity, bond = ASSETS
    weight = level_weight(risk_level, covariance, bounds)
    signal_diff = expected_signal[equity] - expected_signal[bond]
    tilt = float(trust * signal_diff * _tilt_scale(risk_level, weight, covariance, premium))
    w_equity = min(max(weight + tilt, bounds[0]), bounds[1])
    return WeightBreakdown(level_weight=weight, forecast_tilt=tilt, equity=w_equity)


def solve_weights(
    expected_signal: pd.Series,
    covariance: pd.DataFrame,
    *,
    premium: float,
    trust: float = 0.0,
    risk_level: int = DEFAULT_RISK_LEVEL,
    bounds: tuple[float, float] = DEFAULT_WEIGHT_BOUNDS,
) -> pd.Series:
    """The long-only two-asset weights maximising ``w·μ − λ_L/2·σ²(w)``, with
    ``μ = π̄ + β̂·Δs``, subject to the two weights summing to 1 and each lying within
    ``bounds``.

    Two assets collapse that to one free variable with a closed form, then clipped
    to ``bounds``. When the two assets' returns are so alike that moving between them
    changes no risk at all (equal variance, perfect correlation), the covariance can't
    size anything and the lowest-risk weight is held rather than dividing by zero.
    """
    equity, bond = ASSETS
    w_equity = weight_breakdown(
        expected_signal,
        covariance,
        premium=premium,
        trust=trust,
        risk_level=risk_level,
        bounds=bounds,
    ).equity
    return pd.Series({equity: w_equity, bond: 1 - w_equity})


@dataclass(frozen=True)
class SolveInputs:
    """Everything one rebalance's target is solved from, kept so the same solve can be
    run again on a different signal: re-optimising under a shock (FYP-246) changes
    the signal and nothing else."""

    date: pd.Timestamp
    signal: pd.Series
    """Each asset's signal at ``date``: its forecast less its fold's level."""
    covariance: pd.DataFrame
    premium: float
    trust: float
    risk_level: int
    bounds: tuple[float, float]

    def solve(self, signal: pd.Series | None = None) -> WeightBreakdown:
        """``weight_breakdown`` from these inputs, on ``signal`` if given, else on the
        rebalance's own. No no-trade band: this is the target, ``WeightBreakdown.bounded``."""
        return weight_breakdown(
            self.signal if signal is None else signal,
            self.covariance,
            premium=self.premium,
            trust=self.trust,
            risk_level=self.risk_level,
            bounds=self.bounds,
        )


def solve_inputs_at(
    rebalance_date: pd.Timestamp,
    signals: Mapping[TargetRole, pd.Series],
    prices: Mapping[TargetRole, pd.Series],
    *,
    baselines: Mapping[TargetRole, pd.Series],
    horizon: int,
    train_window: int,
    embargo: int,
    risk_level: int = DEFAULT_RISK_LEVEL,
    bounds: tuple[float, float] = DEFAULT_WEIGHT_BOUNDS,
    trust_gate: float = SIGNIFICANCE_T_GATE,
) -> SolveInputs:
    """The inputs ``schedule_breakdowns`` solves ``rebalance_date``'s target from: the
    signal as of that date, the covariance over its training window, the long-run
    premium and ``forecast_trust``."""
    signal = expected_returns(pd.DatetimeIndex([rebalance_date]), signals).iloc[0]
    if signal.isna().any():
        raise OptimizeDataError(
            f"There is no signal for both indices on {rebalance_date:%d %b %Y}."
        )
    pairs = trust_pairs(signals, baselines, prices, horizon=horizon)
    return SolveInputs(
        date=rebalance_date,
        signal=signal,
        covariance=covariance_at_rebalance(
            rebalance_date, prices, horizon=horizon, train_window=train_window, embargo=embargo
        ),
        premium=float(long_run_premium(prices, horizon=horizon).get(rebalance_date, 0.0)),
        trust=forecast_trust(rebalance_date, pairs, horizon=horizon, gate=trust_gate),
        risk_level=risk_level,
        bounds=bounds,
    )


def weight_schedule(
    signals: Mapping[TargetRole, pd.Series],
    prices: Mapping[TargetRole, pd.Series],
    *,
    baselines: Mapping[TargetRole, pd.Series],
    horizon: int,
    train_window: int,
    test_window: int,
    embargo: int,
    risk_level: int = DEFAULT_RISK_LEVEL,
    bounds: tuple[float, float] = DEFAULT_WEIGHT_BOUNDS,
    costs_bps: Mapping[TargetRole, float] = DEFAULT_COSTS_BPS,
    trust: float | None = None,
) -> pd.DataFrame:
    """The full weight schedule ``portfolio.backtest.run_backtest`` chains
    through history: one row of equity/bond weights per rebalance date, from
    ``schedule_breakdowns``."""
    return schedule_weights(
        schedule_breakdowns(
            signals,
            prices,
            baselines=baselines,
            horizon=horizon,
            train_window=train_window,
            test_window=test_window,
            embargo=embargo,
            risk_level=risk_level,
            bounds=bounds,
            costs_bps=costs_bps,
            trust=trust,
        )
    )


def schedule_weights(breakdowns: Mapping[pd.Timestamp, WeightBreakdown]) -> pd.DataFrame:
    """One row of equity/bond weights per rebalance in ``breakdowns``. A held
    rebalance's row is the drifted weights, so the backtest trades nothing there."""
    equity, bond = ASSETS
    weights = np.array([b.equity for b in breakdowns.values()], dtype=float)
    return pd.DataFrame(
        {equity: weights, bond: 1 - weights}, index=pd.DatetimeIndex(list(breakdowns))
    )


def schedule_breakdowns(
    signals: Mapping[TargetRole, pd.Series],
    prices: Mapping[TargetRole, pd.Series],
    *,
    baselines: Mapping[TargetRole, pd.Series],
    horizon: int,
    train_window: int,
    test_window: int,
    embargo: int,
    risk_level: int = DEFAULT_RISK_LEVEL,
    bounds: tuple[float, float] = DEFAULT_WEIGHT_BOUNDS,
    costs_bps: Mapping[TargetRole, float] = DEFAULT_COSTS_BPS,
    trust: float | None = None,
    trust_gate: float = SIGNIFICANCE_T_GATE,
) -> dict[pd.Timestamp, WeightBreakdown]:
    """How each rebalance's equity weight was reached, in date order: the target
    solved from that date's signals, ``forecast_trust`` and the covariance over its
    own training window, then the no-trade band against the weight the last
    allocation drifted to. Zero costs leave no band, so every rebalance moves to its
    target. ``baselines`` are each forecast's fold levels. A ``trust`` given
    is used at every rebalance in place of the estimate; ``trust=0.0`` is the level
    with no forecasts, under the same band. ``trust_gate`` is the t-statistic the
    estimate must clear.

    A rebalance whose signal is missing (a model had no prediction that day) or
    whose training window can't support a covariance estimate is left out rather
    than guessed at; the portfolio drifts on through it.
    """
    rebalance_dates = common_rebalance_dates(
        prices, horizon=horizon, train_window=train_window, test_window=test_window, embargo=embargo
    )
    returns = expected_returns(rebalance_dates, signals)
    premiums = long_run_premium(prices, horizon=horizon)
    pairs = trust_pairs(signals, baselines, prices, horizon=horizon)
    targets: dict[pd.Timestamp, tuple[WeightBreakdown, float]] = {}
    for date in rebalance_dates:
        row = returns.loc[date]
        if row.isna().any():
            continue
        try:
            covariance = covariance_at_rebalance(
                date, prices, horizon=horizon, train_window=train_window, embargo=embargo
            )
        except OptimizeDataError:
            continue
        premium = float(premiums.get(date, 0.0))
        breakdown = weight_breakdown(
            row,
            covariance,
            premium=premium,
            trust=forecast_trust(date, pairs, horizon=horizon, gate=trust_gate)
            if trust is None
            else trust,
            risk_level=risk_level,
            bounds=bounds,
        )
        scale = _tilt_scale(risk_level, breakdown.level_weight, covariance, premium)
        targets[date] = (breakdown, scale)
    if not targets:
        return {}

    cost = sum(costs_bps[role] for role in ASSETS) / 1e4
    daily_returns = asset_returns(prices, pd.DataFrame(index=pd.DatetimeIndex(list(targets))))
    held: np.ndarray | None = None
    breakdowns: dict[pd.Timestamp, WeightBreakdown] = {}
    for i, (day, row) in enumerate(daily_returns[list(ASSETS)].iterrows()):
        if i > 0 and held is not None:
            # Drifted as backtest.chain drifts it, so a held rebalance trades nothing.
            daily = row.to_numpy(dtype=float)
            held = held * (1 + daily) / (1 + float(held @ daily))
        if day not in targets:
            continue
        breakdown, scale = targets[day]
        if held is not None and cost > 0:
            breakdown = _banded(breakdown, float(held[0]), cost * scale, bounds)
        breakdowns[day] = breakdown
        held = np.array([breakdown.equity, 1 - breakdown.equity])
    return breakdowns


def _var_diff(covariance: pd.DataFrame) -> float:
    """``D = σ_e² − 2σ_eb + σ_b²``: the variance of the difference between the
    two indices' returns."""
    equity, bond = ASSETS
    return float(
        covariance.loc[equity, equity]
        - 2 * covariance.loc[equity, bond]
        + covariance.loc[bond, bond]
    )


def _variance(weight: float, covariance: pd.DataFrame) -> float:
    """The variance of a portfolio holding ``weight`` in equity and the rest in bond."""
    equity, bond = ASSETS
    return float(
        weight**2 * covariance.loc[equity, equity]
        + 2 * weight * (1 - weight) * covariance.loc[equity, bond]
        + (1 - weight) ** 2 * covariance.loc[bond, bond]
    )


def _min_variance_weight(covariance: pd.DataFrame) -> float | None:
    """``w_mv = (σ_b² − σ_eb) / D``, or ``None`` where ``D ≤ 0``: every mix is then
    equally risky, so no weight is the minimum."""
    equity, bond = ASSETS
    var_diff = _var_diff(covariance)
    if var_diff <= 0:
        return None
    return float((covariance.loc[bond, bond] - covariance.loc[equity, bond]) / var_diff)


def _lowest_risk_weight(minimum: float | None, bounds: tuple[float, float]) -> float:
    """``w_low``: the minimum-variance weight clipped to ``bounds``, or their midpoint
    where every mix is equally risky."""
    lower, upper = bounds
    return (lower + upper) / 2 if minimum is None else min(max(minimum, lower), upper)


def _tilt_scale(level: int, weight: float, covariance: pd.DataFrame, premium: float) -> float:
    """``(w*_L − w_mv)/π̄``, which is ``1/(λ_L·D)``: the weight a unit of trusted
    signal moves, and the no-trade band per unit of cost. Zero at level 5 (infinite
    risk aversion), where the premium gives no reason to hold equity, and where
    ``w*_L`` is not above a minimum-variance weight."""
    minimum = _min_variance_weight(covariance)
    if level == RISK_LEVELS[-1] or premium <= 0 or minimum is None:
        return 0.0
    return max(weight - minimum, 0.0) / premium


def _banded(
    target: WeightBreakdown, drifted: float, band: float, bounds: tuple[float, float]
) -> WeightBreakdown:
    """Trade to the nearest bound if the portfolio has drifted outside ``bounds``.
    Otherwise hold at ``drifted`` if the target is within ``band`` of it, or move to
    the band's near edge."""
    lower, upper = bounds
    if not lower <= drifted <= upper:
        held, equity = False, min(max(drifted, lower), upper)
    else:
        gap = target.equity - drifted
        held = abs(gap) <= band
        edge = target.equity - math.copysign(band, gap)
        equity = drifted if held else min(max(edge, lower), upper)
    return WeightBreakdown(
        level_weight=target.level_weight,
        forecast_tilt=target.forecast_tilt,
        equity=equity,
        target=target.equity,
        band=band,
        drifted=drifted,
        held=held,
    )
