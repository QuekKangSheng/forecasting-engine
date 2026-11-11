"""What a signal shock does to the optimised portfolio (FYP-33).

A scenario (``scenario.shock``) gives each index a base-case forecast and a
shocked one. This module carries both through to the portfolio: the Monte Carlo
simulation (``risk.monte_carlo``) is run twice, once centred on the base-case
forecasts and once on the shocked ones, holding the same weights through the
same random paths. Everything but the forecasts is identical between the two
runs, so the difference in return, volatility and drawdown is the shock's alone,
and a zero shock reproduces the base case exactly.

The weights are the latest optimised weights as they stand; they are not
re-solved for the shock. Re-optimising under the shock is a separate, explicit
action (FYP-246), so a risk manager first sees what the allocation they already
hold would do.

Re-optimising solves the latest rebalance's target again (``optimize.SolveInputs``)
with one thing changed: each index's signal moves by as much as the shock moved its
forecast. A signal is the forecast less a level the shock doesn't touch, so
the two move together. The covariance, long-run premium, forecast trust, risk level
and weight limits are the rebalance's own. The re-optimised weights are simulated
under the shocked forecasts on the same paths as the held ones, so the two can be
compared directly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.portfolio.backtest import ASSETS
from forecasting_engine.portfolio.optimize import SolveInputs, WeightBreakdown
from forecasting_engine.risk import monte_carlo
from forecasting_engine.risk.monte_carlo import MonteCarloResult, SimulatedTailRisk

PORTFOLIO: str = "optimised"
"""The weights the risk manager holds."""

REOPTIMISED: str = "re-optimised"
"""The weights re-solved from the shocked forecasts (FYP-246)."""


class PortfolioImpactError(ValueError):
    """The impact can't be computed. Written for a risk manager."""


@dataclass(frozen=True)
class PortfolioMetrics:
    """One case's return and risk over the horizon, read off its simulation."""

    mean_return: float
    volatility: float
    max_drawdown: Mapping[float, float]
    """By confidence: the drawdown only the worst ``1 − c`` of paths exceed,
    negative like ``performance.max_drawdown``."""

    @classmethod
    def from_simulation(cls, risk: SimulatedTailRisk) -> PortfolioMetrics:
        return cls(
            mean_return=risk.mean_return,
            volatility=risk.volatility,
            max_drawdown={level.confidence: level.max_drawdown for level in risk.levels},
        )


@dataclass(frozen=True)
class Reoptimised:
    """The latest target solved again from the shocked forecasts, and what it does
    under them."""

    weights: pd.Series
    """By ``TargetRole``, summing to 1."""
    breakdown: WeightBreakdown
    """How the re-solved equity weight was reached."""
    unshocked: WeightBreakdown
    """The same solve on the signals as they were: the rebalance's target before any
    no-trade band, which a zero shock reproduces."""
    signal: pd.Series
    """Each index's shocked signal, the solve's input."""
    metrics: PortfolioMetrics
    """The re-optimised weights simulated under the shocked forecasts."""


@dataclass(frozen=True)
class PortfolioImpact:
    base: PortfolioMetrics
    shocked: PortfolioMetrics
    weights: pd.Series
    """The weights both cases hold, by ``TargetRole``."""
    forecasts: pd.DataFrame
    """Each index's base-case and shocked forecast: a row per role, columns
    ``base`` and ``shocked``."""
    horizon: int
    method: str
    paths: int
    as_of: pd.Timestamp
    reoptimised: Reoptimised | None = None
    """Set only when re-optimising was asked for."""

    @property
    def return_change(self) -> float:
        return self.shocked.mean_return - self.base.mean_return

    @property
    def volatility_change(self) -> float:
        return self.shocked.volatility - self.base.volatility

    def drawdown_change(self, confidence: float) -> float:
        return self.shocked.max_drawdown[confidence] - self.base.max_drawdown[confidence]


def shocked_signal(inputs: SolveInputs, forecasts: pd.DataFrame) -> pd.Series:
    """``inputs``' signal moved by each index's shocked less base-case forecast. An
    index ``forecasts`` doesn't cover is left where it was."""
    shift = forecasts["shocked"] - forecasts["base"]
    return inputs.signal + shift.reindex(inputs.signal.index, fill_value=0.0)


def portfolio_impact(
    history: pd.DataFrame,
    base_forecasts: Mapping[TargetRole, float],
    shocked_forecasts: Mapping[TargetRole, float],
    weights: Mapping[TargetRole, float],
    *,
    horizon: int,
    method: str = monte_carlo.BOOTSTRAP,
    paths: int = monte_carlo.DEFAULT_PATHS,
    seed: int = monte_carlo.SEED,
    reoptimise: SolveInputs | None = None,
) -> PortfolioImpact:
    """The portfolio's return, volatility and drawdown over the next ``horizon``
    days under the base-case forecasts and under the shocked ones.

    ``history`` is the daily returns the simulation draws its moves from
    (``monte_carlo.history_window``). Both forecast mappings must cover every
    index in ``history``, and ``weights`` must sum to 1: a scenario that could
    shock only one index still needs the other's base-case forecast to simulate
    the portfolio.

    Given ``reoptimise``, the latest rebalance's solve inputs, the weights are also
    re-solved on the shocked signals and simulated under the shocked forecasts; the
    held weights' results are unchanged by it.
    """
    assets = list(history.columns)
    for name, forecasts in (("base-case", base_forecasts), ("shocked", shocked_forecasts)):
        missing = [str(role) for role in assets if role not in forecasts]
        if missing:
            raise PortfolioImpactError(
                f"There is no {name} forecast for the {', '.join(missing)} index, so the "
                "portfolio can't be simulated."
            )
    held = pd.Series({role: float(weights[role]) for role in assets})

    def simulate(
        forecasts: Mapping[TargetRole, float], portfolios: Mapping[str, pd.Series]
    ) -> MonteCarloResult:
        return monte_carlo.simulate(
            history,
            pd.Series({role: float(forecasts[role]) for role in assets}),
            portfolios,
            horizon=horizon,
            method=method,
            paths=paths,
            seed=seed,
        )

    forecasts = pd.DataFrame(
        {
            "base": [float(base_forecasts[role]) for role in assets],
            "shocked": [float(shocked_forecasts[role]) for role in assets],
        },
        index=assets,
    )
    shocked_portfolios = {PORTFOLIO: held}
    if reoptimise is not None:
        signal = shocked_signal(reoptimise, forecasts)
        breakdown = reoptimise.solve(signal)
        equity, bond = ASSETS
        resolved = pd.Series({equity: breakdown.equity, bond: 1 - breakdown.equity})
        shocked_portfolios[REOPTIMISED] = resolved.reindex(assets)

    base = simulate(base_forecasts, {PORTFOLIO: held})
    # One run for both portfolios: they share its paths.
    shocked = simulate(shocked_forecasts, shocked_portfolios)

    reoptimised = None
    if reoptimise is not None:
        reoptimised = Reoptimised(
            weights=shocked_portfolios[REOPTIMISED],
            breakdown=breakdown,
            unshocked=reoptimise.solve(),
            signal=signal,
            metrics=PortfolioMetrics.from_simulation(shocked.risk[REOPTIMISED]),
        )
    return PortfolioImpact(
        base=PortfolioMetrics.from_simulation(base.risk[PORTFOLIO]),
        shocked=PortfolioMetrics.from_simulation(shocked.risk[PORTFOLIO]),
        weights=held,
        forecasts=forecasts,
        horizon=horizon,
        method=method,
        paths=paths,
        as_of=base.as_of,
        reoptimised=reoptimised,
    )
