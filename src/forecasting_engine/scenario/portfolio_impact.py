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
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.risk import monte_carlo
from forecasting_engine.risk.monte_carlo import MonteCarloResult, SimulatedTailRisk

PORTFOLIO: str = "optimised"
"""The one portfolio simulated: the weights the risk manager holds."""


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

    @property
    def return_change(self) -> float:
        return self.shocked.mean_return - self.base.mean_return

    @property
    def volatility_change(self) -> float:
        return self.shocked.volatility - self.base.volatility

    def drawdown_change(self, confidence: float) -> float:
        return self.shocked.max_drawdown[confidence] - self.base.max_drawdown[confidence]


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
) -> PortfolioImpact:
    """The portfolio's return, volatility and drawdown over the next ``horizon``
    days under the base-case forecasts and under the shocked ones.

    ``history`` is the daily returns the simulation draws its moves from
    (``monte_carlo.history_window``). Both forecast mappings must cover every
    index in ``history``, and ``weights`` must sum to 1: a scenario that could
    shock only one index still needs the other's base-case forecast to simulate
    the portfolio.
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

    def simulate(forecasts: Mapping[TargetRole, float]) -> MonteCarloResult:
        return monte_carlo.simulate(
            history,
            pd.Series({role: float(forecasts[role]) for role in assets}),
            {PORTFOLIO: held},
            horizon=horizon,
            method=method,
            paths=paths,
            seed=seed,
        )

    base = simulate(base_forecasts)
    shocked = simulate(shocked_forecasts)
    return PortfolioImpact(
        base=PortfolioMetrics.from_simulation(base.risk[PORTFOLIO]),
        shocked=PortfolioMetrics.from_simulation(shocked.risk[PORTFOLIO]),
        weights=held,
        forecasts=pd.DataFrame(
            {"base": base.forecasts, "shocked": shocked.forecasts}, index=assets
        ),
        horizon=horizon,
        method=method,
        paths=paths,
        as_of=base.as_of,
    )
