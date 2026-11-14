"""The Portfolio Optimizer page's last result, kept in session state so Home can
say where the pipeline stands, and the Scenarios page can carry a shock through to
the portfolio, without re-running the optimiser or the backtest.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from forecasting_engine.portfolio.optimize import SolveInputs

#: Session-state key the Portfolio page writes and Home and Scenarios read.
SUMMARY_KEY = "portfolio_summary"


@dataclass(frozen=True)
class PortfolioSummary:
    dataset_fingerprint: str
    """The committed data the run used, to tell Home when it has been replaced."""
    as_of: pd.Timestamp
    """The latest rebalance."""
    equity_weight: float
    bond_weight: float
    net_sharpe: float
    """The optimised portfolio's Sharpe ratio after trading costs."""
    baseline_net_sharpe: float
    """The 50/50 benchmark's, over the same days."""
    risk_level: int | None = None
    """The risk level the run was solved at."""
    forecast_value_label: str | None = None
    """Whether the forecasts added value against the same level with none, after
    costs, as the Portfolio page states it; ``None`` before the page has said."""
    forecast_value_tone: str = "neutral"
    forecast_value_t: str = "—"
    solve_inputs: SolveInputs | None = None
    """What the latest rebalance's target was solved from, so the Scenarios page can
    re-solve it on shocked forecasts (FYP-246). ``None`` where it couldn't be rebuilt."""
