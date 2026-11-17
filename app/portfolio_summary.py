"""The Portfolio Optimizer page's last result, kept in session state so Home can
say where the pipeline stands, and the Scenarios page can carry a shock through to
the portfolio, without re-running the optimiser or the backtest.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.portfolio.optimize import SolveInputs
from forecasting_engine.store.active_model import ActiveModelRecord

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
    active_set_at: Mapping[TargetRole, datetime] | None = None
    """When each active model the run combined was set active, the key every
    active-model row is stored under. Setting another active since leaves these
    weights and Sharpe ratios out of date, as new data does. ``None`` if not
    recorded, which matches no active models."""

    def models_match(self, actives: Mapping[TargetRole, ActiveModelRecord | None]) -> bool:
        """Whether ``actives``, the models active now, are the ones the run combined."""
        now = {role: record.set_at for role, record in actives.items() if record is not None}
        return self.active_set_at is not None and now == dict(self.active_set_at)
