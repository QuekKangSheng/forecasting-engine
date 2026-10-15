"""The Portfolio Optimizer page's last result, kept in session state so Home can
say where the pipeline stands without re-running the optimiser or the backtest.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

#: Session-state key the Portfolio page writes and Home reads.
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
