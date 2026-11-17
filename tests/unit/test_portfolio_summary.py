"""Whether the Portfolio page's last result still describes the models active now:
Home and Scenarios read it without re-running the optimiser."""

from datetime import datetime

import pandas as pd

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.store.active_model import ActiveModelRecord
from portfolio_summary import PortfolioSummary

EQUITY, BOND = TargetRole.EQUITY, TargetRole.BOND
SET_AT = {EQUITY: datetime(2026, 11, 1, 9, 0), BOND: datetime(2026, 11, 1, 9, 5)}


def _record(role: TargetRole, set_at: datetime) -> ActiveModelRecord:
    return ActiveModelRecord(
        role=role,
        model_name="Polynomial (derived)",
        ic=0.05,
        oos_rank_ic=0.04,
        rmse=0.01,
        high_risk=False,
        set_at=set_at,
        horizon=20,
        train_window=504,
        test_window=20,
        dataset_fingerprint="data",
    )


def _summary(active_set_at=None) -> PortfolioSummary:
    return PortfolioSummary(
        dataset_fingerprint="data",
        as_of=pd.Timestamp("2026-10-30"),
        equity_weight=0.6,
        bond_weight=0.4,
        net_sharpe=0.9,
        baseline_net_sharpe=0.8,
        active_set_at=active_set_at,
    )


def test_a_summary_matches_the_models_it_was_computed_from():
    actives = {role: _record(role, set_at) for role, set_at in SET_AT.items()}

    assert _summary(SET_AT).models_match(actives)


def test_a_model_set_active_since_leaves_the_summary_out_of_date():
    actives = {
        EQUITY: _record(EQUITY, datetime(2026, 11, 2, 14, 0)),
        BOND: _record(BOND, SET_AT[BOND]),
    }

    assert not _summary(SET_AT).models_match(actives)


def test_an_index_with_no_active_model_now_does_not_match():
    actives = {EQUITY: _record(EQUITY, SET_AT[EQUITY]), BOND: None}

    assert not _summary(SET_AT).models_match(actives)


def test_a_summary_that_recorded_no_models_matches_none():
    actives = {role: _record(role, set_at) for role, set_at in SET_AT.items()}

    assert not _summary().models_match(actives)
