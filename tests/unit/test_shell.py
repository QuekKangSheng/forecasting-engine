"""What the status bar says about the data, the active models and the allocation."""

from datetime import datetime

import pandas as pd

import shell
from forecasting_engine.extraction.bloomberg_csv import DATE_COLUMN
from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.reporting.polynomial_function import dataset_fingerprint
from forecasting_engine.store.active_model import ActiveModelRecord
from portfolio_summary import PortfolioSummary

EQUITY, BOND = TargetRole.EQUITY, TargetRole.BOND


def _data() -> pd.DataFrame:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-03-28"])
    return pd.DataFrame({DATE_COLUMN: dates, "SPX": [1.0, 2.0, 3.0], "AGG": [3.0, 2.0, 1.0]})


def _record(role: TargetRole, name: str, fingerprint: str) -> ActiveModelRecord:
    return ActiveModelRecord(
        role=role,
        model_name=name,
        ic=0.1,
        oos_rank_ic=0.1,
        rmse=0.01,
        high_risk=False,
        set_at=datetime(2026, 11, 6, 1, 37),
        horizon=20,
        train_window=504,
        test_window=20,
        dataset_fingerprint=fingerprint,
    )


def _summary(fingerprint: str, **changes) -> PortfolioSummary:
    fields = {
        "dataset_fingerprint": fingerprint,
        "as_of": pd.Timestamp("2026-08-11"),
        "equity_weight": 0.23,
        "bond_weight": 0.77,
        "net_sharpe": 0.75,
        "baseline_net_sharpe": 0.83,
        "risk_level": 3,
    }
    return PortfolioSummary(**{**fields, **changes})


def _cells(merged, actives=None, summary=None) -> dict[str, shell.StatusCell]:
    actives = actives or {EQUITY: None, BOND: None}
    return {cell.key: cell for cell in shell.status_cells(merged, actives, summary)}


def test_with_nothing_committed_every_cell_says_what_is_missing():
    cells = _cells(None)

    assert list(cells) == ["Data", "Equity", "Bond", "Allocation"]
    assert cells["Data"].value == "none committed" and cells["Data"].tone == "warning"
    assert cells["Equity"].value == cells["Bond"].value == "not set"
    assert cells["Allocation"].value == "not solved yet"


def test_committed_data_is_named_by_its_first_and_last_day():
    data = _cells(_data())["Data"]

    assert data.value == "02 Jan 2024 → 28 Mar 2024"
    assert data.tone == "success"


def test_an_active_model_fitted_on_the_committed_data_is_named():
    merged = _data()
    fingerprint = str(dataset_fingerprint(merged))

    cells = _cells(merged, {EQUITY: _record(EQUITY, "Polynomial", fingerprint), BOND: None})

    assert (cells["Equity"].value, cells["Equity"].tone) == ("Polynomial", "success")
    assert cells["Bond"].value == "not set"


def test_an_active_model_fitted_on_other_data_is_called_stale_in_words():
    """Its forecasts are not used; the dot's colour alone must not be what says so."""
    cells = _cells(_data(), {EQUITY: _record(EQUITY, "Polynomial", "another dataset"), BOND: None})

    assert cells["Equity"].value == "Polynomial · stale"
    assert cells["Equity"].tone == "warning"


def test_the_last_allocation_is_shown_with_its_level_weights_and_date():
    merged = _data()

    allocation = _cells(merged, summary=_summary(str(dataset_fingerprint(merged))))["Allocation"]

    assert allocation.value == "level 3 · 23% / 77% · 11 Aug 2026"
    assert allocation.tone == "success"


def test_an_allocation_solved_on_other_data_is_called_stale():
    allocation = _cells(_data(), summary=_summary("another dataset"))["Allocation"]

    assert "stale" in allocation.value and allocation.tone == "warning"


def test_the_bar_escapes_what_it_shows():
    """A model's name comes from a saved run, and a run's name from whoever saved it."""
    bar = shell.status_bar([shell.StatusCell("Equity", '<img src=x onerror="alert(1)">')])

    assert "<img" not in bar
    assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;" in bar


def test_an_unknown_tone_falls_back_to_neutral():
    bar = shell.status_bar([shell.StatusCell("Data", "x", tone="not a tone")])

    assert "var(--fe-neutral-fg)" in bar
