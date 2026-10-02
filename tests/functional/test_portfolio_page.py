"""The Portfolio Optimizer page, driven through the real Streamlit page script."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import bloomberg_extraction_panel
from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.reporting.model_metrics import ModelRunResult
from forecasting_engine.store import active_model
from forecasting_engine.validation.crash import CrashDiagnostics

REPO_ROOT = Path(__file__).resolve().parents[2]
PAGE = REPO_ROOT / "app" / "app_pages" / "3_Portfolio.py"

EQUITY, BOND = TargetRole.EQUITY, TargetRole.BOND
EQUITY_COL = "SPX_Index_PX_LAST"
BOND_COL = "LBUSTRUU_Index_TOT_RETURN_INDEX_GROSS_DVDS"
N_DAYS = 60

SETTINGS = {"horizon": 3, "train_window": 15, "test_window": 10, "dataset_fingerprint": "fp-1"}


@pytest.fixture(autouse=True)
def isolated_active_model_db(monkeypatch, tmp_path):
    """The optimiser reads/writes DuckDB at a default, cwd-relative path."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def no_tuning_period(monkeypatch):
    """Rebalance dates are built with the harness's own TUNING_ROWS (504) in
    the loop, same as a real model run — too large for this small fixture."""
    monkeypatch.setattr("forecasting_engine.portfolio.optimize.TUNING_ROWS", 0)


def _committed() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "Date": pd.bdate_range("2024-01-02", periods=N_DAYS),
            EQUITY_COL: 100 * np.cumprod(1 + rng.normal(0, 0.01, N_DAYS)),
            BOND_COL: 100 * np.cumprod(1 + rng.normal(0, 0.01, N_DAYS)),
        }
    )


def _forecast(committed, seed=0, n_dates=40) -> pd.Series:
    dates = pd.DatetimeIndex(committed["Date"])[-n_dates:]
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0, 0.02, n_dates), index=dates)


def _result(forecast=None) -> ModelRunResult:
    return ModelRunResult(
        ic=0.05,
        oos_rank_ic=0.04,
        rmse=0.01,
        pbo=0.3,
        crash=CrashDiagnostics(recall=0.5, precision=0.5, f1=0.5, n_true_tail_days=4),
        forecast=forecast,
    )


def _set_active(role, forecast=None, **overrides):
    active_model.set_active_model(
        role, "Polynomial", _result(forecast=forecast), high_risk=False, **{**SETTINGS, **overrides}
    )


def _page(committed) -> AppTest:
    app = AppTest.from_file(str(PAGE), default_timeout=60)
    app.session_state[bloomberg_extraction_panel.COMMITTED_KEY] = committed
    app.session_state[bloomberg_extraction_panel.COMMITTED_TARGETS_KEY] = {
        EQUITY: EQUITY_COL,
        BOND: BOND_COL,
    }
    return app.run()


def _infos(app) -> list[str]:
    return [i.value for i in app.info]


# --- gating ------------------------------------------------------------------------


def test_with_no_active_models_names_both_as_missing():
    app = _page(_committed())

    assert not app.exception
    assert any("Equity and Bond" in text for text in _infos(app))


def test_with_only_bond_active_names_equity_as_missing():
    _set_active(BOND, forecast=_forecast(_committed()))
    app = _page(_committed())

    assert not app.exception
    assert any("Equity" in text and "Bond" not in text for text in _infos(app))


def test_mismatched_settings_blocks_with_an_explanation():
    committed = _committed()
    _set_active(EQUITY, forecast=_forecast(committed), train_window=15)
    _set_active(BOND, forecast=_forecast(committed), train_window=20)
    app = _page(committed)

    assert not app.exception
    assert any("different settings" in w.value for w in app.warning)


def test_with_no_committed_data_shows_a_guiding_message():
    committed = _committed()
    _set_active(EQUITY, forecast=_forecast(committed))
    _set_active(BOND, forecast=_forecast(committed))
    app = AppTest.from_file(str(PAGE), default_timeout=60).run()

    assert not app.exception
    assert any("Data page" in text for text in _infos(app))


def test_with_no_saved_forecast_shows_a_guiding_message():
    committed = _committed()
    _set_active(EQUITY, forecast=None)
    _set_active(BOND, forecast=_forecast(committed))
    app = _page(committed)

    assert not app.exception
    assert any("wasn't saved" in text for text in _infos(app))


# --- happy path ----------------------------------------------------------------


def test_the_happy_path_shows_latest_weights_summing_to_one():
    committed = _committed()
    _set_active(EQUITY, forecast=_forecast(committed, seed=1))
    _set_active(BOND, forecast=_forecast(committed, seed=2))
    app = _page(committed)

    assert not app.exception
    assert len(app.metric) == 2
    values = [float(m.value.strip("%")) / 100 for m in app.metric]
    assert sum(values) == pytest.approx(1.0, abs=1e-6)
