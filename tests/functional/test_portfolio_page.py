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


# --- FYP-19: performance against the equal-weight benchmark -------------------------


def _happy_page() -> AppTest:
    committed = _committed()
    _set_active(EQUITY, forecast=_forecast(committed, seed=1))
    _set_active(BOND, forecast=_forecast(committed, seed=2))
    return _page(committed)


def _comparison_table(app: AppTest) -> str:
    return next(m.value for m in app.markdown if '<table class="fe-table"' in m.value)


def _captions(app: AppTest) -> str:
    return " ".join(c.value for c in app.caption)


def test_the_four_ratios_are_shown_for_both_portfolios_side_by_side():
    app = _happy_page()

    assert not app.exception
    table = _comparison_table(app)
    for heading in ("Optimised", "Equal-weight benchmark", "Difference"):
        assert f"<th>{heading}</th>" in table
    for metric in ("Sharpe", "Sortino", "Calmar", "Max drawdown"):
        assert f">{metric}<span" in table


def test_the_backtest_date_range_rebalancing_and_costs_are_stated():
    app = _happy_page()

    captions = _captions(app)
    assert "Backtest " in captions and " to " in captions
    assert "every 10 trading days" in captions
    assert "reset to 50/50 monthly" in captions
    assert "3 bp equity, 5 bp bond" in captions
    assert "S&P 500: Polynomial" in captions


def test_after_costs_is_the_default_and_before_costs_can_be_chosen():
    app = _happy_page()
    (basis,) = [c for c in app.segmented_control if c.label == "Returns"]
    assert basis.value == "After costs"
    net = _comparison_table(app)

    basis.set_value("Before costs").run()

    assert not app.exception
    assert _comparison_table(app) != net


def test_historical_var_and_cvar_are_shown_beside_max_drawdown():
    app = _happy_page()

    tables = [m.value for m in app.markdown if '<table class="fe-table"' in m.value]
    assert len(tables) == 2
    tail = tables[1]
    for metric in ("1-day VaR 95%", "1-day CVaR 99%", "Max drawdown", "breach rate"):
        assert metric in tail
    assert any("Historical" in m.value and "Tail risk" in m.value for m in app.markdown)
    assert "one-day losses read from the realised returns" in _captions(app)


def test_the_cumulative_return_chart_is_drawn_beside_the_weights_chart():
    app = _happy_page()

    # The weights bar chart and the cumulative-return line chart.
    assert len(app.get("vega_lite_chart")) == 2


def test_without_active_models_the_page_explains_what_will_appear():
    app = _page(_committed())

    assert any("against the equal-weight benchmark appear here" in t for t in _infos(app))
    assert not [m for m in app.markdown if '<table class="fe-table"' in m.value]
