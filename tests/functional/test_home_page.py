"""The Home page's pipeline status, driven through the real page script (FYP-218)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import bloomberg_extraction_panel
import model_runs
import portfolio_summary
from forecasting_engine.extraction import validation
from forecasting_engine.extraction.bloomberg_csv import ColumnSource
from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.models.base import ModelDescription
from forecasting_engine.reporting.model_metrics import ModelRunResult, ScreeningSummary
from forecasting_engine.reporting.polynomial_function import dataset_fingerprint
from forecasting_engine.store import active_model
from forecasting_engine.validation.crash import CrashDiagnostics

PAGE = Path(__file__).resolve().parents[2] / "app" / "app_pages" / "0_Home.py"
EQUITY, BOND = TargetRole.EQUITY, TargetRole.BOND
SPX = "SPX_Index_TOT_RETURN_INDEX_GROSS_DVDS"
AGG = "LBUSTRUU_Index_TOT_RETURN_INDEX_GROSS_DVDS"
NAIVE, DERIVED, ML = "Naive (training mean)", "Polynomial (derived)", "Machine Learning"


@pytest.fixture(autouse=True)
def isolated_active_model_db(monkeypatch, tmp_path):
    """The active-model store reads DuckDB at a default, cwd-relative path."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def nothing_is_fitted_or_backtested(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("Home must not fit a model or run a backtest")

    monkeypatch.setattr("forecasting_engine.validation.harness.evaluate", refuse)
    monkeypatch.setattr("forecasting_engine.portfolio.backtest.run_backtest", refuse)


def _committed() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 60
    return pd.DataFrame(
        {
            "Date": pd.bdate_range("2024-01-02", periods=n),
            SPX: 4000 + np.cumsum(rng.normal(size=n)),
            AGG: 2200 + np.cumsum(rng.normal(size=n)),
            "VIX_Index_PX_LAST": 15 + rng.normal(size=n),
            "LUACOAS_Index_PX_LAST": 1.2 + rng.normal(scale=0.1, size=n),
        }
    )


def _sources() -> dict:
    return {
        SPX: ColumnSource("SPX Index", "TOT_RETURN_INDEX_GROSS_DVDS"),
        AGG: ColumnSource("LBUSTRUU Index", "TOT_RETURN_INDEX_GROSS_DVDS"),
        "VIX_Index_PX_LAST": ColumnSource("VIX Index", "PX_LAST"),
        "LUACOAS_Index_PX_LAST": ColumnSource("LUACOAS Index", "PX_LAST"),
    }


def _page(committed=None, *, runs=None, summary=None) -> AppTest:
    app = AppTest.from_file(str(PAGE), default_timeout=30)
    if committed is not None:
        app.session_state[bloomberg_extraction_panel.COMMITTED_KEY] = committed
        app.session_state[bloomberg_extraction_panel.COMMITTED_REPORT_KEY] = (
            validation.validate(committed)
        )
        app.session_state[bloomberg_extraction_panel.COMMITTED_TARGETS_KEY] = {
            EQUITY: SPX,
            BOND: AGG,
        }
        app.session_state[bloomberg_extraction_panel.COMMITTED_SOURCES_KEY] = _sources()
    if runs is not None:
        app.session_state[model_runs.RUNS_KEY] = runs
    if summary is not None:
        app.session_state[portfolio_summary.SUMMARY_KEY] = summary
    return app.run()


def _result(signal_rank_ic: float, se: float = 0.01, screening=None) -> ModelRunResult:
    return ModelRunResult(
        ic=0.05,
        oos_rank_ic=signal_rank_ic,
        signal_rank_ic=signal_rank_ic,
        signal_rank_ic_se=se,
        signal_rank_ic_se_test=se,
        rmse=0.01,
        crash=CrashDiagnostics(recall=0.5, precision=0.5, f1=0.5, n_true_tail_days=4),
        screening=screening,
    )


def _run(result: ModelRunResult) -> model_runs.ModelRun:
    return model_runs.ModelRun(result, ModelDescription("Test", (), ()))


def _stored_runs(committed: pd.DataFrame) -> model_runs.StoredRuns:
    screening = ScreeningSummary(
        folds=4,
        fell_back=0,
        counts=(("VIX_Index_PX_LAST", 4), ("LUACOAS_Index_PX_LAST", 1)),
        latest_included=("VIX_Index_PX_LAST",),
    )
    stored = model_runs.StoredRuns((dataset_fingerprint(committed), 20, 504, 20))
    stored.tabs[EQUITY] = model_runs.TabRuns(
        {},
        {
            NAIVE: _run(_result(float("nan"))),
            DERIVED: _run(_result(0.04, screening=screening)),
            ML: _run(_result(0.01, screening=screening)),
        },
    )
    return stored


def _markdown(app: AppTest) -> str:
    return " ".join(m.value for m in app.markdown)


def _captions(app: AppTest) -> str:
    return " ".join(c.value for c in app.caption)


def _next_step(app: AppTest) -> str:
    return next(i.value for i in app.info if i.value.startswith("**Next step:**"))


def test_with_nothing_committed_the_next_step_is_the_data_page():
    app = _page()

    assert not app.exception
    assert "**Data** page" in _next_step(app)
    assert [s.value for s in app.subheader] == ["Data", "Models", "Portfolio Optimizer"]
    assert "Nothing committed yet." in _captions(app)
    assert "Not run yet." in _captions(app)


def test_with_data_only_the_data_card_is_filled_and_the_next_step_is_to_run_models():
    committed = _committed()
    app = _page(committed)

    assert not app.exception
    captions = _captions(app)
    assert "02 Jan 2024 to 25 Mar 2024" in captions
    assert "2 signals" in captions
    assert "quality issue" in captions and "flagged → Data" in captions
    markdown = _markdown(app)
    assert "Equity target" in markdown and "Bond target" in markdown
    assert markdown.count("✓") == 2  # both targets; no active models yet
    assert "run the equity models on the **Models** page" in _next_step(app)


def test_models_that_ran_are_listed_with_their_gate_signal_rank_ic_and_t():
    committed = _committed()
    app = _page(committed, runs=_stored_runs(committed))

    assert not app.exception
    markdown = _markdown(app)
    assert "Passed" in markdown and "Failed" in markdown
    assert "Signal Rank IC 0.040 · t 4.00" in markdown
    assert "Signal Rank IC 0.010 · t 1.00" in markdown
    assert "1 of 2 signals passed screening in the latest fold" in _captions(app)
    assert "set an active equity model on the **Models** page" in _next_step(app)


def test_a_stale_active_model_and_portfolio_run_are_flagged_amber():
    committed = _committed()
    for role in (EQUITY, BOND):
        active_model.set_active_model(
            role,
            DERIVED,
            _result(0.04),
            high_risk=False,
            horizon=20,
            train_window=504,
            test_window=20,
            dataset_fingerprint="an earlier upload",
        )
    summary = portfolio_summary.PortfolioSummary(
        dataset_fingerprint="an earlier upload",
        as_of=pd.Timestamp("2024-03-01"),
        equity_weight=0.6,
        bond_weight=0.4,
        net_sharpe=0.85,
        baseline_net_sharpe=0.83,
    )
    app = _page(committed, runs=_stored_runs(committed), summary=summary)

    assert not app.exception
    warnings = [w.value for w in app.warning]
    assert any("fitted on different data" in w for w in warnings)
    assert any("last run used different data" in w for w in warnings)
    assert "re-run the equity models" in _next_step(app)
    assert "Net Sharpe 0.85 against 0.83 for 50/50." in _captions(app)


def test_an_up_to_date_pipeline_says_there_is_nothing_left_to_do():
    committed = _committed()
    for role in (EQUITY, BOND):
        active_model.set_active_model(
            role,
            DERIVED,
            _result(0.04),
            high_risk=False,
            horizon=20,
            train_window=504,
            test_window=20,
            dataset_fingerprint=str(dataset_fingerprint(committed)),
        )
    summary = portfolio_summary.PortfolioSummary(
        dataset_fingerprint=str(dataset_fingerprint(committed)),
        as_of=pd.Timestamp("2024-03-01"),
        equity_weight=0.6,
        bond_weight=0.4,
        net_sharpe=0.85,
        baseline_net_sharpe=0.83,
    )
    app = _page(committed, runs=_stored_runs(committed), summary=summary)

    assert not app.exception
    assert not app.warning
    assert "nothing" in _next_step(app)
    assert "Latest weights (01 Mar 2024): 60% equity, 40% bond." in _captions(app)
