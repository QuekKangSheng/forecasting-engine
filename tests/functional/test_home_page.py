"""The Home page's pipeline status, driven through the real page script (FYP-218)."""

import html
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from streamlit.testing.v1 import AppTest

import bloomberg_extraction_panel
import model_runs
import portfolio_summary
from forecasting_engine.extraction import validation
from forecasting_engine.extraction.bloomberg_csv import ColumnSource
from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.ingest import index_feed
from forecasting_engine.ingest.index_feed import Candles, IndexFeedError
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
def no_live_feed(monkeypatch):
    """No test asks the network: the live feed does not answer unless a test says so."""

    def offline(span, **kwargs):
        raise IndexFeedError("the live feed did not answer (offline in tests)")

    monkeypatch.setattr(index_feed, "fetch", offline)


def _live_feed(monkeypatch) -> list[str]:
    """The live feed answering with three monthly candles; the spans asked for."""
    asked = []

    def answer(span, **kwargs):
        asked.append(span)
        frame = pd.DataFrame(
            {
                "open": [7500.0, 7690.0, 7650.0],
                "high": [7820.0, 7790.0, 7700.0],
                "low": [7500.0, 7510.0, 7350.0],
                "close": [7686.0, 7650.0, 7418.2],
            },
            index=pd.date_range("2026-08-01", periods=3, freq="MS"),
        )
        return Candles(frame, "M", "Yahoo Finance", True, 7418.2, pd.Timestamp("2026-11-16 16:00"))

    monkeypatch.setattr(index_feed, "fetch", answer)
    return asked


def _charts(app: AppTest) -> list[dict]:
    return [json.loads(chart.proto.spec) for chart in app.get("vega_lite_chart")]


def _candle_rows(app: AppTest) -> pd.DataFrame:
    """The market chart's candles, as sent to the browser beside its specification."""
    (chart,) = app.get("vega_lite_chart")
    (table,) = [d for d in chart.proto.datasets if d.name.startswith("candles")]
    return pa.ipc.open_stream(table.data.data).read_all().to_pandas()


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


def _heading(app: AppTest) -> str:
    """What the opening says of where things stand, under the name, as text."""
    opening = next(m.value for m in app.markdown if 'aria-level="1"' in m.value)
    return html.unescape(re.search(r'class="fe-hero-state[^"]*">(.*?)</div>', opening).group(1))


def _next_step(app: AppTest) -> str:
    """The next step, which is the heading while one is left."""
    opening = next(m.value for m in app.markdown if 'aria-level="1"' in m.value)
    assert 'class="fe-hero-label">Next step<' in opening
    return _heading(app)


def test_with_nothing_committed_the_next_step_is_the_data_page():
    app = _page()

    assert not app.exception
    assert "Data page" in _next_step(app)
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
    assert _next_step(app) == "Run the equity models on the Models page."


def test_models_that_ran_are_listed_with_their_gate_signal_rank_ic_and_t():
    committed = _committed()
    app = _page(committed, runs=_stored_runs(committed))

    assert not app.exception
    markdown = _markdown(app)
    assert "Passed" in markdown and "Failed" in markdown
    assert "Signal Rank IC 0.040 · t 4.00" in markdown
    assert "Signal Rank IC 0.010 · t 1.00" in markdown
    assert "1 of 2 signals passed screening in the latest fold" in _captions(app)
    assert _next_step(app) == "Set an active equity model on the Models page."


def test_a_target_named_with_html_in_its_column_shows_the_name_as_text():
    """Column names come from the uploaded file: a crafted one must not insert an
    element where the page writes its own HTML."""
    hostile = "EVIL_Index_<img src=x onerror=alert(1)>"
    app = AppTest.from_file(str(PAGE), default_timeout=30)
    app.session_state[bloomberg_extraction_panel.COMMITTED_KEY] = _committed().rename(
        columns={SPX: hostile}
    )
    app.session_state[bloomberg_extraction_panel.COMMITTED_TARGETS_KEY] = {
        EQUITY: hostile,
        BOND: AGG,
    }
    app.run()

    raw_html = " ".join(m.value for m in app.markdown if m.proto.allow_html)
    assert not app.exception
    assert "Equity · EVIL Index (&lt;img src=x onerror=alert(1)&gt;)" in raw_html
    assert "<img" not in raw_html


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
    assert _next_step(app).startswith("Re-run the equity models")
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
        active_set_at={role: active_model.get_active_model(role).set_at for role in (EQUITY, BOND)},
    )
    app = _page(committed, runs=_stored_runs(committed), summary=summary)

    assert not app.exception
    assert not app.warning
    assert _heading(app) == "60% equity, 40% bonds."
    assert 'class="fe-hero-label">Next step<' not in _markdown(app)
    assert not app.success  # no banner: the typed line says it
    assert "2 of 2 models active · up to date</span>" in _markdown(app)
    # The latest result leads the page, ahead of the pipeline cards.
    strip = next(m.value for m in app.markdown if '<div class="fe-strip' in m.value)
    assert "0.83 for 50/50" in strip
    assert "Latest weights (01 Mar 2024): 60% equity, 40% bond." in _captions(app)


def test_a_portfolio_run_on_an_active_model_since_replaced_is_flagged_amber():
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
        # The run combined an equity model that another has since replaced.
        active_set_at={
            EQUITY: datetime(2024, 3, 1, 9, 30),
            BOND: active_model.get_active_model(BOND).set_at,
        },
    )
    app = _page(committed, runs=_stored_runs(committed), summary=summary)

    assert not app.exception
    warnings = [w.value for w in app.warning]
    assert any("last run used different active models" in w for w in warnings)
    assert _next_step(app).startswith("Open the Portfolio Optimizer")
    # Its weights no longer lead the page as the latest result.
    assert not [m for m in app.markdown if '<div class="fe-strip' in m.value]


def test_an_active_model_saved_under_its_old_name_shows_the_new_one():
    committed = _committed()
    active_model.set_active_model(
        EQUITY,
        "Research model (fixed inputs)",
        _result(0.04),
        high_risk=False,
        horizon=20,
        train_window=504,
        test_window=20,
        dataset_fingerprint=str(dataset_fingerprint(committed)),
    )
    app = _page(committed, runs=_stored_runs(committed))

    assert not app.exception
    captions = _captions(app)
    assert "Polynomial – fixed economic inputs" in captions
    assert "Research model (fixed inputs)" not in captions


def _set_both_active(committed: pd.DataFrame, targets: dict | None = None) -> None:
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
            target_column=(targets or {}).get(role),
        )


def test_after_the_data_is_cleared_the_last_allocation_is_not_shown_as_current():
    committed = _committed()
    _set_both_active(committed)
    summary = portfolio_summary.PortfolioSummary(
        dataset_fingerprint=str(dataset_fingerprint(committed)),
        as_of=pd.Timestamp("2024-03-01"),
        equity_weight=0.6,
        bond_weight=0.4,
        net_sharpe=0.85,
        baseline_net_sharpe=0.83,
        active_set_at={role: active_model.get_active_model(role).set_at for role in (EQUITY, BOND)},
    )

    app = _page(None, summary=summary)

    assert not app.exception
    assert not [m for m in app.markdown if '<div class="fe-strip' in m.value]
    assert "Data page" in _next_step(app)


def test_a_model_fitted_on_another_target_column_is_flagged_on_home():
    committed = _committed()
    # Fitted when the price field was the equity target; the total return is now.
    _set_both_active(committed, {EQUITY: "SPX_Index_PX_LAST", BOND: AGG})

    app = _page(committed)

    assert not app.exception
    assert any("fitted on different data" in w.value for w in app.warning)
    assert _next_step(app).startswith("Re-run the equity models")

def _long_committed(days: int = 220) -> pd.DataFrame:
    """Enough months for the field to draw, the index rising a little each day."""
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "Date": pd.bdate_range("2024-01-02", periods=days),
            SPX: 4000 + np.arange(days, dtype=float),
            AGG: 2200 + np.cumsum(rng.normal(size=days)),
            "VIX_Index_PX_LAST": 15 + rng.normal(size=days),
            "LUACOAS_Index_PX_LAST": 1.2 + rng.normal(scale=0.1, size=days),
        }
    )


def test_the_page_opens_on_the_name_then_the_step_left_while_one_is():
    app = _page(_committed())

    assert _markdown(app).count('aria-level="1"') == 1
    assert 'aria-level="1" aria-label="Finlytics"' in _markdown(app)
    assert _heading(app) == "Run the equity models on the Models page."
    assert not app.info  # said once, as the heading


def test_the_opening_says_what_the_app_is_working_from():
    markdown = _markdown(_page(_committed()))

    assert ">60 days · 2 signals · 0 of 2 models active</span>" in markdown


def test_with_nothing_committed_and_no_feed_the_opening_says_so_and_draws_nothing():
    app = _page()

    assert ">waiting for data: upload exports on the Data page</span>" in _markdown(app)
    assert '<div class="fe-tape"' not in _markdown(app)
    assert not _charts(app)
    assert "no S&P 500 is committed to draw" in _captions(app)


def test_the_s_and_p_is_drawn_live_from_the_feed_over_five_years_by_default(monkeypatch):
    asked = _live_feed(monkeypatch)
    app = _page(_committed())

    assert not app.exception
    assert asked == ["5Y"]
    assert "candles_hit" in json.dumps(_charts(app))
    assert _candle_rows(app)["close"].tolist() == [7686.0, 7650.0, 7418.2]
    assert "Live · Yahoo Finance · 16 Nov 2026 16:00 New York" in _markdown(app)
    assert ">7,418.20<" in _markdown(app)


def test_picking_a_span_asks_the_feed_for_it(monkeypatch):
    asked = _live_feed(monkeypatch)
    app = _page(_committed())

    (span,) = [c for c in app.segmented_control if c.label == "Span"]
    assert list(span.options) == ["3M", "1Y", "5Y", "10Y"]
    span.set_value("1Y").run()

    assert not app.exception
    assert asked[-1] == "1Y"


def test_without_the_feed_the_committed_s_and_p_is_drawn_and_said_to_be():
    app = _page(_long_committed())

    assert not app.exception
    months = pd.bdate_range("2024-01-02", periods=220).to_period("M").nunique()
    assert len(_candle_rows(app)) == months
    assert "From the committed data · to" in _markdown(app)


def test_the_tape_shows_both_targets_then_each_signal_with_its_change_over_a_horizon():
    markdown = _markdown(_page(_long_committed()))

    tape = markdown[markdown.index('<div class="fe-tape"') :]
    run = tape[: tape.index('<span class="fe-tape-run" aria-hidden="true">')]
    names = ["S&amp;P 500", "US Aggregate Bond", "US IG credit spread", "VIX"]
    places = [run.index(f"<b>{name}</b>") for name in names]
    assert places == sorted(places)
    assert "change over 20 trading days" in tape
    # The index rose a point a day from 4,199 to 4,219 over the last twenty.
    assert "▲ 0.5%" in run


def test_a_model_fitted_on_other_data_is_not_counted_as_active():
    committed = _committed()
    active_model.set_active_model(
        EQUITY,
        ML,
        _result(0.04),
        high_risk=False,
        horizon=20,
        train_window=504,
        test_window=20,
        dataset_fingerprint="some other dataset",
    )
    active_model.set_active_model(
        BOND,
        ML,
        _result(0.04),
        high_risk=False,
        horizon=20,
        train_window=504,
        test_window=20,
        dataset_fingerprint=str(dataset_fingerprint(committed)),
    )

    markdown = _markdown(_page(committed))

    assert "1 of 2 models active</span>" in markdown
