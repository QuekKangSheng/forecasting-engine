"""The Run history page, driven through the real Streamlit page script."""

from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.ingest.provenance import SourceFile
from forecasting_engine.ingest.upload import (
    AcceptedUpload,
    accept_upload,
    frame_csv_bytes,
    frame_sha256,
)
from forecasting_engine.portfolio.backtest import run_backtest
from forecasting_engine.reporting.model_metrics import ModelRunResult
from forecasting_engine.store import active_model, lineage, runs
from forecasting_engine.store.portfolio_runs import PortfolioInputs, record_portfolio_run
from forecasting_engine.store.uploads import record_upload
from forecasting_engine.validation.crash import CrashDiagnostics

PAGE = Path(__file__).resolve().parents[2] / "app" / "app_pages" / "5_Run_History.py"
T0 = datetime(2026, 10, 26, 9, 0)
DATA = pd.DataFrame({"Date": pd.bdate_range("2026-01-05", periods=3), "SPX": [1.0, 2.0, 3.0]})
DATES = pd.bdate_range("2026-01-05", periods=3)


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)


def _result() -> ModelRunResult:
    return ModelRunResult(
        ic=0.05,
        oos_rank_ic=0.04,
        rmse=0.012,
        crash=CrashDiagnostics(recall=0.5, precision=0.5, f1=0.5, n_true_tail_days=4),
        signal_rank_ic=0.06,
        signal_rank_ic_se=0.02,
        signal_rank_ic_se_test=0.03,
        forecast=pd.Series([0.01, 0.02, -0.01], index=DATES),
        baseline=pd.Series(0.005, index=DATES),
        realised=pd.Series([0.02, -0.01, 0.0], index=DATES),
    )


def _record(role=TargetRole.EQUITY, name="Polynomial (derived)", minutes=0, **overrides):
    kwargs = {
        "role": role,
        "model_name": name,
        "started_at": T0 + timedelta(minutes=minutes),
        "finished_at": T0 + timedelta(minutes=minutes, seconds=12),
        "dataset_sha256": frame_sha256(DATA),
        "config_key": "k" * 32,
        "settings": {"horizon": 20, "train_window": 504},
        "code_version": "abc1234",
    }
    if "error" not in overrides:
        kwargs["result"] = _result()
    return runs.record_run(**{**kwargs, **overrides})


def _page() -> AppTest:
    return AppTest.from_file(str(PAGE), default_timeout=30).run()


def _text(app) -> str:
    return " ".join(m.value for m in app.markdown)


def test_with_no_runs_says_where_they_come_from():
    app = _page()

    assert not app.exception
    assert any("Run a model on the Models page" in i.value for i in app.info)


def test_every_run_is_listed_newest_first():
    _record(minutes=0)
    _record(TargetRole.BOND, "Machine Learning", minutes=5)
    app = _page()

    assert not app.exception
    (table,) = app.dataframe[:1]
    assert list(table.value["Model"]) == ["Machine Learning", "Polynomial (derived)"]
    assert "2 runs" in _text(app)


def test_the_target_filter_shows_only_that_targets_runs():
    _record(minutes=0)
    _record(TargetRole.BOND, "Machine Learning", minutes=5)
    app = _page()

    app.segmented_control[0].set_value("Equity").run()

    assert list(app.dataframe[0].value["Target"]) == ["Equity"]


def test_opening_a_run_shows_its_settings_metrics_and_forecasts():
    _record()
    app = _page()

    assert not app.exception
    assert [m.value for m in app.metric][:3] == ["Done", "12.0 s", "abc1234"]
    assert app.json[0].value is not None
    metrics = app.dataframe[1].value
    assert metrics.set_index("Metric").loc["signal_rank_ic", "Value"] == pytest.approx(0.06)
    assert "Out-of-sample forecasts" in _text(app)


def test_a_failed_run_shows_its_error():
    _record(error="not enough complete training rows")
    app = _page()

    assert app.metric[0].value == "Failed"
    assert app.error[0].value == "not enough complete training rows"


def test_a_runs_data_is_traced_to_the_upload_holding_those_bytes(tmp_path):
    accepted = accept_upload("merged.csv", frame_csv_bytes(DATA), uploads_dir=tmp_path / "up")
    record_upload(accepted)
    _record()
    app = _page()

    text = _text(app)
    assert frame_sha256(DATA) in text
    assert "merged.csv, uploaded" in text and "3 rows" in text


def test_a_run_whose_data_was_never_logged_says_so():
    _record(dataset_sha256="f" * 64)
    app = _page()

    assert "no upload with this hash is in the upload log" in _text(app)


def test_the_same_settings_run_twice_is_counted():
    _record(minutes=0)
    _record(minutes=5)
    app = _page()

    assert app.metric[3].value == "2"


def test_a_run_shows_the_exports_its_dataset_was_merged_from(tmp_path):
    def kept(name, data):
        return AcceptedUpload(frame=DATA, source=SourceFile.of(name, data))

    merged = accept_upload("merged.csv", frame_csv_bytes(DATA), uploads_dir=tmp_path / "up")
    lineage.record_merge(
        merged, [(kept("spx.xlsx", b"spx"), lineage.TARGET), (kept("vix.xlsx", b"vix"), "signal")]
    )
    _record()
    app = _page()

    sources = app.dataframe[1].value
    assert list(sources["File"]) == ["spx.xlsx", "vix.xlsx"]
    assert list(sources["Role"]) == ["Target", "Signal"]
    assert "Merged from 2 export(s)" in _text(app)


def test_a_dataset_committed_before_exports_were_logged_says_so():
    _record()
    app = _page()

    assert any("committed before exports were logged" in c.value for c in app.caption)


def test_the_run_behind_an_active_model_is_marked_active():
    used = _record(minutes=0)
    _record(minutes=5)
    active_model.set_active_model(
        TargetRole.EQUITY,
        "Polynomial (derived)",
        _result(),
        high_risk=False,
        horizon=20,
        train_window=504,
        test_window=20,
        dataset_fingerprint="fp",
        run_id=used.run_id,
    )
    app = _page()

    assert list(app.dataframe[0].value["Active"]) == ["", "Equity"]
    assert any(o.endswith("· active") for o in app.selectbox[0].options)


# --- portfolio runs ------------------------------------------------------------------


def _portfolio_tab(app) -> str:
    return " ".join(m.value for m in app.tabs[1].markdown) + " ".join(
        i.value for i in app.tabs[1].info
    )


def test_with_no_portfolio_result_the_tab_says_where_they_come_from():
    app = _page()

    assert "Open the Portfolio Optimizer" in _portfolio_tab(app)


def test_each_distinct_portfolio_result_is_listed_once():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2024-01-31", periods=60)
    prices = {
        TargetRole.EQUITY: pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 60)), index=idx),
        TargetRole.BOND: pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.003, 60)), index=idx),
    }
    schedule = pd.DataFrame({TargetRole.EQUITY: [0.6, 0.5], TargetRole.BOND: [0.4, 0.5]},
                            index=idx[[0, 30]])
    result = run_backtest(prices, schedule)
    inputs = PortfolioInputs(
        dataset_sha256=frame_sha256(DATA), equity_run_id=None, bond_run_id=None,
        equity_set_at=T0, bond_set_at=T0, horizon=20, train_window=504, test_window=20,
        embargo=20, risk_aversion=10.0, bounds=(0.2, 0.8),
        costs_bps={"equity": 3.0, "bond": 5.0}, code_version="abc1234",
    )
    for _ in range(3):  # the page shown three times
        record_portfolio_run(inputs, result, schedule)
    record_portfolio_run(replace(inputs, risk_aversion=30.0), result, schedule)
    app = _page()

    table = app.tabs[1].dataframe[0].value
    assert sorted(table["Risk aversion λ"]) == [10.0, 30.0]
    assert "2 distinct results" in _portfolio_tab(app)
