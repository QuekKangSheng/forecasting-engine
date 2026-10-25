"""The Run history page, driven through the real Streamlit page script."""

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.ingest.upload import accept_upload, frame_csv_bytes, frame_sha256
from forecasting_engine.reporting.model_metrics import ModelRunResult
from forecasting_engine.store import runs
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
