"""Running independent pieces in processes gives exactly the serial answer."""

import sys
import threading
import types

import numpy as np
import pandas as pd
import pytest

from forecasting_engine.ingest.align import FeaturePanel
from forecasting_engine.models.boosted import fits_tuning_period, run_boosted
from forecasting_engine.models.polynomial import run_derived_polynomial
from forecasting_engine.parallel import DEFAULT_MAX_WORKERS, WORKERS_ENV, map_processes, workers
from forecasting_engine.validation.pbo import compute_pbo
from forecasting_engine.validation.splitters import PurgedWalkForward


def _square(x: int) -> int:
    return x * x


def test_workers_follow_the_setting_and_never_exceed_the_tasks(monkeypatch):
    monkeypatch.setenv(WORKERS_ENV, "3")
    assert workers(10) == 3
    assert workers(2) == 2
    monkeypatch.setenv(WORKERS_ENV, "0")
    assert workers(10) == 1


def test_without_a_setting_a_few_workers_at_most(monkeypatch):
    monkeypatch.delenv(WORKERS_ENV)
    assert 1 <= workers(100) <= DEFAULT_MAX_WORKERS


def test_processes_return_results_in_order(monkeypatch):
    monkeypatch.setenv(WORKERS_ENV, "2")
    assert map_processes(_square, [3, 1, 2]) == [9, 1, 4]


def _pbo_configs(n: int = 600) -> dict:
    rng = np.random.default_rng(0)
    index = pd.date_range("2024-01-01", periods=n, freq="D")
    realised = pd.Series(rng.normal(size=n), index=index)
    return {
        f"setup_{i}": (realised * 0.1 * i + pd.Series(rng.normal(size=n), index=index), realised)
        for i in range(4)
    }


def test_pbo_shared_across_processes_matches_the_serial_count(monkeypatch):
    configs = _pbo_configs()
    serial = compute_pbo(configs, n_blocks=8)
    monkeypatch.setenv(WORKERS_ENV, "2")
    shared = compute_pbo(configs, n_blocks=8)

    assert shared == serial


def _panel(n: int = 120) -> FeaturePanel:
    rng = np.random.default_rng(2)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    a, b = rng.normal(size=n), rng.normal(size=n)
    target = 2 * a - b + rng.normal(scale=0.5, size=n)
    frame = pd.DataFrame({"a": a, "b": b, "target": target}, index=idx)
    return FeaturePanel(frame=frame, signals=("a", "b"), targets=("target",), lag_days=1)


def test_the_polynomials_configurations_in_parallel_match_running_them_in_turn(monkeypatch):
    splitter = PurgedWalkForward(train=30, test=8, embargo=2)
    serial, _ = run_derived_polynomial(_panel(), splitter, n_blocks=4)
    monkeypatch.setenv(WORKERS_ENV, "3")
    parallel, _ = run_derived_polynomial(_panel(), splitter, n_blocks=4)

    pd.testing.assert_series_equal(parallel.forecast, serial.forecast)
    assert (parallel.pbo, parallel.signal_rank_ic) == (serial.pbo, serial.signal_rank_ic)


def test_the_two_libraries_tuned_in_parallel_match_tuning_them_in_turn(monkeypatch):
    splitter = PurgedWalkForward(train=20, test=5, embargo=2, tuning_rows=45)
    kwargs = {"n_trials": 3, "retune_trials": 3, "n_blocks": 4}
    serial, _, serial_log = run_boosted(_panel(), splitter, **kwargs)
    monkeypatch.setenv(WORKERS_ENV, "2")
    parallel, _, parallel_log = run_boosted(_panel(), splitter, **kwargs)

    pd.testing.assert_series_equal(parallel.forecast, serial.forecast)
    assert parallel_log == serial_log


@pytest.mark.parametrize(("train", "fits"), [(126, True), (252, True), (504, False)])
def test_machine_learning_needs_room_for_its_folds_inside_the_tuning_period(train, fits):
    assert fits_tuning_period(train, test=20, embargo=20, tuning_rows=504) is fits


def test_workers_never_run_the_page_installed_as_main(monkeypatch, tmp_path):
    # Streamlit installs each page script as __main__, and the Models page's runs
    # start from a background thread; a worker that re-ran the page would crash.
    page = tmp_path / "page.py"
    page.write_text("raise SystemExit('a worker ran the page script')\n")
    fake_main = types.ModuleType("__main__")
    fake_main.__file__ = str(page)
    monkeypatch.setitem(sys.modules, "__main__", fake_main)
    monkeypatch.setenv(WORKERS_ENV, "2")
    results = []

    worker = threading.Thread(target=lambda: results.append(map_processes(_square, [2, 3])))
    worker.start()
    worker.join()

    assert results == [[4, 9]]
    assert sys.modules["__main__"] is fake_main
