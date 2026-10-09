"""Set-up shared by every test."""

import pytest

from forecasting_engine.parallel import WORKERS_ENV


@pytest.fixture(autouse=True)
def serial_runs(monkeypatch):
    """Model runs stay in the test's own process: many tests patch functions
    in-process, which a worker process would never see. Tests of the parallel
    paths set ``FORECASTING_WORKERS`` themselves."""
    monkeypatch.setenv(WORKERS_ENV, "1")
