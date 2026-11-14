"""Set-up shared by every test."""

import pytest
import streamlit as st

from forecasting_engine.parallel import WORKERS_ENV


@pytest.fixture(autouse=True)
def nothing_kept_between_tests():
    """The pages keep solved allocations and store reads between reruns
    (``app/page_cache``), keyed on their inputs' contents. Tests share those inputs
    (the same fixture data, and patched module constants no key can see), so each
    starts with nothing kept."""
    st.cache_data.clear()
    st.cache_resource.clear()


@pytest.fixture(autouse=True)
def serial_runs(monkeypatch):
    """Model runs stay in the test's own process: many tests patch functions
    in-process, which a worker process would never see. Tests of the parallel
    paths set ``FORECASTING_WORKERS`` themselves."""
    monkeypatch.setenv(WORKERS_ENV, "1")
