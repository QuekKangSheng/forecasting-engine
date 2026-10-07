"""Shared set-up for the page tests."""

import pytest

import model_jobs


@pytest.fixture(autouse=True)
def isolated_data_dir(monkeypatch, tmp_path):
    """Saved runs, the last commit and the active-model DuckDB all live under a
    cwd-relative data/ folder; a test must never write to the real one."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def runs_fit_inline(monkeypatch):
    """A pressed Run fits in the test's own thread, so its results are there when
    the script returns, and every test starts with no Run on record."""
    monkeypatch.setattr(model_jobs, "INLINE", True)
    monkeypatch.setattr(model_jobs, "_jobs", {})
