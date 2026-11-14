"""The background warming the app does once as it starts."""

import os
import subprocess
import sys
import threading
from pathlib import Path

import warmup

REPO = Path(__file__).resolve().parents[2]


def _pages(tmp_path: Path, **sources: str) -> Path:
    for name, source in sources.items():
        (tmp_path / f"{name}.py").write_text(source, encoding="utf-8")
    return tmp_path


def test_the_modules_to_warm_are_read_from_the_pages_own_imports(tmp_path):
    pages = _pages(
        tmp_path,
        first="from __future__ import annotations\nimport json\nfrom os import path\n",
        second="import json\nimport csv as c\n\ndef late():\n    import decimal\n",
    )

    assert warmup.page_imports(pages) == ["json", "os", "os.path", "csv", "decimal"]


def test_a_name_imported_from_a_module_is_warmed_if_it_is_a_module_and_skipped_if_not(
    tmp_path, monkeypatch
):
    pages = _pages(tmp_path, page="from os import path\nfrom json import loads\n")
    monkeypatch.setattr(warmup, "_ALSO", ())
    monkeypatch.setattr(warmup, "_touch_store", lambda: None)

    # ``json.loads`` is a function, not a module: nothing to import, and no failure.
    assert warmup.run(pages) == []


def test_every_real_page_is_covered():
    names = warmup.page_imports()

    # The slowest imports a page makes, each the reason a first visit was slow.
    assert "bloomberg_extraction_panel" in names
    assert "forecasting_engine.portfolio.optimize" in names
    assert not [name for name in names if name.startswith("__future__")]


def test_a_step_that_fails_is_reported_and_the_rest_still_run(tmp_path, monkeypatch):
    pages = _pages(tmp_path, page="import json\nimport no_such_module_anywhere\n")
    touched = []
    monkeypatch.setattr(warmup, "_ALSO", ())
    monkeypatch.setattr(warmup, "_touch_store", lambda: touched.append(True))

    assert warmup.run(pages) == ["no_such_module_anywhere"]
    assert touched == [True]


def test_a_store_that_cannot_be_read_does_not_stop_the_app(tmp_path, monkeypatch):
    def broken():
        raise OSError("the database is locked")

    monkeypatch.setattr(warmup, "_ALSO", ())
    monkeypatch.setattr(warmup, "_touch_store", broken)

    assert warmup.run(_pages(tmp_path, page="import json\n")) == ["store"]


def test_warming_starts_once_however_many_sessions_ask(monkeypatch):
    started = []
    release = threading.Event()

    def slow():
        started.append(True)
        release.wait(5)

    monkeypatch.setattr(warmup, "run", slow)
    monkeypatch.setattr(warmup, "_thread", None)

    first, second = warmup.start(), warmup.start()
    release.set()
    first.join(5)

    assert first is second
    assert started == [True]
    assert first.daemon, "a thread still warming must not keep a stopped app alive"


def test_the_models_module_no_longer_loads_shap_until_a_model_is_explained():
    """``shap`` takes seconds to import, and every page imports this module."""
    code = (
        "import sys\n"
        "import forecasting_engine.models.boosted as boosted\n"
        "assert 'shap' not in sys.modules, 'shap was imported with the module'\n"
        "assert boosted.shap.TreeExplainer is not None\n"
        "assert 'shap' in sys.modules\n"
    )

    done = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
        capture_output=True,
        text=True,
        check=False,
    )

    assert done.returncode == 0, done.stderr
