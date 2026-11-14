"""What the app does once, in the background, as its first page is being shown.

The first visit to each page used to pay for things that have nothing to do with
that page's figures: importing the libraries it uses (seconds, for the model and
statistics libraries) and the store checking its tables exist. That made every page
slow the first time it was opened after the app started, which is exactly when a
demonstration opens them. So the first run of the app starts a thread that does that
work while the first page is on screen, and by the time another is opened it is done.

Nothing here computes a result or writes one: it only makes later work start warm.
A page opened before the thread has finished waits on the same imports it would
have done itself, so nothing is done twice and nothing is done wrong.
"""

from __future__ import annotations

import ast
import importlib
import logging
import sys
import threading
from pathlib import Path

from forecasting_engine.extraction.targets import TargetRole

_LOG = logging.getLogger(__name__)

PAGES = Path(__file__).resolve().parent / "app_pages"

#: Imported on first use deep inside a run rather than by a page.
_ALSO = ("shap",)

_started = threading.Lock()
_thread: threading.Thread | None = None


def page_imports(pages: Path = PAGES) -> list[str]:
    """Every module the pages import, read from their source, so a page given a new
    dependency is warmed without being listed here.

    ``from package import name`` may name a module of the package or something
    defined in it, and the source alone can't say which. Both ``package`` and
    ``package.name`` are returned, and ``_load`` settles it."""
    found: dict[str, None] = {}
    for page in sorted(pages.glob("*.py")):
        for node in ast.walk(ast.parse(page.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                found.update(dict.fromkeys(alias.name for alias in node.names))
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                if node.module == "__future__":
                    continue
                found[node.module] = None
                found.update(dict.fromkeys(f"{node.module}.{alias.name}" for alias in node.names))
    return list(found)


def _load(name: str) -> None:
    """Import ``name``, unless it turns out to be something inside a module that is
    already loaded (see ``page_imports``) rather than a module itself."""
    try:
        importlib.import_module(name)
    except ModuleNotFoundError as missing:
        inside = name.rpartition(".")[0]
        if inside in sys.modules and missing.name == name:
            return
        raise


def _touch_store() -> None:
    """One small read from each of the store's logs, so each has opened the database
    and checked its tables before a page asks it for anything."""
    from forecasting_engine.store import active_model, portfolio_runs, runs, uploads

    for role in TargetRole:
        active_model.get_active_model(role)
    runs.list_runs(limit=1)
    portfolio_runs.list_portfolio_runs(limit=1)
    uploads.recent_uploads(limit=1)


def run(pages: Path = PAGES) -> list[str]:
    """Do the warming now, in this thread. Returns what could not be warmed: a step
    that fails is logged and skipped, since the page that needs it will report its
    own error in its own words."""
    failed = []
    for name in (*page_imports(pages), *_ALSO):
        try:
            _load(name)
        except Exception:  # noqa: BLE001 - warming must never break the app
            _LOG.warning("warm-up could not import %s", name, exc_info=True)
            failed.append(name)
    try:
        _touch_store()
    except Exception:  # noqa: BLE001
        _LOG.warning("warm-up could not read the store", exc_info=True)
        failed.append("store")
    return failed


def start() -> threading.Thread:
    """Start the warming in the background, once per process however often called."""
    global _thread
    with _started:
        if _thread is None:
            _thread = threading.Thread(target=run, name="warm-up", daemon=True)
            _thread.start()
        return _thread
