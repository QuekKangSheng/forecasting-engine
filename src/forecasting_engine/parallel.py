"""Run independent pieces of a model run in separate processes.

Only work whose pieces don't depend on each other is split this way — the
derived polynomial's configurations, the two machine-learning libraries' tuning
sequences, PBO's combinations — so the result is the same as running them one
after the other, only sooner.

Processes are started fresh (``spawn``), never forked: a fork taken from the
Models page's background thread, after XGBoost or LightGBM has started its own
threads, can deadlock on Linux, which is where the app is hosted.

A fresh process re-runs whatever ``__main__`` is before doing its work, and
Streamlit installs each page script as ``__main__``. So workers are started with
a blank ``__main__``; otherwise each would try to run the Models page.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import threading
import types
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from typing import TypeVar

T = TypeVar("T")

WORKERS_ENV: str = "FORECASTING_WORKERS"
"""Set to a number to choose how many processes a run may use; 1 runs serially."""

DEFAULT_MAX_WORKERS: int = 4
"""Working default: each process holds its own copy of the libraries (about 200
MB), and Streamlit Community Cloud allows an app about 2.7 GB and 2 cores, so
more than a few processes buys memory pressure, not speed, there. On a laptop,
raise it through ``FORECASTING_WORKERS``."""


def workers(tasks: int) -> int:
    """How many processes to use for ``tasks`` independent pieces of work."""
    configured = os.environ.get(WORKERS_ENV)
    if configured is not None:
        limit = max(1, int(configured))
    else:
        limit = min(DEFAULT_MAX_WORKERS, os.cpu_count() or 1)
    return max(1, min(limit, tasks))


def map_processes(fn: Callable[..., T], *iterables: Iterable) -> list[T]:
    """``[fn(*args) for args in zip(*iterables)]``, spread over processes.

    ``fn`` must be a module-level function and its arguments picklable. With one
    worker, or one task, everything runs in this process.
    """
    calls: Sequence[tuple] = list(zip(*iterables, strict=True))
    count = workers(len(calls))
    if count <= 1:
        return [fn(*args) for args in calls]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=count, mp_context=context) as pool:
        # Submitting starts the workers, so every worker starts inside this block.
        with _blank_main():
            futures = [pool.submit(fn, *args) for args in calls]
        return [future.result() for future in futures]


_MAIN_LOCK = threading.Lock()


@contextmanager
def _blank_main() -> Iterator[None]:
    """``__main__`` replaced by an empty module while workers start, then put
    back, unless something else (a Streamlit rerun) installed its own meanwhile."""
    with _MAIN_LOCK:
        original = sys.modules.get("__main__")
        blank = types.ModuleType("__main__")
        sys.modules["__main__"] = blank
        try:
            yield
        finally:
            if sys.modules.get("__main__") is blank and original is not None:
                sys.modules["__main__"] = original
