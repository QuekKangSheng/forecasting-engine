"""Model runs that carry on in the background, whatever the browser does.

A Run is queued here instead of being fitted inside the page's own script, so
switching tabs, clicking elsewhere or closing the window can't stop it. One
worker thread per server works through the queue in order: two targets fitted
at once would only share the same CPU and both finish later.

Each finished model is also saved to disk, keyed by everything its result
depends on (the dataset, the shared settings, the target, the model and the
model's own setting). A later session that commits the same data under the
same settings picks it up, so a Run left going while the window was closed is
there on return.

That file is a cache: a re-run overwrites it. Each fit is also handed to an
``on_finish`` callback when it ends, done or failed, which the Models page uses
to append it to the run history in DuckDB (``store/runs.py``).
"""

from __future__ import annotations

import hashlib
import pickle
import threading
import time
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

#: Where finished models are saved. Relative, like the active-model DuckDB, so
#: it sits under the app's working directory.
RESULTS_DIR = Path("data") / "model_runs"

#: Set by tests: fit in the caller's thread, so a click's results are there when
#: the script returns.
INLINE = False

QUEUED, RUNNING, DONE, FAILED = "queued", "running", "done", "failed"


@dataclass(frozen=True)
class Finished:
    """One model's fit, as it ended: what ``Job.on_finish`` is given."""

    name: str
    started: float
    finished: float
    """Both ``time.time()``."""
    result: object | None
    """What the runner returned, or ``None`` if it failed."""
    error: str | None


@dataclass
class Job:
    """One target's Run: the models it fits and how far it has got."""

    job_id: str
    target_name: str
    models: tuple[str, ...]
    keys: Mapping[str, str]
    """Model name -> the disk key its result is saved under."""
    runners: Mapping[str, Callable[[], object]] = field(repr=False)
    expected_errors: tuple[type[BaseException], ...] = ()
    """Errors whose message is written for the reader."""
    on_finish: Callable[[Finished], None] | None = field(default=None, repr=False)
    """Called once per model when its fit ends, done or failed, outside the lock."""
    status: dict[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    started: dict[str, float] = field(default_factory=dict)
    finished: dict[str, float] = field(default_factory=dict)
    submitted: float = field(default_factory=time.time)

    @property
    def active(self) -> bool:
        return any(s in (QUEUED, RUNNING) for s in self.status.values())


_lock = threading.Lock()
_jobs: dict[str, Job] = {}
_queue: list[Job] = []
_wake = threading.Condition(_lock)
_worker: threading.Thread | None = None


def result_key(*parts: object) -> str:
    """A stable file-name-safe key for everything a saved result depends on."""
    return hashlib.sha256(repr(parts).encode()).hexdigest()[:32]


def submit(
    job_id: str,
    target_name: str,
    runners: Mapping[str, Callable[[], object]],
    keys: Mapping[str, str],
    *,
    expected_errors: tuple[type[BaseException], ...],
    on_finish: Callable[[Finished], None] | None = None,
) -> Job:
    """Queue ``runners`` (model name -> a function fitting it) as one Run.

    ``expected_errors`` are the ones whose message is written for the reader; any
    other is recorded with its type, so a bug still says something rather than
    nothing. A Run already queued or running under ``job_id`` is returned as it
    is instead of being queued twice."""
    with _lock:
        current = _jobs.get(job_id)
        if current is not None and current.active:
            return current
        job = Job(
            job_id=job_id,
            target_name=target_name,
            models=tuple(runners),
            keys=dict(keys),
            runners=dict(runners),
            expected_errors=expected_errors,
            on_finish=on_finish,
            status=dict.fromkeys(runners, QUEUED),
        )
        _jobs[job_id] = job
        if not INLINE:
            _queue.append(job)
            _ensure_worker()
            _wake.notify()
    if INLINE:
        _fit(job)
    return job


def job(job_id: str) -> Job | None:
    with _lock:
        return _jobs.get(job_id)


def active_jobs() -> list[Job]:
    with _lock:
        return [j for j in _jobs.values() if j.active]


def load(key: str) -> object | None:
    """A saved model result, or ``None`` if nothing was saved under ``key``."""
    path = RESULTS_DIR / f"{key}.pkl"
    if not path.exists():
        return None
    try:
        return pickle.loads(path.read_bytes())
    except Exception:  # an unreadable file is the same as no file
        return None


def _save(key: str, result: object) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = RESULTS_DIR / f"{key}.tmp"
    tmp.write_bytes(pickle.dumps(result))
    tmp.replace(RESULTS_DIR / f"{key}.pkl")


def _ensure_worker() -> None:
    global _worker
    if _worker is None or not _worker.is_alive():
        _worker = threading.Thread(target=_work, name="model-runs", daemon=True)
        _worker.start()


def _work() -> None:
    while True:
        with _lock:
            while not _queue:
                _wake.wait()
            job = _queue.pop(0)
        _fit(job)


def _fit(job: Job) -> None:
    for name in job.models:
        with _lock:
            job.status[name] = RUNNING
            job.started[name] = time.time()
        result = None
        try:
            result = job.runners[name]()
        except job.expected_errors as exc:
            outcome, message = FAILED, str(exc)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            traceback.print_exc()
            outcome, message = FAILED, f"{type(exc).__name__}: {exc}"
        else:
            _save(job.keys[name], result)
            outcome, message = DONE, None
        with _lock:
            job.status[name] = outcome
            job.finished[name] = time.time()
            if message is not None:
                job.errors[name] = message
            ended = Finished(name, job.started[name], job.finished[name], result, message)
        if job.on_finish is not None:
            try:
                job.on_finish(ended)
            except Exception:  # noqa: BLE001 - a failed log must not fail the fit
                traceback.print_exc()
