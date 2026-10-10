"""The Models page's stored runs, and when they stop being valid.

Every run on a tab uses the page's shared settings (dataset, horizon,
walk-forward windows), and some runs also depend on a setting of their own on
that tab (the derived polynomial's term cap, the user-supplied function). So
results are kept together with the settings that produced them and dropped
once those settings change: a shared change clears every tab, a model's own
setting clears only that model's row. Which models are ticked is not a setting —
it only decides what the next Run fits.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.models.base import ModelDescription
from forecasting_engine.models.boosted import TuningLog
from forecasting_engine.models.famafrench import FactorCoverage
from forecasting_engine.reporting.model_metrics import ModelRunResult
from forecasting_engine.reporting.polynomial_function import PolynomialFunction

#: Read by the Data page too, whose "Clear Data" drops every stored run.
RUNS_KEY = "model_runs"


@dataclass(frozen=True)
class ModelRun:
    result: ModelRunResult
    description: ModelDescription
    function: PolynomialFunction | None = None
    """The fitted polynomial, for the Polynomial runs only."""
    coverage: FactorCoverage | None = None
    """What the factor file covered, for the FF5 run only."""
    warning: str | None = None
    """Shown with the result, e.g. that an older saved factor file was used."""
    tuning: TuningLog | None = None
    """Which tune each fold used, for the machine-learning run only."""


@dataclass
class TabRuns:
    model_settings: Mapping[str, object]
    """Model name -> the tab-level setting its run depends on, for the models
    that have one."""
    runs: dict[str, ModelRun] = field(default_factory=dict)
    """Model name (as in ``reporting.model_metrics.MODEL_ORDER``) -> its run."""


@dataclass
class StoredRuns:
    shared_settings: tuple
    tabs: dict[TargetRole, TabRuns] = field(default_factory=dict)


def stored(session: MutableMapping, shared_settings: tuple) -> StoredRuns:
    """The stored runs, emptied first if the shared settings have changed."""
    current = session.get(RUNS_KEY)
    if current is None or current.shared_settings != shared_settings:
        current = StoredRuns(shared_settings)
        session[RUNS_KEY] = current
    return current


def cleared_by(session: MutableMapping, shared_settings: tuple) -> bool:
    """Whether ``stored`` is about to drop results because the shared settings
    changed — so the page can say why they vanished."""
    current = session.get(RUNS_KEY)
    return (
        current is not None
        and current.shared_settings != shared_settings
        and any(tab.runs for tab in current.tabs.values())
    )


def tab(runs: StoredRuns, role: TargetRole, model_settings: Mapping[str, object]) -> TabRuns:
    """One target's runs, without any model whose own setting has changed."""
    current = runs.tabs.get(role)
    if current is None:
        current = TabRuns(dict(model_settings))
        runs.tabs[role] = current
        return current
    for name, value in model_settings.items():
        if current.model_settings.get(name) != value:
            current.runs.pop(name, None)
    current.model_settings = dict(model_settings)
    return current
