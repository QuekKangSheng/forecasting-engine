"""The Models page's stored runs, and when they stop being valid.

Every run on a tab uses the page's shared settings (dataset, horizon,
walk-forward windows) and that tab's polynomial settings, so results are kept
together with the settings that produced them and dropped as a group once
those settings change: a shared change clears every tab, a tab's polynomial
change clears only that tab. Which models are ticked is not a setting — it
only decides what the next Run fits.
"""

from __future__ import annotations

from collections.abc import MutableMapping
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
    """The fitted polynomial, for the Polynomial run only."""
    coverage: FactorCoverage | None = None
    """What the factor file covered, for the FF5 run only."""
    warning: str | None = None
    """Shown with the result, e.g. that an older saved factor file was used."""
    tuning: TuningLog | None = None
    """Which tune each fold used, for the machine-learning run only."""


@dataclass
class TabRuns:
    polynomial_settings: tuple
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


def tab(runs: StoredRuns, role: TargetRole, polynomial_settings: tuple) -> TabRuns:
    """One target's runs, emptied first if its polynomial settings have changed."""
    current = runs.tabs.get(role)
    if current is None or current.polynomial_settings != polynomial_settings:
        current = TabRuns(polynomial_settings)
        runs.tabs[role] = current
    return current
