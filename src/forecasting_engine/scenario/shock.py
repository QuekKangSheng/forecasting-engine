"""Shock a signal and see what the active polynomials forecast (FYP-32).

A shock replaces a signal's last observed value, either with a value entered
directly or with a % change from it. The shocked frame then goes through the
same alignment, transform and one-day lag as the fit (``ingest.align``), so the
model reads the shock exactly as it would read a real day; and the active
model's stored function (``store.active_model.ActiveFunction``) is evaluated on
those inputs, with the clipping and standardisation it was fitted with. Nothing
is refitted.

The base case is the same function on the unshocked inputs: the forecast the
active model would make for the next ``horizon`` days from the latest data, so a
zero shock reproduces it exactly.

Before a scenario runs, every shock is validated (``validate``): it must be a
number, a series that is always above zero (a volatility index, a spread, a
price) must stay above zero, and the move must not exceed ``MAX_RANGE_MULTIPLE``
times the whole range the series has covered. A shocked input outside the range
the model was trained on is allowed, but flagged as an extrapolation.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

from forecasting_engine.ingest.align import Transform, align_and_lag
from forecasting_engine.models.base import ModelDescription
from forecasting_engine.models.polynomial import evaluate_formula, placeholders
from forecasting_engine.reporting.polynomial_function import Origin, from_description
from forecasting_engine.store.active_model import ActiveFunction
from forecasting_engine.validation.splitters import PurgedWalkForward

MAX_RANGE_MULTIPLE: float = 10.0
"""A shock may move a signal by at most this many times the whole range (highest
less lowest value) it has covered in the data. Beyond it the scenario is not a
stress but a typo: VIX at 2,000 rather than 20."""

POSITIVE_TRANSFORMS: frozenset[Transform] = frozenset({Transform.LEVEL, Transform.LOG_RETURN})
"""Series the transform map reads as levels (volatility gauges, credit spreads)
or log returns (prices, total-return indices) are always above zero, so a shock
may not take them to zero or below. Rates and curve slopes, read as differences,
can be negative."""


class ShockKind(StrEnum):
    ABSOLUTE = "absolute"
    """The value entered is the signal's new level."""
    PERCENT = "percent"
    """The value entered is a % change from the last observed level."""


class ScenarioError(ValueError):
    """The scenario can't be evaluated. Written for a risk manager."""


@dataclass(frozen=True)
class Shock:
    signal: str
    kind: ShockKind
    value: str | float
    """As entered: text from a form is parsed by ``validate``."""


@dataclass(frozen=True)
class InputChange:
    """How a shock reached one input of one function."""

    signal: str
    base: float
    """The input the base case reads, after transform and lag."""
    shocked: float
    trained_on: tuple[float, float]
    """The lowest and highest value of the input in the latest fold's training rows."""
    clipped_to: float | None
    """The bound a clipped model holds the shocked input at, or ``None`` if it is
    within the bounds or the model doesn't clip."""

    @property
    def extrapolated(self) -> bool:
        low, high = self.trained_on
        return not low <= self.shocked <= high


@dataclass(frozen=True)
class ScenarioResult:
    base: float
    """The forecast return over the horizon, on the latest inputs."""
    shocked: float
    changes: tuple[InputChange, ...]
    """One per input the shocks moved."""
    as_of: pd.Timestamp
    """The last day of data; the forecast is for the horizon after it."""

    @property
    def change(self) -> float:
        """Shocked less base, in return units."""
        return self.shocked - self.base

    @property
    def relative_change(self) -> float:
        """The change as a share of the base forecast, NaN if the base is zero."""
        return self.change / abs(self.base) if self.base else float("nan")


def function_inputs(
    description: ModelDescription, origin: Origin, columns: Collection[str]
) -> tuple[str, ...]:
    """The signal columns ``description``'s function reads, in the order they first
    appear: the names in a user formula, or every column a fitted term multiplies."""
    if origin is Origin.USER_SUPPLIED:
        return tuple(name for name in placeholders(description.terms[0]) if name in columns)
    fn = from_description(description, origin=origin, target="", horizon=1, columns=columns)
    used = (column for term in fn.terms for column, _ in term.factors)
    return tuple(dict.fromkeys(c for c in used if c in columns))


def last_observed(
    frame: pd.DataFrame, signal: str, as_of: pd.Timestamp | None = None
) -> tuple[pd.Timestamp, float]:
    """The date and value of ``signal``'s last observation in ``frame``, on or before
    ``as_of`` if given: the target's last price, after which nothing is read."""
    observed = frame[signal].dropna()
    if as_of is not None:
        observed = observed[observed.index <= as_of]
    if observed.empty:
        raise ScenarioError(f"{signal} has no observed value to shock.")
    return observed.index[-1], float(observed.iloc[-1])


def validate(
    shocks: Sequence[Shock],
    frame: pd.DataFrame,
    *,
    as_of: pd.Timestamp | None = None,
    positive: Collection[str] = (),
    label: Callable[[str], str] = str,
) -> tuple[dict[str, float], dict[str, str]]:
    """``(shocked levels, errors)``, each keyed by signal. A scenario may run only
    when ``errors`` is empty. A % change is from the last observation on or before
    ``as_of``; ``positive`` names the signals that must stay above zero
    (``POSITIVE_TRANSFORMS``). Each error names its signal by ``label``."""
    levels: dict[str, float] = {}
    errors: dict[str, str] = {}
    for shock in shocks:
        try:
            entered = float(str(shock.value).replace(",", "").strip())
        except ValueError:
            entered = float("nan")
        if not math.isfinite(entered):
            errors[shock.signal] = f"{label(shock.signal)}: {str(shock.value)!r} is not a number."
            continue
        _, last = last_observed(frame, shock.signal, as_of)
        level = entered if shock.kind is ShockKind.ABSOLUTE else last * (1 + entered / 100)
        observed = frame[shock.signal].dropna()
        span = float(observed.max() - observed.min())
        if shock.signal in positive and level <= 0:
            errors[shock.signal] = (
                f"{label(shock.signal)}: {level:,.4g} is not possible, as this series is always "
                "above zero."
            )
        elif abs(level - last) > MAX_RANGE_MULTIPLE * span:
            errors[shock.signal] = (
                f"{label(shock.signal)}: moving from {last:,.4g} to {level:,.4g} is more than "
                f"{MAX_RANGE_MULTIPLE:g} times the whole range it has covered "
                f"({observed.min():,.4g} to {observed.max():,.4g})."
            )
        else:
            levels[shock.signal] = level
    return levels, errors


def shocked_frame(
    frame: pd.DataFrame, levels: Mapping[str, float], as_of: pd.Timestamp | None = None
) -> pd.DataFrame:
    """``frame`` with each signal's last observation on or before ``as_of`` replaced
    by its shocked level."""
    shocked = frame.copy()
    for signal, level in levels.items():
        date, _ = last_observed(frame, signal, as_of)
        shocked.loc[date, signal] = level
    return shocked


def next_inputs(
    frame: pd.DataFrame,
    price_col: str,
    transforms: Mapping[str, Transform],
    *,
    horizon: int,
) -> pd.Series:
    """The inputs the forecast for the trading day after ``frame``'s last price
    reads: each signal in ``transforms`` aligned to the target's calendar,
    transformed and lagged by ``ingest.align.align_and_lag`` itself, on a calendar
    extended by that one day."""
    last = frame[price_col].dropna()
    following = last.index[-1] + pd.offsets.BDay(1)
    extended = pd.concat([frame, pd.DataFrame({price_col: [last.iloc[-1]]}, index=[following])])
    panel = align_and_lag(
        extended, list(transforms), price_col, horizon=horizon, transforms=transforms
    )
    return panel.frame[list(transforms)].iloc[-1]


def evaluate(function: ActiveFunction, inputs: Mapping[str, float]) -> float:
    """The function's forecast on ``inputs``: each input clipped and standardised
    as it was fitted, then the terms applied. No refitting."""
    description = function.description
    missing = [s for s in function.inputs if not math.isfinite(inputs.get(s, float("nan")))]
    if missing:
        raise ScenarioError(
            f"{', '.join(missing)} {'has' if len(missing) == 1 else 'have'} no value on the "
            "last day of data, so the function can't be evaluated."
        )
    intercept = 0.0 if description.intercept is None else float(description.intercept)
    if function.origin is Origin.USER_SUPPLIED:
        (slope,) = description.coefficients
        return intercept + slope * evaluate_formula(description.terms[0], inputs)
    bounds = description.input_bounds or {}
    centre = description.standardisation or {}
    z = {}
    for signal in function.inputs:
        x = float(inputs[signal])
        if signal in bounds:
            low, high = bounds[signal]
            x = min(max(x, low), high)
        if signal in centre:
            mean, sd = centre[signal]
            x = (x - mean) / sd
        z[signal] = x
    fn = from_description(
        description, origin=function.origin, target="", horizon=1, columns=function.inputs
    )
    total = fn.intercept
    for term in fn.terms:
        product = term.coefficient
        for column, power in term.factors:
            if column not in z:
                raise ScenarioError(f"The function's term {column!r} isn't one of its inputs.")
            product *= z[column] ** power
        total += product
    return total


def training_ranges(
    frame: pd.DataFrame,
    price_col: str,
    transforms: Mapping[str, Transform],
    *,
    horizon: int,
    splitter: PurgedWalkForward,
) -> dict[str, tuple[float, float]]:
    """Each input's lowest and highest value over the latest walk-forward fold's
    training rows: the fold whose fit the stored function is."""
    panel = align_and_lag(
        frame, list(transforms), price_col, horizon=horizon, transforms=transforms
    )
    folds = list(splitter.split(panel))
    if not folds:
        raise ScenarioError(splitter.too_short(panel))
    rows = panel.frame.loc[folds[-1][0], list(transforms)]
    return {s: (float(rows[s].min()), float(rows[s].max())) for s in transforms}


def run_scenario(
    function: ActiveFunction,
    frame: pd.DataFrame,
    price_col: str,
    levels: Mapping[str, float],
    *,
    horizon: int,
    splitter: PurgedWalkForward,
) -> ScenarioResult:
    """The function's base and shocked forecasts, with what the shocks did to each
    input it reads. Shocks to signals it doesn't read leave it unchanged."""
    transforms = function.transforms
    as_of = frame[price_col].dropna().index[-1]
    base_inputs = next_inputs(frame, price_col, transforms, horizon=horizon)
    shocked_inputs = next_inputs(
        shocked_frame(frame, levels, as_of), price_col, transforms, horizon=horizon
    )
    ranges = training_ranges(frame, price_col, transforms, horizon=horizon, splitter=splitter)
    bounds = function.description.input_bounds or {}
    changes = []
    for signal in function.inputs:
        base, shocked = float(base_inputs[signal]), float(shocked_inputs[signal])
        if base == shocked or (math.isnan(base) and math.isnan(shocked)):
            continue
        clipped_to = None
        if signal in bounds:
            low, high = bounds[signal]
            if not low <= shocked <= high:
                clipped_to = min(max(shocked, low), high)
        changes.append(InputChange(signal, base, shocked, ranges[signal], clipped_to))
    return ScenarioResult(
        base=evaluate(function, base_inputs),
        shocked=evaluate(function, shocked_inputs),
        changes=tuple(changes),
        as_of=as_of,
    )
