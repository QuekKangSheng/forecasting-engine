"""The bond derived polynomial: the mean of 14 polynomials, each fitting levels apart.

The Models page's Bond tab runs this as its "Polynomial (derived)" row; equity's derived
polynomial is ``polynomial.run_derived_polynomial``, unchanged. It is the candidate the
November 2026 research registered before its hold-out was scored and the one of 11 that
passed it (``s18-holdout-and-structure``, docs/evidence/2026-11-18-bond-holdout.md).

Why. The derived grid's fits give a persistent level (a volatility gauge, a spread) to
the same regression as the daily changes. For a level whose own moves drive the target
that slope is biased in a short sample (Stambaugh 1999), with the sign of the level's
same-day correlation with the target rather than of anything it predicts. The
reduced-bias fit for that case is the bond research model's (``models/carry_research``,
Amihud and Hurvich 2004), so here every level goes to it and every other column to one
of the page's own four fits. And no one fit or window is picked: the window that scored
best swapped between 2008-2026 and the years before 2006, so the 14 are averaged.

What.

1. **Inputs** (``bond_derived_inputs``), by kind of series, never by ticker: the page's
   signals with the page's transforms; for each signal the page reads as a difference
   (a rate), its level, lagged a row like every signal (``level_input``); and the
   calendar counts over each row's label window (``models/calendar_inputs``), which are
   known in advance and not lagged. An input is a **level** if it is read as a level (a
   gauge, a spread, a rate's level); every other one (rate changes, price returns,
   calendar counts) is a **flow** (``is_level``).
2. **A member** (``HybridPolynomial``), per fold, on its screened inputs: the levels go
   to ``CarryResearchPolynomial`` as it is, the flows to one flow fit, and
   ``forecast = level part + flow part − the flow part's mean in-sample forecast``; with
   only one part, that part; with neither, the training-mean target.
3. **The 14** (``MEMBERS``): flow fit in ``FLOW_FITS`` (lasso at degree 1, 2, 3 with the
   page's term cap, and the shrinkage fit) × with and without the calendar counts × the
   page's train window and every earlier row up to ``LONG_TRAIN_WINDOW``
   (``splitters.capped``), less lasso degree 3 with the calendar counts.
4. **The model**: on each test row, the plain mean of the 14 forecasts; per fold, its
   own level (what its signal is measured from) is the mean of the 14 own levels.
   Nothing is selected.

One polynomial. Within one train window the members share each input's clip range and
standardisation, so their mean is one polynomial. Across the two windows an input has
two of each, so the longer window's fits read their inputs under another name, the
input's own with ``LONG_WINDOW_SUFFIX`` (``long_window_input``); the panel holds those
as columns of their own, and the fitted description is one polynomial in which every
input name has one range and one standardisation. ``source_column`` maps any of these
names back to the committed column it reads, which is how a scenario evaluates the
function again.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from forecasting_engine.ingest.align import FeaturePanel, Transform, align_and_lag
from forecasting_engine.models.base import Forecaster, ModelDescription
from forecasting_engine.models.calendar_inputs import (
    CALENDAR_NAMES,
    inputs_for,
    label_window_calendar,
)
from forecasting_engine.models.carry_research import CarryResearchPolynomial
from forecasting_engine.models.polynomial import (
    SHRINKAGE,
    DerivedPolynomial,
    PolynomialConfigError,
)
from forecasting_engine.models.shrinkage import ShrinkageDataError, ShrinkagePolynomial
from forecasting_engine.parallel import map_processes
from forecasting_engine.reporting.model_metrics import ModelRunResult
from forecasting_engine.validation.harness import (
    FoldResult,
    FoldScreening,
    _own_level,
    _training_mean_fold,
    evaluate,
    fold_baseline,
    summarize,
)
from forecasting_engine.validation.splitters import PurgedWalkForward, capped

LONG_TRAIN_WINDOW: int = 2520
"""The longer train window: every row before a test window's embargo, at most this many
(about ten years). The research's ``cap2520``; the shorter is the page's own."""

LEVEL_SUFFIX: str = "_LEVEL"
"""Ends the name of a rate's level, beside the rate's change the page reads."""

LONG_WINDOW_SUFFIX: str = f"_W{LONG_TRAIN_WINDOW}"
"""Ends the name of an input as the fits on the longer window read it."""

FLOW_FITS: tuple[str, ...] = ("lasso1", "lasso2", "lasso3", SHRINKAGE)
"""The flow part's fits: ``DerivedPolynomial`` at degree 1, 2 and 3 with the page's term
cap, and ``ShrinkagePolynomial``. The derived grid's four, unchanged."""

NAME: str = "BondDerivedPolynomial"
"""The fitted description's name, which tells this model's stored function from any
other polynomial's."""

_TOO_FEW = (PolynomialConfigError, ShrinkageDataError)
"""How a fit says its fold has too few complete rows: that part is left out of the fold."""

_ABSENT = "AbsentPart"
"""The description of a part fitted alone that had too few rows to fit."""

_POWER = re.compile(r"^(.+)\^(\d+)$")


class BondDerivedConfigError(PolynomialConfigError):
    """The message is written for a portfolio manager, like the other models' errors."""


@dataclass(frozen=True)
class Member:
    """One of the 14: which flow fit, whether the calendar counts are offered, and which
    train window."""

    fit: str
    calendar: bool
    long_window: bool

    @property
    def name(self) -> str:
        window = f"{LONG_TRAIN_WINDOW} rows" if self.long_window else "page rows"
        return f"hybrid {self.fit}{', calendar' if self.calendar else ''}, {window}"


MEMBERS: tuple[Member, ...] = tuple(
    Member(fit, calendar, long_window)
    for calendar in (False, True)
    for long_window in (False, True)
    for fit in FLOW_FITS
    if not (fit == "lasso3" and calendar)
)
"""The 14, in the research's order. Lasso degree 3 is not offered the calendar counts, as
it was not in the research (a degree-3 expansion of every column is slow and was never
the better fit there)."""


# ── Names ────────────────────────────────────────────────────────────────────


def level_input(column: str) -> str:
    """The name of a rate column's level."""
    return f"{column}{LEVEL_SUFFIX}"


def long_window_input(name: str) -> str:
    """The name the fits on the longer window read ``name`` under."""
    return f"{name}{LONG_WINDOW_SUFFIX}"


def is_long_window(name: str) -> bool:
    return name.endswith(LONG_WINDOW_SUFFIX)


def base_input(name: str) -> str:
    """``name`` as the fits on the page's window read it: the same input, the same
    column, under the same transform."""
    return name.removesuffix(LONG_WINDOW_SUFFIX)


def source_column(name: str, columns: Collection[str]) -> str | None:
    """The committed column an input reads, or ``None`` for a calendar count, which is
    worked out from dates. A committed column is always itself."""
    if name in columns:
        return name
    base = base_input(name)
    if base in columns:
        return base
    if base in CALENDAR_NAMES:
        return None
    rate = base.removesuffix(LEVEL_SUFFIX)
    if base.endswith(LEVEL_SUFFIX) and rate in columns:
        return rate
    raise BondDerivedConfigError(
        f"the function reads {name}, which is neither a column of the committed data nor "
        "an input built from one."
    )


def is_level(panel: FeaturePanel, column: str) -> bool:
    """Whether ``column`` enters as a level: it is read as a level (a gauge, a spread, a
    rate's level). A calendar count has no alignment of its own and is a flow."""
    alignment = panel.alignment.get(column)
    return alignment is not None and alignment.transform is Transform.LEVEL


# ── Panels ───────────────────────────────────────────────────────────────────


def bond_derived_inputs(
    signal_cols: Sequence[str], transforms: Mapping[str, Transform], horizon: int
) -> dict[str, Transform]:
    """Every input the 14 read, with its transform: the page's signals, each rate's level
    and the calendar counts (read as they are), then the same under the longer window's
    names. A calendar count's transform is ``LEVEL``: it is used as it is."""
    page = {c: transforms[c] for c in signal_cols}
    levels = {
        level_input(c): Transform.LEVEL
        for c in signal_cols
        if transforms[c] is Transform.DIFFERENCE
    }
    calendar = dict.fromkeys(inputs_for(horizon), Transform.LEVEL)
    short = {**page, **levels, **calendar}
    return {**short, **{long_window_input(name): t for name, t in short.items()}}


def inputs_panel(
    frame: pd.DataFrame, price_col: str, inputs: Mapping[str, Transform], horizon: int
) -> FeaturePanel:
    """A panel whose signals are ``inputs``, in order: each the committed column it reads
    (``source_column``) under its own transform, aligned and lagged by
    ``align_and_lag``, and each calendar count from the panel's calendar, not lagged.

    ``align_and_lag`` works each signal out on its own, so an input reads exactly as its
    column would in any other panel on this calendar."""
    sources = {name: source_column(name, frame.columns) for name in inputs}
    read = [name for name, column in sources.items() if column is not None]
    work = pd.DataFrame(
        {**{name: frame[sources[name]] for name in read}, price_col: frame[price_col]},
        index=frame.index,
    )
    base = align_and_lag(
        work, read, price_col, horizon=horizon, transforms={n: inputs[n] for n in read}
    )
    counted = [name for name, column in sources.items() if column is None]
    out = base.frame
    if counted:
        counts = label_window_calendar(out.index, horizon)
        missing = [n for n in counted if base_input(n) not in counts.columns]
        if missing:
            raise BondDerivedConfigError(
                f"{', '.join(missing)} {'is' if len(missing) == 1 else 'are'} not a "
                f"calendar count at a {horizon}-day horizon."
            )
        out = out.assign(**{name: counts[base_input(name)] for name in counted})
    order = [*inputs, price_col, base.targets[0]]
    return replace(base, frame=out[order], signals=tuple(inputs))


def bond_derived_panel(
    frame: pd.DataFrame,
    signal_cols: Sequence[str],
    transforms: Mapping[str, Transform],
    price_col: str,
    horizon: int,
) -> FeaturePanel:
    """The panel the 14 read: every input of ``bond_derived_inputs`` as a column."""
    inputs = bond_derived_inputs(signal_cols, transforms, horizon)
    _check_names(frame.columns, signal_cols, price_col, inputs)
    return inputs_panel(frame, price_col, inputs, horizon)


def _check_names(
    columns: Collection[str],
    signal_cols: Sequence[str],
    price_col: str,
    inputs: Mapping[str, Transform],
) -> None:
    """Refuse data whose own column names would be read as an input this model builds."""
    built = [name for name in inputs if name not in signal_cols]
    clashing = [
        column
        for column in [*signal_cols, price_col]
        if column.endswith((LEVEL_SUFFIX, LONG_WINDOW_SUFFIX)) or column in CALENDAR_NAMES
    ] + [name for name in built if name in columns]
    if clashing:
        raise BondDerivedConfigError(
            f"the committed column {clashing[0]} would be mistaken for an input the bond "
            f"derived polynomial builds itself: it names a rate's level by adding "
            f"{LEVEL_SUFFIX} to the rate's column, its fits on up to {LONG_TRAIN_WINDOW:,} "
            f"dates by adding {LONG_WINDOW_SUFFIX}, and reads names such as CAL_MONTH_END "
            "as calendar counts. Rename that series' field in its export, then commit the "
            "data again."
        )


def function_transforms(panel: FeaturePanel) -> dict[str, Transform]:
    """Each of the panel's inputs -> the transform it is read under, ``LEVEL`` for a
    calendar count: what a stored function records for each input."""
    return {
        s: panel.alignment[s].transform if s in panel.alignment else Transform.LEVEL
        for s in panel.signals
    }


def member_columns(panel: FeaturePanel, member: Member) -> tuple[str, ...]:
    """The inputs ``member`` is offered, as its window's fits name them."""
    short = [s for s in panel.signals if not is_long_window(s)]
    chosen = [s for s in short if member.calendar or s not in CALENDAR_NAMES]
    return tuple(long_window_input(s) for s in chosen) if member.long_window else tuple(chosen)


# ── A member ─────────────────────────────────────────────────────────────────


def flow_forecaster(fit: str, max_terms: int | None) -> Callable[[], Forecaster]:
    """A factory for one of ``FLOW_FITS``."""
    if fit == SHRINKAGE:
        return ShrinkagePolynomial
    if fit not in FLOW_FITS:
        raise ValueError(f"no flow fit {fit!r}")
    degree = int(fit.removeprefix("lasso"))
    return lambda: DerivedPolynomial(degree, "lasso", max_terms)


@dataclass
class HybridPolynomial:
    """One member: its fold's level inputs to ``CarryResearchPolynomial``, its flow
    inputs to ``flow``'s fit, ``forecast = level part + flow part − the flow part's mean
    in-sample forecast``. A part with too few complete rows to fit is left out; with
    neither part, the forecast is the training-mean target.

    The definition the run function's faster path is tested against
    (``bond_derived_folds(..., share_level_part=False)``)."""

    flow: Callable[[], Forecaster]
    name: str = "HybridPolynomial"

    def __post_init__(self) -> None:
        self._fitted = False

    def fit(self, panel: FeaturePanel, train: pd.DatetimeIndex) -> None:
        self._mean = float(panel.frame.loc[train, panel.targets[0]].mean())
        self._levels = tuple(s for s in panel.signals if is_level(panel, s))
        self._flows = tuple(s for s in panel.signals if not is_level(panel, s))
        self._level = self._flow = None
        self._flow_mean = 0.0
        if self._levels:
            model = CarryResearchPolynomial()
            try:
                model.fit(replace(panel, signals=self._levels), train)
                self._level = model
            except _TOO_FEW:
                self._level = None
        if self._flows:
            model = self.flow()
            flows = replace(panel, signals=self._flows)
            try:
                model.fit(flows, train)
                self._flow = model
                self._flow_mean = float(model.predict(flows, train).mean())
            except _TOO_FEW:
                self._flow = None
        self._fitted = True

    def predict(self, panel: FeaturePanel, idx: pd.DatetimeIndex) -> pd.Series:
        if not self._fitted:
            raise RuntimeError("predict() called before fit()")
        if self._level is None and self._flow is None:
            return pd.Series(self._mean, index=idx, dtype=float)
        total = pd.Series(0.0, index=idx, dtype=float)
        if self._level is not None:
            total = total + self._level.predict(replace(panel, signals=self._levels), idx)
        if self._flow is not None:
            flow = self._flow.predict(replace(panel, signals=self._flows), idx)
            total = total + (flow - self._flow_mean if self._level is not None else flow)
        return total

    def describe(self) -> ModelDescription:
        if not self._fitted:
            raise RuntimeError("describe() called before fit()")
        level = None if self._level is None else self._level.describe()
        flow = None if self._flow is None else self._flow.describe()
        return _member_description(self.name, self._mean, level, flow, self._flow_mean)


def _member_description(
    name: str,
    mean: float,
    level: ModelDescription | None,
    flow: ModelDescription | None,
    flow_mean: float,
) -> ModelDescription:
    """One member's equation: the level terms and the flow terms, with the intercept
    the forecast's rule gives."""
    terms: list[str] = []
    coefficients: list[float] = []
    bounds: dict[str, tuple[float, float]] = {}
    standardisation: dict[str, tuple[float, float]] = {}
    intercept = mean
    for part in (level, flow):
        if part is None:
            continue
        terms += part.terms
        coefficients += part.coefficients
        bounds |= dict(part.input_bounds or {})
        standardisation |= dict(part.standardisation or {})
    if level is not None:
        intercept = float(level.intercept)
    if flow is not None:
        flow_intercept = float(flow.intercept)
        intercept = intercept + flow_intercept - flow_mean if level is not None else flow_intercept
    kept = [(t, float(c)) for t, c in zip(terms, coefficients, strict=True) if c != 0]
    return ModelDescription(
        name=name,
        terms=tuple(t for t, _ in kept),
        coefficients=tuple(c for _, c in kept),
        intercept=float(intercept),
        input_bounds=bounds,
        standardisation=standardisation,
    )


@dataclass
class _Part:
    """A member's level or flow part fitted on its own, left out (``_ABSENT``) where its
    fold has too few rows, as ``HybridPolynomial`` leaves it out."""

    make: Callable[[], Forecaster]
    name: str = field(default="Part", init=False)

    def fit(self, panel: FeaturePanel, train: pd.DatetimeIndex) -> None:
        model = self.make()
        try:
            model.fit(panel, train)
        except _TOO_FEW:
            model = None
        self._model = model

    def predict(self, panel: FeaturePanel, idx: pd.DatetimeIndex) -> pd.Series:
        if self._model is None:
            return pd.Series(np.nan, index=idx, dtype=float)
        return self._model.predict(panel, idx)

    def describe(self) -> ModelDescription:
        if self._model is None:
            return ModelDescription(_ABSENT, (), ())
        return self._model.describe()


# ── The 14, fold by fold ─────────────────────────────────────────────────────


def _evaluate_member(
    member: Member, panel: FeaturePanel, splitter: PurgedWalkForward, max_terms: int | None
) -> tuple[FoldResult, ...]:
    """One member's screened walk-forward: module-level, so a worker process can run it."""
    flow = flow_forecaster(member.fit, max_terms)
    columns = member_columns(panel, member)
    return evaluate(
        lambda: HybridPolynomial(flow, member.name),
        replace(panel, signals=columns),
        splitter,
        screen=True,
    )


def _part_columns(panel: FeaturePanel, task: tuple) -> tuple[str, ...]:
    kind, member = task
    columns = member_columns(panel, member)
    if kind == "level":
        return tuple(c for c in columns if is_level(panel, c))
    return tuple(c for c in columns if not is_level(panel, c))


def _evaluate_part(
    task: tuple, panel: FeaturePanel, splitter: PurgedWalkForward, max_terms: int | None
) -> tuple[FoldResult, ...] | None:
    """One part's screened walk-forward, or ``None`` if it has no inputs. A level task
    is any member of its window: every member there offers the same levels."""
    kind, member = task
    columns = _part_columns(panel, task)
    if not columns:
        return None
    make = CarryResearchPolynomial if kind == "level" else flow_forecaster(member.fit, max_terms)
    return evaluate(lambda: _Part(make), replace(panel, signals=columns), splitter, screen=True)


def _present(fold: FoldResult | None) -> bool:
    return (
        fold is not None
        and fold.screening is not None
        and not fold.screening.fell_back
        and fold.description.name != _ABSENT
    )


def _member_screening(
    columns: tuple[str, ...], parts: Sequence[FoldResult | None]
) -> FoldScreening:
    """What the member's own screening would have recorded: screening judges each input
    on its own, so it is its parts' screenings put together."""
    included: set[str] = set()
    ics: dict[str, float] = {}
    for part in parts:
        if part is not None and part.screening is not None:
            included |= set(part.screening.included)
            ics |= dict(part.screening.ics)
    return FoldScreening(
        candidates=columns,
        included=tuple(c for c in columns if c in included),
        ics={c: ics[c] for c in columns if c in ics},
    )


def _member_fold(
    member: Member,
    number: int,
    window: tuple[pd.DatetimeIndex, pd.DatetimeIndex],
    level: FoldResult | None,
    flow: FoldResult | None,
    panel: FeaturePanel,
) -> FoldResult:
    """The member's fold ``number``, on its window's ``(train, test)``, from its two
    parts (``None`` for a part with no inputs), exactly as ``HybridPolynomial`` and
    ``harness.evaluate`` would have produced it."""
    train, test = window
    screening = _member_screening(member_columns(panel, member), (level, flow))
    if screening.fell_back:
        return _training_mean_fold(number, train, test, panel, screening)
    target = panel.targets[0]
    mean = float(panel.frame.loc[train, target].mean())
    has_level, has_flow = _present(level), _present(flow)
    flow_mean = float(flow.predicted_train.mean()) if has_level and has_flow else 0.0

    def combined(idx: pd.DatetimeIndex, from_level, from_flow) -> pd.Series:
        if not (has_level or has_flow):
            return pd.Series(mean, index=idx, dtype=float)
        total = pd.Series(0.0, index=idx, dtype=float)
        if has_level:
            total = total + from_level
        if has_flow:
            total = total + (from_flow - flow_mean if has_level else from_flow)
        return total

    predicted = combined(
        test, level.predicted if has_level else None, flow.predicted if has_flow else None
    )
    predicted_train = combined(
        train,
        level.predicted_train if has_level else None,
        flow.predicted_train if has_flow else None,
    )
    description = _member_description(
        member.name,
        mean,
        level.description if has_level else None,
        flow.description if has_flow else None,
        flow_mean,
    )
    return FoldResult(
        fold=number,
        train=train,
        test=test,
        predicted=predicted,
        predicted_train=predicted_train,
        realised=panel.frame.loc[test, target],
        realised_train=panel.frame.loc[train, target],
        description=description,
        screening=screening,
        horizon=panel.horizon,
        own_level=_own_level(predicted_train),
    )


def member_folds(
    panel: FeaturePanel,
    splitter: PurgedWalkForward,
    max_terms: int | None,
    *,
    share_level_part: bool = True,
) -> list[tuple[FoldResult, ...]]:
    """Every member's walk-forward, in ``MEMBERS`` order.

    With ``share_level_part`` (the run's way) each window's level part is fitted once,
    not once per member: all seven members on a window offer the level part the same
    inputs on the same rows, and screening judges each input on its own, so it is the
    same fit. Each part's walk-forward runs in its own process. Without it, each member
    is fitted whole, as ``HybridPolynomial``; the two agree to rounding
    (tests/unit/test_hybrid.py)."""
    splitters = {False: splitter, True: capped(splitter, LONG_TRAIN_WINDOW)}
    if not share_level_part:
        return map_processes(
            _evaluate_member,
            MEMBERS,
            [panel] * len(MEMBERS),
            [splitters[m.long_window] for m in MEMBERS],
            [max_terms] * len(MEMBERS),
        )
    windows = {long: next(m for m in MEMBERS if m.long_window == long) for long in (True, False)}
    tasks = [("level", windows[True]), ("level", windows[False])]
    tasks += [("flow", m) for m in sorted(MEMBERS, key=lambda m: not m.long_window)]
    fits = map_processes(
        _evaluate_part,
        tasks,
        [panel] * len(tasks),
        [splitters[member.long_window] for _, member in tasks],
        [max_terms] * len(tasks),
    )
    parts = dict(zip(tasks, fits, strict=True))
    splits = {long: list(s.split(panel)) for long, s in splitters.items()}
    folds = []
    for member in MEMBERS:
        level = parts[("level", windows[member.long_window])]
        flow = parts[("flow", member)]
        folds.append(
            tuple(
                _member_fold(
                    member,
                    k,
                    window,
                    None if level is None else level[k],
                    None if flow is None else flow[k],
                    panel,
                )
                for k, window in enumerate(splits[member.long_window])
            )
        )
    return folds


# ── The model ────────────────────────────────────────────────────────────────


def bond_derived_folds(
    panel: FeaturePanel,
    splitter: PurgedWalkForward,
    max_terms: int | None,
    *,
    share_level_part: bool = True,
) -> tuple[FoldResult, ...]:
    """The model's walk-forward on ``splitter``'s test windows: per fold, the mean of the
    14 members' forecasts, their mean own level, and their equations as one. Its train
    window, for in-sample forecasts and crash diagnostics, is ``splitter``'s, which every
    longer-window fold's contains. Empty if ``splitter`` has no fold over ``panel``."""
    page = list(splitter.split(panel))
    if not page:
        return ()
    members = member_folds(panel, splitter, max_terms, share_level_part=share_level_part)
    return tuple(
        _combine(k, train, test, [m[k] for m in members], panel)
        for k, (train, test) in enumerate(page)
    )


def _combine(
    number: int,
    train: pd.DatetimeIndex,
    test: pd.DatetimeIndex,
    members: Sequence[FoldResult],
    panel: FeaturePanel,
) -> FoldResult:
    if not all(member.test.equals(test) for member in members):
        raise RuntimeError("the bond derived polynomial's fits disagree on a test window")
    target = panel.targets[0]
    predicted = pd.concat([m.predicted for m in members], axis=1).mean(axis=1, skipna=False)
    predicted_train = pd.concat([m.predicted_train.reindex(train) for m in members], axis=1).mean(
        axis=1, skipna=False
    )
    return FoldResult(
        fold=number,
        train=train,
        test=test,
        predicted=predicted,
        predicted_train=predicted_train,
        realised=panel.frame.loc[test, target],
        realised_train=panel.frame.loc[train, target],
        description=merge_descriptions([m.description for m in members]),
        screening=_combined_screening(panel, members),
        horizon=panel.horizon,
        own_level=float(np.mean([fold_baseline(m) for m in members])),
    )


def _combined_screening(panel: FeaturePanel, members: Sequence[FoldResult]) -> FoldScreening:
    """Every input of both windows, kept if any member kept it."""
    included: set[str] = set()
    ics: dict[str, float] = {}
    for member in members:
        if member.screening is not None:
            included |= set(member.screening.included)
            ics |= dict(member.screening.ics)
    return FoldScreening(
        candidates=panel.signals,
        included=tuple(s for s in panel.signals if s in included),
        ics={s: ics[s] for s in panel.signals if s in ics},
    )


def merge_descriptions(descriptions: Sequence[ModelDescription]) -> ModelDescription:
    """The mean of ``descriptions`` as one polynomial: each term's coefficients summed and
    divided by how many there are, a term whose mean is zero dropped, the intercepts
    averaged. Each input a kept term reads carries the clip range and standardisation
    every description that reads it agrees on; within one train window they always do.

    The terms are in the page's window first, then the longer window's, each by
    degree and then in the order the members first use them."""
    n = len(descriptions)
    sums: dict[str, float] = {}
    bounds: dict[str, tuple[float, float]] = {}
    centre: dict[str, tuple[float, float]] = {}
    for description in descriptions:
        for term, coefficient in zip(description.terms, description.coefficients, strict=True):
            if coefficient != 0:
                sums[term] = sums.get(term, 0.0) + float(coefficient)
        for name in _inputs_read(description):
            _agree(bounds, name, (description.input_bounds or {})[name])
            _agree(centre, name, (description.standardisation or {})[name])
    intercept = sum(float(d.intercept) for d in descriptions) / n
    kept = [(term, total / n) for term, total in sums.items() if total != 0]
    order = {term: position for position, (term, _) in enumerate(kept)}
    kept.sort(key=lambda tc: (_term_window(tc[0]), _degree(tc[0]), order[tc[0]]))
    used = dict.fromkeys(f for term, _ in kept for f, _ in _factors(term))
    return ModelDescription(
        name=NAME,
        terms=tuple(t for t, _ in kept),
        coefficients=tuple(c for _, c in kept),
        intercept=intercept,
        input_bounds={s: bounds[s] for s in used if s in bounds},
        standardisation={s: centre[s] for s in used if s in centre},
    )


def _factors(term: str) -> list[tuple[str, int]]:
    """A ``PolynomialFeatures`` term's ``(input, power)`` factors."""
    out = []
    for part in term.split(" "):
        match = _POWER.match(part)
        out.append((match.group(1), int(match.group(2))) if match else (part, 1))
    return out


def _degree(term: str) -> int:
    return sum(power for _, power in _factors(term))


def _term_window(term: str) -> bool:
    return any(is_long_window(name) for name, _ in _factors(term))


def _inputs_read(description: ModelDescription) -> list[str]:
    """The inputs ``description``'s kept terms read. Every input it standardises, if a
    term can't be split into them (a column whose name holds a space)."""
    own = description.input_bounds or {}
    names = [name for term in description.terms for name, _ in _factors(term)]
    if all(name in own for name in names):
        return list(dict.fromkeys(names))
    return list(own)


def _agree(store: dict[str, tuple[float, float]], name: str, value: tuple[float, float]) -> None:
    if name not in store:
        store[name] = (float(value[0]), float(value[1]))
    elif not np.allclose(store[name], value, rtol=1e-12, atol=0.0):
        raise RuntimeError(
            f"two fits of the bond derived polynomial read {name} with different clip ranges "
            "or standardisations, so their mean is not one polynomial"
        )


def run_bond_derived_polynomial(
    frame: pd.DataFrame,
    signal_cols: Sequence[str],
    transforms: Mapping[str, Transform],
    price_col: str,
    horizon: int,
    splitter: PurgedWalkForward,
    *,
    max_terms: int | None,
) -> tuple[ModelRunResult, ModelDescription, FeaturePanel]:
    """The bond derived polynomial walked forward on ``splitter``'s test windows.

    ``frame`` is the committed data indexed by date, ``signal_cols`` and ``transforms``
    the page's signals and how it reads them, ``max_terms`` the page's cap on each lasso
    fit's terms. Returns the result, the latest fold's equation, and the panel whose
    columns that equation reads (``function_transforms`` gives each one's transform)."""
    panel = bond_derived_panel(frame, signal_cols, transforms, price_col, horizon)
    folds = bond_derived_folds(panel, splitter, max_terms)
    if not folds:
        raise BondDerivedConfigError(splitter.too_short(panel))
    result, description = summarize(folds)
    return result, description, panel
