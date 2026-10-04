"""Formatting rules for the model-comparison table.

``ModelRunResult`` is a placeholder input contract pending the team's
run-store design.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import pandas as pd

from forecasting_engine.validation.crash import CrashDiagnostics
from forecasting_engine.validation.gates import OOS_RANK_IC_GATE, PBO_GATE

MODEL_ORDER: tuple[str, ...] = (
    "Naive (training mean)",
    "FF5 Benchmark",
    "Polynomial (derived)",
    "Polynomial (user-supplied)",
    "Machine Learning",
)
"""The order rows appear in; only models that ran get one."""

NO_CONFIG_SEARCH = "N/A — no configuration search"

SIGNIFICANCE_SE_MULTIPLE: float = 2.0
"""A pooled OOS Rank IC further than this many standard errors from zero is shown
as distinguishable from luck. Diagnostic only: it never enters the gate."""

_COLUMNS: tuple[str, ...] = (
    "IC",
    "OOS Rank IC",
    "RMSE",
    "PBO",
    "Crash Recall",
    "Crash Precision",
    "Crash F1",
    "Rows scored",
    "Rank IC within folds",
    "Beyond 2 s.e.",
    "Constant folds",
)


@dataclass(frozen=True)
class Cell:
    """One table cell. ``tone`` is "success"/"danger" only on a gated
    metric; diagnostics and N/A/Not-run cells stay "neutral"."""

    text: str
    tone: str = field(default="neutral")


@dataclass(frozen=True)
class ScreeningSummary:
    """How many walk-forward folds fit each signal, for a run that screened per fold.

    Counts what each fold was actually fit on. A fold whose screening kept no
    signal fits no model and forecasts its training mean, so it counts towards no
    signal here, and ``fell_back`` says how many folds did that.
    """

    folds: int
    fell_back: int
    counts: tuple[tuple[str, int], ...]
    """``(signal, folds that fit it)``, most-used first. A signal no fold used is
    listed with zero rather than left out — that is what the table exists to show."""
    fold_ics: tuple[Mapping[str, float], ...] = ()
    """Per fold, in order, each candidate's rank IC on that fold's train window."""
    latest_included: tuple[str, ...] = ()
    """The signals the most recent fold's screening kept (before any fallback)."""

    @property
    def latest_ics(self) -> Mapping[str, float]:
        return self.fold_ics[-1] if self.fold_ics else {}


@dataclass(frozen=True)
class FoldTerms:
    """How many of a run's folds ended with any term at all.

    A regularized fit can zero every coefficient on one fold and keep several on
    the next, and a run reports only its most recent fold's equation. Without
    this count that one equation reads as the whole run's answer.
    """

    folds: int
    with_terms: int

    @property
    def every_fold(self) -> bool:
        return self.with_terms == self.folds


@dataclass(frozen=True)
class ConstantForecasts:
    """How many of a run's folds forecast a single value across their test window.

    Such a fold ranks nothing, so it has no Rank IC of its own; pooled with the
    other folds it still counts, through its level alone. The count says how much
    of a run that describes.
    """

    folds: int
    constant: int


@dataclass(frozen=True)
class ModelRunResult:
    """One model family's completed run. ``pbo`` is ``None`` for FF5 (no
    configuration search) — also the signal that row skips the gate badge."""

    ic: float
    oos_rank_ic: float
    rmse: float
    pbo: float | None
    crash: CrashDiagnostics
    screening: ScreeningSummary | None = None
    """Per-fold signal inclusion, or ``None`` for a run that didn't screen (FF5, a
    user-supplied formula): those are handed their features and have nothing to
    filter."""
    terms: FoldTerms | None = None
    """How many folds kept any term. ``None`` only on a result built before this
    field existed (one parked in session state by an older run)."""
    oos_rank_ic_se: float | None = None
    """Newey-West standard error of ``oos_rank_ic`` over h - 1 lags (the overlap
    between labels), or ``None`` if not computed."""
    oos_rank_ic_se_test: float | None = None
    """The same over as many lags as a walk-forward test window has rows, allowing
    for errors shared within a fold's single fit."""
    rows_scored: int | None = None
    """Test rows with both a prediction and a realised value. Each model drops
    rows missing a signal it uses, so this can differ between models."""
    oos_rank_ic_within: float | None = None
    """Rank IC of each fold's forecasts ranked within that fold, pooled. A
    forecast that only shifts level from fold to fold scores nothing here, while
    it can score on ``oos_rank_ic``. Diagnostic; not gated."""
    oos_rank_ic_within_se: float | None = None
    """Newey-West standard error of ``oos_rank_ic_within`` over test-window lags."""
    constant_forecasts: ConstantForecasts | None = None
    forecast: pd.Series | None = None
    """Every fold's out-of-sample prediction, pooled end to end and indexed by
    date — captured here so a consumer can read it without re-running the model."""
    realised: pd.Series | None = None
    """The realised forward return on each of ``forecast``'s dates — what each
    forecast is scored against."""


def build_metrics_rows(
    results: Mapping[str, ModelRunResult], decimals: int = 4
) -> list[dict[str, Cell]]:
    """One row per model in ``results``, in MODEL_ORDER (then any others)."""
    names = [n for n in MODEL_ORDER if n in results] + [
        n for n in results if n not in MODEL_ORDER
    ]
    return [_row(name, results[name], decimals) for name in names]


def _row(name: str, result: ModelRunResult, decimals: int) -> dict[str, Cell]:
    can_be_gated = result.pbo is not None
    rank_ic_text = _fmt(result.oos_rank_ic, decimals)
    errors = [
        f"{name} {_fmt(se, decimals)}"
        for name, se in (
            ("s.e. (h−1 lags)", result.oos_rank_ic_se),
            ("s.e. (test-window lags)", result.oos_rank_ic_se_test),
        )
        if se is not None
    ]
    if errors:
        rank_ic_text += f" ({'; '.join(errors)})"
    oos_rank_ic_cell = (
        _gated_cell(rank_ic_text, result.oos_rank_ic > OOS_RANK_IC_GATE)
        if can_be_gated
        else Cell(rank_ic_text)
    )
    pbo_cell = (
        Cell(NO_CONFIG_SEARCH)
        if result.pbo is None
        else _gated_cell(_fmt(result.pbo, decimals), result.pbo <= PBO_GATE)
    )

    return {
        "Model": Cell(name),
        "IC": Cell(_fmt(result.ic, decimals)),
        "OOS Rank IC": oos_rank_ic_cell,
        "RMSE": Cell(_fmt(result.rmse, decimals)),
        "PBO": pbo_cell,
        "Crash Recall": Cell(_fmt(result.crash.recall, decimals)),
        "Crash Precision": Cell(_fmt(result.crash.precision, decimals)),
        "Crash F1": Cell(_fmt(result.crash.f1, decimals)),
        "Rows scored": Cell("—" if result.rows_scored is None else f"{result.rows_scored:,}"),
        "Rank IC within folds": _within_cell(result, decimals),
        "Beyond 2 s.e.": _significance_cell(result),
        "Constant folds": Cell(
            "—"
            if result.constant_forecasts is None
            else f"{result.constant_forecasts.constant} of {result.constant_forecasts.folds}"
        ),
    }


def _within_cell(result: ModelRunResult, decimals: int) -> Cell:
    within = result.oos_rank_ic_within
    if within is None or within != within:  # NaN: no fold's forecasts ranked any day
        return Cell("—")
    text = _fmt(within, decimals)
    se = result.oos_rank_ic_within_se
    if se is not None and se == se:
        text += f" (s.e. {_fmt(se, decimals)})"
    return Cell(text)


def _significance_cell(result: ModelRunResult) -> Cell:
    """Judged on the larger of the two standard errors: the test-window one also
    allows for errors a fold's single fit shares, so it is the harder bar."""
    errors = [
        se
        for se in (result.oos_rank_ic_se, result.oos_rank_ic_se_test)
        if se is not None and se == se
    ]
    if not errors or result.oos_rank_ic != result.oos_rank_ic:
        return Cell("—")
    beyond = abs(result.oos_rank_ic) > SIGNIFICANCE_SE_MULTIPLE * max(errors)
    return Cell("Yes" if beyond else "No")


def _gated_cell(text: str, passed: bool) -> Cell:
    return Cell(text, "success") if passed else Cell(text, "danger")


def _fmt(value: float, decimals: int) -> str:
    return "—" if value != value else f"{value:.{decimals}f}"  # value != value: NaN
