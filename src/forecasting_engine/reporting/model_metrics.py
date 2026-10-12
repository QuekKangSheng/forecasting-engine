"""Formatting rules for the model-comparison table.

``ModelRunResult`` is a placeholder input contract pending the team's
run-store design.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field

import pandas as pd

from forecasting_engine.validation.crash import CrashDiagnostics
from forecasting_engine.validation.gates import SIGNAL_RANK_IC_GATE, SIGNIFICANCE_T_GATE

MODEL_ORDER: tuple[str, ...] = (
    "Naive (training mean)",
    "FF5 Benchmark",
    "Polynomial (derived)",
    "Research model (fixed inputs)",
    "Polynomial (user-supplied)",
    "Machine Learning",
)
"""The order rows appear in; only models that ran get one."""

_COLUMNS: tuple[str, ...] = (
    "IC",
    "Signal Rank IC",
    "t",
    "OOS Rank IC",
    "RMSE",
    "Crash Recall",
    "Crash Precision",
    "Crash F1",
    "Rows scored",
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
    """One model family's completed run."""

    ic: float
    oos_rank_ic: float
    rmse: float
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
    constant_forecasts: ConstantForecasts | None = None
    forecast: pd.Series | None = None
    """Every fold's out-of-sample prediction, pooled end to end and indexed by
    date — captured here so a consumer can read it without re-running the model."""
    realised: pd.Series | None = None
    """The realised forward return on each of ``forecast``'s dates — what each
    forecast is scored against."""
    signal_rank_ic: float | None = None
    """Pooled Rank IC of each forecast less its fold's training-mean target
    (``validation.harness.signal_forecasts``): what the signals add. The gated
    figure. ``None`` only on a result built before this field existed."""
    signal_rank_ic_se: float | None = None
    """Newey-West standard error of ``signal_rank_ic`` over h - 1 lags."""
    signal_rank_ic_se_test: float | None = None
    """The same over test-window lags."""
    baseline: pd.Series | None = None
    """Each of ``forecast``'s dates' fold training-mean target: ``forecast`` less
    this is the signal's contribution, which the portfolio optimiser tilts on."""

    @property
    def gated_rank_ic(self) -> float:
        """The Rank IC the gate reads: the Signal Rank IC, or NaN (which fails)
        for a result saved before it was computed."""
        return float("nan") if self.signal_rank_ic is None else self.signal_rank_ic

    @property
    def signal_t(self) -> float:
        """The Signal Rank IC over the larger of its two standard errors: the
        test-window one also allows for errors a fold's single fit shares, so it
        is the harder bar. NaN (which fails the gate) without a standard error."""
        errors = [
            se
            for se in (self.signal_rank_ic_se, self.signal_rank_ic_se_test)
            if se is not None and se == se
        ]
        if not errors or max(errors) <= 0:
            return float("nan")
        return self.gated_rank_ic / max(errors)

    @property
    def signal(self) -> pd.Series | None:
        """``forecast`` less ``baseline``, or ``None`` if either is missing."""
        if self.forecast is None or self.baseline is None:
            return None
        return self.forecast - self.baseline


def build_metrics_rows(
    results: Mapping[str, ModelRunResult],
    decimals: int = 4,
    gated: Collection[str] = (),
) -> list[dict[str, Cell]]:
    """One row per model in ``results``, in MODEL_ORDER (then any others).

    A model named in ``gated`` has its Signal Rank IC and t badged against their
    gates; any other (the naive baseline) is shown without badges."""
    names = [n for n in MODEL_ORDER if n in results] + [n for n in results if n not in MODEL_ORDER]
    return [_row(name, results[name], decimals, name in gated) for name in names]


def _row(name: str, result: ModelRunResult, decimals: int, gated: bool) -> dict[str, Cell]:
    signal_text = _with_errors(
        result.gated_rank_ic, result.signal_rank_ic_se, result.signal_rank_ic_se_test, decimals
    )
    t_text = _fmt(result.signal_t, 2)
    if gated:
        signal_cell = _gated_cell(signal_text, result.gated_rank_ic > SIGNAL_RANK_IC_GATE)
        t_cell = _gated_cell(t_text, result.signal_t > SIGNIFICANCE_T_GATE)
    else:
        signal_cell, t_cell = Cell(signal_text), Cell(t_text)

    return {
        "Model": Cell(name),
        "IC": Cell(_fmt(result.ic, decimals)),
        "Signal Rank IC": signal_cell,
        "t": t_cell,
        "OOS Rank IC": Cell(
            _with_errors(
                result.oos_rank_ic, result.oos_rank_ic_se, result.oos_rank_ic_se_test, decimals
            )
        ),
        "RMSE": Cell(_fmt(result.rmse, decimals)),
        "Crash Recall": Cell(_fmt(result.crash.recall, decimals)),
        "Crash Precision": Cell(_fmt(result.crash.precision, decimals)),
        "Crash F1": Cell(_fmt(result.crash.f1, decimals)),
        "Rows scored": Cell("—" if result.rows_scored is None else f"{result.rows_scored:,}"),
        "Constant folds": Cell(
            "—"
            if result.constant_forecasts is None
            else f"{result.constant_forecasts.constant} of {result.constant_forecasts.folds}"
        ),
    }


def _with_errors(value: float, se_h: float | None, se_test: float | None, decimals: int) -> str:
    text = _fmt(value, decimals)
    errors = [
        f"{name} {_fmt(se, decimals)}"
        for name, se in (("s.e. (h−1 lags)", se_h), ("s.e. (test-window lags)", se_test))
        if se is not None
    ]
    return f"{text} ({'; '.join(errors)})" if errors else text


def _gated_cell(text: str, passed: bool) -> Cell:
    return Cell(text, "success") if passed else Cell(text, "danger")


def _fmt(value: float, decimals: int) -> str:
    return "—" if value != value else f"{value:.{decimals}f}"  # value != value: NaN
