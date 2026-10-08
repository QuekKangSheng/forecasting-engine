"""The shared walk-forward harness: fit and predict any Forecaster, fold by fold, and
turn the result into the fixed contract FYP-45/FYP-14 already built the comparison view
and promotion gates against.

FYP-45 already built what happens *after* a model produces fold-by-fold predictions —
``pbo.compute_pbo``, ``crash.crash_diagnostics_over_folds``, ``gates.evaluate_candidate``
— but nothing fit a model and produced those predictions in the first place. This module
is that missing piece: "plug a model into the shared walk-forward harness" means running
it through ``evaluate()``, then ``summarize()`` (one configuration) or
``select_best_candidate()`` + ``summarize()`` (several configurations to compare, like
Polynomial's degree/regularizer grid or Boosted's tuned XGBoost vs. LightGBM).

``evaluate()`` deliberately returns raw predictions and realised values per fold rather
than pre-scored metrics, because each downstream consumer scores differently: crash
diagnostics want the raw series, IC/RankIC/RMSE get averaged across folds, and PBO
needs several *different* models' fold results compared against each other, not one
model's alone (so ``summarize()`` takes an already-computed PBO rather than computing
one itself).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace

import pandas as pd

from forecasting_engine.features.screening import screen_over_folds
from forecasting_engine.ingest.align import FeaturePanel
from forecasting_engine.models.base import Forecaster, ModelDescription
from forecasting_engine.reporting.model_metrics import (
    ConstantForecasts,
    FoldTerms,
    ModelRunResult,
    ScreeningSummary,
)
from forecasting_engine.validation import metrics
from forecasting_engine.validation.crash import (
    CrashDiagnostics,
    crash_diagnostics,
    label_crash_days,
)
from forecasting_engine.validation.pbo import N_BLOCKS, compute_pbo
from forecasting_engine.validation.splitters import PurgedWalkForward

TRAINING_MEAN: str = "TrainingMean"
"""The description name of a fold that kept no signal and forecast its training mean."""


@dataclass(frozen=True)
class FoldScreening:
    """What one fold's screening considered and kept, and so what the fold was fit on."""

    candidates: tuple[str, ...]
    included: tuple[str, ...]
    """May be empty: every signal failed screening on this fold's train window."""
    ics: Mapping[str, float] = field(default_factory=dict)
    """Each candidate's rank IC on this fold's train window."""

    @property
    def fitted(self) -> tuple[str, ...]:
        """The signals this fold was fit on: exactly those screening kept."""
        return self.included

    @property
    def fell_back(self) -> bool:
        """Screening kept no signal, so the fold fell back to forecasting its
        training mean rather than fitting on signals that just failed the gate."""
        return not self.included


@dataclass(frozen=True)
class FoldResult:
    """One walk-forward fold's fit/predict outcome."""

    fold: int
    train: pd.DatetimeIndex
    test: pd.DatetimeIndex
    predicted: pd.Series
    """Predictions over ``test``, indexed the same way."""
    predicted_train: pd.Series
    """Predictions over ``train`` (in-sample, from the same fit). Crash-day
    labelling (``validation/crash.py``) calibrates its flag threshold from a
    model's own in-sample predicted distribution before applying it
    out-of-sample — it needs both, not just the test-window predictions."""
    realised: pd.Series
    """``panel``'s target over ``test`` — what ``predicted`` is scored against."""
    realised_train: pd.Series
    """``panel``'s target over ``train`` — crash-day labelling's tail threshold
    is computed from the *realised* training-window return distribution."""
    description: ModelDescription
    """This fold's fitted model, described — a degree-5 derived fit can pick
    different terms fold to fold, so the description travels with the fold
    rather than being reported once for the whole run."""
    screening: FoldScreening | None = None
    """Which signals this fold's screening kept and fit on, or ``None`` when
    ``evaluate()`` ran without screening."""
    horizon: int = 1
    """The panel's label horizon, which sets how far apart errors stay correlated."""


def evaluate(
    make_forecaster: Callable[[], Forecaster],
    panel: FeaturePanel,
    splitter: PurgedWalkForward,
    *,
    screen: bool = False,
) -> tuple[FoldResult, ...]:
    """Fit a fresh Forecaster per fold on its train window, predict on its test window.

    ``make_forecaster`` is a factory, not a shared instance: each fold fits
    independently, so reusing one fitted instance across folds would let an
    earlier fold's fit leak into a later one's prediction.

    ``screen=True`` re-screens ``panel.signals`` for each fold via
    ``features.screening.screen_over_folds``, using only that fold's own train
    window, and fits/predicts on the signals it included — the walk-forward
    re-evaluation FYP-108/110 call for. A fold whose screening excludes every
    signal fits no model: it forecasts its training window's mean target for
    every row. Off by default: a caller whose Forecaster is given its features directly (a
    user-supplied formula, a fixed factor benchmark) has nothing for screening
    to filter.
    """
    target = panel.targets[0]
    folds = list(splitter.split(panel))
    per_fold_screen = screen_over_folds(panel, folds) if screen else None
    results = []
    for fold, (train_idx, test_idx) in enumerate(folds):
        fold_panel = panel
        screening = None
        if per_fold_screen is not None:
            scores = per_fold_screen[fold]
            screening = FoldScreening(
                candidates=panel.signals,
                included=tuple(s.signal for s in scores if s.included),
                ics={s.signal: s.ic for s in scores},
            )
            if screening.fell_back:
                results.append(_training_mean_fold(fold, train_idx, test_idx, panel, screening))
                continue
            # The fold is fit on exactly what's recorded, so a display built from
            # ``screening`` can't disagree with what the model actually used.
            fold_panel = replace(panel, signals=screening.fitted)
        forecaster = make_forecaster()
        forecaster.fit(fold_panel, train_idx)
        predicted = forecaster.predict(fold_panel, test_idx)
        predicted_train = forecaster.predict(fold_panel, train_idx)
        realised = panel.frame.loc[test_idx, target]
        realised_train = panel.frame.loc[train_idx, target]
        results.append(
            FoldResult(
                fold=fold,
                train=train_idx,
                test=test_idx,
                predicted=predicted,
                predicted_train=predicted_train,
                realised=realised,
                realised_train=realised_train,
                description=forecaster.describe(),
                screening=screening,
                horizon=panel.horizon,
            )
        )
    return tuple(results)


def _training_mean_fold(
    fold: int,
    train: pd.DatetimeIndex,
    test: pd.DatetimeIndex,
    panel: FeaturePanel,
    screening: FoldScreening,
) -> FoldResult:
    """A fold with no screened signal: the training window's mean target, for every
    test and train row, and a description with no terms."""
    target = panel.targets[0]
    mean = float(panel.frame.loc[train, target].mean())
    return FoldResult(
        fold=fold,
        train=train,
        test=test,
        predicted=pd.Series(mean, index=test, dtype=float),
        predicted_train=pd.Series(mean, index=train, dtype=float),
        realised=panel.frame.loc[test, target],
        realised_train=panel.frame.loc[train, target],
        description=ModelDescription(name=TRAINING_MEAN, terms=(), coefficients=(), intercept=mean),
        screening=screening,
        horizon=panel.horizon,
    )


def summarize(
    folds: tuple[FoldResult, ...], *, pbo: float | None = None
) -> tuple[ModelRunResult, ModelDescription]:
    """IC, Rank IC and RMSE over every fold's test predictions pooled together,
    crash diagnostics, and the most recent fold's fitted description — the shape
    every ``run_*`` function bridges its own model into.

    Pooling scores one long out-of-sample series rather than averaging short,
    noisy per-fold scores. Each fold's predictions come from its own fit, so a
    pooled Pearson IC mixes those fits' scales; Rank IC does not care about
    scale. Two Newey-West standard errors of each pooled Rank IC are reported:
    one over h - 1 lags, for the overlap between neighbouring h-day labels, and
    one over as many lags as a test window has rows, for errors a fold's single
    fit shares.

    The gate is set on the Signal Rank IC: the pooled Rank IC of what the
    signals add to each fold's training-mean return (``signal_forecasts``). The
    plain pooled Rank IC also scores the forecast's level, which shifts with
    each fold's training mean whatever the signals say, so a model with no
    signal can score there and one with a real signal can be swamped there.

    Requires at least one fold. Callers should check ``evaluate()``'s output is
    non-empty themselves and raise their own domain-appropriate error message
    (e.g. "the committed dataset is too short for this train/test/embargo
    window") before calling this — the message belongs with the caller who
    knows what a portfolio manager should be told to fix.
    """
    predicted, realised = pooled(folds)
    within = within_fold_ranks(folds)
    signal = signal_forecasts(folds)
    test_window_lags = max(len(f.test) for f in folds)
    result = ModelRunResult(
        ic=metrics.ic(predicted, realised),
        oos_rank_ic=metrics.rank_ic(predicted, realised),
        oos_rank_ic_se=metrics.rank_ic_se(predicted, realised, lags=folds[0].horizon - 1),
        oos_rank_ic_se_test=metrics.rank_ic_se(predicted, realised, lags=test_window_lags),
        rmse=metrics.rmse(predicted, realised),
        pbo=pbo,
        crash=_crash_over_folds(folds),
        screening=_screening_summary(folds),
        terms=FoldTerms(folds=len(folds), with_terms=sum(1 for f in folds if f.description.terms)),
        rows_scored=sum(int((f.predicted.notna() & f.realised.notna()).sum()) for f in folds),
        oos_rank_ic_within=metrics.rank_ic(within, realised),
        oos_rank_ic_within_se=metrics.rank_ic_se(within, realised, lags=test_window_lags),
        constant_forecasts=ConstantForecasts(
            folds=len(folds),
            constant=sum(1 for f in folds if f.predicted.dropna().nunique() <= 1),
        ),
        forecast=predicted,
        realised=realised,
        signal_rank_ic=metrics.rank_ic(signal, realised),
        signal_rank_ic_se=metrics.rank_ic_se(signal, realised, lags=folds[0].horizon - 1),
        signal_rank_ic_se_test=metrics.rank_ic_se(signal, realised, lags=test_window_lags),
        baseline=predicted - signal,
    )
    # FYP-122's "deliverable artifact": the most recent fold's fitted terms
    # and coefficients — a fit can pick different terms fold to fold, so this
    # is what the model would use if deployed today, not an average across a
    # walk-forward run's whole history.
    return result, folds[-1].description


def select_best_candidate(
    per_candidate: dict[str, tuple[FoldResult, ...]], *, n_blocks: int = N_BLOCKS
) -> tuple[str, float]:
    """Compare several named candidates' fold results via PBO, return the name of
    the one with the best pooled OOS Rank IC and the shared PBO score every
    candidate was judged against.

    Used by any ``run_*`` function that has more than one fixed configuration to
    choose between — ``DerivedPolynomial``'s degree/regularizer grid,
    ``BoostedForecaster``'s tuned XGBoost vs. tuned LightGBM. A single
    configuration (FF5, a user-supplied polynomial) has nothing to compare
    against and reports ``pbo=None`` directly to ``summarize()`` instead of
    calling this. Selection, PBO and the gate all use the Signal Rank IC.
    """
    pooled_by_name = {
        name: (signal_forecasts(folds), pooled(folds)[1]) for name, folds in per_candidate.items()
    }
    pbo_result = compute_pbo(pooled_by_name, n_blocks=n_blocks)
    scores = {name: _finite(metrics.rank_ic(*pair)) for name, pair in pooled_by_name.items()}
    best_name = max(scores, key=scores.__getitem__)
    return best_name, pbo_result.pbo


def within_fold_ranks(folds: tuple[FoldResult, ...]) -> pd.Series:
    """Every fold's test predictions as ranks within that fold, centred on zero
    and scaled to [-0.5, 0.5], end to end.

    Pooled Rank IC ranks forecasts across folds, so a forecast whose level merely
    shifts between folds can score without ranking any day within one. These
    ranks carry no level: a fold's forecasts only compete with each other. A fold
    that forecasts one value throughout ranks every day equal, so its ranks are
    exactly zero and it adds nothing.
    """
    parts = []
    for f in folds:
        ranks = f.predicted.rank()
        n = int(ranks.notna().sum())
        parts.append((ranks - (n + 1) / 2) / max(n, 1))
    return pd.concat(parts)


def fold_baseline(fold: FoldResult) -> float:
    """The fold's training-window mean target: what the naive baseline forecasts
    for it, and so the level a forecast is judged relative to."""
    return float(fold.realised_train.mean())


def signal_forecasts(folds: tuple[FoldResult, ...]) -> pd.Series:
    """Every fold's test predictions less that fold's training-window mean target,
    end to end: what the signals add to the naive forecast.

    A forecast's level moves with its fold's training mean whatever the signals
    say, and on daily equity data that trailing mean ranks future returns the
    wrong way (returns mean-revert). Scored pooled, the level can swamp a real
    signal or pass one that has none. Taking the training mean away leaves what
    the model itself contributes. A fold that forecasts its training mean
    contributes exactly zero.
    """
    return pd.concat([f.predicted - fold_baseline(f) for f in folds])


def pooled(folds: tuple[FoldResult, ...]) -> tuple[pd.Series, pd.Series]:
    """Every fold's test predictions and realised values, end to end."""
    return pd.concat([f.predicted for f in folds]), pd.concat([f.realised for f in folds])


def _finite(value: float) -> float:
    return value if value == value else float("-inf")  # NaN ranks last


def _crash_over_folds(folds: tuple[FoldResult, ...]) -> CrashDiagnostics:
    """label_crash_days() per fold using *that fold's own* train+test
    predictions, never another fold's. A walk-forward run refits per fold, so
    consecutive folds' train windows overlap and were scored by different
    models — the shared ``crash_diagnostics_over_folds`` helper assumes one
    continuous prediction series from a single model and would either mix
    fits or need one that doesn't exist here. Concatenating each fold's own
    correctly-scoped labels before scoring reproduces its behaviour without
    that assumption.
    """
    labelled = [
        label_crash_days(
            pd.concat([f.predicted_train, f.predicted]),
            pd.concat([f.realised_train, f.realised]),
            f.train,
            f.test,
        )
        for f in folds
    ]
    combined = pd.concat(labelled)
    return crash_diagnostics(combined["flagged"], combined["true_tail"])


def _screening_summary(folds: tuple[FoldResult, ...]) -> ScreeningSummary | None:
    """Count, per signal, the folds that fit it — ``None`` if no fold screened.

    A fold that kept no signal forecast its training mean, so it counts towards
    none. Every candidate is listed, including one no fold used, since a signal
    screened out everywhere is the most useful row.
    """
    screened = [f.screening for f in folds if f.screening is not None]
    if not screened:
        return None
    counts = dict.fromkeys(screened[0].candidates, 0)
    for screening in screened:
        for signal in screening.fitted:
            counts[signal] = counts.get(signal, 0) + 1
    return ScreeningSummary(
        folds=len(screened),
        fell_back=sum(s.fell_back for s in screened),
        counts=tuple(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        fold_ics=tuple(dict(s.ics) for s in screened),
        latest_included=screened[-1].included,
    )
