"""FYP-44: the machine-learning Forecaster — XGBoost and LightGBM, Optuna-tuned on a
rolling schedule, SHAP feature attribution.

**Refit cadence (FYP-129).** Tuning is expensive, so it does not happen every fold.
The first tune uses the tuning period — the ``splitter.tuning_rows`` rows before the
first test window, which no test window ever overlaps. After that, hyperparameters
are re-tuned every ``RETUNE_EVERY`` rows, each time on the ``tuning_rows`` rows just
before the next test window, so tuning only ever sees the past. Each fold refits
with the most recent tune's *fixed* hyperparameters, the same "fix the
configuration, refit per fold" shape ``DerivedPolynomial`` uses for its grid. Each
tune scores a trial by pooled Rank IC over a mini walk-forward inside its window,
with the main run's train/test/embargo — the same metric selection, PBO and the
gate use.

PBO compares the two tuned candidates (XGBoost vs. LightGBM) and does not count
Optuna's trials: tuning has its own period, so its search never touches the rows
PBO and the reported metrics are computed on.

**FYP-149's SHAP feature attribution** reuses ``ModelDescription``'s existing
terms/coefficients shape (feature name -> mean |SHAP value|) rather than extending the
contract.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace

import numpy as np
import optuna
import pandas as pd
import shap
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from forecasting_engine.ingest.align import FeaturePanel
from forecasting_engine.models.base import ModelDescription
from forecasting_engine.reporting.model_metrics import ModelRunResult
from forecasting_engine.validation.harness import (
    FoldResult,
    evaluate,
    pooled,
    select_best_candidate,
    summarize,
)
from forecasting_engine.validation.metrics import rank_ic
from forecasting_engine.validation.pbo import N_BLOCKS
from forecasting_engine.validation.splitters import PurgedWalkForward

optuna.logging.set_verbosity(optuna.logging.WARNING)

_LIBRARIES: dict[str, type] = {"xgboost": XGBRegressor, "lightgbm": LGBMRegressor}

#: Fixed, non-tuned settings per library. ``subsample_freq`` is what makes
#: LightGBM honour the tuned ``subsample`` at all — without it the fraction is
#: silently ignored.
_FIXED_PARAMS: dict[str, dict] = {
    "xgboost": {"verbosity": 0},
    "lightgbm": {"verbose": -1, "subsample_freq": 1},
}

#: Fewest training rows a leaf may be built on, named per library: XGBoost counts
#: rows through the hessian (one per row for squared error), LightGBM directly.
_LEAF_KEYS: dict[str, str] = {"xgboost": "min_child_weight", "lightgbm": "min_child_samples"}

N_TRIALS: int = 50
"""Optuna trials per library for the first tune."""

RETUNE_TRIALS: int = 20
"""Optuna trials per library for each re-tune, which starts from the previous best."""

RETUNE_EVERY: int = 252
"""Rows between re-tunes: about a year of trading days."""

MIN_TUNING_FOLDS: int = 3
"""Fewest mini walk-forward folds a tuning window must hold to score a trial."""

_MIN_TRAINING_ROWS: int = 15
"""Same threshold FamaFrench5 uses — below this a fit is more noise than signal."""


class BoostedConfigError(ValueError):
    """A tuning/fitting request or the merged data is invalid.

    The message is written for a portfolio manager and is safe to render
    directly in the dashboard — mirrors PolynomialConfigError/FamaFrenchDataError.
    """


@dataclass(frozen=True)
class Tune:
    """One Optuna search and what it chose for each library."""

    first: pd.Timestamp
    last: pd.Timestamp
    rows: int
    trials: int
    params: Mapping[str, Mapping[str, object]]
    """Library -> its tuned search-space parameters."""


@dataclass(frozen=True)
class TuningLog:
    tunes: tuple[Tune, ...]
    fold_tunes: tuple[int, ...]
    """For each walk-forward fold, the index in ``tunes`` of the tune it used."""
    library: str
    """The library the run reports."""


def _search_space(trial: optuna.Trial, library: str) -> dict:
    """Shared by both libraries so PBO's comparison reflects the algorithm, not an
    unevenly-sized search.

    Every parameter means the same thing in both libraries. The sampling
    fractions, L1/L2 penalties and minimum leaf size are there so the search can
    trade fit for simplicity instead of only ever growing a bigger model."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 20, 100),
        "max_depth": trial.suggest_int("max_depth", 2, 4),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        _LEAF_KEYS[library]: trial.suggest_int(_LEAF_KEYS[library], 5, 100, log=True),
    }


def tune_hyperparameters(
    panel: FeaturePanel,
    window: pd.DatetimeIndex,
    library: str,
    splitter: PurgedWalkForward,
    n_trials: int = N_TRIALS,
    seed: int = 0,
    warm_start: Mapping[str, object] | None = None,
) -> dict:
    """Optuna search over ``window`` only, returning the best search-space parameters.

    Each trial runs a mini walk-forward inside ``window`` with ``splitter``'s
    train/test/embargo and is scored by pooled Rank IC; a trial whose Rank IC is
    undefined scores worst. ``warm_start`` is tried first.
    """
    if library not in _LIBRARIES:
        raise BoostedConfigError(f"library must be one of {sorted(_LIBRARIES)}, got {library!r}.")
    inside = _restrict(panel, window)
    mini = PurgedWalkForward(splitter.train, splitter.test, splitter.embargo)
    _require_tuning_folds(inside, mini)

    def objective(trial: optuna.Trial) -> float:
        params = {**_search_space(trial, library), **_FIXED_PARAMS[library]}
        try:
            folds = evaluate(lambda: BoostedForecaster(library, params), inside, mini)
        except BoostedConfigError:
            return float("-inf")
        score = rank_ic(*pooled(folds))
        return score if score == score else float("-inf")

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    if warm_start is not None:
        study.enqueue_trial(dict(warm_start))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return dict(study.best_params)


@dataclass
class BoostedForecaster:
    """One library, fixed (already-tuned) hyperparameters.

    ``fit``/``predict`` mirror ``DerivedPolynomial``'s incomplete-row masking
    for consistency across every model family, even though XGBoost/LightGBM
    handle NaN natively — a comparison exercise across models shouldn't let
    one silently impute differently from the others.
    """

    library: str
    params: Mapping[str, object]
    name: str = field(init=False)

    def __post_init__(self) -> None:
        if self.library not in _LIBRARIES:
            raise BoostedConfigError(
                f"library must be one of {sorted(_LIBRARIES)}, got {self.library!r}."
            )
        self.name = f"Boosted[{self.library}]"
        self._model = None
        self._signals: list[str] | None = None
        self._fit_x: pd.DataFrame | None = None

    def fit(self, panel: FeaturePanel, train: pd.DatetimeIndex) -> None:
        signals = list(panel.signals)
        frame = panel.frame.loc[train, [*signals, panel.targets[0]]].dropna()
        if len(frame) < _MIN_TRAINING_ROWS:
            raise BoostedConfigError(
                f"not enough complete training rows to fit {self.library} "
                f"(need at least {_MIN_TRAINING_ROWS}, got {len(frame)})."
            )
        x, y = frame[signals], frame[panel.targets[0]]
        # Gain, not LightGBM's default split count, so both libraries' per-fold
        # importances measure the same thing.
        params = {**self.params, "importance_type": "gain"}
        leaf = _LEAF_KEYS[self.library]
        if leaf in params:
            # A leaf tuned on a longer window could leave a short one unsplittable.
            params[leaf] = min(params[leaf], max(1, len(x) // 4))
        self._model = _LIBRARIES[self.library](**params).fit(x, y)
        self._signals = signals
        self._fit_x = x

    def predict(self, panel: FeaturePanel, idx: pd.DatetimeIndex) -> pd.Series:
        if self._model is None or self._signals is None:
            raise RuntimeError("predict() called before fit()")
        predicted = pd.Series(np.nan, index=idx, dtype=float)
        raw = panel.frame.loc[idx, self._signals].dropna()
        if raw.empty:
            return predicted
        predicted.loc[raw.index] = self._model.predict(raw)
        return predicted

    def describe(self) -> ModelDescription:
        """The library's own feature importances — cheap enough to run every fold.

        SHAP attribution is reserved for the fold that is displayed; see ``explain``.
        """
        self._require_fit("describe")
        return self._description(self._model.feature_importances_)

    def explain(self) -> ModelDescription:
        """FYP-149: mean |SHAP value| per feature, reusing the terms/coefficients
        contract Polynomial and FF5 already report through."""
        self._require_fit("explain")
        explainer = shap.TreeExplainer(self._model)
        return self._description(np.abs(explainer.shap_values(self._fit_x)).mean(axis=0))

    def _require_fit(self, method: str) -> None:
        if self._model is None or self._signals is None or self._fit_x is None:
            raise RuntimeError(f"{method}() called before fit()")

    def _description(self, values) -> ModelDescription:
        return ModelDescription(
            name=self.name,
            terms=tuple(self._signals),
            coefficients=tuple(float(v) for v in values),
        )


def run_boosted(
    panel: FeaturePanel,
    splitter: PurgedWalkForward,
    n_trials: int = N_TRIALS,
    retune_trials: int = RETUNE_TRIALS,
    retune_every: int = RETUNE_EVERY,
    n_blocks: int = N_BLOCKS,
) -> tuple[ModelRunResult, ModelDescription, TuningLog]:
    """Tunes XGBoost and LightGBM on a rolling schedule (see the module docstring),
    evaluates both through the shared harness with each fold's latest tune,
    compares via PBO, and reports the one with the best pooled OOS Rank IC — the
    same shape ``run_derived_polynomial`` uses, XGBoost/LightGBM standing in for
    a degree/regularizer grid.

    Feature selection is screened per fold (``evaluate(..., screen=True)``).
    Tuning searches over every signal; only which columns a fold is *fit* on is
    screened. Every tuning window is checked before any tuning starts.
    """
    folds = list(splitter.split(panel))
    if not folds:
        raise BoostedConfigError(splitter.too_short(panel))
    if splitter.tuning_rows <= 0:
        raise BoostedConfigError("machine learning needs a tuning period before its first test.")
    index = panel.frame.index
    starts = [index.get_loc(test[0]) for _train, test in folds]
    triggers, fold_tunes = _schedule(starts, retune_every)
    windows = [splitter.window_before(panel, starts[i], splitter.tuning_rows) for i in triggers]
    mini = PurgedWalkForward(splitter.train, splitter.test, splitter.embargo)
    for window in windows:
        _require_tuning_folds(_restrict(panel, window), mini)

    tunes: list[Tune] = []
    for number, window in enumerate(windows):
        trials = n_trials if number == 0 else retune_trials
        previous = tunes[-1].params if tunes else {}
        params = {
            library: tune_hyperparameters(
                panel, window, library, splitter, n_trials=trials, warm_start=previous.get(library)
            )
            for library in _LIBRARIES
        }
        tunes.append(Tune(window[0], window[-1], len(window), trials, params))

    per_candidate = {
        library: _evaluate_by_tune(panel, folds, fold_tunes, tunes, library)
        for library in _LIBRARIES
    }
    best_name, pbo_value = select_best_candidate(per_candidate, n_blocks=n_blocks)
    result, _description = summarize(per_candidate[best_name], pbo=pbo_value)
    latest = _fitted_params(best_name, tunes[fold_tunes[-1]])
    description = _explain_last_fold(panel, per_candidate[best_name][-1], best_name, latest)
    return result, description, TuningLog(tuple(tunes), tuple(fold_tunes), best_name)


def _schedule(starts: list[int], retune_every: int) -> tuple[list[int], list[int]]:
    """Which folds trigger a tune, and each fold's tune. A re-tune happens at the
    first fold whose test window starts ``retune_every`` rows after the last."""
    triggers = [0]
    fold_tunes = []
    for position, start in enumerate(starts):
        if start >= starts[triggers[-1]] + retune_every:
            triggers.append(position)
        fold_tunes.append(len(triggers) - 1)
    return triggers, fold_tunes


def _restrict(panel: FeaturePanel, window: pd.DatetimeIndex) -> FeaturePanel:
    label_end = None if panel.label_end is None else panel.label_end.loc[window]
    return replace(panel, frame=panel.frame.loc[window], label_end=label_end)


def _require_tuning_folds(inside: FeaturePanel, mini: PurgedWalkForward) -> None:
    n_folds = sum(1 for _ in mini.split(inside))
    if n_folds < MIN_TUNING_FOLDS:
        raise BoostedConfigError(
            f"a {len(inside.frame)}-row tuning window holds only {n_folds} walk-forward "
            f"fold(s) of {mini.train} train and {mini.test} test rows with a "
            f"{mini.embargo}-row embargo, and tuning needs at least {MIN_TUNING_FOLDS} — "
            "shorten the train or test window."
        )


def _fitted_params(library: str, tune: Tune) -> dict:
    return {**tune.params[library], **_FIXED_PARAMS[library]}


class _Folds:
    """A fixed list of folds, shaped like a splitter for ``evaluate``."""

    def __init__(self, folds: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]):
        self._folds = folds

    def split(self, panel: FeaturePanel) -> Iterator[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        return iter(self._folds)


def _evaluate_by_tune(
    panel: FeaturePanel,
    folds: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]],
    fold_tunes: list[int],
    tunes: list[Tune],
    library: str,
) -> tuple[FoldResult, ...]:
    """Every fold, each fitted with the hyperparameters of the tune it falls under."""
    results: list[FoldResult] = []
    for number, tune in enumerate(tunes):
        these = [fold for fold, t in zip(folds, fold_tunes, strict=True) if t == number]
        params = _fitted_params(library, tune)
        results.extend(
            evaluate(
                lambda params=params: BoostedForecaster(library, params),
                panel,
                _Folds(these),
                screen=True,
            )
        )
    return tuple(replace(result, fold=position) for position, result in enumerate(results))


def _explain_last_fold(
    panel: FeaturePanel, fold: FoldResult, library: str, params: Mapping[str, object]
) -> ModelDescription:
    """SHAP for the displayed fold only: refit it exactly as ``evaluate`` did and
    explain that fit, rather than running SHAP on every fold of both libraries."""
    signals = fold.screening.fitted if fold.screening is not None else panel.signals
    forecaster = BoostedForecaster(library, params)
    forecaster.fit(replace(panel, signals=signals), fold.train)
    return forecaster.explain()
