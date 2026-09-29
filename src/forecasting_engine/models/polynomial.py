"""FYP-43: the polynomial Forecaster, user-supplied or system-derived.

Two ``Forecaster`` implementations (``UserPolynomial``, ``DerivedPolynomial``) plus the
two functions that run either one through the shared harness and shape the result into
``reporting.model_metrics.ModelRunResult`` — the fixed contract FYP-45/FYP-14 already
built the comparison view and promotion gates against.
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from forecasting_engine.ingest.align import FeaturePanel
from forecasting_engine.models.base import ModelDescription
from forecasting_engine.reporting.model_metrics import ModelRunResult
from forecasting_engine.validation.harness import (
    FoldResult,
    evaluate,
    select_best_candidate,
    summarize,
)
from forecasting_engine.validation.pbo import N_BLOCKS
from forecasting_engine.validation.splitters import PurgedWalkForward

MAX_DEGREE: int = 5
"""FYP-43's third acceptance criterion: a degree above this is rejected."""

_MIN_TRAINING_ROWS: int = 10
"""Below this, a regularized multi-term fit is more noise than signal — reject
with a clear message rather than let sklearn fail on a near-empty design matrix."""


class PolynomialConfigError(ValueError):
    """A user-supplied function or model configuration is invalid.

    The message is written for a portfolio manager and is safe to render
    directly in the dashboard — mirrors ``UploadError`` in ``ingest/upload.py``.
    """


# ── UserPolynomial: applies a supplied function, no fitting ─────────────────

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _parse(formula: str) -> ast.Expression:
    """Parse ``formula`` as a Python expression, then eagerly validate its
    structure against the allow-list below — before any column name is known,
    so a disallowed construct (a function call, attribute access, anything
    outside arithmetic) is rejected at construction, not deferred to predict()."""
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as exc:
        raise PolynomialConfigError(f"{formula!r} is not a valid expression: {exc.msg}.") from exc
    _validate_structure(tree)
    return tree


def _validate_structure(node: ast.AST) -> None:
    """Structural allow-list check, no column values needed: rejects anything
    that isn't a number, a name, or `+ - * /` and integer `**`. Name existence
    is checked separately in ``_safe_eval``, once the panel's actual signal
    columns are known.
    """
    if isinstance(node, ast.Expression):
        _validate_structure(node.body)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            raise PolynomialConfigError(f"only numeric constants are allowed, got {node.value!r}.")
    elif isinstance(node, ast.Name):
        pass
    elif isinstance(node, ast.BinOp):
        if type(node.op) not in _ALLOWED_BINOPS:
            raise PolynomialConfigError(f"operator {type(node.op).__name__} is not supported.")
        if isinstance(node.op, ast.Pow) and not _is_nonnegative_integer_constant(node.right):
            raise PolynomialConfigError("an exponent must be a non-negative whole number.")
        _validate_structure(node.left)
        _validate_structure(node.right)
    elif isinstance(node, ast.UnaryOp):
        if type(node.op) not in _ALLOWED_UNARYOPS:
            raise PolynomialConfigError(f"operator {type(node.op).__name__} is not supported.")
        _validate_structure(node.operand)
    else:
        raise PolynomialConfigError(f"{type(node).__name__} is not a supported expression.")


def _is_nonnegative_integer_constant(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int | float)
        and not isinstance(node.value, bool)
        and float(node.value).is_integer()
        and node.value >= 0
    )


def _safe_eval(node: ast.AST, env: Mapping[str, pd.Series]):
    """Evaluate an already-structurally-validated expression against ``env``.
    The only new failure mode possible here is a name absent from ``env`` —
    every other allow-list check already happened in ``_validate_structure``.
    """
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body, env)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in env:
            known = ", ".join(sorted(env)) or "(none)"
            raise PolynomialConfigError(
                f"{node.id!r} is not a known signal column. Known columns: {known}."
            )
        return env[node.id]
    if isinstance(node, ast.BinOp):
        op = _ALLOWED_BINOPS[type(node.op)]
        return op(_safe_eval(node.left, env), _safe_eval(node.right, env))
    if isinstance(node, ast.UnaryOp):
        op = _ALLOWED_UNARYOPS[type(node.op)]
        return op(_safe_eval(node.operand, env))
    raise PolynomialConfigError(f"{type(node).__name__} is not a supported expression.")


@dataclass
class UserPolynomial:
    """Applies a user-supplied function directly. ``fit`` is a no-op — FYP-43's
    first acceptance criterion is that a valid function is applied with no
    fitting step performed."""

    formula: str
    name: str = field(default="UserPolynomial", init=False)

    def __post_init__(self) -> None:
        self._tree = _parse(self.formula)

    def fit(self, panel: FeaturePanel, train: pd.DatetimeIndex) -> None:
        pass

    def predict(self, panel: FeaturePanel, idx: pd.DatetimeIndex) -> pd.Series:
        env = {name: panel.frame.loc[idx, name] for name in panel.signals}
        result = _safe_eval(self._tree, env)
        if isinstance(result, int | float):
            return pd.Series(float(result), index=idx)
        return result.reindex(idx)

    def describe(self) -> ModelDescription:
        return ModelDescription(name=self.name, terms=(self.formula,), coefficients=(1.0,))


# ── DerivedPolynomial: PolynomialFeatures + LassoCV/ElasticNetCV ────────────

_REGULARIZERS = {"lasso": LassoCV, "elasticnet": ElasticNetCV}


@dataclass
class DerivedPolynomial:
    """Expands the panel's signals with ``PolynomialFeatures`` up to ``degree``,
    fits a regularized linear model, and reports only the terms that survive
    regularization (a non-zero coefficient)."""

    degree: int = 2
    regularizer: str = "lasso"
    max_terms: int | None = None
    name: str = field(default="DerivedPolynomial", init=False)

    def __post_init__(self) -> None:
        if not (1 <= self.degree <= MAX_DEGREE):
            raise PolynomialConfigError(
                f"degree must be between 1 and {MAX_DEGREE}, got {self.degree}."
            )
        if self.regularizer not in _REGULARIZERS:
            known = ", ".join(sorted(_REGULARIZERS))
            raise PolynomialConfigError(
                f"regularizer must be one of {known}, got {self.regularizer!r}."
            )
        self._poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        self._scaler = StandardScaler()
        self._columns: list[str] | None = None
        self._model = None
        self._intercept: float | None = None

    def fit(self, panel: FeaturePanel, train: pd.DatetimeIndex) -> None:
        signals = list(panel.signals)
        frame = panel.frame.loc[train, [*signals, panel.targets[0]]].dropna()
        if len(frame) < _MIN_TRAINING_ROWS:
            raise PolynomialConfigError(
                f"not enough complete training rows to fit a degree-{self.degree} "
                f"polynomial (need at least {_MIN_TRAINING_ROWS}, got {len(frame)})."
            )
        x = pd.DataFrame(
            self._poly.fit_transform(frame[signals]),
            columns=self._poly.get_feature_names_out(signals),
            index=frame.index,
        )
        y = frame[panel.targets[0]]

        if self.max_terms is not None and x.shape[1] > self.max_terms:
            ranked = x.corrwith(y).abs().sort_values(ascending=False)
            self._columns = list(ranked.index[: self.max_terms])
            x = x[self._columns]
        else:
            self._columns = list(x.columns)

        # The penalty acts on coefficients, whose size depends on each expanded
        # term's units, so terms are standardised — on this window only.
        scaled = self._scaler.fit_transform(x)
        cv = _time_series_cv(len(x), gap=panel.horizon)
        model = _REGULARIZERS[self.regularizer](cv=cv, max_iter=10_000)
        model.fit(scaled, y)
        self._model = model
        self._intercept = float(model.intercept_)

    def predict(self, panel: FeaturePanel, idx: pd.DatetimeIndex) -> pd.Series:
        if self._model is None or self._columns is None:
            raise RuntimeError("predict() called before fit()")
        signals = list(panel.signals)
        predicted = pd.Series(np.nan, index=idx, dtype=float)

        # PolynomialFeatures.transform() rejects NaN outright (align_and_lag's
        # lag shift leaves the panel's leading rows NaN), so incomplete rows
        # must be dropped from the *input* before transforming — dropping them
        # from the expanded output would be too late, the transform already
        # raised.
        raw = panel.frame.loc[idx, signals].dropna()
        if raw.empty:
            return predicted

        expanded = pd.DataFrame(
            self._poly.transform(raw),
            columns=self._poly.get_feature_names_out(signals),
            index=raw.index,
        )[self._columns]
        predicted.loc[expanded.index] = self._model.predict(self._scaler.transform(expanded))
        return predicted

    def describe(self) -> ModelDescription:
        if self._model is None or self._columns is None:
            raise RuntimeError("describe() called before fit()")
        # Back in each term's own units, so the displayed equation reproduces
        # predict() on the raw signals.
        raw = self._model.coef_ / self._scaler.scale_
        intercept = self._intercept - float(np.dot(raw, self._scaler.mean_))
        terms, coefficients = [], []
        for term, coefficient in zip(self._columns, raw, strict=True):
            if coefficient != 0:
                terms.append(term)
                coefficients.append(float(coefficient))
        return ModelDescription(
            name=self.name,
            terms=tuple(terms),
            coefficients=tuple(coefficients),
            intercept=intercept,
        )


def _time_series_cv(n_rows: int, gap: int) -> TimeSeriesSplit:
    """Up to five time-ordered folds, each validating after a ``gap`` of rows so
    no training label overlaps it; fewer when the window is too short for five."""
    for n_splits in range(5, 1, -1):
        test_size = n_rows // (n_splits + 1)
        if test_size >= 2 and n_rows - gap - n_splits * test_size >= 2:
            return TimeSeriesSplit(n_splits=n_splits, gap=gap)
    raise PolynomialConfigError(
        f"a {n_rows}-row training window is too short to validate a regularized fit "
        f"in time order with a {gap}-row gap — lengthen the train window."
    )


# ── Bridging to the shared comparison view (ModelRunResult) ─────────────────

CANDIDATE_CONFIGS: tuple[DerivedPolynomial, ...] = tuple(
    DerivedPolynomial(degree=degree, regularizer=regularizer)
    for degree in (1, 2, 3)
    for regularizer in ("lasso", "elasticnet")
)
"""Working default (not sponsor-confirmed): the configuration grid PBO's CSCV
compares against itself for the derived-fit path. Revisit once Alpha Norm gives
compute-budget guidance — degree 4-5 candidates are omitted here to keep a
walk-forward run's wall-clock reasonable, even though FYP-43 allows degree up to 5
for a single fit."""


def run_user_polynomial(
    formula: str, panel: FeaturePanel, splitter: PurgedWalkForward
) -> tuple[ModelRunResult, ModelDescription]:
    """FYP-125: a user-supplied function bypasses fitting and applies directly.
    No configuration search happens, so ``pbo`` is ``None`` — the same "no
    configuration search" case FF5 reports."""
    folds = evaluate(lambda: UserPolynomial(formula), panel, splitter)
    _require_folds(folds, panel, splitter)
    return summarize(folds, pbo=None)


def run_derived_polynomial(
    panel: FeaturePanel,
    splitter: PurgedWalkForward,
    candidates: tuple[DerivedPolynomial, ...] = CANDIDATE_CONFIGS,
    n_blocks: int = N_BLOCKS,
) -> tuple[ModelRunResult, ModelDescription]:
    """Evaluates every candidate in ``candidates``, compares them via PBO
    (``compute_pbo`` needs several configurations' return series — a single
    fitted model has nothing to compute PBO against), then reports the
    candidate with the best pooled OOS Rank IC alongside that shared PBO score.

    ``n_blocks`` is forwarded to ``compute_pbo`` — CSCV's cost is combinatorial
    in it (``C(n_blocks, n_blocks/2)`` splits), so a caller under a tight
    compute budget (an interactive UI, a test) can lower it from the default.

    Feature selection is screened per fold (``evaluate(..., screen=True)``) —
    a signal below FYP-102's inclusion threshold, judged on that fold's own
    train window, is left out of that fold's fit.
    """
    if not candidates:
        raise PolynomialConfigError("deriving a function needs at least one candidate.")
    per_candidate = {
        f"degree{c.degree}_{c.regularizer}": evaluate(
            lambda c=c: DerivedPolynomial(c.degree, c.regularizer, c.max_terms),
            panel,
            splitter,
            screen=True,
        )
        for c in candidates
    }
    _require_folds(next(iter(per_candidate.values()), ()), panel, splitter)
    if len(per_candidate) == 1:
        # PBO asks how often the best of several configurations was luck. One
        # configuration was never chosen from anything, so it reports no PBO
        # rather than failing inside CSCV.
        (only_folds,) = per_candidate.values()
        return summarize(only_folds, pbo=None)
    best_name, pbo_value = select_best_candidate(per_candidate, n_blocks=n_blocks)
    return summarize(per_candidate[best_name], pbo=pbo_value)


def _require_folds(
    folds: tuple[FoldResult, ...], panel: FeaturePanel, splitter: PurgedWalkForward
) -> None:
    if not folds:
        raise PolynomialConfigError(splitter.too_short(panel))
