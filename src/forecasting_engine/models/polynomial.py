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
from sklearn.linear_model import ElasticNet, enet_path
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

CLIP_SD: float = 4.0
"""A derived polynomial's raw inputs are clipped to this many training-window
standard deviations from the mean, so one extreme value can't be raised to a
power into an extreme forecast."""

INNER_CV_SPLITS: int = 5
"""Most time-ordered folds the penalty is validated on."""

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


# ── DerivedPolynomial: PolynomialFeatures + Lasso/ElasticNet ─────────────────

#: Regularizer -> its L1 share. Lasso is all L1; 0.5 is scikit-learn's
#: ElasticNetCV default, kept so the regularizer means what it did before.
_REGULARIZERS: dict[str, float] = {"lasso": 1.0, "elasticnet": 0.5}

#: The penalty grid, as scikit-learn's LassoCV builds it: this many values,
#: spanning from the smallest penalty that zeroes every term down by this factor.
_N_ALPHAS: int = 100
_ALPHA_EPS: float = 1e-3


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
        self._bounds: tuple[pd.Series, pd.Series] | None = None

    def fit(self, panel: FeaturePanel, train: pd.DatetimeIndex) -> None:
        signals = list(panel.signals)
        frame = panel.frame.loc[train, [*signals, panel.targets[0]]].dropna()
        if len(frame) < _MIN_TRAINING_ROWS:
            raise PolynomialConfigError(
                f"not enough complete training rows to fit a degree-{self.degree} "
                f"polynomial (need at least {_MIN_TRAINING_ROWS}, got {len(frame)})."
            )
        # Clipping each signal to its training mean ± CLIP_SD standard deviations
        # is clipping its standardised value to ± CLIP_SD, done in raw units so
        # the displayed equation stays exact inside these bounds.
        mean, sd = frame[signals].mean(), frame[signals].std()
        self._bounds = (mean - CLIP_SD * sd, mean + CLIP_SD * sd)
        x = pd.DataFrame(
            self._poly.fit_transform(self._clip(frame[signals])),
            columns=self._poly.get_feature_names_out(signals),
            index=frame.index,
        )
        y = frame[panel.targets[0]]

        # The penalty is validated on later rows of this window, so the term
        # pick and the scaler are redone inside every validation split, from its
        # earlier rows alone; only then are they redone on the whole window for
        # the final fit. Doing them once up front would let the rows that judge
        # the penalty help choose what they are judging.
        l1_ratio = _REGULARIZERS[self.regularizer]
        cv = _time_series_cv(len(x), gap=panel.horizon)
        alpha = _choose_alpha(x, y, cv, self.max_terms, l1_ratio)
        self._columns, self._scaler = _prepare(x, y, self.max_terms)
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10_000)
        model.fit(self._scaler.transform(x[self._columns]), y)
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
            self._poly.transform(self._clip(raw)),
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
        low, high = self._bounds
        return ModelDescription(
            name=self.name,
            terms=tuple(terms),
            coefficients=tuple(coefficients),
            intercept=intercept,
            input_bounds={s: (float(low[s]), float(high[s])) for s in low.index},
        )

    def _clip(self, signals: pd.DataFrame) -> pd.DataFrame:
        low, high = self._bounds
        return signals.clip(lower=low, upper=high, axis=1)


def _prepare(
    x: pd.DataFrame, y: pd.Series, max_terms: int | None
) -> tuple[list[str], StandardScaler]:
    """The terms to keep, at most ``max_terms`` by absolute correlation with the
    target, and a scaler fitted to them — both from ``x``'s rows only.

    The penalty acts on coefficients, whose size depends on each term's units,
    so terms are standardised before fitting."""
    if max_terms is not None and x.shape[1] > max_terms:
        ranked = x.corrwith(y).abs().sort_values(ascending=False)
        columns = list(ranked.index[:max_terms])
    else:
        columns = list(x.columns)
    return columns, StandardScaler().fit(x[columns])


def _choose_alpha(
    x: pd.DataFrame,
    y: pd.Series,
    cv: TimeSeriesSplit,
    max_terms: int | None,
    l1_ratio: float,
) -> float:
    """The penalty with the lowest mean squared error over ``cv``'s time-ordered
    splits, each split preparing its terms from its own training rows."""
    alphas = _alpha_grid(x, y, l1_ratio)
    errors = np.zeros(len(alphas))
    for train, validate in cv.split(x):
        x_train, y_train = x.iloc[train], y.iloc[train]
        columns, scaler = _prepare(x_train, y_train, max_terms)
        centre = float(y_train.mean())
        _, coefs, _ = enet_path(
            scaler.transform(x_train[columns]),
            y_train.to_numpy() - centre,
            l1_ratio=l1_ratio,
            alphas=alphas,
            max_iter=10_000,
        )
        forecast = scaler.transform(x.iloc[validate][columns]) @ coefs + centre
        errors += ((forecast - y.iloc[validate].to_numpy()[:, None]) ** 2).mean(axis=0)
    return float(alphas[int(np.argmin(errors))])


def _alpha_grid(x: pd.DataFrame, y: pd.Series, l1_ratio: float) -> np.ndarray:
    """``_N_ALPHAS`` penalties, largest first, from the smallest that zeroes every
    standardised term down by ``_ALPHA_EPS``.

    Set from the whole window, as LassoCV sets its grid. That fixes only the
    range searched; which penalty wins is decided by the validation splits."""
    scaled = StandardScaler().fit_transform(x)
    largest = float(np.abs(scaled.T @ (y.to_numpy() - y.mean())).max()) / (len(y) * l1_ratio)
    if not largest > 0:  # a flat target: every penalty gives the same (empty) fit
        return np.array([1.0])
    return np.geomspace(largest, largest * _ALPHA_EPS, _N_ALPHAS)


def _time_series_cv(n_rows: int, gap: int) -> TimeSeriesSplit:
    """Up to ``INNER_CV_SPLITS`` time-ordered folds, each validating after a
    ``gap`` of rows so no training label overlaps it; fewer when the window is
    too short."""
    for n_splits in range(INNER_CV_SPLITS, 1, -1):
        test_size = n_rows // (n_splits + 1)
        if test_size >= 2 and n_rows - gap - n_splits * test_size >= 2:
            return TimeSeriesSplit(n_splits=n_splits, gap=gap)
    raise PolynomialConfigError(
        f"a {n_rows}-row training window is too short to validate a regularized fit "
        f"in time order with a {gap}-row gap — lengthen the train window."
    )


# ── Bridging to the shared comparison view (ModelRunResult) ─────────────────

CANDIDATE_DEGREES: tuple[int, ...] = (1, 2, 3)
CANDIDATE_REGULARIZERS: tuple[str, ...] = ("lasso", "elasticnet")

CANDIDATE_CONFIGS: tuple[DerivedPolynomial, ...] = tuple(
    DerivedPolynomial(degree=degree, regularizer=regularizer)
    for degree in CANDIDATE_DEGREES
    for regularizer in CANDIDATE_REGULARIZERS
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
