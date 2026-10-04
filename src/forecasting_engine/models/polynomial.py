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


# ── UserPolynomial: the user's shape, scaled to the target per fold ─────────

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
    """The user supplies the shape ``f``; each fold fits ``forecast = a + b × f`` by
    ordinary least squares on its training rows (FYP-43 change request).

    The formula alone is a signal's level, not a return: ``VIX_Index_PX_LAST``
    would "forecast" a return of about 20. Fitting a scale and intercept puts the
    forecast on the target's scale while the user still decides its shape.

    ``bindings`` names the signal column each placeholder in ``formula`` stands
    for; a name with no binding is read as a column itself.
    """

    formula: str
    bindings: Mapping[str, str] = field(default_factory=dict)
    name: str = field(default="UserPolynomial", init=False)

    def __post_init__(self) -> None:
        self._tree = _parse(self.formula)
        self._intercept: float | None = None
        self._slope: float | None = None

    @property
    def resolved_formula(self) -> str:
        """The formula with every placeholder replaced by the column it stands for."""
        return ast.unparse(_Rebind(self.bindings).visit(_parse(self.formula)))

    def fit(self, panel: FeaturePanel, train: pd.DatetimeIndex) -> None:
        f = self._shape(panel, train)
        y = panel.frame.loc[train, panel.targets[0]]
        both = f.notna() & y.notna()
        if not both.any():
            raise PolynomialConfigError(
                "no training row has both the function's value and the target, so its "
                "scale can't be fitted."
            )
        f, y = f[both], y[both]
        variance = float(((f - f.mean()) ** 2).mean())
        # A constant shape carries no information to scale: the fold forecasts
        # its training mean.
        slope = float(((f - f.mean()) * (y - y.mean())).mean()) / variance if variance else 0.0
        self._slope = slope
        self._intercept = float(y.mean()) - slope * float(f.mean())

    def predict(self, panel: FeaturePanel, idx: pd.DatetimeIndex) -> pd.Series:
        if self._slope is None:
            raise RuntimeError("predict() called before fit()")
        return self._intercept + self._slope * self._shape(panel, idx)

    def describe(self) -> ModelDescription:
        if self._slope is None:
            raise RuntimeError("describe() called before fit()")
        return ModelDescription(
            name=self.name,
            terms=(self.resolved_formula,),
            coefficients=(self._slope,),
            intercept=self._intercept,
        )

    def _shape(self, panel: FeaturePanel, idx: pd.DatetimeIndex) -> pd.Series:
        """``f``, the formula's own value on ``idx``."""
        env = {name: panel.frame.loc[idx, name] for name in panel.signals}
        for placeholder, column in self.bindings.items():
            if column not in panel.signals:
                raise PolynomialConfigError(
                    f"{placeholder} stands for {column!r}, which is not a signal column."
                )
            env[placeholder] = env[column]
        result = _safe_eval(self._tree, env)
        if isinstance(result, int | float):
            return pd.Series(float(result), index=idx)
        return result.reindex(idx).astype(float)


class _Rebind(ast.NodeTransformer):
    def __init__(self, bindings: Mapping[str, str]) -> None:
        self.bindings = bindings

    def visit_Name(self, node: ast.Name) -> ast.Name:
        return ast.copy_location(ast.Name(self.bindings.get(node.id, node.id), node.ctx), node)


def placeholders(formula: str) -> tuple[str, ...]:
    """The names ``formula`` uses, in the order they first appear."""
    names = sorted(
        (node for node in ast.walk(_parse(formula)) if isinstance(node, ast.Name)),
        key=lambda node: node.col_offset,
    )
    return tuple(dict.fromkeys(node.id for node in names))


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
    """Clips and standardises the panel's signals on the training window, expands
    them with ``PolynomialFeatures`` up to ``degree``, fits a regularized linear
    model, and reports only the terms that survive regularization (a non-zero
    coefficient).

    Expanding standardised signals rather than raw levels matters: a level and
    its square move almost together (VIX and VIX² correlate at about 0.995 over a
    window), so the penalty can't tell them apart and terms swing from fold to
    fold. A centred signal and its square barely correlate."""

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
        self._centre: tuple[pd.Series, pd.Series] | None = None

    def fit(self, panel: FeaturePanel, train: pd.DatetimeIndex) -> None:
        signals = list(panel.signals)
        frame = panel.frame.loc[train, [*signals, panel.targets[0]]].dropna()
        if len(frame) < _MIN_TRAINING_ROWS:
            raise PolynomialConfigError(
                f"not enough complete training rows to fit a degree-{self.degree} "
                f"polynomial (need at least {_MIN_TRAINING_ROWS}, got {len(frame)})."
            )
        # Each signal is clipped to its training mean ± CLIP_SD standard
        # deviations, then standardised with the same mean and SD, so its
        # standardised value lies within ± CLIP_SD. A signal with no spread is
        # only centred.
        mean, sd = frame[signals].mean(), frame[signals].std()
        self._bounds = (mean - CLIP_SD * sd, mean + CLIP_SD * sd)
        self._centre = (mean, sd.where(sd > 0, 1.0))
        x = pd.DataFrame(
            self._poly.fit_transform(self._standardise(frame[signals])),
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
            self._poly.transform(self._standardise(raw)),
            columns=self._poly.get_feature_names_out(signals),
            index=raw.index,
        )[self._columns]
        predicted.loc[expanded.index] = self._model.predict(self._scaler.transform(expanded))
        return predicted

    def describe(self) -> ModelDescription:
        if self._model is None or self._columns is None:
            raise RuntimeError("describe() called before fit()")
        # Undo the expanded terms' scaling, so the displayed equation in the
        # standardised signals reproduces predict().
        raw = self._model.coef_ / self._scaler.scale_
        intercept = self._intercept - float(np.dot(raw, self._scaler.mean_))
        terms, coefficients = [], []
        for term, coefficient in zip(self._columns, raw, strict=True):
            if coefficient != 0:
                terms.append(term)
                coefficients.append(float(coefficient))
        low, high = self._bounds
        mean, sd = self._centre
        return ModelDescription(
            name=self.name,
            terms=tuple(terms),
            coefficients=tuple(coefficients),
            intercept=intercept,
            input_bounds={s: (float(low[s]), float(high[s])) for s in low.index},
            standardisation={s: (float(mean[s]), float(sd[s])) for s in mean.index},
        )

    def _standardise(self, signals: pd.DataFrame) -> pd.DataFrame:
        """``z = (x − mean) / sd`` of each clipped signal, with training statistics."""
        low, high = self._bounds
        mean, sd = self._centre
        return (signals.clip(lower=low, upper=high, axis=1) - mean) / sd


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
    formula: str,
    panel: FeaturePanel,
    splitter: PurgedWalkForward,
    bindings: Mapping[str, str] | None = None,
) -> tuple[ModelRunResult, ModelDescription]:
    """A user-supplied shape, scaled and shifted to the target in each fold.
    There is one configuration and nothing to choose between, so ``pbo`` is
    ``None`` — the same "no configuration search" case FF5 reports."""
    folds = evaluate(lambda: UserPolynomial(formula, dict(bindings or {})), panel, splitter)
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
