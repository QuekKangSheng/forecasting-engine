"""FYP-43: the polynomial Forecaster, user-supplied or system-derived.

Two ``Forecaster`` implementations (``UserPolynomial``, ``DerivedPolynomial``) plus the
two functions that run either one through the shared harness and shape the result into
``reporting.model_metrics.ModelRunResult`` — the fixed contract FYP-45/FYP-14 already
built the comparison view and promotion gates against.
"""

from __future__ import annotations

import ast
import math
import operator
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import enet_path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from forecasting_engine.ingest.align import FeaturePanel
from forecasting_engine.models.base import ModelDescription
from forecasting_engine.models.shrinkage import ShrinkagePolynomial
from forecasting_engine.parallel import map_processes
from forecasting_engine.reporting.model_metrics import ModelRunResult
from forecasting_engine.validation.harness import (
    FoldResult,
    evaluate,
    select_best_candidate,
    summarize,
)
from forecasting_engine.validation.splitters import PurgedWalkForward

MAX_DEGREE: int = 5
"""FYP-43's third acceptance criterion: a degree above this is rejected."""

MAX_FORMULA_NODES: int = 200
"""Most numbers, signals and operations a user function may have: about 25
terms such as ``0.1 * x ** 2 * y``. A function also nests no deeper than this,
which keeps every recursive walk of it (evaluating, rewriting, typesetting) well
inside Python's recursion limit; 1,200 terms once broke the page."""

MAX_FORMULA_LENGTH: int = 10_000
"""Longest user function read, in characters, so that a huge paste is refused
before it is parsed. Far above what ``MAX_FORMULA_NODES`` allows to be typed, so
that a fitted function with each placeholder written out as its column name, as
it is stored and read back, still fits."""

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
    outside arithmetic) is rejected at construction, not deferred to predict().

    Its size, degree and numbers are checked here too, before any of it is worked
    out: a function that takes hours to evaluate or expand holds the one model
    worker, and every queued Run waits behind it. These caps are for a new
    function; one saved before them is read back with ``_read``."""
    if len(formula) > MAX_FORMULA_LENGTH:
        raise PolynomialConfigError(
            f"the function is {len(formula):,} characters long; the most allowed is "
            f"{MAX_FORMULA_LENGTH:,}."
        )
    tree = _syntax(formula)
    size = sum(isinstance(node, ast.expr) for node in ast.walk(tree))
    if size > MAX_FORMULA_NODES:
        raise PolynomialConfigError(
            f"the function has {size} numbers, signals and operations; the most allowed is "
            f"{MAX_FORMULA_NODES}. Write it with fewer terms."
        )
    _validate_structure(tree)
    degree = _degree(tree)
    if degree > MAX_DEGREE:
        raise PolynomialConfigError(
            f"the function's degree is {degree}, above the highest allowed, {MAX_DEGREE}. "
            "Its degree is the most signals multiplied together in one term, counting "
            "powers: x ** 2 * y is 3, and dividing counts as multiplying."
        )
    _constant(tree)
    return tree


def _read(formula: str) -> ast.Expression:
    """A saved function, parsed to be evaluated again: the allow-list alone,
    without ``_parse``'s size, degree and number caps. Those stop a new function
    being queued; a function saved before them has run already.

    So an exponent may be any whole number of 0 or more, as it could be then.
    Raising a signal's value, a float, to a large power is quick: at worst it
    overflows. A plain number raised to one is worked out exactly and can take
    hours, so ``_constant`` refuses one beyond a float before working it out."""
    tree = _syntax(formula)
    _validate_structure(tree, max_exponent=None)
    return tree


def _syntax(formula: str) -> ast.Expression:
    """``formula`` parsed as a Python expression, its syntax errors in plain words."""
    try:
        return ast.parse(formula, mode="eval")
    except SyntaxError as exc:
        # Python won't read a whole number of more than a few thousand digits, and
        # says so in terms of its own settings; for a formula it is a number too large.
        if "integer string conversion" in str(exc.msg):
            raise _too_large() from exc
        raise PolynomialConfigError(f"{formula!r} is not a valid expression: {exc.msg}.") from exc
    except (MemoryError, RecursionError) as exc:  # how the parser refuses deep nesting
        raise _nested_too_deeply() from exc


def _too_large() -> PolynomialConfigError:
    return PolynomialConfigError(
        "a number in the function is too large to work with; use a smaller one."
    )


def _nested_too_deeply() -> PolynomialConfigError:
    return PolynomialConfigError("the function is nested too deeply to read.")


def _validate_structure(node: ast.AST, max_exponent: int | None = MAX_DEGREE) -> None:
    """Structural allow-list check, no column values needed: rejects anything
    that isn't a number, a name, or `+ - * /` and `**` by a whole number up to
    ``max_exponent`` (any, where it is None, as for a saved function). Name
    existence is checked separately in ``_safe_eval``, once the panel's actual
    signal columns are known.
    """
    if isinstance(node, ast.Expression):
        _validate_structure(node.body, max_exponent)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            raise PolynomialConfigError(f"only numeric constants are allowed, got {node.value!r}.")
    elif isinstance(node, ast.Name):
        pass
    elif isinstance(node, ast.BinOp):
        if type(node.op) not in _ALLOWED_BINOPS:
            raise PolynomialConfigError(f"operator {type(node.op).__name__} is not supported.")
        if isinstance(node.op, ast.Pow) and not _is_allowed_exponent(node.right, max_exponent):
            allowed = "of 0 or more" if max_exponent is None else f"from 0 to {max_exponent}"
            raise PolynomialConfigError(
                f"an exponent must be a whole number {allowed}, written as a number, such as "
                "x ** 2."
            )
        _validate_structure(node.left, max_exponent)
        _validate_structure(node.right, max_exponent)
    elif isinstance(node, ast.UnaryOp):
        if type(node.op) not in _ALLOWED_UNARYOPS:
            raise PolynomialConfigError(f"operator {type(node.op).__name__} is not supported.")
        _validate_structure(node.operand, max_exponent)
    else:
        raise PolynomialConfigError(f"{type(node).__name__} is not a supported expression.")


def _is_allowed_exponent(node: ast.AST, max_exponent: int | None) -> bool:
    """A whole number from 0 to ``max_exponent``, or of 0 or more where it is
    None, written as a number. For a new function the cap is ``MAX_DEGREE``: a
    larger power of a signal can't pass the degree cap. A whole number is never
    converted to a float here, as one too large for a float can't be."""
    if not (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int | float)
        and not isinstance(node.value, bool)
    ):
        return False
    value = node.value
    if value < 0 or (max_exponent is not None and value > max_exponent):
        return False
    return isinstance(value, int) or value.is_integer()


def _degree(node: ast.AST) -> int:
    """The function's degree, read from an already-validated structure: a signal
    is 1 and a number 0; a sum takes its larger side's, a product or a quotient
    adds its sides', and a power multiplies its base's by the exponent."""
    if isinstance(node, ast.Expression):
        return _degree(node.body)
    if isinstance(node, ast.Name):
        return 1
    if isinstance(node, ast.UnaryOp):
        return _degree(node.operand)
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Pow):
            return _degree(node.left) * int(node.right.value)
        left, right = _degree(node.left), _degree(node.right)
        return max(left, right) if isinstance(node.op, ast.Add | ast.Sub) else left + right
    return 0


def _constant(node: ast.AST) -> int | float | None:
    """``node``'s value where it reads no signal, else None, worked out one
    operation at a time exactly as ``_safe_eval`` will. A number too large for a
    float is refused as soon as it appears, before a larger one can be built
    from it, and so is a division by zero. A power is refused before it is worked
    out if it would be too large: a saved function's exponent can be any whole
    number, and ``10 ** 1000000000`` alone takes hours."""
    if isinstance(node, ast.Expression):
        return _constant(node.body)
    if isinstance(node, ast.Name):
        return None
    if isinstance(node, ast.Constant):
        value = node.value
    elif isinstance(node, ast.UnaryOp):
        operand = _constant(node.operand)
        if operand is None:
            return None
        value = _ALLOWED_UNARYOPS[type(node.op)](operand)
    else:
        left, right = _constant(node.left), _constant(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise PolynomialConfigError("the function divides by zero.")
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Pow) and _power_beyond_a_float(left, right):
            raise _too_large()
        try:
            value = _ALLOWED_BINOPS[type(node.op)](left, right)
        except OverflowError:
            value = math.inf
    if not _fits_a_float(value):
        raise _too_large()
    return value


def _fits_a_float(value: int | float) -> bool:
    try:
        return math.isfinite(value)
    except OverflowError:  # a whole number beyond the largest float
        return False


_LARGEST_FLOAT_DIGITS: float = math.log10(sys.float_info.max)
"""About 308.25: the largest float has 309 digits before its decimal point."""


def _power_beyond_a_float(base: int | float, exponent: int | float) -> bool:
    """Whether ``base ** exponent`` is too large for a float, told from its digits,
    ``exponent × log10 |base|``, without working it out. ``base`` fits a float,
    and comparing a whole number with a float is exact however large it is."""
    if abs(base) <= 1:  # its powers are no larger than itself
        return False
    return exponent > _LARGEST_FLOAT_DIGITS / math.log10(abs(base))


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
        """``f``, the formula's own value on ``idx``.

        Refused where every signal it reads is present but ``f`` isn't finite:
        dividing by a signal on a day it is zero makes ``f`` infinite there, and
        one such training row would leave every forecast of its fold NaN."""
        env = self._env(panel, idx)
        result = _safe_eval(self._tree, env)
        if isinstance(result, int | float):
            return pd.Series(float(result), index=idx)
        shape = result.reindex(idx).astype(float)
        if not np.isfinite(shape[self._inputs_present(env)]).all():
            raise PolynomialConfigError(self._not_finite(panel))
        return shape

    def _env(self, panel: FeaturePanel, idx: pd.DatetimeIndex) -> dict[str, pd.Series]:
        """Every signal column on ``idx``, and each placeholder as the column it
        stands for."""
        columns = {name: panel.frame.loc[idx, name] for name in panel.signals}
        env = dict(columns)
        for placeholder, column in self.bindings.items():
            if column not in panel.signals:
                raise PolynomialConfigError(
                    f"{placeholder} stands for {column!r}, which is not a signal column."
                )
            # Read from the columns, not from ``env``: a placeholder named after a
            # column re-points that name, and a placeholder bound to that column
            # must still read the column, whichever of the two is bound first.
            env[placeholder] = columns[column]
        return env

    def _inputs_present(self, env: Mapping[str, pd.Series]) -> pd.Series:
        """Whether every name the formula reads has a value, row by row."""
        names = {node.id for node in ast.walk(self._tree) if isinstance(node, ast.Name)}
        return pd.concat([env[name] for name in names], axis=1).notna().all(axis=1)

    def _not_finite(self, panel: FeaturePanel) -> str:
        """Why ``f`` isn't finite on some days whose inputs are all present,
        counted over the whole panel: the division by something that is zero on
        those days, written as the user wrote it, when one explains it."""
        env = self._env(panel, panel.frame.index)
        shape = _safe_eval(self._tree, env)
        bad = self._inputs_present(env) & ~np.isfinite(shape)
        for node in ast.walk(self._tree):
            if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)):
                continue
            divisor = _safe_eval(node.right, env)
            # A divisor that is a plain number was checked to be non-zero in _parse.
            if isinstance(divisor, pd.Series):
                zero = divisor.index[bad & (divisor == 0)]
                if len(zero):
                    return (
                        f"the function divides by {self._named(node.right)}, which is zero "
                        f"on {_days(zero)}, so it can't be worked out there. Divide by "
                        "something that is never zero."
                    )
        return (
            f"the function's value is infinite or undefined on {_days(shape.index[bad])}. "
            "Look for a division by a signal or a very large number."
        )

    def _named(self, node: ast.AST) -> str:
        """``node`` as the user wrote it; a placeholder with the column it stands for."""
        if isinstance(node, ast.Name) and self.bindings.get(node.id, node.id) != node.id:
            return f"{node.id} ({self.bindings[node.id]})"
        return ast.unparse(node)


def _days(dates: pd.DatetimeIndex) -> str:
    """``3 days (the first is 05 Feb 2016)``."""
    return f"{len(dates)} day{'' if len(dates) == 1 else 's'} (the first is {dates[0]:%d %b %Y})"


class _Rebind(ast.NodeTransformer):
    def __init__(self, bindings: Mapping[str, str]) -> None:
        self.bindings = bindings

    def visit_Name(self, node: ast.Name) -> ast.Name:
        return ast.copy_location(ast.Name(self.bindings.get(node.id, node.id), node.ctx), node)


def evaluate_formula(formula: str, values: Mapping[str, float]) -> float:
    """``formula`` at one point, each name it uses read from ``values``: what
    ``UserPolynomial`` computes as ``f`` on one row. NaN where it can't be worked
    out on these values: it divides by zero, or its value is too large for a float.
    A fitted row there is refused, so a caller must not pass NaN off as a forecast.

    ``formula`` is a saved function, so it is read with the allow-list alone
    (``_read``), and an exponent may be any whole number. Evaluating it still
    can't hang. Each value is read as a float, even a whole number, so a power of
    a signal is quick, overflowing at worst. Its plain-number parts are worked out
    first, one step at a time (``_constant``), each step refused if too large for
    a float, a power before it is worked out: no step builds a number of more
    than about 620 digits."""
    try:
        tree = _read(formula)
        _constant(tree)
        floats = {name: float(number) for name, number in values.items()}
        value = float(_safe_eval(tree, floats))
    except (ZeroDivisionError, OverflowError):  # how a float says there is no value
        return float("nan")
    except RecursionError as exc:  # no size cap applies to a saved function
        raise _nested_too_deeply() from exc
    return value if math.isfinite(value) else float("nan")


def placeholders(formula: str) -> tuple[str, ...]:
    """The names ``formula`` uses, in the order they first appear.

    Only its syntax is checked, as nothing in it is evaluated here: a new
    function is checked by constructing ``UserPolynomial``, and a saved one has
    run already, whatever caps were added since."""
    names = sorted(
        (node for node in ast.walk(_syntax(formula)) if isinstance(node, ast.Name)),
        key=lambda node: node.col_offset,
    )
    return tuple(dict.fromkeys(node.id for node in names))


# ── DerivedPolynomial: PolynomialFeatures + Lasso ────────────────────────────

#: Regularizer -> its L1 share. Lasso is all L1.
_REGULARIZERS: dict[str, float] = {"lasso": 1.0}

#: The penalty grid, as scikit-learn's LassoCV builds it: this many values,
#: spanning from the smallest penalty that zeroes every term down by this factor.
_N_ALPHAS: int = 100
_ALPHA_EPS: float = 1e-3

#: Penalties fitted at a time when walking a capped path, so the walk can stop
#: soon after the cap is first exceeded.
_PATH_CHUNK: int = 10


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
        self._coef: np.ndarray | None = None
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

        # The penalty is validated on later rows of this window, so the scaler
        # is refitted inside every validation split from its earlier rows alone.
        # Fitting it once up front would let the rows that judge the penalty
        # help shape what they are judging.
        l1_ratio = _REGULARIZERS[self.regularizer]
        cv = _time_series_cv(len(x), gap=panel.horizon)
        _, self._coef = _choose_alpha(x, y, cv, self.max_terms, l1_ratio)
        self._columns, self._scaler = _prepare(x)
        self._intercept = float(y.mean())

    def predict(self, panel: FeaturePanel, idx: pd.DatetimeIndex) -> pd.Series:
        if self._coef is None or self._columns is None:
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
        predicted.loc[expanded.index] = (
            self._scaler.transform(expanded) @ self._coef + self._intercept
        )
        return predicted

    def describe(self) -> ModelDescription:
        if self._coef is None or self._columns is None:
            raise RuntimeError("describe() called before fit()")
        # Undo the expanded terms' scaling, so the displayed equation in the
        # standardised signals reproduces predict().
        raw = self._coef / self._scaler.scale_
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


def _prepare(x: pd.DataFrame) -> tuple[list[str], StandardScaler]:
    """Every term, and a scaler fitted to them from ``x``'s rows only.

    The penalty acts on coefficients, whose size depends on each term's units,
    so terms are standardised before fitting. No term is dropped here: the
    penalty alone decides which survive."""
    return list(x.columns), StandardScaler().fit(x)


def _choose_alpha(
    x: pd.DataFrame,
    y: pd.Series,
    cv: TimeSeriesSplit,
    max_terms: int | None,
    l1_ratio: float,
) -> tuple[float, np.ndarray]:
    """The penalty with the lowest mean squared error over ``cv``'s time-ordered
    splits, and the whole window's fit at it (coefficients of the standardised
    terms).

    ``max_terms`` caps the terms through the penalty. The whole window's path is
    walked from the largest penalty down, and the penalties before the first
    whose fit has more than ``max_terms`` non-zero coefficients are eligible.
    The path stops there, so the small penalties a cap rules out, which are
    also the slowest to fit, are never computed. If the largest penalty already
    exceeds the cap, it is used."""
    alphas = _alpha_grid(x, y, l1_ratio)
    _, scaler = _prepare(x)
    alphas, path = _capped_path(scaler.transform(x), y, alphas, l1_ratio, max_terms)
    errors = np.zeros(len(alphas))
    for train, validate in cv.split(x):
        x_train, y_train = x.iloc[train], y.iloc[train]
        _, split_scaler = _prepare(x_train)
        coefs = _path(split_scaler.transform(x_train), y_train, alphas, l1_ratio)
        forecast = split_scaler.transform(x.iloc[validate]) @ coefs + float(y_train.mean())
        errors += ((forecast - y.iloc[validate].to_numpy()[:, None]) ** 2).mean(axis=0)
    best = int(np.argmin(errors))
    return float(alphas[best]), path[:, best]


def _capped_path(
    x: np.ndarray, y: pd.Series, alphas: np.ndarray, l1_ratio: float, max_terms: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """The penalties a cap leaves eligible, largest first, and the fit at each.

    Fitted ``_PATH_CHUNK`` penalties at a time, each chunk warm-started from the
    last fit, stopping at the first penalty whose fit keeps more than
    ``max_terms`` terms."""
    if max_terms is None or x.shape[1] <= max_terms:  # the cap can't bind
        return alphas, _path(x, y, alphas, l1_ratio)
    kept, fits = [], []
    coef = None
    for start in range(0, len(alphas), _PATH_CHUNK):
        chunk = alphas[start : start + _PATH_CHUNK]
        coefs = _path(x, y, chunk, l1_ratio, coef_init=coef)
        counts = (coefs != 0).sum(axis=0)
        over = np.flatnonzero(counts > max_terms)
        stop = int(over[0]) if len(over) else len(chunk)
        kept.extend(chunk[:stop])
        fits.extend(coefs[:, :stop].T)
        if len(over):
            break
        coef = coefs[:, -1]
    if not kept:  # even the largest penalty keeps too many terms
        return alphas[:1], _path(x, y, alphas[:1], l1_ratio)
    return np.array(kept), np.array(fits).T


def _path(
    x: np.ndarray,
    y: pd.Series,
    alphas: np.ndarray,
    l1_ratio: float,
    coef_init: np.ndarray | None = None,
) -> np.ndarray:
    """The fit at every penalty in ``alphas``, one column each, on centred ``y``."""
    _, coefs, _ = enet_path(
        x,
        y.to_numpy() - float(y.mean()),
        l1_ratio=l1_ratio,
        alphas=alphas,
        max_iter=10_000,
        coef_init=coef_init,
    )
    return coefs


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
"""The degrees the derived polynomial's grid tries. ``MAX_DEGREE`` still caps a
user-supplied function at 5, but degrees 4 and 5 were never the best configuration
on the Sep 2026 data, and a single degree-5 fold took 6-9 minutes
(docs/evidence/2026-10-12-design-verification.md)."""

CANDIDATE_REGULARIZERS: tuple[str, ...] = ("lasso",)
"""The regularizers the grid tries. Elastic net's forecasts correlated 0.98-1.00
with lasso's at the same degree on both targets, so it added nothing to choose
between (docs/evidence/2026-10-12-design-verification.md)."""

CANDIDATE_CONFIGS: tuple[DerivedPolynomial, ...] = tuple(
    DerivedPolynomial(degree=degree, regularizer=regularizer)
    for degree in CANDIDATE_DEGREES
    for regularizer in CANDIDATE_REGULARIZERS
)
"""The lasso configurations in the derived polynomial's grid, one per degree in
``CANDIDATE_DEGREES``."""

SHRINKAGE: str = "shrinkage"
"""The grid's other candidate: ``shrinkage.ShrinkagePolynomial``, a degree-1 fit
whose slopes are shrunk toward one common slope."""


def run_user_polynomial(
    formula: str,
    panel: FeaturePanel,
    splitter: PurgedWalkForward,
    bindings: Mapping[str, str] | None = None,
) -> tuple[ModelRunResult, ModelDescription]:
    """A user-supplied shape, scaled and shifted to the target in each fold.
    There is one configuration and nothing to choose between."""
    folds = evaluate(lambda: UserPolynomial(formula, dict(bindings or {})), panel, splitter)
    _require_folds(folds, panel, splitter)
    return summarize(folds)


def run_derived_polynomial(
    panel: FeaturePanel,
    splitter: PurgedWalkForward,
    candidates: tuple[DerivedPolynomial, ...] = CANDIDATE_CONFIGS,
    *,
    shrinkage: bool = True,
) -> tuple[ModelRunResult, ModelDescription]:
    """Evaluates every candidate in ``candidates``, and the shrinkage polynomial
    unless ``shrinkage`` is false, then reports the one with the best Signal Rank
    IC.

    Feature selection is screened per fold (``evaluate(..., screen=True)``) —
    a signal below FYP-102's inclusion threshold, judged on that fold's own
    train window, is left out of that fold's fit.
    """
    configs = [(c.degree, c.regularizer, c.max_terms) for c in candidates]
    if shrinkage:
        configs.append((1, SHRINKAGE, None))
    if not configs:
        raise PolynomialConfigError("deriving a function needs at least one candidate.")
    # Each configuration's walk-forward is independent of the others, so they
    # run in parallel and come back in order.
    fits = map_processes(
        _evaluate_candidate, configs, [panel] * len(configs), [splitter] * len(configs)
    )
    per_candidate = {
        SHRINKAGE if regularizer == SHRINKAGE else f"degree{degree}_{regularizer}": folds
        for (degree, regularizer, _), folds in zip(configs, fits, strict=True)
    }
    _require_folds(next(iter(per_candidate.values()), ()), panel, splitter)
    return summarize(per_candidate[select_best_candidate(per_candidate)])


def _evaluate_candidate(
    config: tuple[int, str, int | None], panel: FeaturePanel, splitter: PurgedWalkForward
) -> tuple[FoldResult, ...]:
    """One configuration's screened walk-forward: module-level, so a worker
    process can run it."""
    degree, regularizer, max_terms = config
    if regularizer == SHRINKAGE:
        return evaluate(ShrinkagePolynomial, panel, splitter, screen=True)
    return evaluate(
        lambda: DerivedPolynomial(degree, regularizer, max_terms), panel, splitter, screen=True
    )


def _require_folds(
    folds: tuple[FoldResult, ...], panel: FeaturePanel, splitter: PurgedWalkForward
) -> None:
    if not folds:
        raise PolynomialConfigError(splitter.too_short(panel))
