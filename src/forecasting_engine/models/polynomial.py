"""FYP-43: the user-supplied polynomial Forecaster.

``UserPolynomial`` plus the function that runs it through the shared harness and
shapes the result into ``reporting.model_metrics.ModelRunResult`` — the fixed
contract FYP-45/FYP-14 already built the comparison view and promotion gates
against. The derived polynomial is ``models/sign_ruled.py``.
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Mapping
from dataclasses import dataclass, field

import pandas as pd

from forecasting_engine.ingest.align import FeaturePanel
from forecasting_engine.models.base import ModelDescription
from forecasting_engine.reporting.model_metrics import ModelRunResult
from forecasting_engine.validation.harness import FoldResult, evaluate, summarize
from forecasting_engine.validation.splitters import PurgedWalkForward


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


# ── Bridging to the shared comparison view (ModelRunResult) ─────────────────


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


def _require_folds(
    folds: tuple[FoldResult, ...], panel: FeaturePanel, splitter: PurgedWalkForward
) -> None:
    if not folds:
        raise PolynomialConfigError(splitter.too_short(panel))
