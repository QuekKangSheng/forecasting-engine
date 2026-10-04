"""A fitted polynomial as a readable function: its terms, its equation and its table.

``ModelDescription`` carries a derived fit's terms as ``PolynomialFeatures`` names
(``VIX_Index_PX_LAST^2 LUACOAS_Index_PX_LAST``) and a user-supplied fit as its
formula with the fitted scale ``b`` and intercept ``a`` (``a + b × formula``).
``from_description`` turns either into the same ``PolynomialFunction``, so one
renderer draws both.
"""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

from forecasting_engine.extraction.bloomberg_csv import DATE_COLUMN
from forecasting_engine.models.base import ModelDescription
from forecasting_engine.models.polynomial import PolynomialConfigError, _parse

MAX_EXPANDED_TERMS: int = 50
"""A user formula that expands past this many terms is shown as entered instead."""

SIGNIFICANT_DIGITS: int = 4
"""Significant figures for every coefficient shown. A working default, not yet
team-agreed — change it here."""

_MINUS = "−"
_POWER_RE = re.compile(r"^(.+)\^(\d+)$")

Factors = tuple[tuple[str, int], ...]


class Origin(StrEnum):
    USER_SUPPLIED = "User-Supplied Function"
    DERIVED = "Derived Function"


@dataclass(frozen=True)
class Term:
    coefficient: float
    factors: Factors
    """``((column, power), ...)``, sorted by column."""


@dataclass(frozen=True)
class PolynomialFunction:
    origin: Origin
    target: str
    """The price column the forecast return was built from."""
    horizon: int
    """Trading days."""
    intercept: float
    terms: tuple[Term, ...]
    """Empty when no term survived fitting, or in formula-only mode."""
    formula: str | None
    """Set only when a user formula couldn't be expanded into terms."""
    shape: str | None = None
    """A user function's formula as entered, before its fitted scale."""
    scale: float = 1.0
    """A user function's fitted ``b`` in ``a + b × shape``."""
    offset: float = 0.0
    """A user function's fitted ``a`` in ``a + b × shape``."""


def from_description(
    description: ModelDescription,
    *,
    origin: Origin,
    target: str,
    horizon: int,
    columns: Iterable[str] | None = None,
) -> PolynomialFunction:
    """Build the function a run fitted. Never raises on an odd term or formula.

    ``columns``, when given, are the signal names a derived term may use. The
    field half of a column name comes straight from an export's header and isn't
    sanitised, so it can contain a space; a term whose space-split parts aren't
    all known columns is kept whole rather than split into wrong factors.
    """
    if origin is Origin.USER_SUPPLIED:
        formula = description.terms[0]
        (scale,) = description.coefficients
        offset = 0.0 if description.intercept is None else float(description.intercept)
        expanded = _expand_formula(formula)
        if expanded is None:
            return PolynomialFunction(
                origin, target, horizon, offset, (), formula, formula, scale, offset
            )
        expanded = _scale(expanded, scale)
        intercept = offset + expanded.pop((), 0.0)
        terms = tuple(
            Term(coefficient, factors)
            for factors, coefficient in sorted(
                expanded.items(), key=lambda kv: (sum(p for _, p in kv[0]), kv[0])
            )
        )
        return PolynomialFunction(
            origin, target, horizon, intercept, terms, None, formula, scale, offset
        )

    known = None if columns is None else frozenset(columns)
    terms = tuple(
        Term(float(coefficient), _parse_term(name, known))
        for name, coefficient in zip(description.terms, description.coefficients, strict=True)
    )
    intercept = 0.0 if description.intercept is None else float(description.intercept)
    return PolynomialFunction(origin, target, horizon, intercept, terms, None)


def dataset_fingerprint(frame: pd.DataFrame) -> tuple:
    """Enough of a committed frame to tell when it has been replaced."""
    dates = frame[DATE_COLUMN]
    first, last = (dates.iloc[0], dates.iloc[-1]) if len(frame) else (None, None)
    return (len(frame), first, last, tuple(frame.columns))


# --- derived terms -----------------------------------------------------------


def _parse_term(name: str, known: frozenset[str] | None) -> Factors:
    raw: Factors = ((name, 1),)
    powers: dict[str, int] = {}
    for part in name.split(" "):
        match = _POWER_RE.match(part)
        column, power = (match.group(1), int(match.group(2))) if match else (part, 1)
        if not column or "^" in column or (known is not None and column not in known):
            return raw
        powers[column] = powers.get(column, 0) + power
    return tuple(sorted(powers.items()))


# --- user formulas -----------------------------------------------------------


class _Stop(Exception):
    """Expansion can't continue; the formula is shown as entered."""


Polynomial = dict[Factors, float]


def _expand_formula(formula: str) -> Polynomial | None:
    try:
        return _expand(_parse(formula).body)
    except (_Stop, PolynomialConfigError):
        return None


def _expand(node: ast.AST) -> Polynomial:
    if isinstance(node, ast.Constant):
        return _tidy({(): float(node.value)})
    if isinstance(node, ast.Name):
        return {((node.id, 1),): 1.0}
    if isinstance(node, ast.UnaryOp):
        operand = _expand(node.operand)
        return _scale(operand, -1.0) if isinstance(node.op, ast.USub) else operand
    if not isinstance(node, ast.BinOp):
        raise _Stop
    if isinstance(node.op, ast.Pow):
        base = _expand(node.left)
        result: Polynomial = {(): 1.0}
        for _ in range(int(node.right.value)):  # validated: a whole-number constant
            result = _multiply(result, base)
        return result
    left, right = _expand(node.left), _expand(node.right)
    if isinstance(node.op, ast.Add):
        return _add(left, right)
    if isinstance(node.op, ast.Sub):
        return _add(left, _scale(right, -1.0))
    if isinstance(node.op, ast.Mult):
        return _multiply(left, right)
    if isinstance(node.op, ast.Div):
        if any(factors for factors in right) or not right.get((), 0.0):
            raise _Stop  # divides by a signal, or by zero
        return _scale(left, 1.0 / right[()])
    raise _Stop


def _add(a: Polynomial, b: Polynomial) -> Polynomial:
    out = dict(a)
    for factors, coefficient in b.items():
        out[factors] = out.get(factors, 0.0) + coefficient
    return _tidy(out)


def _scale(a: Polynomial, by: float) -> Polynomial:
    return _tidy({factors: coefficient * by for factors, coefficient in a.items()})


def _multiply(a: Polynomial, b: Polynomial) -> Polynomial:
    out: Polynomial = {}
    for fa, ca in a.items():
        for fb, cb in b.items():
            powers = dict(fa)
            for column, power in fb:
                powers[column] = powers.get(column, 0) + power
            key = tuple(sorted(powers.items()))
            out[key] = out.get(key, 0.0) + ca * cb
            if len(out) > MAX_EXPANDED_TERMS:
                raise _Stop
    return _tidy(out)


def _tidy(poly: Polynomial) -> Polynomial:
    out = {factors: coefficient for factors, coefficient in poly.items() if coefficient != 0.0}
    if len(out) > MAX_EXPANDED_TERMS:
        raise _Stop
    return out


# --- rendering ---------------------------------------------------------------


def significant(value: float, digits: int = SIGNIFICANT_DIGITS) -> str:
    """``value`` to ``digits`` significant figures, trailing zeros kept.

    Scientific form (``1.234 × 10^-5``) below 1e-4 or from 1e6, a true minus sign.
    """
    if not math.isfinite(value):
        return str(value)
    if value == 0:
        return "0"
    sign = _MINUS if value < 0 else ""
    mantissa, exponent = f"{abs(value):.{digits - 1}e}".split("e")
    rounded = float(f"{mantissa}e{exponent}")
    power = int(exponent)
    if rounded < 1e-4 or rounded >= 1e6:
        return f"{sign}{mantissa} × 10^{power}"
    return f"{sign}{rounded:.{max(digits - 1 - power, 0)}f}"


def to_latex(fn: PolynomialFunction, label: Callable[[str], str]) -> str:
    lead = r"\hat{y} = "
    if fn.formula is not None:
        return shape_latex(fn, label)

    pieces: list[tuple[bool, str]] = []  # (negative, body)
    if fn.intercept != 0 or not fn.terms:
        pieces.append((fn.intercept < 0, _latex_number(abs(fn.intercept))))
    for term in fn.terms:
        factors = r"\,".join(_latex_factor(column, power, label) for column, power in term.factors)
        magnitude = abs(term.coefficient)
        body = factors if significant(magnitude) == significant(1.0) else (
            rf"{_latex_number(magnitude)}\,{factors}"
        )
        pieces.append((term.coefficient < 0, body))

    out = []
    for i, (negative, body) in enumerate(pieces):
        if i == 0:
            out.append(f"-{body}" if negative else body)
        else:
            out.append(f" - {body}" if negative else f" + {body}")
    return lead + "".join(out)


def shape_latex(fn: PolynomialFunction, label: Callable[[str], str]) -> str:
    """A user function as fitted: ``a + b · (formula)``, with its own formula
    written back as entered. Without a fitted scale, the formula alone."""
    body = _latex_node(_parse(fn.shape or fn.formula).body, label)
    if fn.scale == 1.0 and fn.offset == 0.0:
        return r"\hat{y} = " + body
    sign = "-" if fn.scale < 0 else "+"
    return (
        rf"\hat{{y}} = {_latex_signed(fn.offset)} {sign} "
        rf"{_latex_number(abs(fn.scale))} \cdot ({body})"
    )


def _latex_signed(value: float) -> str:
    return f"-{_latex_number(abs(value))}" if value < 0 else _latex_number(value)


def term_rows(fn: PolynomialFunction, label: Callable[[str], str]) -> list[dict[str, str]]:
    """One row per term of the equation, in the same order, every value a string."""
    if fn.formula is not None:
        return []
    rows = []
    if fn.intercept != 0 or not fn.terms:
        rows.append(
            {"Factor": "(intercept)", "Exponent": "—", "Coefficient": significant(fn.intercept)}
        )
    for term in fn.terms:
        rows.append(
            {
                "Factor": " × ".join(label(column) for column, _ in term.factors),
                "Exponent": " × ".join(str(power) for _, power in term.factors),
                "Coefficient": significant(term.coefficient),
            }
        )
    return rows


def _latex_number(value: float) -> str:
    text = significant(value)
    if " × 10^" in text:
        mantissa, power = text.split(" × 10^")
        text = rf"{mantissa} \times 10^{{{power}}}"
    return text.replace(_MINUS, "-")


def _latex_factor(column: str, power: int, label: Callable[[str], str]) -> str:
    text = rf"\text{{{_escape(label(column))}}}"
    return text if power == 1 else f"{text}^{{{power}}}"


_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _escape(text: str) -> str:
    return "".join(_ESCAPES.get(char, char) for char in text)


_PRECEDENCE = {ast.Add: 1, ast.Sub: 1, ast.Mult: 2, ast.Div: 2}
_UNARY, _POWER, _ATOM = 3, 4, 5
_SYMBOLS = {ast.Add: "+", ast.Sub: "-", ast.Mult: r"\cdot", ast.Div: "/"}


def _latex_node(node: ast.AST, label: Callable[[str], str]) -> str:
    """Write a formula's syntax tree back out, bracketing only where precedence needs it."""
    if isinstance(node, ast.Constant):
        return f"{node.value:g}"
    if isinstance(node, ast.Name):
        return rf"\text{{{_escape(label(node.id))}}}"
    if isinstance(node, ast.UnaryOp):
        sign = "-" if isinstance(node.op, ast.USub) else "+"
        return sign + _bracket(node.operand, _UNARY, label)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        return f"{_bracket(node.left, _ATOM, label)}^{{{node.right.value:g}}}"
    assert isinstance(node, ast.BinOp)
    precedence = _PRECEDENCE[type(node.op)]
    # Subtraction and division aren't associative: a right operand of equal
    # precedence needs brackets, a - (b + c), where a left one doesn't.
    right_needs = precedence + 1 if isinstance(node.op, ast.Sub | ast.Div) else precedence
    left = _bracket(node.left, precedence, label)
    right = _bracket(node.right, right_needs, label)
    return f"{left} {_SYMBOLS[type(node.op)]} {right}"


def _bracket(node: ast.AST, needs: int, label: Callable[[str], str]) -> str:
    text = _latex_node(node, label)
    return f"({text})" if _node_precedence(node) < needs else text


def _node_precedence(node: ast.AST) -> int:
    if isinstance(node, ast.BinOp):
        return _POWER if isinstance(node.op, ast.Pow) else _PRECEDENCE[type(node.op)]
    if isinstance(node, ast.UnaryOp):
        return _UNARY
    return _ATOM
