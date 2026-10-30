"""Exact per-signal SHAP attribution for a fitted polynomial function.

Mirrors the ML models' mean |SHAP value| table, but computed exactly from
the polynomial's own terms rather than sampled: each term's contribution
decomposes into its factors' Shapley values in closed form, relative to
each factor's mean over the training rows.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from forecasting_engine.models.base import ModelDescription
from forecasting_engine.reporting.polynomial_function import PolynomialFunction, Term


def polynomial_shap(
    fn: PolynomialFunction,
    training: pd.DataFrame,
    input_bounds: Mapping[str, tuple[float, float]] | None = None,
) -> ModelDescription | None:
    """Mean |SHAP value| per signal over ``training``'s rows, in ML's
    terms/coefficients table shape. ``None`` if ``fn`` has no terms (a
    formula that divides by a signal, or expands past 50 terms).

    Each term's value decomposes across its factors by the exact Shapley
    value for a product: a degree-1 term reduces to coefficient × (value −
    training mean); a square attributes wholly to its one signal; an
    interaction splits between its two signals, summing back to the term's
    own contribution.
    """
    if fn.formula is not None:
        return None
    needed = dict.fromkeys(column for term in fn.terms for column, _ in term.factors)
    clean = training[list(needed)].dropna() if needed else training.iloc[:0]
    totals: dict[str, np.ndarray] = {}
    for term in fn.terms:
        for signal, share in _term_shapley(term, fn.standardisation, input_bounds, clean).items():
            totals[signal] = totals.get(signal, 0.0) + share
    return ModelDescription(
        name=fn.origin,
        terms=tuple(totals),
        coefficients=tuple(float(np.abs(values).mean()) for values in totals.values()),
    )


def _term_shapley(
    term: Term,
    standardisation: Mapping[str, tuple[float, float]] | None,
    input_bounds: Mapping[str, tuple[float, float]] | None,
    training: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """One term's coefficient × factor-product, decomposed across its distinct
    signals by the exact Shapley value for a product of ``k`` players."""
    signals = [column for column, _ in term.factors]
    values: dict[str, np.ndarray] = {}
    baselines: dict[str, float] = {}
    for column, power in term.factors:
        raw = training[column].to_numpy(dtype=float)
        if input_bounds is not None and column in input_bounds:
            low, high = input_bounds[column]
            raw = np.clip(raw, low, high)
        if standardisation is not None and column in standardisation:
            mean, sd = standardisation[column]
            raw = (raw - mean) / sd
        value = raw**power
        values[column] = value
        baselines[column] = float(value.mean())

    k = len(signals)
    shares: dict[str, np.ndarray] = {}
    for i, signal in enumerate(signals):
        others = [s for j, s in enumerate(signals) if j != i]
        diff = values[signal] - baselines[signal]
        weight = np.zeros(len(training))
        for size in range(k):
            for subset in itertools.combinations(others, size):
                mix = np.ones(len(training))
                for s in subset:
                    mix = mix * values[s]
                for s in others:
                    if s not in subset:
                        mix = mix * baselines[s]
                weight = weight + mix / math.comb(k - 1, size)
        shares[signal] = term.coefficient * diff * weight / k
    return shares
