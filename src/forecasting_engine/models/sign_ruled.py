"""The sign-ruled polynomial: a degree-1 polynomial in risk gauges oriented by economics,
whose slopes are shrunk toward one common positive slope.

Run on the Models page's Equity tab as its own row, "Polynomial – fixed economic
inputs", exactly as the research ran it. Its inputs, their signs and its level were chosen by
looking at results over Aug 2016 - Sep 2024, which overlaps the scored period, so its
scores there overstate it; the page's derived polynomial uses the same shrinkage
(``models/shrinkage.py``) with every choice made per fold instead. See
docs/methodology.md.

From the October 2026 model research (its model ``bayes/eq_exch``, chosen from about
300 trials under a selection rule written before the held-back data was scored):
- for equity, HY OAS, IG OAS, VIX and the 2s10s slope all say "a higher value, a higher
  expected return" (a risk premium);
- they move together, so a lasso keeps one and drops the rest; a prior that pulls the
  four slopes toward a shared value keeps all four and averages their noise;
- no term above degree 1 helped out of sample in any technique tried (up to degree 5);
- the edge needs every earlier row: in a 252-row rolling window it disappeared.

Per fold, on the training rows only:
1. each input is turned by its economic sign, clipped to its mean ± ``CLIP_SD`` sd and
   standardised, ``z = (x − mean) / sd``;
2. the forecast's level is the mean of the first ``LEVEL_ROWS`` training labels, not the
   training mean, which drifts and ranks equity returns the wrong way;
3. the slopes are the posterior mean of a Bayesian regression with an exchangeable
   horseshoe prior, ``β_j ~ N(m, σ²τ²λ_j²)``, with the common slope ``m ≥ 0``
   (prior scale ``COMMON_SLOPE_SCALE``). The likelihood is divided by the labels'
   overlap, ``1 / (1 − lag-1 autocorrelation)``, so ``h``-day labels count as about
   ``n / h`` independent periods;
4. the posterior is drawn by Gibbs sampling: ``CHAINS`` chains, ``BURN_IN`` sweeps
   discarded and ``KEPT_SWEEPS`` kept, from ``SEED``, so a fit is reproducible.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from forecasting_engine.extraction.bloomberg_csv import ColumnSource
from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.ingest.align import FeaturePanel, Transform, align_and_lag
from forecasting_engine.models.base import ModelDescription
from forecasting_engine.models.polynomial import PolynomialConfigError
from forecasting_engine.models.shrinkage import (  # noqa: F401  (overlap is re-exported)
    CLIP_SD,
    SEED,
    _posterior_mean_slopes,
    overlap,
)
from forecasting_engine.reporting.model_metrics import ModelRunResult
from forecasting_engine.validation.harness import evaluate, summarize
from forecasting_engine.validation.splitters import (  # noqa: F401  (re-exported)
    PurgedWalkForward,
    expanding,
)

ECONOMIC_INPUTS: Mapping[TargetRole, Mapping[str, int]] = {
    TargetRole.EQUITY: {"LF98OAS": 1, "LUACOAS": 1, "VIX": 1, "USYC2Y10": 1},
}
"""Per target, each input's ticker and economic sign (+1: a higher value means a higher
expected return). Read as levels. No bond set beat chance in the research, so the
method is equity-only."""

LEVEL_ROWS: int = 252
"""The forecast's level is the mean of this many earliest training labels."""

_MIN_TRAINING_ROWS: int = 30

VERSION: int = 1
"""Raise when a change above alters a fit, so results saved under the old one refit."""


class SignRuledInputsError(PolynomialConfigError):
    """The committed data lacks an input the sign-ruled polynomial needs."""


# ── The Forecaster ───────────────────────────────────────────────────────────


@dataclass
class SignRuledPolynomial:
    """``forecast = level + Σ β_j · z_j``, one term per input, ``z_j`` the clipped,
    standardised input, with the slopes from ``_posterior_mean_slopes``."""

    signs: Mapping[str, int]
    """Input column -> economic sign (+1 or −1)."""
    seed: int = SEED
    name: str = field(default="SignRuledPolynomial", init=False)

    def __post_init__(self) -> None:
        self._columns = list(self.signs)
        self._sign = pd.Series(self.signs, dtype=float)
        self._coef: np.ndarray | None = None
        self._intercept: float | None = None

    def fit(self, panel: FeaturePanel, train: pd.DatetimeIndex) -> None:
        target = panel.targets[0]
        frame = panel.frame.loc[train, [*self._columns, target]].dropna()
        if len(frame) < _MIN_TRAINING_ROWS:
            raise PolynomialConfigError(
                f"not enough complete training rows for the sign-ruled polynomial (need at "
                f"least {_MIN_TRAINING_ROWS}, got {len(frame)})."
            )
        x = frame[self._columns] * self._sign
        self._mean, sd = x.mean(), x.std()
        self._sd = sd.where(sd > 0, 1.0)
        self._low, self._high = self._mean - CLIP_SD * self._sd, self._mean + CLIP_SD * self._sd
        z = self._z(x)
        z_mean, z_sd = z.mean(0), z.std(0)
        z_sd[z_sd == 0] = 1.0
        design = (z - z_mean) / z_sd
        y = frame[target]
        values = y.to_numpy(float)
        y_mean, y_sd = values.mean(), values.std()
        scaled = (values - y_mean) / y_sd if y_sd > 0 else np.zeros_like(values)
        weight = 1.0 / overlap(y)
        beta = _posterior_mean_slopes(
            weight * design.T @ design,
            weight * design.T @ scaled,
            weight * float(scaled @ scaled),
            weight * len(scaled),
            np.random.default_rng(self.seed),
        )
        self._coef = y_sd * beta / z_sd
        level = float(y.iloc[:LEVEL_ROWS].mean())
        self._intercept = level - float(self._coef @ z_mean)

    def predict(self, panel: FeaturePanel, idx: pd.DatetimeIndex) -> pd.Series:
        if self._coef is None:
            raise RuntimeError("predict() called before fit()")
        predicted = pd.Series(np.nan, index=idx, dtype=float)
        x = panel.frame.loc[idx, self._columns].dropna()
        if not x.empty:
            predicted.loc[x.index] = self._intercept + self._z(x * self._sign) @ self._coef
        return predicted

    def describe(self) -> ModelDescription:
        """The equation in each input's own direction: a sign of −1 turns ``z`` around,
        so its slope is reported negated, with the unturned input's mean and bounds."""
        if self._coef is None:
            raise RuntimeError("describe() called before fit()")
        sign = self._sign
        mean = self._mean * sign
        low = np.minimum(self._low * sign, self._high * sign)
        high = np.maximum(self._low * sign, self._high * sign)
        return ModelDescription(
            name=self.name,
            terms=tuple(self._columns),
            coefficients=tuple(float(c * s) for c, s in zip(self._coef, sign, strict=True)),
            intercept=self._intercept,
            input_bounds={c: (float(low[c]), float(high[c])) for c in self._columns},
            standardisation={c: (float(mean[c]), float(self._sd[c])) for c in self._columns},
        )

    def _z(self, signed: pd.DataFrame) -> np.ndarray:
        clipped = signed[self._columns].clip(lower=self._low, upper=self._high, axis=1)
        return ((clipped - self._mean) / self._sd).to_numpy(float)


# ── Inputs and the run ───────────────────────────────────────────────────────


def economic_signs(
    columns: Sequence[str], sources: Mapping[str, ColumnSource], role: TargetRole
) -> dict[str, int]:
    """Signal column -> economic sign for ``role``'s inputs, found by each column's ticker."""
    wanted = ECONOMIC_INPUTS.get(role)
    if wanted is None:
        raise SignRuledInputsError(
            "the sign-ruled polynomial is for the equity target only: no bond input set "
            "beat chance in testing."
        )
    by_ticker: dict[str, list[str]] = {}
    for c in columns:
        if c in sources:
            by_ticker.setdefault(sources[c].ticker, []).append(c)
    missing = [t for t in wanted if t not in by_ticker]
    if missing:
        raise SignRuledInputsError(
            f"the sign-ruled polynomial needs {', '.join(missing)} in the committed data."
        )
    # Two securities can share a ticker (VIX Index, VIX Curncy): refuse rather than
    # guess which one is the gauge.
    ambiguous = {t: by_ticker[t] for t in wanted if len(by_ticker[t]) > 1}
    if ambiguous:
        listed = "; ".join(f"{t}: {', '.join(cols)}" for t, cols in ambiguous.items())
        raise SignRuledInputsError(
            f"the sign-ruled polynomial found more than one column for an input ({listed}). "
            "Keep one per input."
        )
    return {by_ticker[t][0]: sign for t, sign in wanted.items()}


def run_sign_ruled_polynomial(
    frame: pd.DataFrame,
    signs: Mapping[str, int],
    price_col: str,
    horizon: int,
    splitter: PurgedWalkForward,
) -> tuple[ModelRunResult, ModelDescription, FeaturePanel]:
    """Walk the sign-ruled polynomial forward on ``splitter``'s test windows.

    Its inputs are read as levels whatever the page's transform map says (the 2s10s
    slope is otherwise differenced), and every fold trains on all earlier rows. One
    configuration. Returns the panel too, for the equation's columns."""
    panel = align_and_lag(
        frame,
        list(signs),
        price_col,
        horizon=horizon,
        transforms=dict.fromkeys(signs, Transform.LEVEL),
    )
    folds = evaluate(lambda: SignRuledPolynomial(dict(signs)), panel, expanding(splitter))
    if not folds:
        raise PolynomialConfigError(expanding(splitter).too_short(panel))
    result, description = summarize(folds)
    return result, description, panel
