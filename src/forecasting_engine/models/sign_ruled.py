"""The sign-ruled polynomial: a degree-1 polynomial in risk gauges oriented by economics,
whose slopes are shrunk toward one common positive slope.

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
from scipy.special import log_ndtr, ndtr, ndtri

from forecasting_engine.extraction.bloomberg_csv import ColumnSource
from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.ingest.align import FeaturePanel, Transform, align_and_lag
from forecasting_engine.models.base import ModelDescription
from forecasting_engine.models.polynomial import PolynomialConfigError
from forecasting_engine.reporting.model_metrics import ModelRunResult
from forecasting_engine.validation.harness import evaluate, summarize
from forecasting_engine.validation.splitters import PurgedWalkForward

ECONOMIC_INPUTS: Mapping[TargetRole, Mapping[str, int]] = {
    TargetRole.EQUITY: {"LF98OAS": 1, "LUACOAS": 1, "VIX": 1, "USYC2Y10": 1},
}
"""Per target, each input's ticker and economic sign (+1: a higher value means a higher
expected return). Read as levels. No bond set beat chance in the research, so the
method is equity-only."""

CLIP_SD: float = 4.0
"""Each input is clipped to this many training standard deviations from its mean, so
one extreme value can't drive an extreme forecast."""

LEVEL_ROWS: int = 252
"""The forecast's level is the mean of this many earliest training labels."""

COMMON_SLOPE_SCALE: float = 0.2
"""Prior scale of the common slope ``m``, in standardised units of the target."""

MAX_OVERLAP: float = 63.0
"""Cap on the estimated label overlap the likelihood is divided by."""

CHAINS: int = 24
BURN_IN: int = 150
KEPT_SWEEPS: int = 150
SEED: int = 20261008

_MIN_TRAINING_ROWS: int = 30

VERSION: int = 1
"""Raise when a change above alters a fit, so results saved under the old one refit."""


class SignRuledInputsError(PolynomialConfigError):
    """The committed data lacks an input the sign-ruled polynomial needs."""


# ── The sampler ──────────────────────────────────────────────────────────────


def _log_mass(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``log(Φ(b) − Φ(a))`` elementwise, ``-inf`` where the interval is empty."""
    flip = a > 0
    hi = np.where(flip, -a, b)
    lo = np.where(flip, -b, a)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        lh, ll = log_ndtr(hi), log_ndtr(lo)
        out = lh + np.log1p(-np.exp(ll - lh))
    ok = (b > a) & np.isfinite(lh)
    return np.where(ok & ~np.isnan(out), out, -np.inf)


def _truncated_normal(a: np.ndarray, b: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Standard normal draws truncated to ``[a, b]``, by inversion in their own tail."""
    flip = a > 0
    a2 = np.where(flip, -b, a)
    b2 = np.where(flip, -a, b)
    pa, pb = ndtr(a2), ndtr(b2)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = ndtri(pa + u * (pb - pa))
    z = np.clip(np.where(flip, -z, z), a, b)
    fallback = np.where(np.isfinite(a), a, np.where(np.isfinite(b), b, 0.0))
    return np.where(np.isfinite(z), z, fallback)


def _piece(q, b, precision, centre, lo, hi):
    """One side (``[lo, hi]``) of a slope's conditional: its log mass, mean and sd, and
    the standardised bounds."""
    a = q + precision
    linear = b + precision * centre
    mean = linear / a
    sd = 1.0 / np.sqrt(a)
    with np.errstate(invalid="ignore"):
        low, high = (lo - mean) / sd, (hi - mean) / sd
    log_mass = (
        -0.5 * np.log(a) + linear**2 / (2 * a) - precision * centre**2 / 2 + _log_mass(low, high)
    )
    return log_mass, mean, sd, low, high


def _posterior_mean_slopes(
    gram: np.ndarray, cross: np.ndarray, sum_sq: float, n_eff: float, rng: np.random.Generator
) -> np.ndarray:
    """Posterior mean of ``β`` for ``y ~ N(Zβ, σ²)`` from the tempered sufficient
    statistics ``Z'Z``, ``Z'y`` and ``y'y``, under the exchangeable horseshoe prior
    (Makalic & Schmidt's auxiliary-variable sampler).

    Each slope is drawn from its two sides, ``β ≥ 0`` and ``β ≤ 0``, which together
    are its unrestricted conditional; the draws are kept in that form so a fit is
    exactly reproducible from ``SEED``."""
    p, c = len(cross), CHAINS
    tau0 = 1.0 / np.sqrt(n_eff)
    beta = np.zeros((p, c))
    lam2, nu = np.ones((p, c)), np.ones((p, c))
    tau2 = np.full(c, tau0**2)
    xi = np.ones(c)
    sig2, common = np.ones(c), np.zeros(c)
    inf = np.full(c, np.inf)
    total, draws = np.zeros(p), 0
    for sweep in range(BURN_IN + KEPT_SWEEPS):
        for j in range(p):
            r = cross[j] - gram[j] @ beta + gram[j, j] * beta[j]
            q, b = gram[j, j] / sig2, r / sig2
            precision = 1.0 / (sig2 * tau2 * lam2[j])
            lp, mp, sp, ap, bp = _piece(q, b, precision, common, np.zeros(c), inf)
            up = mp + sp * _truncated_normal(ap, bp, rng.random(c))
            ln, mn, sn, an, bn = _piece(q, b, precision, common, -inf, np.zeros(c))
            down = mn + sn * _truncated_normal(an, bn, rng.random(c))
            both = np.logaddexp(lp, ln)
            with np.errstate(invalid="ignore"):
                take_up = np.log(rng.random(c)) < lp - both
            new = np.where(take_up, up, down)
            beta[j] = np.where(np.isfinite(both) & np.isfinite(new), new, beta[j])
        weight = 1.0 / (tau2 * lam2)
        precision = (1.0 / COMMON_SLOPE_SCALE**2 + weight.sum(0)) / sig2
        mean = (weight * beta).sum(0) / sig2 / precision
        sd = 1.0 / np.sqrt(precision)
        common = mean + sd * _truncated_normal(-mean / sd, inf, rng.random(c))
        spread = (beta - common) ** 2
        lam2 = (1.0 / nu + spread / (2 * sig2 * tau2)) / rng.exponential(size=(p, c))
        nu = (1.0 + 1.0 / lam2) / rng.exponential(size=(p, c))
        rate = 1.0 / xi + (spread / lam2).sum(0) / (2 * sig2)
        tau2 = rate / rng.gamma((p + 1) / 2, size=c)
        xi = (1.0 / tau0**2 + 1.0 / tau2) / rng.exponential(size=c)
        rss = sum_sq - 2 * cross @ beta + (beta * (gram @ beta)).sum(0)
        penalty = (spread / (tau2 * lam2)).sum(0) + common**2 / COMMON_SLOPE_SCALE**2
        shape = (n_eff + p + 1) / 2
        sig2 = (np.maximum(rss, 1e-9) + penalty) / 2 / rng.gamma(shape, size=c)
        if sweep >= BURN_IN:
            total += beta.sum(1)
            draws += c
    return total / draws


def overlap(labels: pd.Series) -> float:
    """How many rows a label effectively shares with its neighbours,
    ``1 / (1 − lag-1 autocorrelation)``, between 1 and ``MAX_OVERLAP``."""
    ac = labels.autocorr(1)
    if not np.isfinite(ac) or ac <= 0:
        return 1.0
    return float(np.clip(1.0 / (1.0 - ac), 1.0, MAX_OVERLAP))


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


def expanding(splitter: PurgedWalkForward) -> PurgedWalkForward:
    """``splitter``'s test windows, each trained on every row before its embargo.

    The first test window opens where ``splitter``'s does, after the longer of its
    train window and tuning period, so every model on the page is scored on the same
    dates."""
    return PurgedWalkForward(
        train=None,
        test=splitter.test,
        embargo=splitter.embargo,
        tuning_rows=max(splitter.train or 0, splitter.tuning_rows),
    )


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
    configuration, so no PBO. Returns the panel too, for the equation's columns."""
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
    result, description = summarize(folds, pbo=None)
    return result, description, panel
