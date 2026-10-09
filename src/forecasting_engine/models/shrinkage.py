"""Shrinkage polynomial: a degree-1 polynomial whose slopes are shrunk toward one
common slope, every choice made on the fold's own training rows.

It keeps the idea of the October 2026 model research (models/sign_ruled.py) that
moved together, risk gauges are better averaged than picked between, which a
lasso does not do: it keeps one of a correlated group and drops the rest. The
research fixed four inputs and their signs by hand, after trying about 300
variants on data that overlaps the scored period; here nothing is fixed:

1. the signals are those per-fold screening kept (``harness.evaluate(screen=True)``),
   each transformed as the page transforms it;
2. each signal's direction is the sign of its Rank IC on the training rows;
3. oriented, clipped to its mean ± ``CLIP_SD`` sd and standardised;
4. the slopes are the posterior mean of a Bayesian regression with an exchangeable
   horseshoe prior, ``β_j ~ N(m, σ²τ²λ_j²)``, with the common slope ``m ≥ 0``
   (prior scale ``COMMON_SLOPE_SCALE``). The likelihood is divided by the labels'
   overlap, ``1 / (1 − lag-1 autocorrelation)``, so ``h``-day labels count as about
   ``n / h`` independent periods;
5. the posterior is drawn by Gibbs sampling: ``CHAINS`` chains, ``BURN_IN`` sweeps
   discarded and ``KEPT_SWEEPS`` kept, from ``SEED``, so a fit is reproducible;
6. the forecast's level is the training mean, like every other model's, so what it
   adds to that level is its slopes alone.

It is one candidate in the derived polynomial's grid (``polynomial.run_derived_polynomial``),
chosen only when it beats the lasso and elastic-net fits on Signal Rank IC, and
counted by PBO like them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import log_ndtr, ndtr, ndtri

from forecasting_engine.ingest.align import FeaturePanel
from forecasting_engine.models.base import ModelDescription
from forecasting_engine.validation.metrics import rank_ic

CLIP_SD: float = 4.0
"""Each input is clipped to this many training standard deviations from its mean, so
one extreme value can't drive an extreme forecast."""

COMMON_SLOPE_SCALE: float = 0.2
"""Prior scale of the common slope ``m``, in standardised units of the target."""

MAX_OVERLAP: float = 63.0
"""Cap on the estimated label overlap the likelihood is divided by."""

CHAINS: int = 24
BURN_IN: int = 150
KEPT_SWEEPS: int = 150
SEED: int = 20261008

_MIN_TRAINING_ROWS: int = 30


class ShrinkageDataError(ValueError):
    """The message is written for a portfolio manager, like the other models' errors."""


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
class ShrinkagePolynomial:
    """``forecast = training mean + Σ β_j · z_j``, one term per screened signal, ``z_j``
    the oriented, clipped, standardised signal, with the slopes from
    ``_posterior_mean_slopes``."""

    seed: int = SEED
    name: str = field(default="ShrinkagePolynomial", init=False)

    def __post_init__(self) -> None:
        self._coef: np.ndarray | None = None

    def fit(self, panel: FeaturePanel, train: pd.DatetimeIndex) -> None:
        signals, target = list(panel.signals), panel.targets[0]
        frame = panel.frame.loc[train, [*signals, target]].dropna()
        if len(frame) < _MIN_TRAINING_ROWS:
            raise ShrinkageDataError(
                f"not enough complete training rows for the shrinkage polynomial (need at "
                f"least {_MIN_TRAINING_ROWS}, got {len(frame)})."
            )
        y = frame[target]
        # A signal with no rank IC (no variation) counts as pointing upward; its
        # standardised value is constant, so it adds nothing either way.
        self._sign = pd.Series(
            {s: -1.0 if rank_ic(frame[s], y) < 0 else 1.0 for s in signals}, dtype=float
        )
        self._columns = signals
        x = frame[signals] * self._sign
        self._mean, sd = x.mean(), x.std()
        self._sd = sd.where(sd > 0, 1.0)
        self._low, self._high = self._mean - CLIP_SD * self._sd, self._mean + CLIP_SD * self._sd
        z = self._z(x)
        z_mean, z_sd = z.mean(0), z.std(0)
        z_sd[z_sd == 0] = 1.0
        design = (z - z_mean) / z_sd
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
        self._intercept = y_mean - float(self._coef @ z_mean)

    def predict(self, panel: FeaturePanel, idx: pd.DatetimeIndex) -> pd.Series:
        if self._coef is None:
            raise RuntimeError("predict() called before fit()")
        predicted = pd.Series(np.nan, index=idx, dtype=float)
        x = panel.frame.loc[idx, self._columns].dropna()
        if not x.empty:
            predicted.loc[x.index] = self._intercept + self._z(x * self._sign) @ self._coef
        return predicted

    def describe(self) -> ModelDescription:
        """The equation in each signal's own direction: a sign of −1 turns ``z`` around,
        so its slope is reported negated, with the unturned signal's mean and bounds."""
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
