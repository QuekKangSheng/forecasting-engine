"""Prediction-quality metrics shared across screening and validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.stats.sandwich_covariance import weights_uniform


def rank_ic(signal: pd.Series, target: pd.Series) -> float:
    """Spearman rank correlation between a signal and its target.

    Pairs with a NaN in either series are dropped before scoring. Returns NaN
    if fewer than two valid pairs remain.

    Computed as the Pearson correlation of ranks rather than
    ``Series.corr(method="spearman")`` directly: on the installed pandas
    version that path imports scipy internally, which isn't a project
    dependency. Ranking first and correlating is the textbook definition of
    Spearman's rho and needs no new dependency.
    """
    paired = pd.concat([signal, target], axis=1).dropna()
    if len(paired) < 2:
        return float("nan")
    ic = paired.iloc[:, 0].rank().corr(paired.iloc[:, 1].rank())
    # Pearson-of-ranks lands a hair off +/-1.0 on clean monotonic data due to
    # float rounding; correlation is bounded to [-1, 1] so 15 decimal places
    # is far more precision than the statistic carries anyway.
    return round(ic, 15)


def ic(signal: pd.Series, target: pd.Series) -> float:
    """Plain (Pearson) correlation between a signal and its target.

    Pairs with a NaN in either series are dropped before scoring. Returns NaN
    if fewer than two valid pairs remain. Unlike ``rank_ic``, this is
    sensitive to the signal's actual scale and outliers, not just its order —
    the two are reported side by side rather than one replacing the other.
    """
    paired = pd.concat([signal, target], axis=1).dropna()
    if len(paired) < 2:
        return float("nan")
    return round(paired.iloc[:, 0].corr(paired.iloc[:, 1]), 15)


def rmse(predicted: pd.Series, actual: pd.Series) -> float:
    """Root-mean-squared error between predictions and the realised target.

    Pairs with a NaN in either series are dropped before scoring, the same
    convention ``rank_ic``/``ic`` use. Returns NaN if no valid pairs remain.
    """
    paired = pd.concat([predicted, actual], axis=1).dropna()
    if paired.empty:
        return float("nan")
    errors = paired.iloc[:, 0] - paired.iloc[:, 1]
    return float((errors**2).mean() ** 0.5)


def overlap_fit(y: np.ndarray, x: np.ndarray, *, lags: int) -> RegressionResultsWrapper:
    """The OLS of ``y`` on ``x`` with an intercept, its covariance allowing for errors
    that overlap over ``lags`` rows: every autocovariance up to ``lags`` counted in
    full (Hansen & Hodrick 1980).

    h-day labels a day apart share h - 1 days, so their errors are correlated out to
    h - 1 rows and no further. Newey-West's weights shrink an autocovariance as its
    lag grows, by ``1 - lag / (lags + 1)``, which leaves the variance too small when
    every one of those lags matters. On simulated 20-day labels with no
    predictability and a slow-moving signal, the Newey-West error was 0.84 of the
    Rank IC's real spread and t exceeded 2 in 4.6% of samples, twice the 2.3% it
    should; counted in full, 0.98 and 2.9%.

    Equal weights do not guarantee a positive variance, and on a sample not much
    longer than ``lags`` they can give one far too small (over every lag a sample
    has, the scores' autocovariances sum to exactly zero). So the fit returned is
    never surer than Newey-West's over the same lags: where the slope's standard
    error counted in full is smaller, or not a positive number, the fit carries the
    Newey-West covariance instead.
    """
    model = sm.OLS(y, sm.add_constant(x))
    newey_west = model.fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    with np.errstate(invalid="ignore"):
        in_full = model.fit(
            cov_type="HAC", cov_kwds={"maxlags": lags, "weights_func": weights_uniform}
        )
        wider = bool(in_full.bse[1] > newey_west.bse[1])  # False for a NaN
    return in_full if wider else newey_west


def rank_ic_se(
    predicted: pd.Series, realised: pd.Series, lags: int, *, overlap: bool = False
) -> float:
    """Standard error of ``rank_ic(predicted, realised)``, allowing for errors
    correlated over ``lags`` rows.

    Regresses standardised ranks of the realised values on standardised ranks
    of the predictions, whose slope is the rank IC, with a HAC covariance over
    ``lags`` lags. With ``overlap``, for the overlap between neighbouring h-day
    labels (``lags`` is h - 1), every lag counts in full (``overlap_fit``);
    without, the weights are Newey-West's, which decline with the lag. Returns
    NaN with fewer than three pairs.
    """
    paired = pd.concat([predicted, realised], axis=1).dropna()
    if len(paired) < 3:
        return float("nan")
    ranks = paired.rank()
    standardised = (ranks - ranks.mean()) / ranks.std(ddof=0)
    if not np.isfinite(standardised.to_numpy()).all():
        return float("nan")
    x, y = standardised.iloc[:, 0].to_numpy(), standardised.iloc[:, 1].to_numpy()
    if overlap:
        return float(overlap_fit(y, x, lags=lags).bse[1])
    fit = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(fit.bse[1])
