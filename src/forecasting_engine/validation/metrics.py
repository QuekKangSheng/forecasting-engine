"""Prediction-quality metrics shared across screening and validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


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


def rank_ic_se(predicted: pd.Series, realised: pd.Series, lags: int) -> float:
    """Newey-West standard error of ``rank_ic(predicted, realised)``.

    Regresses standardised ranks of the realised values on standardised ranks
    of the predictions, whose slope is the rank IC, with a HAC covariance over
    ``lags`` lags: overlapping h-day labels make neighbouring errors
    correlated, so ``lags`` is h - 1. Returns NaN with fewer than three pairs.
    """
    paired = pd.concat([predicted, realised], axis=1).dropna()
    if len(paired) < 3:
        return float("nan")
    ranks = paired.rank()
    standardised = (ranks - ranks.mean()) / ranks.std(ddof=0)
    if not np.isfinite(standardised.to_numpy()).all():
        return float("nan")
    x = sm.add_constant(standardised.iloc[:, 0].to_numpy())
    fit = sm.OLS(standardised.iloc[:, 1].to_numpy(), x).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags}
    )
    return float(fit.bse[1])
