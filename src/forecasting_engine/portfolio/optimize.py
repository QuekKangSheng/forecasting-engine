"""Turn the active equity and bond models' forecasts into a mean-variance weight
schedule — the input ``portfolio.backtest.run_backtest`` chains through history.

Each active model's forecast was captured fold by fold, one walk-forward test
window at a time (``store.active_model``); a rebalance happens once per test
window, not once per day, so the dates here are each fold's first date, not
every date a forecast exists for.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.ingest.align import FeaturePanel
from forecasting_engine.portfolio.backtest import ASSETS
from forecasting_engine.validation.splitters import TUNING_ROWS, PurgedWalkForward


class OptimizeDataError(ValueError):
    """The message is written for a portfolio manager, like the other models' errors."""


DEFAULT_RISK_AVERSION: float = 4.0
"""Working default, not sponsor-confirmed: a moderately risk-averse investor, in
line with standard CAPM-style calibrations. Revisit once Alpha Norm states a
risk-aversion preference."""

DEFAULT_WEIGHT_BOUNDS: tuple[float, float] = (0.2, 0.8)
"""Working default, not sponsor-confirmed: stops either asset being pushed to a
corner by a small, noisy difference between the two forecasts. Revisit once
Alpha Norm states how far the allocation may tilt from 50/50."""


def _common_price_calendar(prices: Mapping[TargetRole, pd.Series]) -> pd.DatetimeIndex:
    calendars = [prices[role].dropna().index for role in ASSETS]
    common = calendars[0]
    for other in calendars[1:]:
        common = common.intersection(other)
    return common.sort_values()


def common_rebalance_dates(
    prices: Mapping[TargetRole, pd.Series],
    *,
    horizon: int,
    train_window: int,
    test_window: int,
    embargo: int,
) -> pd.DatetimeIndex:
    """One date per walk-forward test window, on the calendar both targets share.

    Equity's and bond's own active models were each walk-forward split over
    their *own* calendar, so if one target's price history starts even a few
    days before the other's, their independent "every test_window-th day"
    counts drift out of phase and rarely land on the same date twice. Splitting
    one shared calendar instead keeps both targets' rebalance cadence in sync
    throughout, the same way their settings are already required to match.
    """
    calendar = _common_price_calendar(prices)
    panel = FeaturePanel(
        frame=pd.DataFrame(index=calendar), signals=(), targets=(), horizon=horizon
    )
    splitter = PurgedWalkForward(
        train=train_window, test=test_window, embargo=embargo, tuning_rows=TUNING_ROWS
    )
    return pd.DatetimeIndex([test[0] for _, test in splitter.split(panel) if len(test)])


def expected_returns(
    rebalance_dates: pd.DatetimeIndex, forecasts: Mapping[TargetRole, pd.Series]
) -> pd.DataFrame:
    """Each target's own forecast as of each rebalance date — the mean-variance
    solver's expected-return input, one row per rebalance, a column per asset.

    Reads the most recent forecast at or before the rebalance date rather than
    requiring an exact match: equity's and bond's own saved forecasts can fall
    on slightly different calendars, and a model's most recent view as of that
    date is what it would actually be using if asked that day — never a date
    after it, so this stays as leakage-free as an exact match would be.
    """
    return pd.DataFrame(
        {
            role: forecasts[role].sort_index().reindex(rebalance_dates, method="ffill")
            for role in ASSETS
        }
    )


def training_window(
    rebalance_date: pd.Timestamp,
    calendar: pd.DatetimeIndex,
    *,
    horizon: int,
    train_window: int,
    embargo: int,
) -> pd.DatetimeIndex:
    """The ``train_window`` dates on ``calendar`` ending ``embargo`` days before
    ``rebalance_date`` — the same purge gap a forecasting model itself trains
    under, so covariance never reaches into days the model couldn't see either.
    """
    if rebalance_date not in calendar:
        raise OptimizeDataError(
            f"{rebalance_date:%d %b %Y} is not a trading day on the shared equity/bond calendar."
        )
    panel = FeaturePanel(
        frame=pd.DataFrame(index=calendar), signals=(), targets=(), horizon=horizon
    )
    splitter = PurgedWalkForward(train=train_window, test=0, embargo=embargo)
    position = calendar.get_loc(rebalance_date)
    return splitter.window_before(panel, position, train_window)


def covariance_at_rebalance(
    rebalance_date: pd.Timestamp,
    prices: Mapping[TargetRole, pd.Series],
    *,
    horizon: int,
    train_window: int,
    embargo: int,
) -> pd.DataFrame:
    """Equity/bond daily-return covariance over the training window ending before
    ``rebalance_date``, scaled from daily to ``horizon``-day terms."""
    calendar = _common_price_calendar(prices)
    train_dates = training_window(
        rebalance_date, calendar, horizon=horizon, train_window=train_window, embargo=embargo
    )
    returns = pd.DataFrame(
        {role: prices[role].reindex(train_dates).pct_change() for role in ASSETS}
    ).dropna()
    if len(returns) < 2:
        raise OptimizeDataError(
            f"Not enough trading days before {rebalance_date:%d %b %Y} to estimate a "
            f"covariance (need at least 2 daily returns, got {len(returns)})."
        )
    return returns.cov() * horizon


def solve_weights(
    expected_return: pd.Series,
    covariance: pd.DataFrame,
    *,
    risk_aversion: float = DEFAULT_RISK_AVERSION,
    bounds: tuple[float, float] = DEFAULT_WEIGHT_BOUNDS,
) -> pd.Series:
    """The long-only two-asset mean-variance weights maximising
    ``w @ expected_return - risk_aversion / 2 * w @ covariance @ w``, subject to
    the two weights summing to 1 and each lying within ``bounds``.

    Two assets collapse that constrained problem to one free variable — equity's
    weight, bond being its complement — which has a closed form: maximising
    over the free variable gives the unconstrained optimum below, then clipped
    to ``bounds``. When the two assets' returns are so alike that splitting
    between them changes no risk at all (equal variance, perfect correlation),
    there is nothing for the covariance term to decide between them, and the
    midpoint of ``bounds`` is used rather than dividing by zero.
    """
    equity, bond = ASSETS
    mu_diff = expected_return[equity] - expected_return[bond]
    var_ee, var_bb = covariance.loc[equity, equity], covariance.loc[bond, bond]
    cov_eb = covariance.loc[equity, bond]
    var_diff = var_ee - 2 * cov_eb + var_bb
    if var_diff <= 0:
        w_equity = sum(bounds) / 2
    else:
        w_equity = (mu_diff / risk_aversion - (cov_eb - var_bb)) / var_diff
    w_equity = min(max(w_equity, bounds[0]), bounds[1])
    return pd.Series({equity: w_equity, bond: 1 - w_equity})


def weight_schedule(
    forecasts: Mapping[TargetRole, pd.Series],
    prices: Mapping[TargetRole, pd.Series],
    *,
    horizon: int,
    train_window: int,
    test_window: int,
    embargo: int,
    risk_aversion: float = DEFAULT_RISK_AVERSION,
    bounds: tuple[float, float] = DEFAULT_WEIGHT_BOUNDS,
) -> pd.DataFrame:
    """The full weight schedule ``portfolio.backtest.run_backtest`` chains
    through history: one row of equity/bond weights per rebalance date, solved
    from each active model's forecast and the equity/bond covariance over that
    rebalance's own training window.

    A rebalance whose expected return is missing (a model had no prediction
    that day) or whose training window can't support a covariance estimate is
    left out rather than guessed at.
    """
    rebalance_dates = common_rebalance_dates(
        prices, horizon=horizon, train_window=train_window, test_window=test_window, embargo=embargo
    )
    returns = expected_returns(rebalance_dates, forecasts)
    rows: dict[pd.Timestamp, pd.Series] = {}
    for date in rebalance_dates:
        row = returns.loc[date]
        if row.isna().any():
            continue
        try:
            covariance = covariance_at_rebalance(
                date, prices, horizon=horizon, train_window=train_window, embargo=embargo
            )
        except OptimizeDataError:
            continue
        rows[date] = solve_weights(row, covariance, risk_aversion=risk_aversion, bounds=bounds)
    return pd.DataFrame(rows).T.sort_index()
