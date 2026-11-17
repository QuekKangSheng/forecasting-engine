"""The bond calendar model: a degree-1 polynomial in what kind of days a label window
holds, month-ends above all, fitted as the derived polynomial's shrinkage candidate.

The Models page's bond research row, "Polynomial – calendar inputs (research)". It
replaced the carry model (``models/carry_research``) on 17 Nov 2026, under a comparison
fixed in advance (``s15-bond-grids-honest``, part A): it passes the gate in both decades
at h = 1 and 5, carry at no horizon. Its inputs were fixed from published research
before any result was looked at, but its form was then picked from 98 runs whose results
over Oct 2008 - Nov 2026 were looked at (the November 2026 research, ``s14-bond-short``),
so its scores there overstate it. See docs/methodology.md.

Why. Treasury returns are higher on the last trading day of a month, when index trackers
extend duration to match their index's month-end rebalance (Hartley & Schwarz,
"Predictable end-of-month Treasury returns"). That was chosen from the literature, sign
included, before looking. On the data from 31 Aug 2006 the bond index's mean 1-day
return is +7.4 bp on a month's last trading day against +0.9 bp on other days, higher in
each of the 21 calendar years, and the effect is gone when the flag is moved to other
days of the month. Which days a label window holds is known in advance, since trading
calendars are published years ahead, so these inputs are not lagged like signals. The
other a-priori idea, jobs Fridays (Savor & Wilson 2013), went the wrong way; it is kept
because the rule fixed the columns before the runs, and the fit decides its slope.

What. Per row ``t``, the columns count the label window's trading days ``t+1 .. t+h``
that are each kind of day (``label_window_calendar``): ``CALENDAR_INPUTS`` always, and at
``h = 1`` the weekday too (``WEEKDAY_INPUTS``). The trading days are the target's own
calendar, the dates it has a price; past its last date, the label windows of the latest
rows run into days the data does not hold yet, which are taken from ``HOLIDAY_CALENDAR``
(``coming_sessions``), so those rows get a forecast too. Then, per fold, exactly as the
derived polynomial's shrinkage candidate: the columns are screened on the fold's
training rows (``harness.evaluate(screen=True)``) and the kept ones fitted by
``ShrinkagePolynomial``, on the page's own splitter (the latest ``TRAIN_WINDOW`` rows,
the same test windows and embargo as every other model).

It carries information at ``h = 1`` and ``h = 5`` only: every 10- or 20-day window holds
about the same number of month-ends, so at those horizons the columns barely vary.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from forecasting_engine.ingest.align import FeaturePanel, align_and_lag
from forecasting_engine.models.base import ModelDescription
from forecasting_engine.models.polynomial import PolynomialConfigError
from forecasting_engine.models.shrinkage import ShrinkagePolynomial
from forecasting_engine.models.sign_ruled import HOLDOUT_BEFORE
from forecasting_engine.reporting.model_metrics import ModelRunResult
from forecasting_engine.validation.harness import evaluate, holdout, summarize
from forecasting_engine.validation.splitters import PurgedWalkForward

CALENDAR_INPUTS: tuple[str, ...] = ("CAL_MONTH_END", "CAL_MONTH_START", "CAL_JOBS_FRIDAY")
"""Per row, how many of the label window's trading days are a month's last trading day,
a month's first, and a Friday in a month's first seven days (when the US jobs report is
usually published). Fixed by the research before any run, in this order, which the
shrinkage sampler's draws follow."""

WEEKDAY_INPUTS: tuple[str, ...] = ("CAL_MON", "CAL_TUE", "CAL_WED", "CAL_THU")
"""At ``h = 1`` only: whether the next trading day is a Monday, Tuesday, Wednesday or
Thursday (Friday is the left-out day). Over a longer window every weekday appears about
equally often."""

HOLIDAY_CALENDAR: str = "SIFMA_US"
"""The ``pandas_market_calendars`` calendar the coming trading days are read from, past
the data's last date: the US bond market's, which closes on Columbus Day and Veterans Day
when the NYSE trades and on Good Friday like it. Of the calendars to hand it is the one
the bond index's own dates match best (docs/methodology.md, §9.3)."""

VERSION: int = 1
"""Raise when a change above alters a fit, so results saved under the old one refit."""


class CalendarDataError(PolynomialConfigError):
    """The message is written for a portfolio manager, like the other models' errors."""


# ── The columns ──────────────────────────────────────────────────────────────


def inputs_for(horizon: int) -> tuple[str, ...]:
    """The columns the model is offered at ``horizon``."""
    return CALENDAR_INPUTS + (WEEKDAY_INPUTS if horizon == 1 else ())


def coming_sessions(after: pd.Timestamp, count: int) -> pd.DatetimeIndex:
    """The first ``count`` trading days of ``HOLIDAY_CALENDAR`` after ``after``."""
    start = pd.Timestamp(after).normalize() + pd.Timedelta(days=1)
    # Two calendar days a session, plus a fortnight, covers any run of holidays.
    end = start + pd.Timedelta(days=2 * count + 14)
    sessions = mcal.get_calendar(HOLIDAY_CALENDAR).valid_days(start_date=start, end_date=end)
    found = pd.DatetimeIndex(sessions.tz_localize(None)[:count])
    if len(found) < count:
        raise CalendarDataError(
            f"the {HOLIDAY_CALENDAR} calendar has only {len(found)} trading days in the "
            f"{(end - start).days} days after {after:%d %b %Y}; the calendar model needs "
            f"{count}."
        )
    return found


def label_window_calendar(index: pd.DatetimeIndex, horizon: int) -> pd.DataFrame:
    """Per row ``t`` of ``index``, how many of its label window's trading days
    ``t+1 .. t+h`` are each kind of day in ``inputs_for(horizon)``.

    ``index`` is the target's own calendar, and gives the trading days as far as it
    runs. The latest rows' windows run past its end, so ``h + 1`` coming trading days
    are taken from ``HOLIDAY_CALENDAR``: ``h`` to fill the last row's window, and one
    more to tell whether the last of them ends its month. A day's month-end flag reads
    the next trading day and its month-start flag the one before; the first row's
    month-start flag reads no day before it, but no row's window holds the first row."""
    index = pd.DatetimeIndex(index)
    if index.empty:
        return pd.DataFrame(index=index, columns=list(inputs_for(horizon)), dtype=float)
    days = index.append(coming_sessions(index[-1], horizon + 1))
    month = np.asarray(days.year * 12 + days.month)
    following = np.r_[month[1:], -1]  # the last day's is never read: no window reaches it
    previous = np.r_[-1, month[:-1]]
    flags = {
        "CAL_MONTH_END": month != following,
        "CAL_MONTH_START": month != previous,
        "CAL_JOBS_FRIDAY": (days.dayofweek == 4) & (days.day <= 7),
    }
    if horizon == 1:
        flags |= {name: days.dayofweek == d for d, name in enumerate(WEEKDAY_INPUTS)}
    per_day = pd.DataFrame(flags, index=days).astype(float)
    window = per_day.rolling(horizon).sum().shift(-horizon)
    return window.iloc[: len(index)].set_axis(index)[list(inputs_for(horizon))]


def calendar_panel(frame: pd.DataFrame, price_col: str, horizon: int) -> FeaturePanel:
    """The target's panel with the calendar columns as its signals.

    The target, its calendar and its label are ``align_and_lag``'s; the columns describe
    the label window, which is known in advance, so they are not lagged."""
    base = align_and_lag(frame, [], price_col, horizon=horizon)
    columns = label_window_calendar(base.frame.index, horizon)
    return replace(
        base, frame=pd.concat([columns, base.frame], axis=1), signals=tuple(columns.columns)
    )


# ── The run ──────────────────────────────────────────────────────────────────


def run_calendar_research_polynomial(
    frame: pd.DataFrame,
    price_col: str,
    horizon: int,
    splitter: PurgedWalkForward,
) -> tuple[ModelRunResult, ModelDescription, FeaturePanel]:
    """Walk the calendar model forward on ``splitter``, its own train window included.

    The columns are screened per fold and the kept ones fitted by
    ``ShrinkagePolynomial``. One configuration. Its result carries its pre-2016 score
    (``holdout``), which for this model is not out of sample: its research looked at
    those years too. Returns the panel too, for the equation's columns."""
    panel = calendar_panel(frame, price_col, horizon)
    folds = evaluate(ShrinkagePolynomial, panel, splitter, screen=True)
    if not folds:
        raise PolynomialConfigError(splitter.too_short(panel))
    result, description = summarize(folds)
    return replace(result, holdout=holdout(panel, folds, HOLDOUT_BEFORE)), description, panel
