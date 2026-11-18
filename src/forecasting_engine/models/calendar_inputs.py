"""Calendar counts over a label window: what kind of days the next ``h`` trading days are.

Shared by the models that read them (the bond derived polynomial, ``models/hybrid``),
and written so the bond calendar research model can import them too.

Why. Treasury returns are higher on the last trading day of a month, when index trackers
extend duration to match their index's month-end rebalance (Hartley & Schwarz,
"Predictable end-of-month Treasury returns"). On the data from 31 Aug 2006 the bond
index's mean 1-day return is +7.4 bp on a month's last trading day against +0.9 bp on
other days, and on public data for Dec 1986 – Aug 2006 a bond fund's is +9.1 bp against
+2.4 bp (docs/evidence/2026-11-18-bond-holdout.md). Which days a label window holds is
known in advance, since trading calendars are published years ahead, so these columns
are not lagged like signals.

What. Per row ``t``, the columns count the label window's trading days ``t+1 .. t+h``
that are each kind of day (``label_window_calendar``): ``CALENDAR_INPUTS`` always, and at
``h = 1`` the weekday too (``WEEKDAY_INPUTS``). The trading days are the target's own
calendar, the dates it has a price; past its last date, the label windows of the latest
rows run into days the data does not hold yet, which are taken from ``HOLIDAY_CALENDAR``
(``coming_sessions``), so those rows get a value, and a forecast, too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from forecasting_engine.models.polynomial import PolynomialConfigError

CALENDAR_INPUTS: tuple[str, ...] = ("CAL_MONTH_END", "CAL_MONTH_START", "CAL_JOBS_FRIDAY")
"""Per row, how many of the label window's trading days are a month's last trading day,
a month's first, and a Friday in a month's first seven days (when the US jobs report is
usually published). Fixed by the research before any run, in this order."""

WEEKDAY_INPUTS: tuple[str, ...] = ("CAL_MON", "CAL_TUE", "CAL_WED", "CAL_THU")
"""At ``h = 1`` only: whether the next trading day is a Monday, Tuesday, Wednesday or
Thursday (Friday is the left-out day). Over a longer window every weekday appears about
equally often."""

CALENDAR_NAMES: frozenset[str] = frozenset(CALENDAR_INPUTS + WEEKDAY_INPUTS)
"""Every calendar column's name: no committed column may take one."""

HOLIDAY_CALENDAR: str = "SIFMA_US"
"""The ``pandas_market_calendars`` calendar the coming trading days are read from, past
the data's last date: the US bond market's, which closes on Columbus Day and Veterans Day
when the NYSE trades and on Good Friday like it."""


class CalendarDataError(PolynomialConfigError):
    """The message is written for a portfolio manager, like the models' errors."""


def inputs_for(horizon: int) -> tuple[str, ...]:
    """The calendar columns at ``horizon``."""
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
            f"{(end - start).days} days after {after:%d %b %Y}; the calendar counts need "
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
