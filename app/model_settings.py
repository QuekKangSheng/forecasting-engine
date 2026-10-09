"""The Models page's settings and their defaults.

Kept out of the page script so they can be imported — a page filename starting
with a digit isn't a valid Python module name — and checked against
docs/methodology.md.
"""

from __future__ import annotations

from forecasting_engine.extraction.targets import TargetRole

#: The forecast horizons the validation framework asks for, reported separately
#: and never averaged. Capped at about a month (20 trading days) for now.
HORIZONS: tuple[int, ...] = (1, 5, 10, 20)

#: One embargo shared by every horizon, equal to the longest of them, rather than
#: one per horizon. Picking h=1 used to drop the embargo to 1 as well.
EMBARGO_DAYS: int = max(HORIZONS)

#: Working default, not sponsor-confirmed. On the Sep 2026 Bloomberg data, at the
#: default windows, the equity signal held up best over 20 days (Signal Rank IC
#: +0.075; +0.034 at 5 days, about zero at 10), and the optimiser rebalances every
#: horizon days, so a 20-day forecast is held for its own 20 days.
DEFAULT_HORIZON: int = 20
#: Every model's train window but one, fixed rather than a setting: trying windows
#: until one scores well is overfitting no single run's PBO can see. About a year.
#: On the Sep 2026 data two years scored no better for bond (Signal Rank IC +0.05
#: against +0.07, failing PBO) and leaves machine learning no room to tune inside
#: its 504-day tuning period.
TRAIN_WINDOW: int = 252

#: The targets whose derived polynomial trains on every earlier date (an expanding
#: window) instead. On the Sep 2026 data an expanding window lifted every equity
#: model's Signal Rank IC (lasso +0.08 to +0.11, shrinkage +0.11 to +0.14) and
#: lowered every bond model's (lasso +0.07 to -0.04).
EXPANDING_DERIVED: frozenset[TargetRole] = frozenset({TargetRole.EQUITY})

DEFAULT_TEST_WINDOW: int = 20
DEFAULT_MAX_TERMS: int = 10
