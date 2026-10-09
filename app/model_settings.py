"""The Models page's settings and their defaults.

Kept out of the page script so they can be imported — a page filename starting
with a digit isn't a valid Python module name — and checked against
docs/methodology.md.
"""

from __future__ import annotations

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
#: Working default, not sponsor-confirmed: about a year. Shorter windows' fold
#: means and signal slopes swung the most on the Sep 2026 data.
DEFAULT_TRAIN_WINDOW: int = 252
DEFAULT_TEST_WINDOW: int = 20
