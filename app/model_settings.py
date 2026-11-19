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

#: Working default, not sponsor-confirmed. On the Bloomberg data from 31 Aug 2006
#: the equity signal is most significant over 5 days: the derived polynomial's Signal
#: Rank IC is +0.083 with t 3.20, its highest t of the four horizons, and the research
#: model and machine learning pass the gate there too. At 20 days, the default until
#: 19 Nov 2026, the derived polynomial scores higher (+0.103) but with t 1.98, and no
#: model passes. The optimiser rebalances every horizon days, so a 5-day forecast is
#: held for its own 5 days.
DEFAULT_HORIZON: int = 5
#: Every model's train window on both targets, fixed rather than a setting: trying
#: windows until one scores well is overfitting no single run can see. About two
#: years. On the Sep 2026 data it scored best for the equity derived polynomial
#: (Signal Rank IC +0.145 against +0.095 at 252 rows), neither expanding variant
#: beat it by a standard error, and machine learning scored higher at 504 on both
#: targets (docs/evidence/2026-10-12-design-verification.md).
TRAIN_WINDOW: int = 504

#: Every model's test window, fixed like the train window.
DEFAULT_TEST_WINDOW: int = 20
DEFAULT_MAX_TERMS: int = 10
