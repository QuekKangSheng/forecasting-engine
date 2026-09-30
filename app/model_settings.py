"""The Models page's settings and their defaults.

Kept out of the page script so they can be imported — a page filename starting
with a digit isn't a valid Python module name — and checked against
docs/methodology.md.
"""

from __future__ import annotations

#: The forecast horizons the validation framework asks for, reported separately
#: and never averaged.
HORIZONS: tuple[int, ...] = (1, 5)

#: One embargo shared by every horizon, equal to the longest of them, rather than
#: one per horizon. Picking h=1 used to drop the embargo to 1 as well.
EMBARGO_DAYS: int = max(HORIZONS)

DEFAULT_HORIZON: int = max(HORIZONS)
DEFAULT_TRAIN_WINDOW: int = 120
DEFAULT_TEST_WINDOW: int = 20
DEFAULT_MAX_TERMS: int = 10
