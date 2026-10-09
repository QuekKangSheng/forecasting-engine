"""PurgedWalkForward: the only splitter in the codebase.

Rolls a train/test window forward across a FeaturePanel's index, dropping an
embargo gap between train and test so overlapping forward-return labels can't
leak across the split. The train window is either a fixed number of rows or,
with ``train=None``, every row before the embargo (an expanding window).
"""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd

from forecasting_engine.ingest.align import FeaturePanel

#: Rows at the start of the target calendar reserved for tuning. No test window
#: starts before they end (plus the embargo), so tuning never touches a score.
TUNING_ROWS: int = 504


class PurgedWalkForward:
    def __init__(self, train: int | None, test: int, embargo: int, tuning_rows: int = 0):
        """``train=None`` trains each fold on every row before its embargo."""
        self.train = train
        self.test = test
        self.embargo = embargo
        self.tuning_rows = tuning_rows

    def split(
        self, panel: FeaturePanel
    ) -> Iterator[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        index = panel.frame.index
        test_start = max(self.train or 0, self.tuning_rows) + self.embargo
        while test_start + self.test <= len(index):
            yield (
                self.window_before(panel, test_start, self.train),
                index[test_start : test_start + self.test],
            )
            test_start += self.test

    def window_before(
        self, panel: FeaturePanel, test_start: int, rows: int | None
    ) -> pd.DatetimeIndex:
        """Up to ``rows`` rows (every row, if ``None``) ending ``embargo`` rows before
        position ``test_start``, less any whose label reaches the test window.

        A row's label looks ``panel.horizon`` days past its own date. If that
        reaches the test window, the label needs a price the model isn't
        supposed to see yet — those rows are purged regardless of how large
        ``embargo`` is, rather than trusting the caller to have picked
        embargo >= horizon. Training windows and tuning windows both use this.
        """
        index = panel.frame.index
        natural_end = test_start - self.embargo
        start = 0 if rows is None else max(0, natural_end - rows)
        purge_boundary = min(natural_end, test_start - panel.horizon)
        if panel.label_end is not None:
            reaching = _first_label_reaching(
                panel.label_end, index, start, natural_end, index[test_start]
            )
            purge_boundary = min(purge_boundary, reaching)
        return index[start : max(start, purge_boundary)]

    def too_short(self, panel: FeaturePanel) -> str:
        """Why ``split`` produced no folds, for a portfolio manager."""
        needed = max(self.train or 0, self.tuning_rows) + self.embargo + self.test
        if self.train is None:
            before = f"a {self.tuning_rows}-row tuning period (training uses every earlier row)"
        else:
            before = (
                f"the longer of a {self.tuning_rows}-row tuning period and a "
                f"{self.train}-row train window"
            )
        return (
            f"the committed data has {len(panel.frame):,} target dates, too few for one "
            f"walk-forward fold: the first test window needs {needed:,} ({before}, a "
            f"{self.embargo}-row embargo, then {self.test} test rows)."
        )


def _first_label_reaching(
    label_end: pd.Series,
    index: pd.DatetimeIndex,
    start: int,
    stop: int,
    test_start_date: pd.Timestamp,
) -> int:
    """Position of the first row in ``[start, stop)`` whose label's price is dated
    on or after the test window opens, or ``stop`` if none is.

    Counting ``horizon`` rows back from the test window is not enough on a merged
    frame: a day the target's market was shut is a row but not a trading day, so
    a label crossing it reaches further than ``horizon`` rows. ``label_end`` says
    where each label really ends. Label ends only increase with row order, so
    everything from the first reaching row onward is purged, and the rows before
    it are safe.
    """
    reaching = (label_end.reindex(index[start:stop]) >= test_start_date).to_numpy()
    return start + int(reaching.argmax()) if reaching.any() else stop
