import pandas as pd

from forecasting_engine.ingest.align import FeaturePanel, align_and_lag
from forecasting_engine.validation.splitters import PurgedWalkForward


def _panel(n: int) -> FeaturePanel:
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    frame = pd.DataFrame({"signal_a": range(n), "fwd_return_1d": range(n)}, index=idx)
    return FeaturePanel(
        frame=frame, signals=("signal_a",), targets=("fwd_return_1d",), lag_days=1
    )


def test_split_yields_expected_fold_sizes():
    panel = _panel(20)
    folds = list(PurgedWalkForward(train=10, test=3, embargo=2).split(panel))
    assert len(folds[0][0]) == 10
    assert len(folds[0][1]) == 3


def test_embargo_gap_excluded_from_both_sides():
    panel = _panel(20)
    train_idx, test_idx = next(PurgedWalkForward(train=10, test=3, embargo=2).split(panel))
    gap = panel.frame.index[10:12]
    assert not gap.isin(train_idx).any()
    assert not gap.isin(test_idx).any()


def test_folds_roll_forward():
    panel = _panel(20)
    folds = list(PurgedWalkForward(train=10, test=3, embargo=2).split(panel))
    assert len(folds) == 2
    assert folds[1][0][0] == panel.frame.index[3]


def test_stops_when_not_enough_data_left():
    panel = _panel(15)
    folds = list(PurgedWalkForward(train=10, test=3, embargo=2).split(panel))
    assert len(folds) == 1


def test_purges_training_rows_whose_label_reaches_into_the_test_window():
    # horizon (5) > embargo (1): without purging, the last few training rows'
    # labels would need prices from inside the test window.
    idx = pd.date_range("2024-01-01", periods=40, freq="D")
    frame = pd.DataFrame({"signal_a": range(40), "fwd_return_5d": range(40)}, index=idx)
    panel = FeaturePanel(
        frame=frame, signals=("signal_a",), targets=("fwd_return_5d",), lag_days=1, horizon=5
    )

    train_idx, test_idx = next(PurgedWalkForward(train=20, test=5, embargo=1).split(panel))

    test_start_pos = frame.index.get_loc(test_idx[0])
    for date in train_idx:
        train_pos = frame.index.get_loc(date)
        assert train_pos + panel.horizon < test_start_pos
    assert len(train_idx) == 16  # 4 rows purged: their labels would reach into the test window


def test_no_purge_needed_when_embargo_already_covers_the_horizon():
    # horizon (1, the default) <= embargo (2): purging changes nothing.
    panel = _panel(20)
    train_idx, _ = next(PurgedWalkForward(train=10, test=3, embargo=2).split(panel))
    assert len(train_idx) == 10


# --- purging by where a label's price actually comes from -------------------


def _gappy_panel() -> FeaturePanel:
    """A merged frame where the target's market is shut on every 4th row."""
    idx = pd.bdate_range("2024-01-01", periods=60)
    price = [100.0 + i if i % 4 != 3 else None for i in range(60)]
    frame = pd.DataFrame({"signal_a": range(60), "price": price}, index=idx)
    return align_and_lag(frame, ["signal_a"], "price", horizon=5)


def test_a_label_crossing_a_closed_day_is_purged_even_when_row_count_says_it_is_safe():
    # This is the leak. The last training row sits 5 rows before the test
    # window, so counting rows says its 5-day label ends in time. But one of
    # those rows is a day the target's market was shut, so the label's price
    # actually comes from inside the test window.
    panel = _gappy_panel()
    train_idx, test_idx = next(PurgedWalkForward(train=30, test=5, embargo=5).split(panel))

    for date in train_idx:
        end = panel.label_end.loc[date]
        assert pd.isna(end) or end < test_idx[0], f"{date.date()} labels from {end.date()}"


def test_no_training_label_ever_reaches_its_test_window_across_every_fold():
    panel = _gappy_panel()
    for train_idx, test_idx in PurgedWalkForward(train=20, test=5, embargo=1).split(panel):
        ends = panel.label_end.reindex(train_idx).dropna()
        assert (ends < test_idx[0]).all()


def test_date_based_purging_is_never_looser_than_counting_rows():
    # The same panel without label_end falls back to counting rows. Dating the
    # purge must only ever remove more training rows, never fewer.
    panel = _gappy_panel()
    rows_only = FeaturePanel(
        frame=panel.frame,
        signals=panel.signals,
        targets=panel.targets,
        lag_days=panel.lag_days,
        horizon=panel.horizon,
    )
    for (dated, _), (counted, _) in zip(
        PurgedWalkForward(train=30, test=5, embargo=1).split(panel),
        PurgedWalkForward(train=30, test=5, embargo=1).split(rows_only),
        strict=True,
    ):
        assert len(dated) <= len(counted)


# --- the tuning period ----------------------------------------------------------


def test_no_test_window_starts_before_the_tuning_period_and_embargo_end():
    panel = _panel(60)
    folds = list(PurgedWalkForward(train=10, test=3, embargo=2, tuning_rows=30).split(panel))

    assert folds[0][1][0] == panel.frame.index[32]
    # Training windows may reach back into the tuning period.
    assert folds[0][0][0] == panel.frame.index[20]


def test_a_tuning_period_shorter_than_the_train_window_changes_nothing():
    panel = _panel(40)
    plain = list(PurgedWalkForward(train=10, test=3, embargo=2).split(panel))
    tuned = list(PurgedWalkForward(train=10, test=3, embargo=2, tuning_rows=5).split(panel))
    assert [t[0] for _, t in plain] == [t[0] for _, t in tuned]


def test_window_before_purges_labels_that_reach_the_test_window():
    panel = _gappy_panel()
    splitter = PurgedWalkForward(train=20, test=5, embargo=1)

    window = splitter.window_before(panel, 40, rows=30)

    assert (panel.label_end.reindex(window).dropna() < panel.frame.index[40]).all()
    assert len(window) <= 30


def test_too_short_says_how_many_dates_the_first_fold_needs():
    message = PurgedWalkForward(train=10, test=3, embargo=2, tuning_rows=30).too_short(_panel(20))
    assert "20 target dates" in message
    assert "needs 35" in message
