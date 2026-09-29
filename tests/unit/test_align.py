import pandas as pd
import pytest

from forecasting_engine.ingest.align import PRODUCTION_LAG_DAYS, FeaturePanel, align_and_lag


def _frame() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    return pd.DataFrame(
        {"signal_a": range(10), "price": [100 + i for i in range(10)]},
        index=idx,
    )


def test_align_and_lag_shifts_signals_forward():
    panel = align_and_lag(_frame(), ["signal_a"], "price", horizon=2)
    assert panel.frame["signal_a"].iloc[2] == 1
    assert pd.isna(panel.frame["signal_a"].iloc[0])


def test_align_and_lag_computes_forward_return_target():
    panel = align_and_lag(_frame(), ["signal_a"], "price", horizon=2)
    assert panel.targets == ("fwd_return_2d",)
    expected = (102 / 100) - 1
    assert panel.frame["fwd_return_2d"].iloc[0] == pytest.approx(expected)
    assert pd.isna(panel.frame["fwd_return_2d"].iloc[-1])


def test_lag_days_is_fixed_at_the_production_lag():
    assert align_and_lag(_frame(), ["signal_a"], "price").lag_days == PRODUCTION_LAG_DAYS
    with pytest.raises(ValueError):
        FeaturePanel(frame=_frame(), signals=("signal_a",), targets=("price",), lag_days=2)


def test_target_cannot_also_be_a_signal():
    with pytest.raises(ValueError):
        FeaturePanel(
            frame=_frame(),
            signals=("fwd_return_2d",),
            targets=("fwd_return_2d",),
            lag_days=1,
        )


# --- the horizon is counted in the TARGET'S trading days, not merged rows ---
#
# A merged frame has a row for every date any series traded. Columbus Day
# (Mon 14 Oct 2024) is a row because the NYSE was open, but the US bond market
# was shut, so the bond target is blank there. pct_change(h) counts rows, so a
# "5-day" label spanning that row was really a 4-day return. These numbers are
# real LBUSTRUU-shaped levels from that week.

_COLUMBUS = pd.DatetimeIndex(
    pd.to_datetime(
        [
            "2024-10-08",
            "2024-10-09",
            "2024-10-10",
            "2024-10-11",
            "2024-10-14",
            "2024-10-15",
            "2024-10-16",
            "2024-10-17",
            "2024-10-18",
            "2024-10-21",
            "2024-10-22",
        ]
    )
)
_BOND = [
    2231.40,
    2229.10,
    2226.85,
    2224.60,
    None,
    2232.90,
    2236.10,
    2233.40,
    2235.80,
    2228.70,
    2229.95,
]


def _bond_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"vix": [21.4, 20.9, 20.9, 20.5, 19.7, 20.6, 19.6, 19.1, 18.0, 18.4, 18.2], "bond": _BOND},
        index=_COLUMBUS,
    )


def test_a_five_day_label_spans_five_trading_days_of_the_target():
    panel = align_and_lag(_bond_frame(), ["vix"], "bond", horizon=5)
    # 8 Oct -> 16 Oct is five bond trading days (9, 10, 11, 15, 16). Counting
    # rows instead would stop at 15 Oct, four trading days on.
    assert panel.frame.loc["2024-10-08", "fwd_return_5d"] == pytest.approx(2236.10 / 2231.40 - 1)


def test_a_day_the_target_market_was_shut_gets_no_label():
    panel = align_and_lag(_bond_frame(), ["vix"], "bond", horizon=1)
    assert pd.isna(panel.frame.loc["2024-10-14", "fwd_return_1d"])


def test_the_real_move_across_a_closed_day_is_kept_not_dropped():
    # Fri 11 -> Tue 15 is one bond trading day. Counting rows lost it entirely,
    # because the row in between was blank.
    panel = align_and_lag(_bond_frame(), ["vix"], "bond", horizon=1)
    assert panel.frame.loc["2024-10-11", "fwd_return_1d"] == pytest.approx(2232.90 / 2224.60 - 1)


def test_each_label_records_the_date_its_price_comes_from():
    panel = align_and_lag(_bond_frame(), ["vix"], "bond", horizon=5)
    assert panel.label_end.loc["2024-10-08"] == pd.Timestamp("2024-10-16")
    assert panel.label_end.loc["2024-10-11"] == pd.Timestamp("2024-10-21")


def test_no_label_end_where_there_is_no_label():
    panel = align_and_lag(_bond_frame(), ["vix"], "bond", horizon=5)
    assert pd.isna(panel.label_end.loc["2024-10-14"])  # market shut
    assert pd.isna(panel.label_end.loc["2024-10-22"])  # past the end of the data


def test_a_frame_with_no_gaps_is_unchanged_by_the_fix():
    panel = align_and_lag(_frame(), ["signal_a"], "price", horizon=2)
    expected = _frame()["price"].pct_change(2).shift(-2)
    pd.testing.assert_series_equal(panel.frame["fwd_return_2d"], expected, check_names=False)


def test_signals_are_still_lagged_by_row():
    panel = align_and_lag(_bond_frame(), ["vix"], "bond", horizon=1)
    assert panel.frame.loc["2024-10-15", "vix"] == 19.7  # the Columbus Day value


def test_a_hand_built_panel_has_no_label_end():
    panel = FeaturePanel(frame=_frame(), signals=("signal_a",), targets=("price",), lag_days=1)
    assert panel.label_end is None
