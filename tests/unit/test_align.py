import numpy as np
import pandas as pd
import pytest

from forecasting_engine.ingest.align import (
    MAX_STALENESS,
    PRODUCTION_LAG_DAYS,
    ColumnSource,
    FeaturePanel,
    Transform,
    align_and_lag,
    select_signals,
    transform_for,
)


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


def test_a_day_the_target_market_was_shut_has_no_row():
    panel = align_and_lag(_bond_frame(), ["vix"], "bond", horizon=1)
    assert pd.Timestamp("2024-10-14") not in panel.frame.index
    assert len(panel.frame) == len(_BOND) - 1


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
    assert pd.isna(panel.label_end.loc["2024-10-22"])  # past the end of the data


def test_a_frame_with_no_gaps_is_unchanged_by_the_fix():
    panel = align_and_lag(_frame(), ["signal_a"], "price", horizon=2)
    expected = _frame()["price"].pct_change(2).shift(-2)
    pd.testing.assert_series_equal(
        panel.frame["fwd_return_2d"], expected, check_names=False, check_freq=False
    )


def test_signals_are_lagged_one_row_of_the_target_calendar():
    panel = align_and_lag(_bond_frame(), ["vix"], "bond", horizon=1)
    assert panel.frame.loc["2024-10-15", "vix"] == 20.5  # Fri 11 Oct, the previous bond day


def test_the_change_after_a_target_holiday_spans_the_gap():
    panel = align_and_lag(
        _bond_frame(), ["vix"], "bond", horizon=1, transforms={"vix": Transform.DIFFERENCE}
    )
    # Tue 15 Oct's change is from Fri 11 Oct, across Columbus Day; lagged a row,
    # it is what Wed 16 Oct sees.
    assert panel.frame.loc["2024-10-16", "vix"] == pytest.approx(20.6 - 20.5)


def test_the_target_is_never_filled():
    frame = _bond_frame()
    frame.loc["2024-10-16", "bond"] = None
    panel = align_and_lag(frame, ["vix"], "bond", horizon=1)
    assert pd.Timestamp("2024-10-16") not in panel.frame.index
    assert panel.frame["bond"].notna().all()


def _stale_frame(last_signal_day: int) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=8, freq="D")
    signal = [float(i) if i <= last_signal_day else None for i in range(8)]
    return pd.DataFrame({"s": signal, "price": [100.0 + i for i in range(8)]}, index=idx)


def test_a_signal_is_carried_forward_up_to_the_staleness_limit_then_missing():
    panel = align_and_lag(_stale_frame(2), ["s"], "price", horizon=1)
    lagged = panel.frame["s"]
    # Last observed on row 2; carried to rows 3-5, missing from row 6. Lagged a row.
    assert list(lagged.iloc[1:7]) == [0.0, 1.0, 2.0, 2.0, 2.0, 2.0]
    assert pd.isna(lagged.iloc[7])
    assert panel.alignment["s"].carried_forward == MAX_STALENESS


def test_a_log_return_signal_is_the_log_change_on_the_calendar():
    frame = _frame()
    panel = align_and_lag(
        frame, ["signal_a"], "price", horizon=1, transforms={"signal_a": Transform.LOG_RETURN}
    )
    assert panel.frame["signal_a"].iloc[3] == pytest.approx(np.log(2 / 1))
    assert panel.alignment["signal_a"].transform is Transform.LOG_RETURN


def test_an_exact_column_is_never_carried_forward():
    panel = align_and_lag(_stale_frame(2), ["s"], "price", horizon=1, exact={"s"})
    assert pd.isna(panel.frame["s"].iloc[4])
    assert panel.alignment["s"].carried_forward == 0


def _sources(**fields: str) -> dict[str, ColumnSource]:
    return {
        column: ColumnSource(security=f"{column.split('_')[0]} Index", field=field)
        for column, field in fields.items()
    }


def test_every_field_of_a_target_security_is_excluded_from_signals():
    sources = _sources(
        SPX_TOT="TOT_RETURN_INDEX_GROSS_DVDS",
        SPX_BID="PX_BID",
        SPX_LAST="PX_LAST",
        VIX_LAST="PX_LAST",
    )
    assert select_signals(list(sources), ["SPX_TOT"], sources) == ["VIX_LAST"]


def test_a_security_keeps_one_field_preferring_total_return_then_last_price():
    sources = _sources(
        LF98TRUU_LAST="PX_LAST",
        LF98TRUU_TOT="TOT_RETURN_INDEX_GROSS_DVDS",
        LUACOAS_LAST="PX_LAST",
        LUACOAS_BID="PX_BID",
        USGGBE10_BID="PX_BID",
    )
    assert select_signals(list(sources), [], sources) == ["LF98TRUU_TOT", "LUACOAS_LAST"]


def test_a_column_with_no_known_source_is_kept_as_its_own_signal():
    assert select_signals(["a", "b", "t"], ["t"], {}) == ["a", "b"]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (ColumnSource("SPX Index", "PX_LAST"), Transform.LOG_RETURN),
        (ColumnSource("VIX Index", "PX_LAST"), Transform.DIFFERENCE),
        (ColumnSource("ABC Index", "TOT_RETURN_INDEX_NET_DVDS"), Transform.LOG_RETURN),
        (ColumnSource("ABC Index", "PX_LAST"), Transform.DIFFERENCE),
        (None, Transform.DIFFERENCE),
    ],
)
def test_the_transform_follows_the_ticker_map_with_difference_as_default(source, expected):
    assert transform_for(source) is expected


def test_a_hand_built_panel_has_no_label_end():
    panel = FeaturePanel(frame=_frame(), signals=("signal_a",), targets=("price",), lag_days=1)
    assert panel.label_end is None
