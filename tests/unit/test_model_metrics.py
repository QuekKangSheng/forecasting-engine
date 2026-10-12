import math
from dataclasses import replace

from forecasting_engine.reporting.model_metrics import (
    Cell,
    ConstantForecasts,
    ModelRunResult,
    build_metrics_rows,
)
from forecasting_engine.validation.crash import CrashDiagnostics

GATED = ("Polynomial", "Machine Learning", "Polynomial (derived)", "FF5 Benchmark")


def _result(
    ic, signal_rank_ic, rmse, se=None, recall=0.6, precision=0.5, f1=0.5455, n_events=5
):
    """A result whose Signal Rank IC and plain OOS Rank IC are both ``signal_rank_ic``,
    with both of its standard errors ``se``."""
    return ModelRunResult(
        ic=ic,
        oos_rank_ic=signal_rank_ic,
        signal_rank_ic=signal_rank_ic,
        signal_rank_ic_se=se,
        signal_rank_ic_se_test=se,
        rmse=rmse,
        crash=CrashDiagnostics(
            recall=recall, precision=precision, f1=f1, n_true_tail_days=n_events
        ),
    )


def test_only_models_that_ran_get_a_row():
    assert build_metrics_rows({}) == []
    rows = build_metrics_rows({"Machine Learning": _result(0.03, 0.03, 0.02)})
    assert [row["Model"].text for row in rows] == ["Machine Learning"]


def test_row_order_is_fixed_regardless_of_input_order():
    results = {
        "Machine Learning": _result(0.03, 0.03, 0.02),
        "Polynomial (user-supplied)": _result(0.01, 0.01, 0.02),
        "FF5 Benchmark": _result(0.02, 0.025, 0.02),
        "Polynomial (derived)": _result(0.03, 0.03, 0.015),
    }
    rows = build_metrics_rows(results)
    names = [row["Model"].text for row in rows]
    assert names == [
        "FF5 Benchmark",
        "Polynomial (derived)",
        "Polynomial (user-supplied)",
        "Machine Learning",
    ]


def test_a_model_not_named_as_gated_gets_no_badges():
    row = build_metrics_rows({"Naive (training mean)": _result(0.02, 0.025, 0.02, se=0.01)})[0]

    assert row["Signal Rank IC"].tone == "neutral"
    assert row["t"] == Cell("2.50")


def test_every_gated_model_is_badged_the_same_way_ff5_included():
    results = {
        "Polynomial (derived)": _result(0.03, 0.03, 0.015, se=0.01),
        "FF5 Benchmark": _result(0.02, 0.025, 0.02, se=0.01),
    }
    ff5, derived = build_metrics_rows(results, gated=GATED)

    assert derived["Signal Rank IC"].tone == "success"
    assert derived["t"] == Cell("3.00", "success")
    assert ff5["Signal Rank IC"].tone == "success"
    assert ff5["t"] == Cell("2.50", "success")


def test_success_tone_when_both_gates_pass():
    row = build_metrics_rows({"Polynomial": _result(0.03, 0.03, 0.015, se=0.01)}, gated=GATED)[0]

    assert row["Signal Rank IC"].tone == "success"
    assert row["t"] == Cell("3.00", "success")
    assert row["Crash Recall"] == Cell("0.6000")
    assert row["Crash Precision"] == Cell("0.5000")
    assert row["Crash F1"] == Cell("0.5455")


def test_danger_tone_when_both_gates_miss():
    row = build_metrics_rows(
        {"Machine Learning": _result(0.03, 0.01, 0.02, se=0.01)}, gated=GATED
    )[0]

    assert row["Signal Rank IC"].tone == "danger"
    assert row["t"] == Cell("1.00", "danger")


def test_a_score_above_the_ic_gate_can_still_fail_on_t():
    row = build_metrics_rows({"Polynomial": _result(0.03, 0.03, 0.015, se=0.05)}, gated=GATED)[0]

    assert row["Signal Rank IC"].tone == "success"
    assert row["t"] == Cell("0.60", "danger")


def test_t_is_judged_on_the_larger_standard_error():
    # 0.10 is 2.5 h-1 standard errors (0.04) but under 2 test-window ones (0.06);
    # the larger is the one that allows for a fold's shared errors.
    result = replace(
        _result(0.03, 0.10, 0.02), signal_rank_ic_se=0.04, signal_rank_ic_se_test=0.06
    )
    assert result.signal_t == 0.10 / 0.06
    assert build_metrics_rows({"Polynomial": result}, gated=GATED)[0]["t"] == Cell(
        "1.67", "danger"
    )


def test_no_standard_error_means_no_t_and_the_t_gate_fails():
    result = _result(0.03, 0.10, 0.02)
    assert math.isnan(result.signal_t)
    assert build_metrics_rows({"Polynomial": result}, gated=GATED)[0]["t"] == Cell("—", "danger")


def test_values_rounded_to_requested_decimals():
    row = build_metrics_rows({"Polynomial": _result(0.03456, 0.03, 0.015)}, decimals=2)[0]

    assert row["IC"] == Cell("0.03")


def test_nan_metric_renders_as_an_em_dash():
    row = build_metrics_rows({"Polynomial": _result(0.03, 0.03, 0.015, recall=math.nan)})[0]

    assert row["Crash Recall"] == Cell("—")


def test_rows_scored_is_shown_when_known():
    result = _result(0.03, 0.03, 0.015)
    assert build_metrics_rows({"Polynomial": result})[0]["Rows scored"] == Cell("—")
    counted = replace(result, rows_scored=1234)
    assert build_metrics_rows({"Polynomial": counted})[0]["Rows scored"] == Cell("1,234")


def test_both_rank_ic_standard_errors_are_shown_beside_it_and_labelled():
    with_se = replace(
        _result(0.03, 0.03, 0.015), signal_rank_ic_se=0.012, signal_rank_ic_se_test=0.019
    )
    row = build_metrics_rows({"Polynomial": with_se}, gated=GATED)[0]
    assert row["Signal Rank IC"] == Cell(
        "0.0300 (s.e. (h−1 lags) 0.0120; s.e. (test-window lags) 0.0190)", "success"
    )


def test_the_plain_oos_rank_ic_is_shown_but_never_gated():
    result = replace(_result(0.03, 0.03, 0.015, se=0.01), oos_rank_ic=-0.2)
    row = build_metrics_rows({"Polynomial": result}, gated=GATED)[0]
    assert row["OOS Rank IC"] == Cell("-0.2000")
    assert row["Signal Rank IC"].tone == "success"


def test_a_result_saved_before_the_signal_rank_ic_fails_its_gate_with_a_dash():
    result = replace(_result(0.03, 0.03, 0.015), signal_rank_ic=None)
    row = build_metrics_rows({"Polynomial": result}, gated=GATED)[0]
    assert row["Signal Rank IC"] == Cell("—", "danger")


def test_constant_folds_are_counted_against_all_folds():
    result = replace(
        _result(0.03, 0.10, 0.02), constant_forecasts=ConstantForecasts(folds=119, constant=119)
    )
    assert build_metrics_rows({"Polynomial": result})[0]["Constant folds"] == Cell("119 of 119")


def test_an_older_result_without_constant_folds_shows_a_dash():
    assert build_metrics_rows({"Polynomial": _result(0.03, 0.10, 0.02)})[0][
        "Constant folds"
    ] == Cell("—")


def test_the_table_has_no_pbo_or_within_fold_columns():
    row = build_metrics_rows({"Polynomial": _result(0.03, 0.10, 0.02)})[0]
    assert set(row) == {
        "Model",
        "IC",
        "Signal Rank IC",
        "t",
        "OOS Rank IC",
        "RMSE",
        "Crash Recall",
        "Crash Precision",
        "Crash F1",
        "Rows scored",
        "Constant folds",
    }
