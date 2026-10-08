import math
from dataclasses import replace

from forecasting_engine.reporting.model_metrics import (
    NO_CONFIG_SEARCH,
    Cell,
    ConstantForecasts,
    ModelRunResult,
    build_metrics_rows,
)
from forecasting_engine.validation.crash import CrashDiagnostics


def _result(ic, signal_rank_ic, rmse, pbo, recall=0.6, precision=0.5, f1=0.5455, n_events=5):
    """A result whose Signal Rank IC and plain OOS Rank IC are both ``signal_rank_ic``."""
    return ModelRunResult(
        ic=ic,
        oos_rank_ic=signal_rank_ic,
        signal_rank_ic=signal_rank_ic,
        rmse=rmse,
        pbo=pbo,
        crash=CrashDiagnostics(
            recall=recall, precision=precision, f1=f1, n_true_tail_days=n_events
        ),
    )


def test_only_models_that_ran_get_a_row():
    assert build_metrics_rows({}) == []
    rows = build_metrics_rows({"Machine Learning": _result(0.03, 0.03, 0.02, 0.3)})
    assert [row["Model"].text for row in rows] == ["Machine Learning"]


def test_row_order_is_fixed_regardless_of_input_order():
    results = {
        "Machine Learning": _result(0.03, 0.03, 0.02, 0.3),
        "Polynomial (user-supplied)": _result(0.01, 0.01, 0.02, None),
        "FF5 Benchmark": _result(0.02, 0.025, 0.02, None),
        "Polynomial (derived)": _result(0.03, 0.03, 0.015, 0.4),
    }
    rows = build_metrics_rows(results)
    names = [row["Model"].text for row in rows]
    assert names == [
        "FF5 Benchmark",
        "Polynomial (derived)",
        "Polynomial (user-supplied)",
        "Machine Learning",
    ]


def test_ff5_gets_no_gate_badge_and_na_pbo():
    results = {"FF5 Benchmark": _result(0.02, 0.025, 0.02, None)}
    row = build_metrics_rows(results)[0]

    assert row["Signal Rank IC"] == Cell("0.0250")
    assert row["PBO"] == Cell(NO_CONFIG_SEARCH)


def test_polynomial_gets_success_tone_when_both_gates_pass():
    results = {"Polynomial": _result(0.03, 0.03, 0.015, 0.4)}
    row = build_metrics_rows(results)[0]

    assert row["Signal Rank IC"] == Cell("0.0300", "success")
    assert row["PBO"] == Cell("0.4000", "success")
    assert row["Crash Recall"] == Cell("0.6000")
    assert row["Crash Precision"] == Cell("0.5000")
    assert row["Crash F1"] == Cell("0.5455")


def test_ml_gets_danger_tone_when_both_gates_miss():
    results = {"Machine Learning": _result(0.03, 0.01, 0.02, 0.6)}
    row = build_metrics_rows(results)[0]

    assert row["Signal Rank IC"] == Cell("0.0100", "danger")
    assert row["PBO"] == Cell("0.6000", "danger")


def test_values_rounded_to_requested_decimals():
    results = {"Polynomial": _result(0.03456, 0.03, 0.015, 0.4)}
    row = build_metrics_rows(results, decimals=2)[0]

    assert row["IC"] == Cell("0.03")


def test_nan_metric_renders_as_an_em_dash():
    results = {"Polynomial": _result(0.03, 0.03, 0.015, 0.4, recall=math.nan)}
    row = build_metrics_rows(results)[0]

    assert row["Crash Recall"] == Cell("—")


def test_rows_scored_is_shown_when_known():
    result = _result(0.03, 0.03, 0.015, 0.4)
    assert build_metrics_rows({"Polynomial": result})[0]["Rows scored"] == Cell("—")
    counted = ModelRunResult(**{**result.__dict__, "rows_scored": 1234})
    assert build_metrics_rows({"Polynomial": counted})[0]["Rows scored"] == Cell("1,234")


def test_both_rank_ic_standard_errors_are_shown_beside_it_and_labelled():
    result = _result(0.03, 0.03, 0.015, 0.4)
    with_se = ModelRunResult(
        **{**result.__dict__, "signal_rank_ic_se": 0.012, "signal_rank_ic_se_test": 0.019}
    )
    row = build_metrics_rows({"Polynomial": with_se})[0]
    assert row["Signal Rank IC"] == Cell(
        "0.0300 (s.e. (h−1 lags) 0.0120; s.e. (test-window lags) 0.0190)", "success"
    )


def test_the_gate_ignores_the_standard_errors():
    result = _result(0.03, 0.021, 0.015, 0.4)
    wide = replace(result, signal_rank_ic_se=0.5, signal_rank_ic_se_test=0.9)
    assert build_metrics_rows({"Polynomial": wide})[0]["Signal Rank IC"].tone == "success"


def test_the_plain_oos_rank_ic_is_shown_but_never_gated():
    result = replace(_result(0.03, 0.03, 0.015, 0.4), oos_rank_ic=-0.2)
    row = build_metrics_rows({"Polynomial": result})[0]
    assert row["OOS Rank IC"] == Cell("-0.2000")
    assert row["Signal Rank IC"].tone == "success"


def test_a_result_saved_before_the_signal_rank_ic_fails_its_gate_with_a_dash():
    result = replace(_result(0.03, 0.03, 0.015, 0.4), signal_rank_ic=None)
    assert build_metrics_rows({"Polynomial": result})[0]["Signal Rank IC"] == Cell("—", "danger")


# --- diagnostics beside the gate: level-free score, significance, constant folds


def _diagnosed(**fields) -> ModelRunResult:
    return replace(_result(0.03, 0.10, 0.02, 0.3), **fields)


def _row(result: ModelRunResult) -> dict[str, Cell]:
    return build_metrics_rows({"Polynomial": result})[0]


def test_the_within_fold_score_is_shown_with_its_standard_error():
    row = _row(_diagnosed(oos_rank_ic_within=0.3312, oos_rank_ic_within_se=0.0451))
    assert row["Rank IC within folds"] == Cell("0.3312 (s.e. 0.0451)")


def test_a_forecast_with_no_ranking_within_folds_shows_a_dash():
    assert _row(_diagnosed(oos_rank_ic_within=math.nan))["Rank IC within folds"] == Cell("—")


def test_a_score_more_than_two_standard_errors_from_zero_is_marked():
    row = _row(_diagnosed(signal_rank_ic=0.10, signal_rank_ic_se=0.03, signal_rank_ic_se_test=0.04))
    assert row["Beyond 2 s.e."] == Cell("Yes")


def test_significance_is_judged_on_the_larger_standard_error():
    # 0.10 is beyond two of the h-1 s.e. (0.04) but not of the test-window one
    # (0.06); the larger is the one that allows for a fold's shared errors.
    row = _row(_diagnosed(signal_rank_ic=0.10, signal_rank_ic_se=0.04, signal_rank_ic_se_test=0.06))
    assert row["Beyond 2 s.e."] == Cell("No")


def test_significance_counts_a_reliably_negative_score_too():
    row = _row(
        _diagnosed(signal_rank_ic=-0.10, signal_rank_ic_se=0.03, signal_rank_ic_se_test=0.04)
    )
    assert row["Beyond 2 s.e."] == Cell("Yes")


def test_no_standard_error_means_no_verdict():
    assert _row(_diagnosed())["Beyond 2 s.e."] == Cell("—")


def test_significance_never_changes_the_gate():
    # Diagnostic only: an insignificant score above the gate still meets it.
    row = _row(_diagnosed(signal_rank_ic=0.03, signal_rank_ic_se=0.05, signal_rank_ic_se_test=0.05))
    assert row["Beyond 2 s.e."] == Cell("No")
    assert row["Signal Rank IC"].tone == "success"


def test_constant_folds_are_counted_against_all_folds():
    row = _row(_diagnosed(constant_forecasts=ConstantForecasts(folds=119, constant=119)))
    assert row["Constant folds"] == Cell("119 of 119")


def test_an_older_result_without_the_diagnostics_shows_dashes():
    row = _row(_result(0.03, 0.10, 0.02, 0.3))
    assert (
        row["Rank IC within folds"],
        row["Beyond 2 s.e."],
        row["Constant folds"],
    ) == (Cell("—"), Cell("—"), Cell("—"))
