"""The downloadable run report: what it says, and how it says what it doesn't have."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.extraction.validation import ValidationReport
from forecasting_engine.portfolio.backtest import run_backtest
from forecasting_engine.reporting.portfolio_comparison import NOT_RUN
from forecasting_engine.reporting.run_report import (
    line_chart_svg,
    report_html,
    thinned,
    weights_explanation,
)
from forecasting_engine.reporting.run_results import data_quality_summary, run_results
from forecasting_engine.store.portfolio_runs import SavedRun

EQUITY, BOND = TargetRole.EQUITY, TargetRole.BOND


def _prices():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2024-01-31", periods=300)
    return {
        EQUITY: pd.Series(100 * np.cumprod(1 + rng.normal(0.001, 0.01, 300)), index=idx),
        BOND: pd.Series(100 * np.cumprod(1 + rng.normal(0.0002, 0.003, 300)), index=idx),
    }


PRICES = _prices()
INDEX = PRICES[EQUITY].index
DATES = INDEX[::20]
SCHEDULE = pd.DataFrame({EQUITY: [0.7] * len(DATES), BOND: [0.3] * len(DATES)}, index=DATES)
BACKTEST = run_backtest(PRICES, SCHEDULE, active_models=("S&P 500: Polynomial",))
SETTINGS = {
    "risk_level": 3,
    "bounds": [0.2, 0.8],
    "costs_bps": {"equity": 3.0, "bond": 5.0},
    "horizon": 20,
    "train_window": 504,
    "test_window": 20,
    "embargo": 20,
    "rebalance_frequency": "monthly",
}
NAMES = {"equity": "S&P 500", "bond": "US Aggregate"}
MODEL = {
    "name": "S&P 500",
    "model": "Polynomial",
    "set_at": "2026-10-20T09:00:00",
    "high_risk": False,
    "signal_rank_ic": 0.05,
    "signal_t": 2.5,
    "gate_met": True,
    "gates_failed": [],
    "origin": "Derived Function",
    "formula": None,
    "terms": [{"Factor": "VIX", "Exponent": "1", "Coefficient": "0.0123"}],
    "standardisation": [],
    "attribution": {"VIX": 0.004},
}
QUALITY = data_quality_summary(
    ValidationReport(0, (), 2, ("2024-01-06",), {"VIX": 1.5, "SPX": 0.0}, pd.DataFrame(), []),
    pd.DataFrame({"Date": INDEX, "SPX": 1.0, "VIX": 2.0}),
    ["EMPTY FIELD"],
)


def _results(**overrides):
    return run_results(
        BACKTEST,
        BACKTEST,
        names={EQUITY: "S&P 500", BOND: "US Aggregate"},
        settings=SETTINGS,
        latest_date=INDEX[-1],
        latest_weights={EQUITY: 0.7, BOND: 0.3},
        how_set=["How this was set: level 3 targets 60% S&P 500."],
        windows={"rolling_volatility": 63, "rolling_sharpe": 252},
        **{"data_quality": QUALITY, "models": [MODEL], **overrides},
    )


def _run(name=None, code="abc1234", results=None):
    return SavedRun(
        run_key="k",
        run_number=7,
        name=name,
        recorded_at=datetime(2026, 11, 2, 14, 30),
        dataset_sha256="d" * 64,
        code_version=code,
        results=_results() if results is None else results,
    )


def test_the_report_names_the_run_number_timestamp_and_commit():
    text = report_html(_run(name="Before the cut"), BACKTEST)

    assert "Run 7: Before the cut" in text
    assert "Run number 7" in text
    assert "02 Nov 2026 14:30" in text
    assert "Commit abc1234" in text


def test_a_run_name_cannot_inject_markup_into_the_report():
    text = report_html(_run(name="<script>alert(1)</script>"), BACKTEST)

    assert "<script>" not in text
    assert "&lt;script&gt;" in text


def test_the_report_needs_nothing_from_the_internet():
    text = report_html(_run(), BACKTEST)

    assert "http://" not in text.replace("http://www.w3.org", "") and "https://" not in text
    assert "<script" not in text and "<link" not in text


@pytest.mark.parametrize(
    "heading",
    [
        "Data quality",
        "Active models",
        "Optimiser settings",
        "How the weights were set",
        "Backtest",
        "Spread t-statistics",
        "Rolling volatility, 63-day window",
        "Drawdown",
        "VaR and CVaR",
        "Monte Carlo",
    ],
)
def test_the_report_has_each_section_the_story_asks_for(heading):
    assert heading in report_html(_run(), BACKTEST)


def test_the_data_quality_summary_gives_the_flags_as_they_stand():
    text = report_html(_run(), BACKTEST)

    assert "<td>Weekend rows</td><td>2</td>" in text
    assert "<td>Rows</td><td>300</td>" in text
    assert "<td>VIX</td><td>1.5%</td>" in text
    assert "EMPTY FIELD" in text
    assert "SPX</td><td>0.0%" not in text, "a complete column isn't listed as missing"


def test_a_model_shows_its_gate_results_function_and_shap():
    text = report_html(_run(), BACKTEST)

    assert "Gate met. Signal Rank IC +0.050" in text
    assert "t 2.50" in text
    assert "<td>VIX</td><td>1</td><td>0.0123</td>" in text
    assert "<td>VIX</td><td>0.004000</td>" in text


def test_a_failed_gate_says_which_and_that_the_model_was_set_anyway():
    failed = {
        **MODEL,
        "gate_met": False,
        "gates_failed": ["signal_rank_ic", "t"],
        "high_risk": True,
    }
    text = report_html(_run(results=_results(models=[failed])), BACKTEST)

    assert "Gate failed on Signal Rank IC and t." in text
    assert "Set active despite failing both gates." in text


def test_a_model_that_is_not_a_polynomial_says_so_rather_than_showing_an_empty_table():
    machine = {**MODEL, "model": "Machine Learning", "terms": None, "origin": None}
    text = report_html(_run(results=_results(models=[machine])), BACKTEST)

    assert "not a polynomial" in text


def test_what_an_older_record_did_not_keep_is_said_not_passed_off_as_zero():
    older = {
        **MODEL,
        "signal_t": None,
        "gate_met": None,
        "attribution": None,
    }
    text = report_html(_run(results=_results(models=[older])), BACKTEST)

    assert "Gate result not saved for this model." in text
    assert "t not saved" in text
    assert "Per-signal SHAP: not saved" in text


def test_a_run_saved_before_the_report_existed_still_produces_a_report():
    old = _results()
    del old["data_quality"], old["models"]

    text = report_html(_run(results=old), BACKTEST)

    assert "Not saved for this run." in text
    assert "Backtest" in text


def test_without_a_simulation_monte_carlo_reads_not_run():
    text = report_html(_run(), BACKTEST)

    assert f"<h2>Monte Carlo</h2><p>{NOT_RUN}.</p>" in text


def test_a_missing_commit_is_stated():
    assert "Commit not recorded" in report_html(_run(code=None), BACKTEST)


def test_the_weights_explanation_is_written_from_the_settings():
    text = weights_explanation(SETTINGS, NAMES)

    assert "at least 20% and at most 80%" in text
    assert "Every 20 trading days" in text
    assert "Risk level 3 of 5" in text
    assert "504 trading days" in text
    assert "3 bp on S&P 500 and 5 bp on US Aggregate" in text
    assert "tilt" in text


def test_level_five_is_explained_as_ignoring_the_forecasts():
    text = weights_explanation({**SETTINGS, "risk_level": 5}, NAMES)

    assert "whatever the forecasts say" in text
    assert "tilt" not in text


# --- the charts --------------------------------------------------------------------


def _frame(days=2000):
    index = pd.bdate_range("2018-01-01", periods=days)
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {"A": rng.normal(size=days).cumsum(), "B": rng.normal(size=days).cumsum()}, index=index
    )


def test_thinning_leaves_a_short_frame_alone():
    frame = _frame(100)

    pd.testing.assert_frame_equal(thinned(frame), frame)


def test_thinning_keeps_about_the_limit_and_every_lines_true_extremes():
    frame = _frame()

    thin = thinned(frame, 100)

    assert len(thin) <= 104
    assert thin.index[0] == frame.index[0] and thin.index[-1] == frame.index[-1]
    assert thin.min().equals(frame.min()) and thin.max().equals(frame.max())


def test_a_chart_draws_one_line_per_column_with_a_legend():
    svg = line_chart_svg(_frame(), percent=False, title="Rolling Sharpe, 252-day window")

    assert svg.count("<polyline") == 2
    assert 'aria-label="Rolling Sharpe, 252-day window"' in svg
    assert "</i>A</span>" in svg and "</i>B</span>" in svg
    assert ">2019</text>" in svg


def test_a_percent_chart_labels_its_axis_in_percent():
    frame = pd.DataFrame(
        {"A": np.linspace(-0.2, 0.0, 50)}, index=pd.bdate_range("2024-01-01", periods=50)
    )

    assert ">-20%</text>" in line_chart_svg(frame, percent=True, title="Drawdown")


def test_a_day_with_no_figure_breaks_the_line_rather_than_joining_across_it():
    frame = _frame(50)
    frame.iloc[20:25, 0] = np.nan

    assert line_chart_svg(frame, percent=False, title="x").count("<polyline") == 3


def test_a_flat_line_still_draws():
    frame = pd.DataFrame({"A": 0.5}, index=pd.bdate_range("2024-01-01", periods=10))

    assert line_chart_svg(frame, percent=False, title="x").count("<polyline") == 1


def test_a_frame_too_short_to_draw_says_so():
    assert "too short" in line_chart_svg(_frame().iloc[:1], percent=True, title="x")
    assert "too short" in line_chart_svg(_frame().iloc[:0], percent=True, title="x")
