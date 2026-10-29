"""The Portfolio Optimizer page, driven through the real Streamlit page script."""

import html
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import bloomberg_extraction_panel
import model_settings
from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.reporting.model_metrics import ModelRunResult
from forecasting_engine.reporting.polynomial_function import dataset_fingerprint
from forecasting_engine.store import active_model
from forecasting_engine.validation.crash import CrashDiagnostics

REPO_ROOT = Path(__file__).resolve().parents[2]
PAGE = REPO_ROOT / "app" / "app_pages" / "3_Portfolio.py"

EQUITY, BOND = TargetRole.EQUITY, TargetRole.BOND
EQUITY_COL = "SPX_Index_PX_LAST"
BOND_COL = "LBUSTRUU_Index_TOT_RETURN_INDEX_GROSS_DVDS"
N_DAYS = 60


@pytest.fixture(autouse=True)
def isolated_active_model_db(monkeypatch, tmp_path):
    """The optimiser reads/writes DuckDB at a default, cwd-relative path."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def no_tuning_period(monkeypatch):
    """Rebalance dates are built with the harness's own TUNING_ROWS (504) in
    the loop, same as a real model run — too large for this small fixture."""
    monkeypatch.setattr("forecasting_engine.portfolio.optimize.TUNING_ROWS", 0)


@pytest.fixture(autouse=True)
def short_train_window(monkeypatch):
    """The covariance is estimated over the models' fixed train window (252 days),
    longer than this 60-day fixture."""
    monkeypatch.setattr(model_settings, "TRAIN_WINDOW", 15)


def _committed() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "Date": pd.bdate_range("2024-01-02", periods=N_DAYS),
            EQUITY_COL: 100 * np.cumprod(1 + rng.normal(0, 0.01, N_DAYS)),
            BOND_COL: 100 * np.cumprod(1 + rng.normal(0, 0.01, N_DAYS)),
        }
    )


#: What the active models are stamped with: the committed dataset below.
SETTINGS = {
    "horizon": 3,
    "train_window": 15,
    "test_window": 10,
    "dataset_fingerprint": str(dataset_fingerprint(_committed())),
}


def _forecast(committed, seed=0, n_dates=40) -> pd.Series:
    dates = pd.DatetimeIndex(committed["Date"])[-n_dates:]
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0, 0.02, n_dates), index=dates)


def _result(forecast=None, *, with_baseline=True) -> ModelRunResult:
    """A result whose forecast sits 0.1% above its fold's training mean throughout,
    so its signal is ``forecast`` less 0.001."""
    baseline = None
    if forecast is not None and with_baseline:
        baseline = forecast * 0 + 0.001
    return ModelRunResult(
        ic=0.05,
        oos_rank_ic=0.04,
        rmse=0.01,
        crash=CrashDiagnostics(recall=0.5, precision=0.5, f1=0.5, n_true_tail_days=4),
        forecast=forecast,
        baseline=baseline,
        signal_rank_ic=0.04,
    )


def _set_active(role, forecast=None, *, with_baseline=True, **overrides):
    active_model.set_active_model(
        role,
        "Polynomial",
        _result(forecast=forecast, with_baseline=with_baseline),
        high_risk=False,
        **{**SETTINGS, **overrides},
    )


def _page(committed) -> AppTest:
    app = AppTest.from_file(str(PAGE), default_timeout=60)
    app.session_state[bloomberg_extraction_panel.COMMITTED_KEY] = committed
    app.session_state[bloomberg_extraction_panel.COMMITTED_TARGETS_KEY] = {
        EQUITY: EQUITY_COL,
        BOND: BOND_COL,
    }
    return app.run()


def _infos(app) -> list[str]:
    return [i.value for i in app.info]


# --- gating ------------------------------------------------------------------------


def test_with_no_active_models_names_both_as_missing():
    app = _page(_committed())

    assert not app.exception
    assert any("Equity and Bond" in text for text in _infos(app))


def test_with_only_bond_active_names_equity_as_missing():
    _set_active(BOND, forecast=_forecast(_committed()))
    app = _page(_committed())

    assert not app.exception
    assert any("Equity" in text and "Bond" not in text for text in _infos(app))


def test_mismatched_settings_blocks_with_an_explanation():
    committed = _committed()
    _set_active(EQUITY, forecast=_forecast(committed), test_window=10)
    _set_active(BOND, forecast=_forecast(committed), test_window=5)
    app = _page(committed)

    assert not app.exception
    assert any("different settings" in w.value for w in app.warning)


def test_an_equity_model_saved_with_no_train_window_combines_with_a_bond_model():
    # An active model set before every model moved to one rolling window can carry
    # no train window (an expanding one); both still forecast the same days.
    committed = _committed()
    _set_active(EQUITY, forecast=_forecast(committed, seed=1), train_window=None)
    _set_active(BOND, forecast=_forecast(committed, seed=2))
    app = _page(committed)

    assert not app.exception
    assert not any("different settings" in w.value for w in app.warning)
    assert len(app.metric) == 2


def test_active_models_set_on_another_dataset_are_refused():
    committed = _committed()
    stale = {"dataset_fingerprint": "an earlier upload"}
    _set_active(EQUITY, forecast=_forecast(committed), **stale)
    _set_active(BOND, forecast=_forecast(committed), **stale)
    app = _page(committed)

    assert not app.exception
    assert any("different dataset than the one committed now" in w.value for w in app.warning)
    assert not app.metric


def test_with_no_committed_data_shows_a_guiding_message():
    committed = _committed()
    _set_active(EQUITY, forecast=_forecast(committed))
    _set_active(BOND, forecast=_forecast(committed))
    app = AppTest.from_file(str(PAGE), default_timeout=60).run()

    assert not app.exception
    assert any("Data page" in text for text in _infos(app))


def test_with_no_saved_forecast_shows_a_guiding_message():
    committed = _committed()
    _set_active(EQUITY, forecast=None)
    _set_active(BOND, forecast=_forecast(committed))
    app = _page(committed)

    assert not app.exception
    assert any("wasn't saved" in text for text in _infos(app))


def test_a_model_set_active_before_signals_were_saved_asks_to_be_set_again():
    committed = _committed()
    _set_active(EQUITY, forecast=_forecast(committed), with_baseline=False)
    _set_active(BOND, forecast=_forecast(committed))
    app = _page(committed)

    assert not app.exception
    assert any("set it active again" in text for text in _infos(app))


# --- happy path ----------------------------------------------------------------


def test_the_happy_path_shows_latest_weights_summing_to_one():
    committed = _committed()
    _set_active(EQUITY, forecast=_forecast(committed, seed=1))
    _set_active(BOND, forecast=_forecast(committed, seed=2))
    app = _page(committed)

    assert not app.exception
    assert len(app.metric) == 2
    values = [float(m.value.strip("%")) / 100 for m in app.metric]
    assert sum(values) == pytest.approx(1.0, abs=1e-6)


# --- FYP-19: performance against the equal-weight benchmark -------------------------


def _happy_page() -> AppTest:
    committed = _committed()
    _set_active(EQUITY, forecast=_forecast(committed, seed=1))
    _set_active(BOND, forecast=_forecast(committed, seed=2))
    return _page(committed)


def _comparison_table(app: AppTest) -> str:
    return next(m.value for m in app.markdown if '<table class="fe-table"' in m.value)


def _captions(app: AppTest) -> str:
    return " ".join(c.value for c in app.caption)


def test_the_four_ratios_are_shown_for_both_portfolios_side_by_side():
    app = _happy_page()

    assert not app.exception
    table = _comparison_table(app)
    for heading in ("Optimised", "Equal-weight benchmark", "Difference"):
        assert f"<th>{heading}</th>" in table
    for metric in ("Sharpe", "Sortino", "Calmar", "Max drawdown"):
        assert f">{metric}<span" in table


def test_the_backtest_date_range_rebalancing_and_costs_are_stated():
    app = _happy_page()

    captions = _captions(app)
    assert "Backtest " in captions and " to " in captions
    assert "every 3 trading days, the forecast horizon" in captions
    assert "reset to 50/50 monthly" in captions
    assert "3 bp equity, 5 bp bond" in captions
    assert "S&P 500: Polynomial" in captions


def test_after_costs_is_the_default_and_before_costs_can_be_chosen():
    app = _happy_page()
    (basis,) = [c for c in app.segmented_control if c.label == "Returns"]
    assert basis.value == "After costs"
    net = _comparison_table(app)

    basis.set_value("Before costs").run()

    assert not app.exception
    assert _comparison_table(app) != net


def test_the_latest_weights_say_how_they_were_set():
    captions = _captions(_happy_page())

    assert "How this was set: the benchmark holds 50%" in captions
    assert "at risk aversion 3" in captions


def test_risk_aversion_is_a_1_to_5_scale_defaulting_to_the_middle():
    app = _happy_page()
    (scale,) = [s for s in app.select_slider if s.label == "Risk aversion"]
    assert list(scale.options) == ["1 · risk-loving", "2", "3", "4", "5 · risk-averse"]
    assert scale.value == 3
    tilt = _solved_weight(app) - 0.5

    scale.set_value(5).run()

    assert not app.exception
    # The solved weight, before the no-trade band: a held rebalance's weight is
    # wherever the portfolio drifted to, whatever the risk aversion.
    assert abs(_solved_weight(app) - 0.5) <= abs(tilt)
    assert "Level 5 is λ = 100." in _captions(app)


def _solved_weight(app) -> float:
    """The latest rebalance's equity weight before any bound or band, from the
    "How this was set" caption."""
    caption = next(c.value for c in app.caption if c.value.startswith("How this was set"))
    return float(re.search(r"giving (-?\d+)%", caption).group(1)) / 100


def test_a_minimum_in_each_index_bounds_both_weights():
    app = _happy_page()
    (floor,) = [s for s in app.slider if s.label == "Minimum in each index"]
    assert floor.value == pytest.approx(0.2)

    floor.set_value(0.4).run()

    assert not app.exception
    values = [float(m.value.strip("%")) / 100 for m in app.metric]
    assert all(0.4 - 1e-9 <= v <= 0.6 + 1e-9 for v in values)
    assert "Each index stays between 40% and 60%." in _captions(app)


def _tail_table(app: AppTest) -> str:
    """Found by content, not position, so another table on the page can't shift it."""
    (tail,) = [
        m.value
        for m in app.markdown
        if '<table class="fe-table"' in m.value and "1-day CVaR" in m.value
    ]
    return tail


def test_historical_var_and_cvar_are_shown_beside_max_drawdown():
    app = _happy_page()

    tail = _tail_table(app)
    for metric in ("1-day VaR 95%", "1-day CVaR 99%", "Max drawdown", "breach rate"):
        assert metric in tail
    assert any("Historical" in m.value and "Tail risk" in m.value for m in app.markdown)
    assert "one-day losses read from the realised returns" in _captions(app)


def test_every_var_method_is_labelled_and_monte_carlo_shows_not_run():
    app = _happy_page()

    tail = _tail_table(app)
    for method in ("Historical", "Monte Carlo (Parametric / Bootstrap)"):
        assert method in tail
    assert "GARCH" not in tail  # optional, and not built: no row (FYP-251)
    assert ">Not run</td>" in tail
    assert "Max drawdown" in tail


def test_the_tail_table_follows_the_before_costs_toggle():
    app = _happy_page()
    (basis,) = [c for c in app.segmented_control if c.label == "Returns"]

    basis.set_value("Before costs").run()

    assert not app.exception
    assert "Not run" in _tail_table(app)


def test_the_caption_explains_that_not_run_is_not_zero_risk():
    assert "not a zero risk" in _captions(_happy_page())


def test_the_cumulative_return_chart_is_drawn_beside_the_weights_chart():
    app = _happy_page()

    # The weights bar chart and the cumulative-return line chart.
    assert len(app.get("vega_lite_chart")) == 2


def test_without_active_models_the_page_explains_what_will_appear():
    app = _page(_committed())

    assert any("performance against the equal-weight benchmark" in t for t in _infos(app))
    assert not [m for m in app.markdown if '<table class="fe-table"' in m.value]


def test_the_no_trade_band_is_explained_with_the_share_of_rebalances_it_held():
    app = _happy_page()

    assert not app.exception
    captions = _captions(app)
    assert "were held by the no-trade band." in captions
    assert "Held: within the no-trade band." in captions or "no-trade band of ±" in captions


# --- FYP-20 / FYP-52: is the outperformance statistically significant? --------------

_VERDICTS = (
    "Statistically significant at 95% confidence",
    "Significantly worse than the benchmark at 95% confidence",
    "Not statistically significant",
    "Cannot be tested",
)


def _markdown(app: AppTest) -> str:
    return " ".join(m.value for m in app.markdown)


def test_the_spread_t_statistic_is_shown_with_a_plain_language_verdict():
    app = _happy_page()

    assert not app.exception
    text = _markdown(app)
    assert "Is the outperformance genuine?" in text
    assert re.search(r"Spread t-statistic.*t = (-?\d+\.\d\d|—)", text)
    assert sum(verdict in text for verdict in _VERDICTS) == 1


def test_the_significance_caption_says_how_the_test_was_run():
    captions = _captions(_happy_page())

    assert "Paired test on" in captions or "no difference to test" in captions
    assert "After costs, the same basis as the table above" in captions


def test_the_t_statistic_follows_the_before_costs_toggle():
    app = _happy_page()
    (basis,) = [c for c in app.segmented_control if c.label == "Returns"]
    basis.set_value("Before costs").run()

    assert not app.exception
    assert "Before costs, the same basis as the table above" in _captions(app)


def test_the_t_statistic_explains_itself_on_hover():
    import glossary

    hint = html.escape(glossary.term("Spread t-statistic"), quote=True)
    assert hint in _markdown(_happy_page())


def test_without_performance_data_the_page_guides_instead_of_testing():
    app = _page(_committed())

    assert not app.exception
    assert any("statistically significant appear here" in t for t in _infos(app))
    assert "Spread t-statistic" not in _markdown(app)


# --- FYP-36: Monte Carlo VaR, CVaR and drawdown over the next horizon --------------


def _simulation_table(app: AppTest) -> str | None:
    return next(
        (m.value for m in app.markdown if "Max drawdown 95%" in m.value and "<table" in m.value),
        None,
    )


def _run_simulation(app: AppTest) -> AppTest:
    (button,) = [b for b in app.button if b.label == "Run simulation"]
    button.click().run()
    assert not app.exception
    return app


def test_the_simulation_waits_to_be_run_and_defaults_to_bootstrap_and_100000_paths():
    app = _happy_page()

    assert not app.exception
    assert any("Monte Carlo over the next 3 days" in m.value for m in app.markdown)
    (method,) = [c for c in app.segmented_control if c.label == "Sampling method"]
    assert method.value == "bootstrap"
    assert list(method.options) == ["Bootstrap (historical days)", "Parametric (fitted normal)"]
    (paths,) = [n for n in app.number_input if n.label == "Simulated paths"]
    assert paths.value == 100_000
    assert _simulation_table(app) is None
    assert "Choose a sampling method, then run it." in _captions(app)


def test_running_the_simulation_shows_var_cvar_and_drawdown_at_both_levels():
    app = _run_simulation(_happy_page())

    table = _simulation_table(app)
    for metric in (
        "Mean 3-day return",
        "3-day VaR 95%",
        "3-day CVaR 95%",
        "3-day VaR 99%",
        "3-day CVaR 99%",
        "Max drawdown 95%",
        "Max drawdown 99%",
    ):
        assert f">{metric}<span" in table
    for heading in ("Optimised", "Equal-weight benchmark"):
        assert f"<th>{heading}</th>" in table
    captions = _captions(app)
    assert "100,000 paths of the next 3 trading days" in captions
    assert "by resampling whole days" in captions
    assert "the one-day Monte Carlo rows above are the first day of these same paths" in captions


def test_the_path_count_is_adjustable_under_advanced_settings():
    app = _happy_page()
    (paths,) = [n for n in app.number_input if n.label == "Simulated paths"]
    paths.set_value(5_000).run()

    _run_simulation(app)

    assert "5,000 paths of the next 3 trading days" in _captions(app)


def test_parametric_sampling_can_be_chosen_and_gives_other_figures():
    app = _run_simulation(_happy_page())
    bootstrap = _simulation_table(app)
    (method,) = [c for c in app.segmented_control if c.label == "Sampling method"]

    method.set_value("parametric").run()

    # The bootstrap figures are not shown as if they answered the new settings.
    assert _simulation_table(app) is None
    assert any("changed since the last simulation" in i.value for i in app.info)

    _run_simulation(app)

    assert _simulation_table(app) != bootstrap
    assert "from a normal distribution fitted to the days" in _captions(app)


def test_a_new_allocation_hides_the_old_simulation_until_it_is_run_again():
    app = _run_simulation(_happy_page())
    (floor,) = [s for s in app.slider if s.label == "Minimum in each index"]

    floor.set_value(0.45).run()

    assert _simulation_table(app) is None
    assert any("changed since the last simulation" in i.value for i in app.info)


def _tail_table(app: AppTest) -> str:
    return next(m.value for m in app.markdown if "<th>Method</th>" in m.value)


def _method_rows(table: str, method: str) -> list[str]:
    return [row for row in table.split("<tr>") if f">{method}<span" in row]


def test_running_the_simulation_fills_the_monte_carlo_rows_of_the_tail_table():
    app = _happy_page()
    before = _method_rows(_tail_table(app), "Monte Carlo (Parametric / Bootstrap)")
    assert len(before) == 4 and all("Not run" in row for row in before)

    _run_simulation(app)

    table = _tail_table(app)
    after = _method_rows(table, "Monte Carlo (Parametric / Bootstrap)")
    assert len(after) == 4
    assert not any("Not run" in row for row in after)
    assert "the first day of each simulated path below" in _captions(app)


def test_a_changed_setting_puts_the_monte_carlo_rows_back_to_not_run():
    app = _run_simulation(_happy_page())
    (method,) = [c for c in app.segmented_control if c.label == "Sampling method"]

    method.set_value("parametric").run()

    rows = _method_rows(_tail_table(app), "Monte Carlo (Parametric / Bootstrap)")
    assert all("Not run" in row for row in rows)
