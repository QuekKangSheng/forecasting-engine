"""The Scenarios page, driven through the real Streamlit page script (FYP-32)."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import bloomberg_extraction_panel
from forecasting_engine.extraction.bloomberg_csv import ColumnSource
from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.ingest.align import Transform
from forecasting_engine.models.base import ModelDescription
from forecasting_engine.reporting.factor_labels import labeller
from forecasting_engine.reporting.model_metrics import ModelRunResult
from forecasting_engine.reporting.polynomial_function import Origin, dataset_fingerprint
from forecasting_engine.store import active_model
from forecasting_engine.validation.crash import CrashDiagnostics

PAGE = Path(__file__).resolve().parents[2] / "app" / "app_pages" / "4_Scenarios.py"
EQUITY, BOND = TargetRole.EQUITY, TargetRole.BOND
SPX = "SPX_Index_PX_LAST"
AGG = "LEGATRUU_Index_TOT_RETURN_INDEX_GROSS_DVDS"
VIX = "VIX_Index_PX_LAST"
RATE = "USGG10YR_Index_PX_LAST"
N_DAYS = 120


@pytest.fixture(autouse=True)
def isolated_active_model_db(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def no_tuning_period(monkeypatch):
    """The latest fold is found with the harness's TUNING_ROWS (504) in the loop —
    longer than this fixture."""
    monkeypatch.setattr("forecasting_engine.validation.splitters.TUNING_ROWS", 0)


def _committed() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    vix = np.linspace(12, 28, N_DAYS)
    vix[-1] = 20.0
    return pd.DataFrame(
        {
            "Date": pd.bdate_range("2025-01-02", periods=N_DAYS),
            SPX: 4000 * np.cumprod(1 + rng.normal(0, 0.01, N_DAYS)),
            AGG: 100 * np.cumprod(1 + rng.normal(0, 0.003, N_DAYS)),
            VIX: vix,
            RATE: 4 + np.cumsum(rng.normal(0, 0.02, N_DAYS)),
        }
    )


SOURCES = {
    VIX: ColumnSource("VIX Index", "PX_LAST"),
    RATE: ColumnSource("USGG10YR Index", "PX_LAST"),
}

#: forecast = 1% − 0.2% × z, z = (clipped VIX − 20) / 5, VIX clipped to 5–35.
LINEAR = active_model.ActiveFunction(
    Origin.DERIVED,
    ModelDescription(
        name="DerivedPolynomial",
        terms=(VIX,),
        coefficients=(-0.002,),
        intercept=0.01,
        input_bounds={VIX: (5.0, 35.0)},
        standardisation={VIX: (20.0, 5.0)},
    ),
    {VIX: Transform.LEVEL},
)
#: forecast = 0.2% + 0.5 × the rate's one-day change.
BOND_FN = active_model.ActiveFunction(
    Origin.USER_SUPPLIED,
    ModelDescription(name="UserPolynomial", terms=(RATE,), coefficients=(0.5,), intercept=0.002),
    {RATE: Transform.DIFFERENCE},
)


def _result() -> ModelRunResult:
    return ModelRunResult(
        ic=0.05,
        oos_rank_ic=0.04,
        rmse=0.01,
        crash=CrashDiagnostics(recall=0.5, precision=0.5, f1=0.5, n_true_tail_days=4),
        signal_rank_ic=0.04,
    )


def _set_active(role, name="Polynomial (derived)", function=None, fingerprint=None):
    active_model.set_active_model(
        role,
        name,
        _result(),
        high_risk=False,
        horizon=5,
        train_window=60,
        test_window=10,
        dataset_fingerprint=fingerprint or str(dataset_fingerprint(_committed())),
        function=function,
    )


def _page(committed=None) -> AppTest:
    app = AppTest.from_file(str(PAGE), default_timeout=60)
    app.session_state[bloomberg_extraction_panel.COMMITTED_KEY] = (
        _committed() if committed is None else committed
    )
    app.session_state[bloomberg_extraction_panel.COMMITTED_TARGETS_KEY] = {EQUITY: SPX, BOND: AGG}
    app.session_state[bloomberg_extraction_panel.COMMITTED_SOURCES_KEY] = SOURCES
    return app.run()


def _shock(app, signal_label, value, kind=None):
    app.multiselect[0].set_value([*app.multiselect[0].value, *_options(app, signal_label)]).run()
    if kind is not None:
        (control,) = [c for c in app.segmented_control if c.key == f"shock_kind_{signal_label}"]
        control.set_value(kind).run()
    (box,) = [t for t in app.text_input if t.key == f"shock_value_{signal_label}"]
    box.input(str(value)).run()
    return app


def _options(app, signal):
    return [] if signal in app.multiselect[0].value else [signal]


def _run(app):
    (button,) = [b for b in app.button if b.label == "Run scenario"]
    assert not button.disabled
    button.click().run()
    assert not app.exception
    return app


def _table(app) -> str | None:
    return next(
        (m.value for m in app.markdown if ">Shocked<span" in m.value), None
    )


def _run_button(app):
    return next(b for b in app.button if b.label == "Run scenario")


# --- gating --------------------------------------------------------------------------


def test_with_no_committed_data_shows_a_guiding_message():
    app = AppTest.from_file(str(PAGE), default_timeout=60).run()

    assert not app.exception
    assert any("visit the Data page" in i.value for i in app.info)


def test_with_no_active_polynomial_says_why_for_each_index():
    _set_active(EQUITY, "Machine Learning")
    app = _page()

    assert not app.exception
    (info,) = app.info
    assert "needs an active polynomial" in info.value
    assert "Machine Learning, has no stored polynomial function" in info.value
    assert "No active model is set for" in info.value


def test_a_model_set_on_other_data_is_not_used():
    _set_active(EQUITY, function=LINEAR, fingerprint="another dataset")
    app = _page()

    assert any("set on another dataset" in i.value for i in app.info)


# --- shocks and validation ------------------------------------------------------------


def test_only_the_signals_the_active_functions_read_can_be_shocked():
    _set_active(EQUITY, function=LINEAR)
    _set_active(BOND, "Polynomial (user-supplied)", function=BOND_FN)
    app = _page()

    label = labeller([c for c in _committed().columns if c != "Date"])
    # Not SPX or the bond index (targets), and nothing the two functions don't read.
    assert app.multiselect[0].options == [label(VIX), label(RATE)]


def test_run_waits_for_a_value_for_each_chosen_signal():
    _set_active(EQUITY, function=LINEAR)
    app = _page()
    assert _run_button(app).disabled  # nothing chosen yet

    app.multiselect[0].set_value([VIX]).run()

    assert _run_button(app).disabled
    assert any("Enter a value for" in c.value for c in app.caption)
    assert any("Last observed 20 on" in m.value for m in app.markdown)


def test_a_value_that_is_not_a_number_is_refused_naming_the_signal_and_blocks_run():
    _set_active(EQUITY, function=LINEAR)
    app = _shock(_page(), VIX, "forty")

    (error,) = app.error
    assert error.value.startswith("VIX") and "not a number" in error.value
    assert _run_button(app).disabled


def test_a_negative_vix_is_refused():
    _set_active(EQUITY, function=LINEAR)
    app = _shock(_page(), VIX, "-5")

    assert "always above zero" in app.error[0].value
    assert _run_button(app).disabled


def test_a_move_beyond_ten_times_the_range_is_refused():
    _set_active(EQUITY, function=LINEAR)
    app = _shock(_page(), VIX, "5000")

    assert "more than 10 times the whole range" in app.error[0].value
    assert _run_button(app).disabled


# --- running it ------------------------------------------------------------------------


def test_a_scenario_shows_base_and_shocked_forecasts_for_both_indices():
    _set_active(EQUITY, function=LINEAR)
    _set_active(BOND, "Polynomial (user-supplied)", function=BOND_FN)
    app = _run(_shock(_page(), VIX, "25"))

    table = _table(app)
    # Equity: 1% base; VIX 20 → 25 is z +1, so 1% − 0.2% = 0.8%.
    assert "+1.00%" in table and "+0.80%" in table and "-0.20 pts" in table
    assert "-20.00%" in table
    # Bond reads only the rate, which was not shocked, so it does not move.
    assert table.count("+0.00 pts") == 1
    assert "Polynomial (user-supplied)" in table
    assert any("Returns over the next 5 trading days" in c.value for c in app.caption)


def test_a_percent_shock_is_applied_to_the_last_observed_value():
    _set_active(EQUITY, function=LINEAR)
    app = _run(_shock(_page(), VIX, "+25", kind="percent"))

    assert "+0.80%" in _table(app)  # 20 × 1.25 = 25


def test_an_index_without_a_polynomial_is_listed_with_the_reason():
    _set_active(EQUITY, function=LINEAR)
    _set_active(BOND, "Machine Learning")
    app = _run(_shock(_page(), VIX, "25"))

    assert "has no stored polynomial function" in _table(app)


def test_a_shock_outside_the_training_range_is_flagged_as_an_extrapolation():
    _set_active(EQUITY, function=LINEAR)
    app = _run(_shock(_page(), VIX, "60"))

    (warning,) = app.warning
    assert warning.value.startswith("Extrapolation: VIX")
    assert "clips this input to 35" in warning.value


def test_a_clipped_input_inside_the_training_range_still_says_it_is_clipped():
    # The latest fold trained on VIX of about 16 to 24 (rows 30 to 89 of a 12-to-28
    # ramp), but this model clips it to 5 to 21: 23 is inside the one, beyond the other.
    narrow = replace(
        LINEAR, description=replace(LINEAR.description, input_bounds={VIX: (5.0, 21.0)})
    )
    _set_active(EQUITY, function=narrow)
    app = _run(_shock(_page(), VIX, "23"))

    assert not app.warning
    assert any("clips this input to 21" in i.value for i in app.info)


def test_changing_a_shock_after_a_run_hides_the_old_result():
    _set_active(EQUITY, function=LINEAR)
    app = _run(_shock(_page(), VIX, "25"))
    (box,) = [t for t in app.text_input if t.key == f"shock_value_{VIX}"]

    box.input("26").run()

    assert _table(app) is None
    assert any("shocks have changed" in i.value for i in app.info)
