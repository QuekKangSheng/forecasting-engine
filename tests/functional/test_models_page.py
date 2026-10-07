"""The Models page, driven through the real Streamlit page script.

Most of these render stored runs the way the page shows a previous result,
rather than fitting one: a derived fit is slow and nondeterministic, and what's
under test here is what the page shows, not the fitting. Tests that press Run
stub the machine-learning fit and the factor download.
"""

import pickle
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import model_jobs
import model_runs
from forecasting_engine.extraction.bloomberg_csv import ColumnSource
from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.ingest import fama_french
from forecasting_engine.ingest.fama_french import FactorFetchError, FactorFile, ResolvedFactors
from forecasting_engine.ingest.provenance import SourceFile
from forecasting_engine.models import boosted
from forecasting_engine.models.base import ModelDescription
from forecasting_engine.models.boosted import Tune, TuningLog
from forecasting_engine.models.famafrench import FACTOR_COLUMNS, FactorCoverage
from forecasting_engine.reporting.model_metrics import (
    FoldTerms,
    ModelRunResult,
    ScreeningSummary,
)
from forecasting_engine.reporting.polynomial_function import (
    Origin,
    dataset_fingerprint,
    from_description,
)
from forecasting_engine.store import active_model
from forecasting_engine.validation import splitters
from forecasting_engine.validation.crash import CrashDiagnostics

REPO_ROOT = Path(__file__).resolve().parents[2]
PAGE = REPO_ROOT / "app" / "app_pages" / "2_Models.py"

EQUITY, BOND = TargetRole.EQUITY, TargetRole.BOND
SPX = "SPX_Index_PX_LAST"
AGG = "LBUSTRUU_Index_TOT_RETURN_INDEX_GROSS_DVDS"
POLY = "Polynomial (derived)"
USER_POLY = "Polynomial (user-supplied)"
#: The page's defaults: 5-day horizon, 120/20 walk-forward windows, a 10-term
#: cap on the derived polynomial and a blank user-supplied function.
DEFAULT_SHARED = (5, 120, 20)
DEFAULT_MODEL_SETTINGS = {POLY: 10, USER_POLY: ("", ())}
NAIVE = "Naive (training mean)"


@pytest.fixture(autouse=True)
def short_tuning_period(monkeypatch):
    """The fixture data is 200 rows, far short of the real 504-row tuning period."""
    monkeypatch.setattr(splitters, "TUNING_ROWS", 40)


@pytest.fixture(autouse=True)
def isolated_active_model_db(monkeypatch, tmp_path):
    """The active-model picker reads/writes DuckDB at a default, cwd-relative path."""
    monkeypatch.chdir(tmp_path)


def _committed(*, bond: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    frame = pd.DataFrame(
        {
            "Date": pd.bdate_range("2024-01-01", periods=n),
            SPX: 4000 + np.cumsum(rng.normal(size=n)),
            "VIX_Index_PX_LAST": 15 + rng.normal(size=n),
            "LUACOAS_Index_PX_LAST": 1.2 + rng.normal(scale=0.1, size=n),
        }
    )
    if bond:
        frame[AGG] = 2200 + np.cumsum(rng.normal(size=n))
    return frame


def _result(
    screening: ScreeningSummary | None = None,
    *,
    pbo: float | None = 0.3,
    oos_rank_ic: float = 0.04,
    terms: FoldTerms | None = None,
) -> ModelRunResult:
    return ModelRunResult(
        ic=0.05,
        oos_rank_ic=oos_rank_ic,
        rmse=0.01,
        pbo=pbo,
        crash=CrashDiagnostics(recall=0.5, precision=0.5, f1=0.5, n_true_tail_days=4),
        screening=screening,
        terms=terms,
        rows_scored=180,
    )


DERIVED = ModelDescription(
    name="DerivedPolynomial",
    terms=("VIX_Index_PX_LAST^2", "LUACOAS_Index_PX_LAST VIX_Index_PX_LAST"),
    coefficients=(0.0004521, -0.000231),
    intercept=0.001234,
)


def _polynomial_run(
    result: ModelRunResult | None = None, description: ModelDescription = DERIVED
) -> model_runs.ModelRun:
    fn = from_description(description, origin=Origin.DERIVED, target=SPX, horizon=5)
    return model_runs.ModelRun(result or _result(), description, function=fn)


def _ml_run(result: ModelRunResult | None = None) -> model_runs.ModelRun:
    description = ModelDescription("Boosted[xgboost]", ("VIX_Index_PX_LAST",), (0.3,))
    return model_runs.ModelRun(result or _result(), description)


def _page(
    tabs: dict[TargetRole, dict[str, model_runs.ModelRun]] | None = None,
    *,
    bond: bool = False,
    sources: dict | None = None,
    committed: pd.DataFrame | None = None,
) -> AppTest:
    """The page with ``tabs``' runs stored under the default settings."""
    committed = _committed(bond=bond) if committed is None else committed
    app = AppTest.from_file(str(PAGE), default_timeout=60)
    app.session_state["extraction_committed"] = committed
    app.session_state["extraction_committed_targets"] = (
        {EQUITY: SPX, BOND: AGG} if bond else {EQUITY: SPX}
    )
    if sources is not None:
        app.session_state["extraction_committed_sources"] = sources
    if tabs:
        stored = model_runs.StoredRuns((dataset_fingerprint(committed), *DEFAULT_SHARED))
        for role, runs in tabs.items():
            stored.tabs[role] = model_runs.TabRuns(dict(DEFAULT_MODEL_SETTINGS), dict(runs))
        app.session_state[model_runs.RUNS_KEY] = stored
    return app.run()


def _stored_runs(app: AppTest, role: TargetRole = EQUITY) -> dict:
    stored = app.session_state[model_runs.RUNS_KEY]
    return stored.tabs[role].runs if role in stored.tabs else {}


def _markdown(app: AppTest) -> str:
    return " ".join(m.value for m in app.markdown)


def _captions(app: AppTest) -> str:
    return " ".join(c.value for c in app.caption)


def _table(app: AppTest) -> str:
    return next(m.value for m in app.markdown if '<table class="fe-table"' in m.value)


def _has_table(app: AppTest) -> bool:
    return any('<table class="fe-table"' in m.value for m in app.markdown)


def _expander_labels(app: AppTest) -> list[str]:
    return [e.label for e in app.expander]


DERIVE, OWN = "Derive automatically", "Use your own function"


def _source(app: AppTest, tab: int = 0):
    return [r for r in app.radio if r.label == "Function source"][tab]


def _choose_own(app: AppTest) -> AppTest:
    """Switch every tab's polynomial to the user-supplied function."""
    for radio in [r for r in app.radio if r.label == "Function source"]:
        radio.set_value(OWN)
    return app.run()


def _untick(app: AppTest, *labels: str) -> AppTest:
    for label in labels:
        next(c for c in app.checkbox if c.label == label).uncheck()
    return app.run()


# --- guards and layout -----------------------------------------------------------


def test_no_target_resolved_shows_a_guiding_message():
    app = AppTest.from_file(str(PAGE), default_timeout=30)
    app.session_state["extraction_committed"] = _committed()
    app.run()

    assert not app.exception
    assert any("No target resolved yet" in i.value for i in app.info)
    assert not app.tabs


def test_there_is_a_tab_per_resolved_target_named_after_it():
    app = _page(bond=True)

    assert not app.exception
    assert [t.label for t in app.tabs] == ["Equity · S&P 500", "Bond · US Aggregate Bond"]


def test_the_old_family_and_target_pickers_and_trials_input_are_gone():
    app = _page()

    assert not [r for r in app.radio if r.label in ("Model family", "Target")]
    assert all("optuna" not in n.label.lower() for n in app.number_input)


def test_the_model_checkboxes_default_to_every_family_and_none_is_compulsory():
    app = _page()

    boxes = {c.label: c for c in app.checkbox}
    assert set(boxes) == {"Polynomial", "Fama-French 5", "Machine learning"}
    assert all(box.value and not box.disabled for box in boxes.values())


def test_settings_hold_the_horizon_windows_and_embargo():
    app = _page()

    (settings,) = [e for e in app.expander if e.label == "Settings"]
    (horizon,) = settings.segmented_control
    assert list(horizon.options) == ["1 day", "5 days"]
    assert horizon.value == 5
    windows = {n.label: n.value for n in settings.number_input}
    assert windows == {
        "Walk-forward train window (days)": 120,
        "Walk-forward test window (days)": 20,
    }
    assert any(c.value.startswith("Embargo is fixed") for c in settings.caption)


def test_the_embargo_and_signal_lag_are_not_inputs():
    app = _page()
    labels = [n.label.lower() for n in app.number_input]
    assert all("embargo" not in label and "lag" not in label for label in labels)
    assert all("lag-shift" not in e.label.lower() for e in app.expander)


@pytest.fixture
def splitter_calls(monkeypatch):
    """Record the arguments PurgedWalkForward is actually built with.

    The page imports the name when its script runs, so patching the module
    attribute beforehand is what the page picks up.
    """
    calls: list[dict] = []
    real = splitters.PurgedWalkForward

    class Recording(real):
        def __init__(self, train: int, test: int, embargo: int, tuning_rows: int = 0):
            calls.append(
                {"train": train, "test": test, "embargo": embargo, "tuning_rows": tuning_rows}
            )
            super().__init__(train, test, embargo, tuning_rows)

    monkeypatch.setattr(splitters, "PurgedWalkForward", Recording)
    return calls


@pytest.mark.parametrize("chosen", [1, 5])
def test_the_embargo_is_five_whichever_horizon_is_chosen(splitter_calls, chosen):
    app = _page()
    (horizon,) = [c for c in app.segmented_control if c.label == "Forecast horizon"]
    horizon.set_value(chosen).run()

    assert not app.exception
    assert splitter_calls[-1]["embargo"] == 5


def test_every_model_is_scored_only_after_the_tuning_period(splitter_calls):
    app = _page()

    assert not app.exception
    assert splitter_calls[-1]["tuning_rows"] == splitters.TUNING_ROWS
    assert "tuning period" in _captions(app)


def test_a_target_securitys_other_fields_are_not_signals_and_transforms_are_shown():
    committed = _committed()
    committed["SPX_Index_PX_BID"] = committed[SPX] - 0.1
    sources = {
        f"{ticker}_Index_{field}": ColumnSource(f"{ticker} Index", field)
        for ticker, field in [
            ("SPX", "PX_LAST"),
            ("SPX", "PX_BID"),
            ("VIX", "PX_LAST"),
            ("LUACOAS", "PX_LAST"),
        ]
    }
    app = _page(committed=committed, sources=sources)

    assert not app.exception
    alignment = next(d.value for d in app.dataframe if "Carried forward" in d.value.columns)
    assert dict(zip(alignment["Signal"], alignment["Transform"], strict=True)) == {
        "VIX_Index_PX_LAST": "level",
        "LUACOAS_Index_PX_LAST": "level",
    }
    assert not app.warning


def test_an_unclassified_signal_is_named_in_a_warning():
    sources = {
        SPX: ColumnSource("SPX Index", "PX_LAST"),
        "VIX_Index_PX_LAST": ColumnSource("VIX Index", "PX_LAST"),
        "LUACOAS_Index_PX_LAST": ColumnSource("NEWTICK Index", "PX_LAST"),
    }
    app = _page(sources=sources)

    (warning,) = app.warning
    assert "LUACOAS_Index_PX_LAST" in warning.value
    assert "VIX_Index_PX_LAST" not in warning.value
    assert "difference" in warning.value


# --- results: only what ran, named for its target -------------------------------


def test_nothing_run_yet_shows_no_results():
    app = _page()

    assert not app.exception
    assert "Nothing has run for S&P 500" in _captions(app)
    assert not _has_table(app)


def test_results_render_only_for_the_models_that_ran():
    app = _page({EQUITY: {POLY: _polynomial_run()}})

    assert not app.exception
    table = _table(app)
    assert table.count("<tr>") == 2  # header + one model
    assert "Polynomial" in table and "Machine Learning" not in table
    labels = _expander_labels(app)
    assert "Polynomial (derived) · S&P 500" in labels
    assert not [e for e in labels if e.startswith(("Fama-French 5", "Machine learning"))]


def test_every_results_header_names_the_target():
    app = _page({EQUITY: {POLY: _polynomial_run(), "Machine Learning": _ml_run()}})

    assert "Results · S&P 500" in [s.value for s in app.subheader]
    for heading in ("Signal screening", POLY, "Machine learning"):
        assert f"{heading} · S&P 500" in _expander_labels(app)


def test_the_table_shows_rows_scored():
    table = _table(_page({EQUITY: {POLY: _polynomial_run()}}))
    assert "Rows scored" in table
    assert "<td>180</td>" in table


def test_each_model_gets_a_one_line_gate_summary():
    failing = _result(oos_rank_ic=0.01, pbo=0.7)
    benchmark = model_runs.ModelRun(
        _result(pbo=None),
        ModelDescription("FamaFrench5", FACTOR_COLUMNS, (0.1,) * 5),
        coverage=FactorCoverage(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-08-30"), 150, 5),
    )
    app = _page({EQUITY: {POLY: _polynomial_run(failing), "FF5 Benchmark": benchmark}})

    text = _markdown(app)
    assert "**Polynomial (derived)**: gate failed on OOS Rank IC and PBO." in text
    assert "**FF5 Benchmark**: not gated" in text


def test_each_targets_results_stay_in_its_own_tab():
    app = _page({EQUITY: {POLY: _polynomial_run()}}, bond=True)

    assert not app.exception
    assert "Results · S&P 500" in [s.value for s in app.subheader]
    assert "Nothing has run for US Aggregate Bond" in _captions(app)


# --- active model ------------------------------------------------------------------


def test_no_active_model_shows_a_guiding_caption():
    app = _page({EQUITY: {POLY: _polynomial_run()}})

    assert not app.exception
    assert "No active model set yet for S&P 500." in _captions(app)
    assert active_model.get_active_model(EQUITY) is None


def test_setting_a_model_active_persists_it_and_shows_it():
    app = _page({EQUITY: {POLY: _polynomial_run()}})
    next(b for b in app.button if b.label == "Set as active").click().run()

    assert not app.exception
    assert "Active model · S&P 500" in _markdown(app)
    assert "Polynomial" in _markdown(app)
    stored = active_model.get_active_model(EQUITY)
    assert stored.model_name == POLY
    assert stored.high_risk is False


def test_setting_a_new_active_model_replaces_the_prior_one():
    app = _page({EQUITY: {POLY: _polynomial_run(), "Machine Learning": _ml_run()}})
    (select,) = app.selectbox
    select.select("Machine Learning").run()
    next(b for b in app.button if b.label == "Set as active").click().run()

    assert active_model.get_active_model(EQUITY).model_name == "Machine Learning"

    select = app.selectbox[0]
    select.select(POLY).run()
    next(b for b in app.button if b.label == "Set as active").click().run()

    assert active_model.get_active_model(EQUITY).model_name == POLY


def test_the_active_model_picker_excludes_benchmarks():
    benchmark = model_runs.ModelRun(
        _result(pbo=None),
        ModelDescription("FamaFrench5", FACTOR_COLUMNS, (0.1,) * 5),
        coverage=FactorCoverage(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-08-30"), 150, 5),
    )
    app = _page(
        {
            EQUITY: {
                NAIVE: model_runs.ModelRun(_result(), ModelDescription("Naive", (), ())),
                "FF5 Benchmark": benchmark,
                POLY: _polynomial_run(),
            }
        }
    )

    (select,) = app.selectbox
    assert list(select.options) == [POLY]


def test_with_no_forecasting_model_run_the_picker_shows_a_caption_instead():
    naive_run = model_runs.ModelRun(_result(), ModelDescription("Naive", (), ()))
    app = _page({EQUITY: {NAIVE: naive_run}})

    assert not app.exception
    assert not app.selectbox
    assert "No forecasting model (Polynomial or ML) has run yet for S&P 500." in _captions(app)


def test_a_high_risk_model_asks_for_confirmation_before_being_set_active():
    failing = _result(oos_rank_ic=0.01, pbo=0.7)
    app = _page({EQUITY: {POLY: _polynomial_run(failing)}})
    next(b for b in app.button if b.label == "Set as active").click().run()

    assert not app.exception
    assert any("failed both promotion gates" in w.value for w in app.warning)
    assert any(b.label == "Set active anyway" for b in app.button)
    assert active_model.get_active_model(EQUITY) is None


def test_confirming_a_high_risk_model_sets_it_active():
    failing = _result(oos_rank_ic=0.01, pbo=0.7)
    app = _page({EQUITY: {POLY: _polynomial_run(failing)}})
    next(b for b in app.button if b.label == "Set as active").click().run()
    next(b for b in app.button if b.label == "Set active anyway").click().run()

    assert not app.exception
    active = active_model.get_active_model(EQUITY)
    assert active is not None and active.model_name == POLY and active.high_risk


def test_cancelling_a_high_risk_model_leaves_nothing_active():
    failing = _result(oos_rank_ic=0.01, pbo=0.7)
    app = _page({EQUITY: {POLY: _polynomial_run(failing)}})
    next(b for b in app.button if b.label == "Set as active").click().run()
    next(b for b in app.button if b.label == "Cancel").click().run()

    assert active_model.get_active_model(EQUITY) is None
    assert not any("failed both promotion gates" in w.value for w in app.warning)


def test_a_model_that_only_fails_one_gate_is_not_treated_as_high_risk():
    one_gate_failing = _result(oos_rank_ic=0.01, pbo=0.3)
    app = _page({EQUITY: {POLY: _polynomial_run(one_gate_failing)}})
    next(b for b in app.button if b.label == "Set as active").click().run()

    assert not app.exception
    assert not any("promotion gates" in w.value for w in app.warning)
    assert active_model.get_active_model(EQUITY).model_name == POLY


# --- signal screening -------------------------------------------------------------


@pytest.fixture
def screened() -> ScreeningSummary:
    return ScreeningSummary(
        folds=8,
        fell_back=0,
        counts=(("VIX_Index_PX_LAST", 8), ("LUACOAS_Index_PX_LAST", 0)),
        fold_ics=({"VIX_Index_PX_LAST": 0.05, "LUACOAS_Index_PX_LAST": 0.001},),
        latest_included=("VIX_Index_PX_LAST",),
    )


def _screening_table(app: AppTest) -> pd.DataFrame:
    return next(d.value for d in app.dataframe if "Included in" in d.value.columns)


def test_screening_shows_transform_latest_fold_and_folds_included(screened):
    app = _page({EQUITY: {POLY: _polynomial_run(_result(screened))}})

    table = _screening_table(app).set_index("Signal")
    assert table.loc["VIX_Index_PX_LAST", "Transform"] == "difference"
    assert table.loc["VIX_Index_PX_LAST", "Latest fold"] == "In"
    assert table.loc["LUACOAS_Index_PX_LAST", "Latest fold"] == "Out"
    assert table.loc["VIX_Index_PX_LAST", "Latest-fold IC"] == pytest.approx(0.05)
    assert table.loc["LUACOAS_Index_PX_LAST", "Included in"] == "0 of 8 folds"


def test_screening_is_taken_from_ml_when_the_polynomial_did_not_screen(screened):
    app = _page(
        {EQUITY: {POLY: _polynomial_run(), "Machine Learning": _ml_run(_result(screened))}}
    )
    assert _screening_table(app)["Signal"].tolist() == [
        "VIX_Index_PX_LAST",
        "LUACOAS_Index_PX_LAST",
    ]


def test_folds_with_no_screened_signal_are_called_out_under_each_model():
    screening = ScreeningSummary(folds=8, fell_back=3, counts=(("VIX_Index_PX_LAST", 5),))
    runs = {
        POLY: _polynomial_run(_result(screening)),
        "Machine Learning": _ml_run(_result(screening)),
    }
    app = _page({EQUITY: runs})

    message = "3 of 8 folds had no signal pass screening; those folds forecast the training mean."
    for heading in (f"{POLY} · S&P 500", "Machine learning · S&P 500"):
        (expander,) = [e for e in app.expander if e.label == heading]
        assert message in " ".join(c.value for c in expander.caption)


def test_nothing_screened_says_so():
    app = _page({EQUITY: {POLY: _polynomial_run()}})

    assert "Nothing was screened" in _captions(app)
    assert all("Included in" not in d.value.columns for d in app.dataframe)


# --- the fitted polynomial as a labelled function --------------------------------


def _terms_table(app: AppTest) -> pd.DataFrame:
    return next(d.value for d in app.dataframe if "Exponent" in d.value.columns)


def test_the_function_says_what_it_forecasts():
    app = _page({EQUITY: {POLY: _polynomial_run()}})
    assert "Forecasts: S&P 500, 5-day return" in _captions(app)
    assert "Derived Function" in _markdown(app)


def test_the_equation_is_typeset_with_labels():
    (latex,) = [lx.value for lx in _page({EQUITY: {POLY: _polynomial_run()}}).latex]
    assert r"\hat{y} = 0.001234 + 0.0004521" in latex
    assert r"\text{VIX}^{2}" in latex


def test_the_clip_bounds_are_shown_beside_the_equation():
    clipped = replace(DERIVED, input_bounds={"VIX_Index_PX_LAST": (9.5, 31.25)})
    app = _page({EQUITY: {POLY: _polynomial_run(description=clipped)}})

    assert "mean ± 4 standard deviations: VIX 9.5 to 31.25" in _captions(app)


def test_a_standardised_equation_says_how_each_z_is_made():
    standardised = replace(
        DERIVED,
        standardisation={"VIX_Index_PX_LAST": (18.2, 6.1), "LUACOAS_Index_PX_LAST": (1.25, 0.3)},
    )
    app = _page({EQUITY: {POLY: _polynomial_run(description=standardised)}})

    (equation,) = [e.value for e in app.latex]
    assert r"z_{\text{VIX}}^{2}" in equation
    captions = _captions(app)
    assert "z(VIX) = (VIX − 18.20) / 6.100" in captions
    assert "z(US IG credit spread) = (US IG credit spread − 1.250) / 0.3000" in captions


def test_the_term_table_uses_labels_not_raw_column_codes():
    table = _terms_table(_page({EQUITY: {POLY: _polynomial_run()}}))

    assert table["Factor"].tolist() == ["(intercept)", "VIX", "US IG credit spread × VIX"]
    assert table["Coefficient"].tolist() == ["0.001234", "0.0004521", "−0.0002310"]


def test_a_run_whose_folds_mostly_kept_nothing_says_so():
    run = _polynomial_run(_result(terms=FoldTerms(folds=10, with_terms=4)))
    captions = _captions(_page({EQUITY: {POLY: run}}))

    assert "4 of 10 walk-forward folds kept any term at all" in captions
    assert "most recent fold's fit" in captions


def test_a_run_where_every_fold_kept_terms_says_that_instead():
    run = _polynomial_run(_result(terms=FoldTerms(folds=10, with_terms=10)))
    captions = _captions(_page({EQUITY: {POLY: run}}))

    assert "Every one of the 10 walk-forward folds kept at least one term." in captions


def test_an_empty_equation_still_explains_itself():
    empty = ModelDescription(name="DerivedPolynomial", terms=(), coefficients=(), intercept=0.002)
    run = _polynomial_run(_result(terms=FoldTerms(folds=10, with_terms=3)), empty)
    captions = _captions(_page({EQUITY: {POLY: run}}))

    assert "No terms survived fitting" in captions
    assert "3 of 10 walk-forward folds kept any term at all" in captions


# --- clearing: results always match the settings shown ---------------------------


def _two_tabs() -> dict:
    return {EQUITY: {POLY: _polynomial_run()}, BOND: {POLY: _polynomial_run()}}


def test_changing_a_shared_setting_clears_every_tabs_results():
    app = _page(_two_tabs(), bond=True)
    (train,) = [n for n in app.number_input if n.label.startswith("Walk-forward train")]
    train.set_value(130).run()

    assert not app.exception
    assert _stored_runs(app, EQUITY) == {}
    assert _stored_runs(app, BOND) == {}
    assert not _has_table(app)


def test_changing_the_horizon_clears_every_tabs_results():
    app = _page(_two_tabs(), bond=True)
    (horizon,) = [c for c in app.segmented_control if c.label == "Forecast horizon"]
    horizon.set_value(1).run()

    assert _stored_runs(app, EQUITY) == {}
    assert _stored_runs(app, BOND) == {}


def test_changing_a_function_clears_only_that_tabs_user_supplied_row():
    tabs = {
        EQUITY: {POLY: _polynomial_run(), USER_POLY: _polynomial_run()},
        BOND: {POLY: _polynomial_run(), USER_POLY: _polynomial_run()},
    }
    app = _choose_own(_page(tabs, bond=True))
    (formula, _bond_formula) = [t for t in app.text_input if t.label.startswith("Function")]
    formula.set_value("2 * VIX_Index_PX_LAST").run()

    assert set(_stored_runs(app, EQUITY)) == {POLY}
    assert set(_stored_runs(app, BOND)) == {POLY, USER_POLY}


def test_changing_the_term_cap_clears_only_that_tabs_derived_row():
    tabs = {EQUITY: {POLY: _polynomial_run(), "Machine Learning": _ml_run()}}
    app = _page(tabs)
    (cap,) = [n for n in app.number_input if n.label.startswith("Max terms")]
    cap.set_value(5).run()

    assert set(_stored_runs(app)) == {"Machine Learning"}


def test_ticking_or_unticking_a_model_clears_nothing():
    app = _untick(_page(_two_tabs(), bond=True), "Machine learning", "Polynomial")

    assert POLY in _stored_runs(app, EQUITY)
    assert POLY in _stored_runs(app, BOND)


def test_switching_the_polynomial_source_clears_nothing_and_keeps_each_input():
    tabs = {EQUITY: {POLY: _polynomial_run(), USER_POLY: _polynomial_run()}}
    app = _page(tabs)
    _source(app).set_value(OWN).run()
    assert not [n for n in app.number_input if n.label.startswith("Max terms")]
    _source(app).set_value(DERIVE).run()

    assert set(_stored_runs(app)) == {POLY, USER_POLY}
    (cap,) = [n for n in app.number_input if n.label.startswith("Max terms")]
    assert cap.value == 10


def test_only_the_chosen_sources_input_is_shown():
    app = _page()
    assert [n for n in app.number_input if n.label.startswith("Max terms")]
    assert not [t for t in app.text_input if t.label.startswith("Function")]

    _source(app).set_value(OWN).run()

    assert not [n for n in app.number_input if n.label.startswith("Max terms")]
    assert [t for t in app.text_input if t.label.startswith("Function")]


def test_committing_a_new_dataset_clears_everything():
    app = _page(_two_tabs(), bond=True)
    app.session_state["extraction_committed"] = _committed(bond=True).iloc[:-1]
    app.run()

    assert _stored_runs(app, EQUITY) == {}
    assert _stored_runs(app, BOND) == {}


# --- pressing Run ------------------------------------------------------------------


@pytest.fixture
def stub_ml(monkeypatch):
    calls = []

    def fake(panel, splitter):
        calls.append(panel)
        return _result(), _ml_run().description, _tuning_log()

    monkeypatch.setattr(boosted, "run_boosted", fake)
    return calls


def _tuning_log() -> TuningLog:
    def tune(first: str, last: str, trials: int, depth: int) -> Tune:
        params = {"max_depth": depth, "learning_rate": 0.05}
        return Tune(
            pd.Timestamp(first), pd.Timestamp(last), 40, trials, {"xgboost": params}
        )

    return TuningLog(
        tunes=(tune("2024-01-01", "2024-02-23", 50, 3), tune("2024-03-01", "2024-04-25", 20, 4)),
        fold_tunes=(0, 0, 1),
        library="xgboost",
    )


def _factors(dates: pd.Series) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {"Date": dates, **{c: rng.normal(size=len(dates)) for c in FACTOR_COLUMNS}}
    )


def _run_equity(app: AppTest, formula: str = "2 * VIX_Index_PX_LAST") -> AppTest:
    """Run the equity tab with ``formula`` as the user-supplied polynomial."""
    _choose_own(app)
    next(t for t in app.text_input if t.label.startswith("Function")).set_value(formula).run()
    return _press_run(app)


def _press_run(app: AppTest) -> AppTest:
    next(b for b in app.button if b.label == "Run").click().run()
    return app


def test_run_fits_every_ticked_model_and_one_failure_does_not_stop_the_rest(
    monkeypatch, stub_ml
):
    def unavailable():
        raise FactorFetchError("OSError: no route to host")

    monkeypatch.setattr(fama_french, "resolve", unavailable)
    app = _run_equity(_page())

    assert not app.exception
    assert set(_stored_runs(app)) == {NAIVE, USER_POLY, "Machine Learning"}
    assert "no route to host" in " ".join(e.value for e in app.error)
    assert len(stub_ml) == 1


def test_the_naive_baseline_runs_every_time_on_the_polynomials_rows(stub_ml):
    app = _press_run(_untick(_page(), "Fama-French 5", "Machine learning"))

    runs = _stored_runs(app)
    assert runs[NAIVE].result.rows_scored == runs[POLY].result.rows_scored
    assert "**Naive (training mean)**: the baseline to beat — not gated." in _markdown(app)
    assert "Naive (training mean)" in _table(app)


def test_an_unticked_model_is_not_run(monkeypatch, stub_ml):
    monkeypatch.setattr(fama_french, "resolve", lambda: pytest.fail("FF5 was unticked"))
    app = _run_equity(_untick(_page(), "Fama-French 5", "Machine learning"))

    assert set(_stored_runs(app)) == {NAIVE, USER_POLY}
    assert not stub_ml


def test_unticking_polynomial_runs_no_polynomial_row(monkeypatch, stub_ml):
    monkeypatch.setattr(fama_french, "resolve", lambda: pytest.fail("FF5 was unticked"))
    app = _press_run(_untick(_page(), "Polynomial", "Fama-French 5"))

    assert not app.exception
    assert set(_stored_runs(app)) == {NAIVE, "Machine Learning"}
    assert not [r for r in app.radio if r.label == "Function source"]


def test_machine_learning_can_run_on_its_own(stub_ml):
    app = _press_run(_untick(_page(), "Polynomial", "Fama-French 5"))

    assert not app.exception
    assert len(stub_ml) == 1
    assert "Machine Learning" in _table(app)


def test_run_is_disabled_with_nothing_but_the_baseline_ticked():
    app = _untick(_page(), "Polynomial", "Fama-French 5", "Machine learning")

    (run,) = [b for b in app.button if b.label == "Run"]
    assert run.disabled
    assert "Tick a model besides the naive baseline to run" in _captions(app)


def test_only_the_equity_tab_offers_ff5():
    app = _page(bond=True)

    labels = [c.label for c in app.checkbox]
    assert labels.count("Fama-French 5") == 1
    assert labels.count("Polynomial") == labels.count("Machine learning") == 2


def test_each_tab_has_its_own_model_checkboxes():
    app = _untick(_page(bond=True), "Polynomial", "Fama-French 5", "Machine learning")

    (equity_run, bond_run) = [b for b in app.button if b.label == "Run"]
    assert equity_run.disabled
    assert not bond_run.disabled


@pytest.mark.parametrize(("source", "ran"), [("derive", POLY), ("own", USER_POLY)])
def test_only_the_chosen_polynomial_runs(stub_ml, source, ran):
    app = _untick(_page(), "Fama-French 5", "Machine learning")
    app = _run_equity(app) if source == "own" else _press_run(app)

    assert set(_stored_runs(app)) == {NAIVE, ran}


def test_ff5_never_runs_on_the_bond_tab(monkeypatch, stub_ml):
    monkeypatch.setattr(fama_french, "resolve", lambda: pytest.fail("FF5 ran for bonds"))
    app = _choose_own(_page(bond=True))
    (_equity_formula, bond_formula) = [t for t in app.text_input if t.label.startswith("Function")]
    bond_formula.set_value("2 * VIX_Index_PX_LAST").run()
    (_equity_run, bond_run) = [b for b in app.button if b.label == "Run"]
    bond_run.click().run()

    assert not app.exception
    assert set(_stored_runs(app, BOND)) == {NAIVE, USER_POLY, "Machine Learning"}


def test_ff5_shows_a_fallback_warning_and_the_factor_coverage(monkeypatch, stub_ml):
    dates = _committed()["Date"]
    resolved = ResolvedFactors(
        FactorFile(_factors(dates[:-10]), SourceFile.of("ff.csv", b"x")),
        warning="Using the saved copy.",
    )
    monkeypatch.setattr(fama_french, "resolve", lambda: resolved)
    app = _run_equity(_page())

    assert not app.exception
    assert "Fama-French 5 · S&P 500" in _expander_labels(app)
    assert "Using the saved copy." in " ".join(w.value for w in app.warning)
    assert "10 target dates have no factor row" in _captions(app)


def test_a_blank_own_function_runs_no_polynomial_and_says_why(stub_ml):
    app = _press_run(_choose_own(_untick(_page(), "Fama-French 5")))

    assert not app.exception
    assert not app.error
    assert set(_stored_runs(app)) == {NAIVE, "Machine Learning"}
    assert "Enter a function for the user-supplied polynomial to run." in _captions(app)


def test_a_broken_function_fails_its_own_row_but_the_rest_still_run(stub_ml):
    app = _run_equity(_untick(_page(), "Fama-French 5"), formula="2 *")

    assert USER_POLY in " ".join(e.value for e in app.error)
    assert set(_stored_runs(app)) == {NAIVE, "Machine Learning"}


def test_ml_results_show_shap_and_the_coarse_pbo_note(stub_ml):
    app = _run_equity(_untick(_page(), "Fama-French 5"))

    assert "Feature attribution (SHAP)" in _markdown(app)
    assert "only two candidates" in _captions(app)


def test_ml_results_show_which_tune_each_fold_used_and_the_latest_settings(stub_ml):
    app = _run_equity(_untick(_page(), "Fama-French 5"))

    tunes = next(d.value for d in app.dataframe if "Tuned on" in d.value.columns)
    assert tunes["Folds"].tolist() == ["1–2", "3"]
    assert tunes["Trials"].tolist() == [50, 20]
    settings = next(d.value for d in app.dataframe if "Setting" in d.value.columns)
    assert dict(zip(settings["Setting"], settings["Value"], strict=True))["max_depth"] == "4"
    assert "latest tune (xgboost)" in _captions(app)


# --- FYP-43 change request: the user's shape, scaled to the target ------------------

VIX, IG = "VIX_Index_PX_LAST", "LUACOAS_Index_PX_LAST"


def _bindings(app: AppTest) -> dict[str, str]:
    return {s.label.split(" ")[0]: s.value for s in app.selectbox if s.label.endswith("stands for")}


def test_each_placeholder_gets_a_signal_dropdown_starting_with_the_first_signals():
    app = _choose_own(_page())
    next(t for t in app.text_input if t.label.startswith("Function")).set_value("x - y ** 2").run()

    assert _bindings(app) == {"x": VIX, "y": IG}


def test_a_placeholder_named_after_a_signal_defaults_to_that_signal():
    app = _choose_own(_page())
    formula = next(t for t in app.text_input if t.label.startswith("Function"))
    formula.set_value(f"{IG} + x").run()

    assert _bindings(app) == {IG: IG, "x": VIX}


def test_the_example_uses_the_first_two_real_signals():
    app = _choose_own(_page())
    formula = next(t for t in app.text_input if t.label.startswith("Function"))

    assert formula.placeholder == "e.g. x - 0.5 * y ** 2"
    assert "x and y would start as VIX and US IG credit spread" in _captions(app)


def test_the_signal_table_lists_every_signal_with_its_transform_and_lag():
    sources = {
        VIX: ColumnSource("VIX Index", "PX_LAST"),
        IG: ColumnSource("LUACOAS Index", "PX_LAST"),
    }
    app = _choose_own(_page(sources=sources))

    table = next(d.value for d in app.dataframe if "Latest value" in d.value.columns)
    assert table["Column"].tolist() == [VIX, IG]
    assert table["Signal"].tolist() == ["VIX", "US IG credit spread"]
    assert table["Security"].tolist() == ["VIX Index", "LUACOAS Index"]
    assert table["Transform"].tolist() == ["level", "level"]
    assert set(table["Lag"]) == {"1 day"}


def test_running_binds_each_placeholder_to_its_chosen_signal(stub_ml):
    app = _untick(_page(), "Fama-French 5", "Machine learning")
    app = _run_equity(app, formula="2 * x + y")
    next(s for s in app.selectbox if s.label == "y stands for").set_value(VIX).run()
    app = _press_run(app)

    description = _stored_runs(app)[USER_POLY].description
    assert description.terms == (f"2 * {VIX} + {VIX}",)
    assert description.intercept is not None


def test_changing_a_placeholders_signal_clears_only_the_user_supplied_row():
    tabs = {EQUITY: {POLY: _polynomial_run(), USER_POLY: _polynomial_run()}}
    app = _choose_own(_page(tabs))
    next(t for t in app.text_input if t.label.startswith("Function")).set_value("x").run()
    app.session_state[model_runs.RUNS_KEY].tabs[EQUITY].runs[USER_POLY] = _polynomial_run()
    next(s for s in app.selectbox if s.label == "x stands for").set_value(IG).run()

    assert set(_stored_runs(app)) == {POLY}


def test_a_user_function_is_shown_as_its_fitted_scale_and_intercept():
    description = ModelDescription("UserPolynomial", (f"{VIX} / {IG}",), (0.0004,), -0.0012)
    fn = from_description(description, origin=Origin.USER_SUPPLIED, target=SPX, horizon=5)
    run = model_runs.ModelRun(_result(pbo=None), description, function=fn)
    app = _page({EQUITY: {USER_POLY: run}})

    (equation,) = [e.value for e in app.latex]
    assert r"-0.001200 + 0.0004000 \cdot (" in equation
    assert "scale and intercept are fitted by least squares" in _captions(app)


# --- Runs fit in the background and are kept --------------------------------------


def _wait_for_runs(timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while model_jobs.active_jobs():
        assert time.time() < deadline, "the background Run never finished"
        time.sleep(0.1)


def test_a_run_fits_in_the_background_and_its_results_appear_when_done(monkeypatch):
    monkeypatch.setattr(model_jobs, "INLINE", False)
    release = threading.Event()

    def slow(panel, splitter):
        release.wait(10)
        return _result(), _ml_run().description, _tuning_log()

    monkeypatch.setattr(boosted, "run_boosted", slow)
    app = _press_run(_untick(_page(), "Polynomial", "Fama-French 5"))

    assert not app.exception
    progress = " ".join(i.value for i in app.info)
    assert "Running in the background for S&P 500" in progress
    assert "**Machine Learning**: fitting" in progress or "waiting" in progress
    assert next(b for b in app.button if b.label == "Run").disabled

    deadline = time.time() + 10
    while model_jobs.job(next(iter(model_jobs._jobs))).status[NAIVE] != model_jobs.DONE:
        assert time.time() < deadline
        time.sleep(0.05)
    app.run()
    assert "Naive (training mean)" in _table(app)  # shown while ML is still fitting
    assert "Machine Learning" not in _table(app)

    release.set()
    _wait_for_runs()
    app.run()

    assert "Machine Learning" in _table(app)
    assert not any("Running in the background" in i.value for i in app.info)


def test_finished_runs_are_there_in_a_new_session_on_the_same_data(stub_ml):
    _press_run(_untick(_page(), "Polynomial", "Fama-French 5"))

    later = _page()  # a new session: nothing stored in it

    assert set(_stored_runs(later)) == {NAIVE, "Machine Learning"}
    assert "Machine Learning" in _table(later)


def test_saved_runs_are_not_shown_under_other_settings(stub_ml):
    _press_run(_untick(_page(), "Polynomial", "Fama-French 5"))

    app = _page()
    (horizon,) = [c for c in app.segmented_control if c.label == "Forecast horizon"]
    horizon.set_value(1).run()

    assert _stored_runs(app) == {}


def test_a_failing_model_reports_its_error_after_a_background_run(monkeypatch):
    def broken(panel, splitter):
        raise ValueError("unexpected")

    monkeypatch.setattr(boosted, "run_boosted", broken)
    app = _press_run(_untick(_page(), "Polynomial", "Fama-French 5"))

    assert "Machine Learning: ValueError: unexpected" in " ".join(e.value for e in app.error)
    assert set(_stored_runs(app)) == {NAIVE}


def test_run_all_targets_queues_every_tabs_ticked_models(stub_ml):
    app = _page(bond=True)
    for box in [c for c in app.checkbox if c.label in ("Polynomial", "Fama-French 5")]:
        box.uncheck()
    app.run()
    next(b for b in app.button if b.label == "Run all targets").click().run()

    assert not app.exception
    assert set(_stored_runs(app, EQUITY)) == {NAIVE, "Machine Learning"}
    assert set(_stored_runs(app, BOND)) == {NAIVE, "Machine Learning"}
    assert len(stub_ml) == 2


def test_a_new_window_can_pick_up_the_data_committed_last_time(stub_ml):
    _press_run(_untick(_page(), "Polynomial", "Fama-French 5"))  # commits nothing itself
    committed = _committed()
    bundle = {
        "extraction_committed": committed,
        "extraction_committed_targets": {EQUITY: SPX},
    }
    path = Path("data") / "last_commit.pkl"
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(pickle.dumps(bundle))

    app = AppTest.from_file(str(PAGE), default_timeout=60).run()  # a new window
    assert "committed in an earlier window" in " ".join(i.value for i in app.info)
    next(b for b in app.button if b.label == "Use the data committed last time").click().run()

    assert not app.exception
    assert "Machine Learning" in _table(app)


def test_run_all_targets_is_only_offered_with_more_than_one_target():
    app = _page()

    assert not [b for b in app.button if b.label == "Run all targets"]


# --- FYP-161: one run, one set of settings, stated beside the results ---------------


def test_the_results_state_the_settings_every_row_was_run_under():
    captions = _captions(_page({EQUITY: {POLY: _polynomial_run()}}))

    assert (
        "Every row was run under: 5-day horizon · walk-forward train 120 / test 20 days · "
        "embargo 5 days · signals lagged 1 day"
    ) in captions


def test_the_derived_and_user_supplied_polynomials_get_a_row_each():
    app = _page({EQUITY: {POLY: _polynomial_run(), USER_POLY: _polynomial_run()}})

    table = _table(app)
    assert table.count("<tr>") == 3  # header + two models
    assert POLY in table and USER_POLY in table
    labels = _expander_labels(app)
    assert f"{POLY} · S&P 500" in labels
    assert f"{USER_POLY} · S&P 500" in labels


def test_a_user_supplied_function_can_be_set_active():
    app = _page({EQUITY: {POLY: _polynomial_run(), USER_POLY: _polynomial_run()}})

    (select,) = [s for s in app.selectbox if s.label == "Model to set active"]
    assert list(select.options) == [POLY, USER_POLY]


def test_a_shared_change_says_why_the_results_went():
    app = _page(_two_tabs(), bond=True)
    (train,) = [n for n in app.number_input if n.label.startswith("Walk-forward train")]
    train.set_value(130).run()

    assert "were cleared because the data or a shared setting changed" in _captions(app)


def test_a_fresh_page_does_not_claim_anything_was_cleared():
    assert "were cleared" not in _captions(_page())


# --- FYP-162: would the forecast's direction have paid? ------------------------------


def _oos(values, start="2024-06-03") -> pd.Series:
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def _with_forecast(forecast: pd.Series, realised: pd.Series, **kwargs) -> ModelRunResult:
    return replace(_result(**kwargs), forecast=forecast, realised=realised)


def _directional_page(**kwargs) -> AppTest:
    forecast = _oos([0.01, -0.01] * 50)
    realised = _oos([0.02, -0.03] * 50)
    return _page(
        {EQUITY: {POLY: _polynomial_run(_with_forecast(forecast, realised, **kwargs))}}
    )


def _metric(app: AppTest, label: str) -> str:
    return next(m.value for m in app.metric if m.label == label)


def test_the_directional_check_shows_strategy_buy_and_hold_hit_rate_and_days_invested():
    app = _directional_page()

    assert not app.exception
    assert "Would the forecast's direction have paid?" in _markdown(app)
    # Calls alternate rise/fall and the forecast is right every time: in on each of
    # the 10 rises (+2%), out on each of the 10 falls (-3%).
    assert _metric(app, "Hit rate") == "100%"
    assert _metric(app, "Days invested") == "50%"
    assert _metric(app, "Long/cash strategy") == f"{1.02**10 - 1:+.2%}"
    assert _metric(app, "Buy and hold") == f"{1.02**10 * 0.97**10 - 1:+.2%}"


def test_the_directional_check_steps_five_days_at_a_time_at_h5():
    app = _directional_page()

    captions = _captions(app)
    assert "all 100 out-of-sample trading days, 20 calls" in captions
    assert "one call every 5 days, never overlapping" in captions
    assert "Gross of transaction costs" in captions


def test_the_directional_check_replays_the_whole_period_with_no_window_to_set():
    app = _directional_page()

    assert not [n for n in app.number_input if "Window" in n.label]
    captions = _captions(app)
    assert "all 100 out-of-sample trading days" in captions
    assert "scroll to zoom, drag to pan and double-click to reset" in captions


def test_a_window_picked_with_the_slider_recomputes_every_figure():
    app = _directional_page()
    (window,) = [s for s in app.slider if s.label == "Window"]
    dates = pd.bdate_range("2024-06-03", periods=100)
    # The whole period by default: from the first 5-day step to the last.
    assert window.value == (dates[0].date(), dates[95].date())

    window.set_value((dates[40].date(), dates[59].date())).run()

    assert not app.exception
    captions = _captions(app)
    assert "the chosen window of 20 out-of-sample trading days, 4 calls" in captions
    assert _metric(app, "Hit rate") == "100%"


def test_the_directional_check_says_how_to_see_the_other_horizon():
    assert "switch to 1 day in Settings and run again" in _captions(_directional_page())


def test_the_directional_check_defaults_to_the_active_model():
    forecast, realised = _oos([0.01] * 30), _oos([0.01] * 30)
    runs = {
        POLY: _polynomial_run(_with_forecast(forecast, realised)),
        "Machine Learning": _ml_run(_with_forecast(forecast, realised)),
    }
    active_model.set_active_model(
        EQUITY,
        "Machine Learning",
        runs["Machine Learning"].result,
        high_risk=False,
        horizon=5,
        train_window=120,
        test_window=20,
        dataset_fingerprint="fp",
    )
    app = _page({EQUITY: runs})

    (picker,) = [s for s in app.selectbox if s.label == "Forecasts from"]
    assert picker.value == "Machine Learning"


def test_a_run_without_saved_forecasts_shows_no_directional_check():
    app = _page({EQUITY: {POLY: _polynomial_run()}})

    assert "Would the forecast's direction have paid?" not in _markdown(app)
    assert not [s for s in app.selectbox if s.label == "Forecasts from"]
