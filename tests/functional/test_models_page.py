"""The Models page, driven through the real Streamlit page script.

Most of these render stored runs the way the page shows a previous result,
rather than fitting one: a derived fit is slow and nondeterministic, and what's
under test here is what the page shows, not the fitting. Tests that press Run
stub the machine-learning fit and the factor download.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

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
from forecasting_engine.validation import splitters
from forecasting_engine.validation.crash import CrashDiagnostics

REPO_ROOT = Path(__file__).resolve().parents[2]
PAGE = REPO_ROOT / "app" / "app_pages" / "3_Models.py"

EQUITY, BOND = TargetRole.EQUITY, TargetRole.BOND
SPX = "SPX_Index_PX_LAST"
AGG = "LBUSTRUU_Index_TOT_RETURN_INDEX_GROSS_DVDS"
#: The page's defaults: 5-day horizon, 120/20 walk-forward windows, and a
#: blank user-supplied function.
DEFAULT_SHARED = (5, 120, 20)
DEFAULT_POLYNOMIAL = ("Enter a function", "")


@pytest.fixture(autouse=True)
def short_tuning_period(monkeypatch):
    """The fixture data is 200 rows, far short of the real 504-row tuning period."""
    monkeypatch.setattr(splitters, "TUNING_ROWS", 40)


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
            stored.tabs[role] = model_runs.TabRuns(DEFAULT_POLYNOMIAL, dict(runs))
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


def test_the_model_checkboxes_default_to_every_family_with_polynomial_always_on():
    app = _page()

    boxes = {c.label: c for c in app.checkbox}
    assert boxes["Polynomial"].value and boxes["Polynomial"].disabled
    assert boxes["Fama-French 5"].value
    assert boxes["Machine learning"].value


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
        "VIX_Index_PX_LAST": "difference",
        "LUACOAS_Index_PX_LAST": "difference",
    }


# --- results: only what ran, named for its target -------------------------------


def test_nothing_run_yet_shows_no_results():
    app = _page()

    assert not app.exception
    assert "Nothing has run for S&P 500" in _captions(app)
    assert not _has_table(app)


def test_results_render_only_for_the_models_that_ran():
    app = _page({EQUITY: {"Polynomial": _polynomial_run()}})

    assert not app.exception
    table = _table(app)
    assert table.count("<tr>") == 2  # header + one model
    assert "Polynomial" in table and "Machine Learning" not in table
    labels = _expander_labels(app)
    assert "Polynomial · S&P 500" in labels
    assert not [e for e in labels if e.startswith(("Fama-French 5", "Machine learning"))]


def test_every_results_header_names_the_target():
    app = _page({EQUITY: {"Polynomial": _polynomial_run(), "Machine Learning": _ml_run()}})

    assert "Results · S&P 500" in [s.value for s in app.subheader]
    for heading in ("Signal screening", "Polynomial", "Machine learning"):
        assert f"{heading} · S&P 500" in _expander_labels(app)


def test_the_table_shows_rows_scored():
    table = _table(_page({EQUITY: {"Polynomial": _polynomial_run()}}))
    assert "Rows scored" in table
    assert "<td>180</td>" in table


def test_each_model_gets_a_one_line_gate_summary():
    failing = _result(oos_rank_ic=0.01, pbo=0.7)
    benchmark = model_runs.ModelRun(
        _result(pbo=None),
        ModelDescription("FamaFrench5", FACTOR_COLUMNS, (0.1,) * 5),
        coverage=FactorCoverage(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-08-30"), 150, 5),
    )
    app = _page({EQUITY: {"Polynomial": _polynomial_run(failing), "FF5 Benchmark": benchmark}})

    text = _markdown(app)
    assert "**Polynomial**: gate failed on OOS Rank IC and PBO." in text
    assert "**FF5 Benchmark**: not gated" in text


def test_each_targets_results_stay_in_its_own_tab():
    app = _page({EQUITY: {"Polynomial": _polynomial_run()}}, bond=True)

    assert not app.exception
    assert "Results · S&P 500" in [s.value for s in app.subheader]
    assert "Nothing has run for US Aggregate Bond" in _captions(app)


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
    app = _page({EQUITY: {"Polynomial": _polynomial_run(_result(screened))}})

    table = _screening_table(app).set_index("Signal")
    assert table.loc["VIX_Index_PX_LAST", "Transform"] == "difference"
    assert table.loc["VIX_Index_PX_LAST", "Latest fold"] == "In"
    assert table.loc["LUACOAS_Index_PX_LAST", "Latest fold"] == "Out"
    assert table.loc["VIX_Index_PX_LAST", "Latest-fold IC"] == pytest.approx(0.05)
    assert table.loc["LUACOAS_Index_PX_LAST", "Included in"] == "0 of 8 folds"


def test_screening_is_taken_from_ml_when_the_polynomial_did_not_screen(screened):
    app = _page(
        {EQUITY: {"Polynomial": _polynomial_run(), "Machine Learning": _ml_run(_result(screened))}}
    )
    assert _screening_table(app)["Signal"].tolist() == [
        "VIX_Index_PX_LAST",
        "LUACOAS_Index_PX_LAST",
    ]


def test_folds_that_fell_back_to_every_signal_are_called_out():
    screening = ScreeningSummary(folds=8, fell_back=3, counts=(("VIX_Index_PX_LAST", 8),))
    app = _page({EQUITY: {"Polynomial": _polynomial_run(_result(screening))}})
    assert "3 of 8 folds kept no signal" in _captions(app)


def test_nothing_screened_says_so():
    app = _page({EQUITY: {"Polynomial": _polynomial_run()}})

    assert "Nothing was screened" in _captions(app)
    assert all("Included in" not in d.value.columns for d in app.dataframe)


# --- the fitted polynomial as a labelled function --------------------------------


def _terms_table(app: AppTest) -> pd.DataFrame:
    return next(d.value for d in app.dataframe if "Exponent" in d.value.columns)


def test_the_function_says_what_it_forecasts():
    app = _page({EQUITY: {"Polynomial": _polynomial_run()}})
    assert "Forecasts: S&P 500, 5-day return" in _captions(app)
    assert "Derived Function" in _markdown(app)


def test_the_equation_is_typeset_with_labels():
    (latex,) = [lx.value for lx in _page({EQUITY: {"Polynomial": _polynomial_run()}}).latex]
    assert r"\hat{y} = 0.001234 + 0.0004521" in latex
    assert r"\text{VIX}^{2}" in latex


def test_the_term_table_uses_labels_not_raw_column_codes():
    table = _terms_table(_page({EQUITY: {"Polynomial": _polynomial_run()}}))

    assert table["Factor"].tolist() == ["(intercept)", "VIX", "US IG credit spread × VIX"]
    assert table["Coefficient"].tolist() == ["0.001234", "0.0004521", "−0.0002310"]


def test_a_run_whose_folds_mostly_kept_nothing_says_so():
    run = _polynomial_run(_result(terms=FoldTerms(folds=10, with_terms=4)))
    captions = _captions(_page({EQUITY: {"Polynomial": run}}))

    assert "4 of 10 walk-forward folds kept any term at all" in captions
    assert "most recent fold's fit" in captions


def test_a_run_where_every_fold_kept_terms_says_that_instead():
    run = _polynomial_run(_result(terms=FoldTerms(folds=10, with_terms=10)))
    captions = _captions(_page({EQUITY: {"Polynomial": run}}))

    assert "Every one of the 10 walk-forward folds kept at least one term." in captions


def test_an_empty_equation_still_explains_itself():
    empty = ModelDescription(name="DerivedPolynomial", terms=(), coefficients=(), intercept=0.002)
    run = _polynomial_run(_result(terms=FoldTerms(folds=10, with_terms=3)), empty)
    captions = _captions(_page({EQUITY: {"Polynomial": run}}))

    assert "No terms survived fitting" in captions
    assert "3 of 10 walk-forward folds kept any term at all" in captions


# --- clearing: results always match the settings shown ---------------------------


def _two_tabs() -> dict:
    return {EQUITY: {"Polynomial": _polynomial_run()}, BOND: {"Polynomial": _polynomial_run()}}


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


def test_changing_one_tabs_polynomial_settings_clears_only_that_tab():
    app = _page(_two_tabs(), bond=True)
    (formula, _bond_formula) = [t for t in app.text_input if t.label.startswith("Function")]
    formula.set_value("2 * VIX_Index_PX_LAST").run()

    assert _stored_runs(app, EQUITY) == {}
    assert "Polynomial" in _stored_runs(app, BOND)


def test_ticking_or_unticking_a_model_clears_nothing():
    app = _page(_two_tabs(), bond=True)
    (ml,) = [c for c in app.checkbox if c.label == "Machine learning"]
    ml.uncheck().run()

    assert "Polynomial" in _stored_runs(app, EQUITY)
    assert "Polynomial" in _stored_runs(app, BOND)


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


def _untick(app: AppTest, *labels: str) -> AppTest:
    for label in labels:
        next(c for c in app.checkbox if c.label == label).uncheck()
    return app.run()


def _run_equity(app: AppTest, formula: str = "2 * VIX_Index_PX_LAST") -> AppTest:
    next(t for t in app.text_input if t.label.startswith("Function")).set_value(formula).run()
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
    assert set(_stored_runs(app)) == {"Polynomial", "Machine Learning"}
    assert "no route to host" in " ".join(e.value for e in app.error)
    assert len(stub_ml) == 1


def test_an_unticked_model_is_not_run(monkeypatch, stub_ml):
    monkeypatch.setattr(fama_french, "resolve", lambda: pytest.fail("FF5 was unticked"))
    app = _run_equity(_untick(_page(), "Fama-French 5", "Machine learning"))

    assert set(_stored_runs(app)) == {"Polynomial"}
    assert not stub_ml


def test_ff5_never_runs_on_the_bond_tab(monkeypatch, stub_ml):
    monkeypatch.setattr(fama_french, "resolve", lambda: pytest.fail("FF5 ran for bonds"))
    app = _page(bond=True)
    (_equity_formula, bond_formula) = [t for t in app.text_input if t.label.startswith("Function")]
    bond_formula.set_value("2 * VIX_Index_PX_LAST").run()
    (_equity_run, bond_run) = [b for b in app.button if b.label == "Run"]
    bond_run.click().run()

    assert not app.exception
    assert set(_stored_runs(app, BOND)) == {"Polynomial", "Machine Learning"}


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


def test_a_blank_function_fails_the_polynomial_but_the_rest_still_run(monkeypatch, stub_ml):
    app = _untick(_page(), "Fama-French 5")
    next(b for b in app.button if b.label == "Run").click().run()

    assert "enter a function above first" in " ".join(e.value for e in app.error)
    assert set(_stored_runs(app)) == {"Machine Learning"}


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
