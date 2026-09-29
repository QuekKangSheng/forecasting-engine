"""The Models page, driven through the real Streamlit page script.

These render a stored run the way the page shows a previous result, rather than
fitting one: a derived fit is slow and nondeterministic, and what's under test
here is what the page shows, not the fitting.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.models.base import ModelDescription
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
from forecasting_engine.validation.crash import CrashDiagnostics

REPO_ROOT = Path(__file__).resolve().parents[2]
PAGE = REPO_ROOT / "app" / "app_pages" / "3_Models.py"


def _committed() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame(
        {
            "Date": pd.bdate_range("2024-01-01", periods=n),
            "SPX_Index_PX_LAST": 4000 + np.cumsum(rng.normal(size=n)),
            "VIX_Index_PX_LAST": 15 + rng.normal(size=n),
            "LUACOAS_Index_PX_LAST": 1.2 + rng.normal(scale=0.1, size=n),
        }
    )


#: The committed data's target, resolved the way the page requires (Task
#: 4.2) — every helper below seeds this rather than a free column pick.
_TARGETS = {TargetRole.EQUITY: "SPX_Index_PX_LAST"}


def _result(screening: ScreeningSummary | None) -> ModelRunResult:
    return ModelRunResult(
        ic=0.05,
        oos_rank_ic=0.04,
        rmse=0.01,
        pbo=0.3,
        crash=CrashDiagnostics(recall=0.5, precision=0.5, f1=0.5, n_true_tail_days=4),
        screening=screening,
    )


def _page(screening: ScreeningSummary | None) -> AppTest:
    app = AppTest.from_file(str(PAGE), default_timeout=30)
    app.session_state["extraction_committed"] = _committed()
    app.session_state["extraction_committed_targets"] = _TARGETS
    app.session_state["polynomial_result_equity"] = _result(screening)
    app.session_state["polynomial_description_equity"] = ModelDescription(
        name="Derived polynomial", terms=("VIX_Index_PX_LAST",), coefficients=(0.2,)
    )
    return app.run()


def _markdown(app: AppTest) -> str:
    return " ".join(m.value for m in app.markdown)


def _captions(app: AppTest) -> str:
    return " ".join(c.value for c in app.caption)


@pytest.fixture
def screened() -> ScreeningSummary:
    return ScreeningSummary(
        folds=8,
        fell_back=0,
        counts=(("VIX_Index_PX_LAST", 8), ("LUACOAS_Index_PX_LAST", 0)),
    )


def test_a_screened_run_shows_per_fold_signal_inclusion(screened):
    app = _page(screened)

    assert not app.exception
    assert "Signal inclusion across folds" in _markdown(app)


def test_each_signal_is_shown_against_the_number_of_folds(screened):
    app = _page(screened)

    tables = [d.value for d in app.dataframe]
    inclusion = next(t for t in tables if "Folds" in t.columns)
    assert dict(zip(inclusion["Signal"], inclusion["Folds"], strict=True)) == {
        "VIX_Index_PX_LAST": "8/8",
        "LUACOAS_Index_PX_LAST": "0/8",
    }


def test_a_signal_no_fold_used_is_still_shown(screened):
    app = _page(screened)

    inclusion = next(d.value for d in app.dataframe if "Folds" in d.value.columns)
    assert "0/8" in inclusion["Folds"].tolist()


def test_folds_that_fell_back_to_every_signal_are_called_out():
    app = _page(ScreeningSummary(folds=8, fell_back=3, counts=(("VIX_Index_PX_LAST", 8),)))
    assert "3 of 8 folds kept no signal" in _captions(app)


def test_no_fallback_note_when_no_fold_fell_back(screened):
    app = _page(screened)
    assert "kept no signal" not in _captions(app)


def test_a_run_without_screening_shows_no_inclusion_table():
    # FF5 and a user-supplied formula are handed their features; there is
    # nothing per-fold to report.
    app = _page(None)

    assert not app.exception
    assert "Signal inclusion across folds" not in _markdown(app)
    assert all("Folds" not in d.value.columns for d in app.dataframe)


# --- Phase 4.3: the horizon, the embargo, and where the lag lives -----------


@pytest.fixture
def splitter_calls(monkeypatch):
    """Record the arguments PurgedWalkForward is actually built with.

    The page imports the name when its script runs, so patching the module
    attribute beforehand is what the page picks up.
    """
    import forecasting_engine.validation.splitters as splitters

    calls: list[dict] = []
    real = splitters.PurgedWalkForward

    class Recording(real):
        def __init__(self, train: int, test: int, embargo: int):
            calls.append({"train": train, "test": test, "embargo": embargo})
            super().__init__(train, test, embargo)

    monkeypatch.setattr(splitters, "PurgedWalkForward", Recording)
    return calls


def _bare_page() -> AppTest:
    app = AppTest.from_file(str(PAGE), default_timeout=30)
    app.session_state["extraction_committed"] = _committed()
    app.session_state["extraction_committed_targets"] = _TARGETS
    return app.run()


def test_the_horizon_is_a_choice_of_exactly_one_or_five_days():
    app = _bare_page()

    (horizon,) = [c for c in app.segmented_control if c.label == "Forecast horizon"]
    assert list(horizon.options) == ["1 day", "5 days"]


def test_the_horizon_defaults_to_five_days():
    app = _bare_page()
    (horizon,) = [c for c in app.segmented_control if c.label == "Forecast horizon"]
    assert horizon.value == 5


def test_the_free_numeric_horizon_input_is_gone():
    app = _bare_page()
    assert all("horizon" not in n.label.lower() for n in app.number_input)


@pytest.mark.parametrize("chosen", [1, 5])
def test_the_embargo_is_five_whichever_horizon_is_chosen(splitter_calls, chosen):
    # Before, picking h=1 silently dropped the embargo to 1 as well.
    app = _bare_page()
    (horizon,) = [c for c in app.segmented_control if c.label == "Forecast horizon"]
    horizon.set_value(chosen)
    app.run()

    assert not app.exception
    assert splitter_calls, "the page must build a splitter"
    assert splitter_calls[-1]["embargo"] == 5


def test_the_embargo_is_no_longer_an_editable_input():
    app = _bare_page()
    assert all("embargo" not in n.label.lower() for n in app.number_input)


def test_the_signal_lag_is_not_an_input():
    app = _bare_page()
    assert all("lag" not in n.label.lower() for n in app.number_input)
    assert all("lag" not in e.label.lower() for e in app.expander)


# --- the fitted polynomial as a labelled function ----------------------------

DERIVED = ModelDescription(
    name="DerivedPolynomial",
    terms=("VIX_Index_PX_LAST^2", "LUACOAS_Index_PX_LAST VIX_Index_PX_LAST"),
    coefficients=(0.0004521, -0.000231),
    intercept=0.001234,
)


def _function_page(
    *, target: str = "SPX_Index_PX_LAST", horizon: int = 5, frame: pd.DataFrame | None = None
) -> AppTest:
    """A page holding a stored derived function, fitted for ``target``/``horizon`` on
    ``frame`` (the committed data, by default)."""
    committed = _committed()
    fn = from_description(DERIVED, origin=Origin.DERIVED, target=target, horizon=horizon)
    app = AppTest.from_file(str(PAGE), default_timeout=30)
    app.session_state["extraction_committed"] = committed
    app.session_state["extraction_committed_targets"] = _TARGETS
    app.session_state["polynomial_result_equity"] = _result(None)
    app.session_state["polynomial_description_equity"] = DERIVED
    app.session_state["polynomial_function_equity"] = (
        fn,
        dataset_fingerprint(committed if frame is None else frame),
    )
    return app.run()


def _terms_table(app: AppTest) -> pd.DataFrame:
    return next(d.value for d in app.dataframe if "Exponent" in d.value.columns)


def test_a_stored_derived_function_is_headed_as_derived():
    app = _function_page()

    assert not app.exception
    assert [s.value for s in app.subheader].count("Derived Function") == 1


def test_the_function_says_what_it_forecasts():
    assert "Forecasts: S&P 500, 5-day return" in _captions(_function_page())


def test_no_target_resolved_shows_a_guiding_message_not_a_free_picker():
    app = AppTest.from_file(str(PAGE), default_timeout=30)
    app.session_state["extraction_committed"] = _committed()
    # No "extraction_committed_targets" at all — nothing resolved on the Data page.
    app.run()

    assert not app.exception
    assert any("No target resolved yet" in i.value for i in app.info)
    assert not app.radio  # no "Target" picker, and no "Model family" either


def test_only_the_resolved_targets_are_offered():
    app = AppTest.from_file(str(PAGE), default_timeout=30)
    app.session_state["extraction_committed"] = _committed()
    app.session_state["extraction_committed_targets"] = {TargetRole.EQUITY: "SPX_Index_PX_LAST"}
    app.run()

    (target,) = [r for r in app.radio if r.label == "Target"]
    assert list(target.options) == ["Equity — S&P 500 (total return)"]


def test_fama_french_is_not_offered_for_the_bond_target():
    app = AppTest.from_file(str(PAGE), default_timeout=30)
    app.session_state["extraction_committed"] = _committed().rename(
        columns={"SPX_Index_PX_LAST": "LBUSTRUU_Index_TOT_RETURN_INDEX_GROSS_DVDS"}
    )
    app.session_state["extraction_committed_targets"] = {
        TargetRole.BOND: "LBUSTRUU_Index_TOT_RETURN_INDEX_GROSS_DVDS"
    }
    app.run()

    (family,) = [r for r in app.radio if r.label == "Model family"]
    assert "Fama-French 5-Factor" not in family.options


def test_a_target_column_never_appears_as_a_signal_even_for_the_other_role():
    # Both targets present: selecting Equity must not let the Bond column
    # (or vice versa) ride along as a candidate signal.
    committed = _committed().rename(
        columns={"LUACOAS_Index_PX_LAST": "LBUSTRUU_Index_TOT_RETURN_INDEX_GROSS_DVDS"}
    )
    app = AppTest.from_file(str(PAGE), default_timeout=30)
    app.session_state["extraction_committed"] = committed
    app.session_state["extraction_committed_targets"] = {
        TargetRole.EQUITY: "SPX_Index_PX_LAST",
        TargetRole.BOND: "LBUSTRUU_Index_TOT_RETURN_INDEX_GROSS_DVDS",
    }
    app.run()

    assert not app.exception
    (mode,) = [r for r in app.radio if r.label == "Function source"]
    # "Enter a function" is the default; the caption right above the formula
    # box names every column offered as a signal.
    captions = " ".join(c.value for c in app.caption)
    assert "LBUSTRUU_Index_TOT_RETURN_INDEX_GROSS_DVDS" not in captions
    assert "SPX_Index_PX_LAST" not in captions


def test_switching_target_does_not_clobber_the_other_targets_result():
    committed = _committed().rename(
        columns={"LUACOAS_Index_PX_LAST": "LBUSTRUU_Index_TOT_RETURN_INDEX_GROSS_DVDS"}
    )
    app = AppTest.from_file(str(PAGE), default_timeout=30)
    app.session_state["extraction_committed"] = committed
    app.session_state["extraction_committed_targets"] = {
        TargetRole.EQUITY: "SPX_Index_PX_LAST",
        TargetRole.BOND: "LBUSTRUU_Index_TOT_RETURN_INDEX_GROSS_DVDS",
    }
    app.session_state["polynomial_result_equity"] = _result(None)
    app.session_state["polynomial_description_equity"] = ModelDescription(
        name="Derived polynomial", terms=("VIX_Index_PX_LAST",), coefficients=(0.2,)
    )
    app.run()

    # Nothing about loading the Bond side of the page should touch the
    # Equity result parked under its own namespaced key.
    assert app.session_state["polynomial_result_equity"] is not None
    assert "polynomial_result_bond" not in app.session_state


def test_the_equation_is_typeset_with_labels():
    (latex,) = [lx.value for lx in _function_page().latex]
    assert r"\hat{y} = 0.001234 + 0.0004521" in latex
    assert r"\text{VIX}^{2}" in latex


def test_the_term_table_uses_labels_not_raw_column_codes():
    table = _terms_table(_function_page())

    assert table["Factor"].tolist() == ["(intercept)", "VIX", "US IG credit spread × VIX"]
    assert table["Coefficient"].tolist() == ["0.001234", "0.0004521", "−0.0002310"]
    assert not table.astype(str).apply(lambda col: col.str.contains("_Index_")).any().any()


def test_the_raw_fitted_terms_table_is_replaced():
    app = _function_page()
    assert "Fitted terms" not in _markdown(app)
    assert all("Term" not in d.value.columns for d in app.dataframe)


def test_a_current_function_has_no_stale_warning():
    assert not _function_page().warning


def test_a_function_for_another_horizon_is_flagged_stale():
    (warning,) = _function_page(horizon=1).warning
    assert warning.value == (
        "Fitted for S&P 500, 1-day return. The inputs above have changed, so run again "
        "to update."
    )


def test_a_function_for_another_target_is_flagged_stale():
    (warning,) = _function_page(target="VIX_Index_PX_LAST").warning
    assert warning.value.startswith("Fitted for VIX, 5-day return.")


def test_a_function_fitted_on_other_data_is_flagged_stale_first():
    other = _committed().iloc[:-1]
    (warning,) = _function_page(horizon=1, frame=other).warning
    assert warning.value == "Fitted on a previous dataset. Run again to update."


def test_a_result_without_a_stored_function_still_renders():
    # A session from before this change holds a result but no function.
    app = _page(None)

    assert not app.exception
    assert app.metric
    assert "Derived Function" not in [s.value for s in app.subheader]


def test_applying_a_formula_replaces_the_stored_derived_function():
    app = _function_page()
    (mode,) = [r for r in app.radio if r.label == "Function source"]
    mode.set_value("Enter a function").run()
    (formula,) = [t for t in app.text_input if t.label.startswith("Function")]
    formula.set_value("2 * VIX_Index_PX_LAST + LUACOAS_Index_PX_LAST ** 2").run()
    (apply,) = [b for b in app.button if b.label == "Apply"]
    apply.click().run()

    assert not app.exception
    assert not app.error
    headings = [s.value for s in app.subheader]
    assert "User-Supplied Function" in headings
    assert "Derived Function" not in headings
    assert _terms_table(app)["Factor"].tolist() == ["VIX", "US IG credit spread"]
    assert not app.warning


# --- how typical the shown equation is of the whole run ----------------------


def _page_with_terms(with_terms: int, folds: int = 10, description=DERIVED) -> AppTest:
    committed = _committed()
    app = AppTest.from_file(str(PAGE), default_timeout=30)
    app.session_state["extraction_committed"] = committed
    app.session_state["extraction_committed_targets"] = _TARGETS
    app.session_state["polynomial_result_equity"] = ModelRunResult(
        ic=0.05,
        oos_rank_ic=0.04,
        rmse=0.01,
        pbo=0.3,
        crash=CrashDiagnostics(recall=0.5, precision=0.5, f1=0.5, n_true_tail_days=4),
        terms=FoldTerms(folds=folds, with_terms=with_terms),
    )
    app.session_state["polynomial_description_equity"] = description
    app.session_state["polynomial_function_equity"] = (
        from_description(
            description, origin=Origin.DERIVED, target="SPX_Index_PX_LAST", horizon=5
        ),
        dataset_fingerprint(committed),
    )
    return app.run()


def test_a_run_whose_folds_mostly_kept_nothing_says_so():
    app = _page_with_terms(with_terms=4, folds=10)

    assert not app.exception
    assert "4 of 10 walk-forward folds kept any term at all" in _captions(app)
    assert "most recent fold's fit" in _captions(app)


def test_a_run_where_every_fold_kept_terms_says_that_instead():
    app = _page_with_terms(with_terms=10, folds=10)

    assert "Every one of the 10 walk-forward folds kept at least one term." in _captions(app)
    assert "kept any term at all" not in _captions(app)


def test_an_older_result_without_the_count_still_renders():
    # A result parked in session state before this field existed.
    app = _page(None)

    assert not app.exception
    assert "walk-forward folds kept" not in _captions(app)


def test_an_empty_equation_still_explains_itself():
    empty = ModelDescription(
        name="DerivedPolynomial", terms=(), coefficients=(), intercept=0.002
    )
    app = _page_with_terms(with_terms=3, folds=10, description=empty)

    assert "No terms survived fitting" in _captions(app)
    assert "3 of 10 walk-forward folds kept any term at all" in _captions(app)
