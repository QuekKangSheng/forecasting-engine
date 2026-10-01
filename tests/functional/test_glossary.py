"""Every piece of jargon on the dashboard explains itself on hover."""

import html
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import glossary
import model_runs
from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.models.base import ModelDescription
from forecasting_engine.reporting.model_metrics import ModelRunResult, ScreeningSummary
from forecasting_engine.reporting.polynomial_function import (
    Origin,
    dataset_fingerprint,
    from_description,
)
from forecasting_engine.validation.crash import CrashDiagnostics

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_PAGE = REPO_ROOT / "app" / "app_pages" / "3_Models.py"


def _committed() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame(
        {
            "Date": pd.bdate_range("2024-01-01", periods=n),
            "SPX_Index_PX_LAST": 4000 + np.cumsum(rng.normal(size=n)),
            "VIX_Index_PX_LAST": 15 + rng.normal(size=n),
        }
    )


def _result() -> ModelRunResult:
    return ModelRunResult(
        ic=0.05,
        oos_rank_ic=0.04,
        rmse=0.01,
        pbo=0.3,
        crash=CrashDiagnostics(recall=0.5, precision=0.5, f1=0.5, n_true_tail_days=4),
        screening=ScreeningSummary(folds=4, fell_back=0, counts=(("VIX_Index_PX_LAST", 4),)),
    )


@pytest.fixture
def models_page() -> AppTest:
    description = ModelDescription(
        name="DerivedPolynomial",
        terms=("VIX_Index_PX_LAST^2",),
        coefficients=(0.0004521,),
        intercept=0.001234,
    )
    committed = _committed()
    fn = from_description(description, origin=Origin.DERIVED, target="SPX_Index_PX_LAST", horizon=5)
    stored = model_runs.StoredRuns((dataset_fingerprint(committed), 5, 120, 20))
    stored.tabs[TargetRole.EQUITY] = model_runs.TabRuns(
        ("Enter a function", ""),
        {"Polynomial": model_runs.ModelRun(_result(), description, function=fn)},
    )
    app = AppTest.from_file(str(MODELS_PAGE), default_timeout=30)
    app.session_state["extraction_committed"] = committed
    app.session_state["extraction_committed_targets"] = {
        TargetRole.EQUITY: "SPX_Index_PX_LAST"
    }
    app.session_state[model_runs.RUNS_KEY] = stored
    return app.run()


def _helps(app: AppTest) -> dict[str, str]:
    """Every widget/heading/caption on the page that carries hover help."""
    labelled = [
        *((w.label, w.help) for w in app.selectbox),
        *((w.label, w.help) for w in app.radio),
        *((w.label, w.help) for w in app.number_input),
        *((w.label, w.help) for w in app.metric),
        *((w.label, w.help) for w in app.segmented_control),
        *((w.value, w.help) for w in app.subheader),
        *((w.value, w.help) for w in app.caption),
    ]
    return {label: help_text for label, help_text in labelled if help_text}


@pytest.mark.parametrize("label", ["Forecast horizon", "Function source"])
def test_the_controls_that_name_a_concept_explain_it(models_page, label):
    assert label in _helps(models_page)


@pytest.mark.parametrize("metric", ["IC", "OOS Rank IC", "RMSE", "PBO"])
def test_every_headline_metric_explains_itself(models_page, metric):
    hint = html.escape(glossary.term(metric), quote=True)
    assert f'<th title="{hint}">{metric}' in _table(models_page)
    assert len(glossary.term(metric)) > 80, "a tooltip should say what it is and why it matters"


def test_the_walk_forward_windows_explain_what_walk_forward_means(models_page):
    helps = _helps(models_page)
    assert "Walk-forward train window (days)" in helps
    assert "Walk-forward test window (days)" in helps


def test_the_embargo_caption_explains_the_leak_it_prevents(models_page):
    (embargo,) = [h for label, h in _helps(models_page).items() if label.startswith("Embargo")]
    assert "leak" in embargo


def test_the_signal_lag_caption_explains_itself(models_page):
    helps = _helps(models_page)
    assert glossary.term("Signal lag") in helps.values()


def test_crash_diagnostics_explain_recall_and_precision(models_page):
    hint = html.escape(glossary.term("Crash diagnostics"), quote=True)
    assert f'<th title="{hint}">Crash Recall' in _table(models_page)
    assert "precision" in glossary.term("Crash diagnostics")


def test_the_fitted_function_heading_explains_what_it_shows(models_page):
    hint = html.escape(glossary.term("Fitted terms"), quote=True)
    assert any("Derived Function" in m.value and hint in m.value for m in models_page.markdown)


def test_the_screening_caption_explains_itself(models_page):
    assert glossary.term("Signal inclusion across folds") in _helps(models_page).values()


def _table(app: AppTest) -> str:
    return next(m.value for m in app.markdown if '<table class="fe-table"' in m.value)


def test_the_comparison_table_explains_every_column_it_can(models_page):
    table = _table(models_page)
    assert not models_page.exception
    for column in ("IC", "OOS Rank IC", "RMSE", "PBO", "Crash Recall"):
        assert "<th title=" in table
        assert column in table
    explained = [
        "IC",
        "OOS Rank IC",
        "RMSE",
        "PBO",
        "Crash Recall",
        "Crash Precision",
        "Crash F1",
        "Rank IC within folds",
        "Beyond 2 s.e.",
        "Constant folds",
    ]
    for column in explained:
        assert column in table
    assert table.count('class="fe-eyebrow-help"') == len(explained)


def test_a_tooltip_is_escaped_so_it_cannot_break_the_table(models_page):
    # The hint is written into an HTML attribute; a quote in the wording would
    # otherwise end the attribute early.
    table = _table(models_page)
    assert "&#x27;" in table or all('"' not in glossary.term(t) for t in glossary.TERMS)


def test_an_unknown_term_fails_loudly():
    with pytest.raises(KeyError):
        glossary.term("Sharpe ratio")


@pytest.mark.parametrize("name", sorted(glossary.TERMS))
def test_every_entry_says_what_it_is_and_why_it_matters(name):
    text = glossary.TERMS[name]
    assert len(text) > 60, "too short to explain anything"
    assert len(text) < 700, "a tooltip nobody reads is not an explanation"
