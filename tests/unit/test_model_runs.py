"""The Models page's per-session store of runs (``app/model_runs.py``): when a
target's results are kept and when they are dropped."""

import model_runs
from forecasting_engine.extraction.targets import TargetRole

EQUITY = TargetRole.EQUITY
NAIVE = "Naive (training mean)"
INPUTS = ("SPX_Index_PX_LAST", (("VIX_Index_PX_LAST", "level"),))


def _stored_with_a_run() -> model_runs.StoredRuns:
    stored = model_runs.StoredRuns(("fingerprint", 20, 504, 20))
    model_runs.tab(stored, EQUITY, {}, inputs=INPUTS).runs[NAIVE] = object()
    return stored


def test_a_tab_keeps_its_runs_while_what_it_is_fitted_on_stays_the_same():
    stored = _stored_with_a_run()

    tab = model_runs.tab(stored, EQUITY, {}, inputs=INPUTS)

    assert set(tab.runs) == {NAIVE}


def test_a_tab_drops_its_runs_when_its_target_column_changes():
    stored = _stored_with_a_run()
    another_field = ("SPX_Index_TOT_RETURN_INDEX_GROSS_DVDS", INPUTS[1])

    tab = model_runs.tab(stored, EQUITY, {}, inputs=another_field)

    assert not tab.runs
    assert tab.inputs == another_field


def test_a_tab_drops_its_runs_when_its_signals_or_their_transforms_change():
    stored = _stored_with_a_run()
    another_transform = (INPUTS[0], (("VIX_Index_PX_LAST", "difference"),))

    tab = model_runs.tab(stored, EQUITY, {}, inputs=another_transform)

    assert not tab.runs


def test_a_tab_with_nothing_recorded_keeps_its_runs_and_records_what_it_is_given():
    stored = model_runs.StoredRuns(("fingerprint", 20, 504, 20))
    stored.tabs[EQUITY] = model_runs.TabRuns({}, {NAIVE: object()})

    tab = model_runs.tab(stored, EQUITY, {}, inputs=INPUTS)

    assert set(tab.runs) == {NAIVE}
    assert tab.inputs == INPUTS


def test_a_models_own_setting_still_drops_only_that_model():
    stored = model_runs.StoredRuns(("fingerprint", 20, 504, 20))
    stored.tabs[EQUITY] = model_runs.TabRuns({"A": 1, "B": 1}, {"A": object(), "B": object()})

    tab = model_runs.tab(stored, EQUITY, {"A": 2, "B": 1}, inputs=INPUTS)

    assert set(tab.runs) == {"B"}
