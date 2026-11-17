import numpy as np
import pandas as pd
import pytest

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.ingest.align import FeaturePanel
from forecasting_engine.models.polynomial import (
    CANDIDATE_CONFIGS,
    ONLINE_SELECTION_DEFAULT,
    SHRINKAGE,
    PolynomialConfigError,
    _evaluate_candidate,
    run_derived_polynomial,
    run_user_polynomial,
)
from forecasting_engine.parallel import WORKERS_ENV
from forecasting_engine.reporting.model_metrics import build_metrics_rows
from forecasting_engine.validation.gates import evaluate_candidate
from forecasting_engine.validation.harness import (
    MIN_SELECTION_HISTORY,
    select_best_candidate,
    summarize,
)
from forecasting_engine.validation.splitters import PurgedWalkForward


def _panel(n: int = 80) -> FeaturePanel:
    rng = np.random.default_rng(2)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    sig = rng.normal(size=n)
    target = 2 * sig + rng.normal(scale=0.1, size=n)
    frame = pd.DataFrame({"sig": sig, "target": target}, index=idx)
    return FeaturePanel(frame=frame, signals=("sig",), targets=("target",), lag_days=1)


def _splitter() -> PurgedWalkForward:
    return PurgedWalkForward(train=30, test=8, embargo=2)


def test_run_user_polynomial_is_scored_like_any_model():
    result, description = run_user_polynomial("2 * sig", _panel(), _splitter())
    assert result.oos_rank_ic == result.oos_rank_ic  # a real number, not NaN
    assert description.terms == ("2 * sig",)


def test_run_user_polynomial_result_matches_the_shared_comparison_contract():
    """Regression test: a single-configuration result must plug into
    evaluate_candidate() and build_metrics_rows() without raising."""
    result, _description = run_user_polynomial("2 * sig", _panel(), _splitter())

    outcome = evaluate_candidate(result.gated_rank_ic, result.signal_t)
    assert outcome.promoted in (True, False)

    rows = build_metrics_rows({"Polynomial": result})
    assert "t" in rows[0]


def test_run_user_polynomial_raises_when_the_split_produces_no_folds():
    too_short = _panel(n=20)
    with pytest.raises(PolynomialConfigError):
        run_user_polynomial("sig", too_short, _splitter())


def test_run_derived_polynomial_reports_the_winning_candidate():
    result, description = run_derived_polynomial(_panel(), _splitter())
    assert result.signal_rank_ic == result.signal_rank_ic  # a real number, not NaN
    assert description.terms  # the winning candidate's surviving terms


def test_run_derived_polynomial_result_matches_the_shared_comparison_contract():
    """Regression test against FYP-14's contract: a Polynomial run must plug
    into evaluate_candidate() and build_metrics_rows() completely unmodified."""
    result, _description = run_derived_polynomial(_panel(), _splitter())

    outcome = evaluate_candidate(result.gated_rank_ic, result.signal_t)
    assert outcome.promoted in (True, False)

    rows = build_metrics_rows({"Polynomial": result})
    assert rows[0]["Model"].text == "Polynomial"
    assert "t" in rows[0]


# --- which candidate the grid reports: per window for bond, whole period for equity ----


def _long_panel(n: int = 360) -> FeaturePanel:
    """A signal that forecasts the target for a stretch and then reverses."""
    rng = np.random.default_rng(5)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    sig = rng.normal(size=n)
    sign = np.where(np.arange(n) < n // 2, 1.0, -1.0)
    target = sign * sig + rng.normal(size=n)
    frame = pd.DataFrame({"sig": sig, "other": rng.normal(size=n), "target": target}, index=idx)
    return FeaturePanel(frame=frame, signals=("sig", "other"), targets=("target",), lag_days=1)


def _long_splitter() -> PurgedWalkForward:
    return PurgedWalkForward(train=60, test=8, embargo=2)


@pytest.fixture
def one_process(monkeypatch):
    monkeypatch.setenv(WORKERS_ENV, "1")


def test_only_the_bond_target_chooses_per_window():
    assert set(ONLINE_SELECTION_DEFAULT) == {TargetRole.BOND}
    assert ONLINE_SELECTION_DEFAULT[TargetRole.BOND] == SHRINKAGE


def test_for_bond_each_window_uses_the_candidate_chosen_from_earlier_windows(one_process):
    panel, splitter = _long_panel(), _long_splitter()

    result, description = run_derived_polynomial(panel, splitter, role=TargetRole.BOND)

    choice = result.selection
    folds = len(list(splitter.split(panel)))
    assert [name for name, _ in choice.windows] == [
        "degree1_lasso",
        "degree2_lasso",
        "degree3_lasso",
        SHRINKAGE,
    ]
    assert sum(count for _, count in choice.windows) == folds
    assert choice.default == SHRINKAGE
    assert dict(choice.windows)[SHRINKAGE] >= MIN_SELECTION_HISTORY  # the default comes first
    configs = {f"degree{c.degree}_{c.regularizer}": c for c in CANDIDATE_CONFIGS}
    chosen = configs.get(choice.latest)
    config = (
        (1, SHRINKAGE, None)
        if chosen is None
        else (chosen.degree, chosen.regularizer, chosen.max_terms)
    )
    assert description == _evaluate_candidate(config, panel, splitter)[-1].description


def test_equity_and_a_run_with_no_target_keep_the_whole_period_pick(one_process):
    panel, splitter = _long_panel(), _long_splitter()
    configs = [(c.degree, c.regularizer, c.max_terms) for c in CANDIDATE_CONFIGS]
    per_candidate = {
        f"degree{d}_{r}": _evaluate_candidate((d, r, m), panel, splitter) for d, r, m in configs
    }
    per_candidate[SHRINKAGE] = _evaluate_candidate((1, SHRINKAGE, None), panel, splitter)
    expected, expected_description = summarize(per_candidate[select_best_candidate(per_candidate)])

    for role in (TargetRole.EQUITY, None):
        result, description = run_derived_polynomial(panel, splitter, role=role)

        assert result.selection is None
        assert result.signal_rank_ic == expected.signal_rank_ic
        pd.testing.assert_series_equal(result.forecast, expected.forecast)
        assert description == expected_description
