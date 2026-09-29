import numpy as np
import pandas as pd
import pytest
import shap

from forecasting_engine.ingest.align import FeaturePanel, align_and_lag
from forecasting_engine.models import boosted
from forecasting_engine.models.boosted import BoostedConfigError, run_boosted
from forecasting_engine.reporting.model_metrics import build_metrics_rows
from forecasting_engine.validation.gates import evaluate_candidate
from forecasting_engine.validation.splitters import PurgedWalkForward

_N_TRIALS = 3  # kept tiny — every trial runs a mini walk-forward


def _panel(n: int = 120) -> FeaturePanel:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    sig_a = rng.normal(size=n)
    sig_b = rng.normal(size=n)
    target = 2 * sig_a - sig_b + rng.normal(scale=0.05, size=n)
    frame = pd.DataFrame({"sig_a": sig_a, "sig_b": sig_b, "target": target}, index=idx)
    return FeaturePanel(frame=frame, signals=("sig_a", "sig_b"), targets=("target",), lag_days=1)


def _splitter(tuning_rows: int = 45) -> PurgedWalkForward:
    return PurgedWalkForward(train=20, test=5, embargo=2, tuning_rows=tuning_rows)


def _run(panel=None, splitter=None, **kwargs):
    return run_boosted(
        panel if panel is not None else _panel(),
        splitter or _splitter(),
        n_trials=_N_TRIALS,
        retune_trials=_N_TRIALS,
        n_blocks=4,
        **kwargs,
    )


def test_run_boosted_computes_a_real_pbo():
    # Unlike FF5/UserPolynomial, there are always two candidates here (tuned
    # XGBoost vs. tuned LightGBM), so pbo is never None.
    result, description, _tuning = _run()

    assert result.pbo is not None
    assert 0.0 <= result.pbo <= 1.0
    assert result.oos_rank_ic == result.oos_rank_ic  # a real number, not NaN
    assert description.terms == ("sig_a", "sig_b")


def test_run_boosted_raises_when_the_split_produces_no_folds():
    with pytest.raises(BoostedConfigError, match="too few for one"):
        _run(panel=_panel(n=40))


def test_run_boosted_result_matches_the_shared_comparison_contract():
    """Regression test against FYP-14's contract: an ML run must plug into
    evaluate_candidate() and build_metrics_rows() completely unmodified."""
    result, _description, _tuning = _run()

    outcome = evaluate_candidate(result.oos_rank_ic, result.pbo)
    assert outcome.promoted in (True, False)

    rows = build_metrics_rows({"Machine Learning": result})
    assert rows[0]["Model"].text == "Machine Learning"


def test_run_boosted_runs_shap_once_not_once_per_fold(monkeypatch):
    calls = []
    real = shap.TreeExplainer

    def counting(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr("forecasting_engine.models.boosted.shap.TreeExplainer", counting)
    assert len(list(_splitter().split(_panel()))) > 1

    _result, description, _tuning = _run()

    assert len(calls) == 1
    assert all(c >= 0 for c in description.coefficients)


# --- the tuning period and the rolling re-tune ----------------------------------


def test_no_test_window_starts_inside_the_tuning_period():
    panel = _panel()
    folds = list(_splitter().split(panel))
    first_test = panel.frame.index.get_loc(folds[0][1][0])
    assert first_test >= 45 + 2


def test_retunes_follow_the_schedule_and_each_fold_uses_the_latest():
    _result, _description, tuning = _run(retune_every=30)

    panel = _panel()
    starts = [panel.frame.index.get_loc(t[0]) for _, t in _splitter().split(panel)]
    # A re-tune at the first fold whose test window opens 30+ rows after the last.
    expected, tune, point = [], 0, starts[0]
    for start in starts:
        if start >= point + 30:
            tune, point = tune + 1, start
        expected.append(tune)
    assert list(tuning.fold_tunes) == expected
    assert len(tuning.tunes) == max(tuning.fold_tunes) + 1 > 1
    assert tuning.tunes[0].trials == tuning.tunes[1].trials == _N_TRIALS


def test_the_first_tune_uses_only_the_tuning_period():
    _result, _description, tuning = _run()
    panel = _panel()
    assert tuning.tunes[0].first == panel.frame.index[0]
    assert tuning.tunes[0].last < panel.frame.index[45]


def test_no_tuning_label_reaches_the_test_window_it_tunes_for():
    # Horizon 5 with a 2-row embargo: without purging on label_end, the last
    # tuning rows' labels would read prices inside the next test window.
    rng = np.random.default_rng(3)
    n = 140
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    frame = pd.DataFrame(
        {
            "sig_a": rng.normal(size=n),
            "sig_b": rng.normal(size=n),
            "price": 100 + np.cumsum(rng.normal(size=n)),
        },
        index=idx,
    )
    panel = align_and_lag(frame, ["sig_a", "sig_b"], "price", horizon=5)
    splitter = _splitter()
    starts = [panel.frame.index.get_loc(t[0]) for _, t in splitter.split(panel)]

    _result, _description, tuning = _run(panel=panel, splitter=splitter, retune_every=30)

    assert len(tuning.tunes) > 1
    triggers = [tuning.fold_tunes.index(number) for number in range(len(tuning.tunes))]
    for tune, fold in zip(tuning.tunes, triggers, strict=True):
        test_opens = panel.frame.index[starts[fold]]
        assert panel.label_end.loc[tune.last] < test_opens


def test_a_retune_starts_from_the_previous_best(monkeypatch):
    warm_starts = []
    real = boosted.tune_hyperparameters

    def recording(*args, warm_start=None, **kwargs):
        warm_starts.append(warm_start)
        return real(*args, warm_start=warm_start, **kwargs)

    monkeypatch.setattr(boosted, "tune_hyperparameters", recording)
    _result, _description, tuning = _run(retune_every=30)

    assert warm_starts[:2] == [None, None]  # the first tune, one per library
    assert warm_starts[2] == tuning.tunes[0].params["xgboost"]


def test_the_explained_last_fold_uses_the_latest_tune(monkeypatch):
    seen = []
    real = boosted.BoostedForecaster.explain

    def recording(self):
        seen.append(dict(self.params))
        return real(self)

    monkeypatch.setattr(boosted.BoostedForecaster, "explain", recording)
    _result, _description, tuning = _run(retune_every=30)

    latest = tuning.tunes[tuning.fold_tunes[-1]].params[tuning.library]
    (params,) = seen
    assert {k: params[k] for k in latest} == dict(latest)


def test_too_few_tuning_folds_is_refused_before_any_tuning(monkeypatch):
    monkeypatch.setattr(
        boosted, "tune_hyperparameters", lambda *a, **k: pytest.fail("tuning started")
    )
    splitter = PurgedWalkForward(train=40, test=5, embargo=2, tuning_rows=45)

    with pytest.raises(BoostedConfigError, match="at least 3"):
        _run(splitter=splitter)


def test_machine_learning_needs_a_tuning_period():
    with pytest.raises(BoostedConfigError, match="tuning period"):
        _run(splitter=_splitter(tuning_rows=0))
