from dataclasses import replace

import pandas as pd

from forecasting_engine.ingest.align import FeaturePanel
from forecasting_engine.models.base import ModelDescription
from forecasting_engine.validation import metrics
from forecasting_engine.validation.harness import (
    FoldResult,
    evaluate,
    select_best_candidate,
    summarize,
)
from forecasting_engine.validation.splitters import PurgedWalkForward


def _panel(n: int = 20) -> FeaturePanel:
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    frame = pd.DataFrame({"signal_a": range(n), "fwd_return_1d": range(n)}, index=idx)
    return FeaturePanel(frame=frame, signals=("signal_a",), targets=("fwd_return_1d",), lag_days=1)


def _panel_with_varying_folds(n: int = 20) -> FeaturePanel:
    """Predicted (signal_a) and realised (fwd_return_1d) differ, and differ
    *differently* fold to fold, so a per-fold mean is distinguishable from
    just reading one fold's value — degenerate perfect-prediction data
    (predicted == realised everywhere) wouldn't catch a bug like that."""
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    signal = list(range(n))
    target = [v + (1 if i % 2 == 0 else -3) for i, v in enumerate(signal)]
    frame = pd.DataFrame({"signal_a": signal, "fwd_return_1d": target}, index=idx)
    return FeaturePanel(frame=frame, signals=("signal_a",), targets=("fwd_return_1d",), lag_days=1)


class _EchoForecaster:
    """A deterministic stub: predicts the signal column unchanged, so fold
    results are trivial to check against the panel by hand."""

    name = "Echo"

    def __init__(self) -> None:
        self.fit_calls: list[pd.DatetimeIndex] = []

    def fit(self, panel: FeaturePanel, train: pd.DatetimeIndex) -> None:
        self.fit_calls.append(train)

    def predict(self, panel: FeaturePanel, idx: pd.DatetimeIndex) -> pd.Series:
        return panel.frame.loc[idx, "signal_a"].astype(float)

    def describe(self) -> ModelDescription:
        return ModelDescription(name=self.name, terms=("signal_a",), coefficients=(1.0,))


class _InverseForecaster:
    """Predicts the *negative* of the signal — deliberately anti-correlated
    with the target, so its rank IC is a real, clearly worse number (not NaN,
    which would make max()'s comparison order-dependent and unreliable)."""

    name = "Inverse"

    def fit(self, panel: FeaturePanel, train: pd.DatetimeIndex) -> None:
        pass

    def predict(self, panel: FeaturePanel, idx: pd.DatetimeIndex) -> pd.Series:
        return -panel.frame.loc[idx, "signal_a"].astype(float)

    def describe(self) -> ModelDescription:
        return ModelDescription(name=self.name, terms=("signal_a",), coefficients=(-1.0,))


def test_evaluate_produces_one_fold_result_per_split():
    panel = _panel()
    splitter = PurgedWalkForward(train=10, test=3, embargo=2)
    results = evaluate(_EchoForecaster, panel, splitter)
    assert len(results) == len(list(splitter.split(panel)))


def test_fold_predicted_and_realised_match_the_panel_slice():
    panel = _panel()
    splitter = PurgedWalkForward(train=10, test=3, embargo=2)
    results = evaluate(_EchoForecaster, panel, splitter)

    fold = results[0]
    assert list(fold.predicted) == list(panel.frame.loc[fold.test, "signal_a"])
    assert list(fold.realised) == list(panel.frame.loc[fold.test, "fwd_return_1d"])
    assert list(fold.predicted_train) == list(panel.frame.loc[fold.train, "signal_a"])
    assert list(fold.realised_train) == list(panel.frame.loc[fold.train, "fwd_return_1d"])


def test_each_fold_fits_a_fresh_instance_on_only_its_own_train_window():
    panel = _panel()
    splitter = PurgedWalkForward(train=10, test=3, embargo=2)
    seen_train_windows = []

    class _RecordingForecaster(_EchoForecaster):
        def fit(self, panel: FeaturePanel, train: pd.DatetimeIndex) -> None:
            super().fit(panel, train)
            seen_train_windows.append(train)

    results = evaluate(_RecordingForecaster, panel, splitter)
    # one fit() call per fold, each on exactly that fold's own train window —
    # a shared/reused instance would instead show every fold's window piling
    # up against the first fold's forecaster.
    assert seen_train_windows == [f.train for f in results]


def test_description_travels_with_its_own_fold():
    panel = _panel()
    splitter = PurgedWalkForward(train=10, test=3, embargo=2)
    results = evaluate(_EchoForecaster, panel, splitter)
    for fold in results:
        assert fold.description == ModelDescription(
            name="Echo", terms=("signal_a",), coefficients=(1.0,)
        )


def test_summarize_scores_ic_rank_ic_rmse_once_over_pooled_folds():
    panel = _panel_with_varying_folds()
    splitter = PurgedWalkForward(train=10, test=3, embargo=2)
    folds = evaluate(_EchoForecaster, panel, splitter)
    assert len(folds) > 1, "fixture must produce more than one fold to test pooling"

    result, _description = summarize(folds, pbo=None)

    predicted = pd.concat([f.predicted for f in folds])
    realised = pd.concat([f.realised for f in folds])
    assert result.ic == metrics.ic(predicted, realised)
    assert result.oos_rank_ic == metrics.rank_ic(predicted, realised)
    assert result.rmse == metrics.rmse(predicted, realised)
    per_fold_mean = sum(metrics.rank_ic(f.predicted, f.realised) for f in folds) / len(folds)
    assert result.oos_rank_ic != per_fold_mean


def test_summarize_reports_a_newey_west_se_using_the_panels_horizon():
    panel = replace(_panel_with_varying_folds(n=60), horizon=5)
    folds = evaluate(_EchoForecaster, panel, PurgedWalkForward(train=10, test=5, embargo=5))

    result, _description = summarize(folds, pbo=None)

    predicted = pd.concat([f.predicted for f in folds])
    realised = pd.concat([f.realised for f in folds])
    assert result.oos_rank_ic_se == metrics.rank_ic_se(predicted, realised, lags=4)


def test_summarize_passes_pbo_through_unchanged():
    panel = _panel()
    splitter = PurgedWalkForward(train=10, test=3, embargo=2)
    folds = evaluate(_EchoForecaster, panel, splitter)

    assert summarize(folds, pbo=0.42)[0].pbo == 0.42
    assert summarize(folds, pbo=None)[0].pbo is None


def test_summarize_returns_the_most_recent_folds_description():
    panel = _panel()
    splitter = PurgedWalkForward(train=10, test=3, embargo=2)
    folds = evaluate(_EchoForecaster, panel, splitter)

    _result, description = summarize(folds, pbo=None)

    assert description == folds[-1].description


def test_summarize_computes_crash_diagnostics():
    panel = _panel_with_varying_folds()
    splitter = PurgedWalkForward(train=10, test=3, embargo=2)
    folds = evaluate(_EchoForecaster, panel, splitter)

    result, _description = summarize(folds, pbo=None)

    assert result.crash.n_true_tail_days >= 0


class _SignalRecordingForecaster:
    """Records the signal set it was fit with per fold; predicts zeros
    unconditionally, since these tests only care what ``evaluate()`` handed
    to ``fit()``, not prediction quality."""

    name = "Recorder"

    def __init__(self) -> None:
        self.seen_signals: list[tuple[str, ...]] = []

    def fit(self, panel: FeaturePanel, train: pd.DatetimeIndex) -> None:
        self.seen_signals.append(panel.signals)

    def predict(self, panel: FeaturePanel, idx: pd.DatetimeIndex) -> pd.Series:
        return pd.Series(0.0, index=idx)

    def describe(self) -> ModelDescription:
        return ModelDescription(name=self.name, terms=(), coefficients=())


def test_screen_true_drops_a_signal_that_fails_screening_per_fold():
    # "strong" is a strictly monotonic transform of the target (rank IC == 1.0
    # on any window); "flat" is constant (rank IC is undefined -> NaN ->
    # excluded) — both deterministic regardless of which fold's window is used,
    # so the assertion below can't flake on a particular random draw.
    idx = pd.date_range("2024-01-01", periods=60, freq="D")
    target = list(range(60))
    frame = pd.DataFrame(
        {"strong": [v * 3 for v in target], "flat": 1.0, "fwd_return_1d": target}, index=idx
    )
    panel = FeaturePanel(
        frame=frame, signals=("strong", "flat"), targets=("fwd_return_1d",), lag_days=1
    )
    splitter = PurgedWalkForward(train=30, test=5, embargo=2)
    recorder = _SignalRecordingForecaster()

    evaluate(lambda: recorder, panel, splitter, screen=True)

    assert recorder.seen_signals, "fixture must produce at least one fold"
    assert all(signals == ("strong",) for signals in recorder.seen_signals)


def test_screen_false_leaves_every_signal_unfiltered():
    idx = pd.date_range("2024-01-01", periods=60, freq="D")
    target = list(range(60))
    frame = pd.DataFrame(
        {"strong": [v * 3 for v in target], "flat": 1.0, "fwd_return_1d": target}, index=idx
    )
    panel = FeaturePanel(
        frame=frame, signals=("strong", "flat"), targets=("fwd_return_1d",), lag_days=1
    )
    splitter = PurgedWalkForward(train=30, test=5, embargo=2)
    recorder = _SignalRecordingForecaster()

    evaluate(lambda: recorder, panel, splitter)  # screen defaults to False

    assert recorder.seen_signals
    assert all(signals == ("strong", "flat") for signals in recorder.seen_signals)


def test_screen_true_falls_back_to_every_signal_if_all_excluded():
    # Both signals are constant -> both score NaN -> both excluded. A fold
    # left with zero features would break every model family's fit(), so
    # evaluate() must fall back to the full signal set instead.
    idx = pd.date_range("2024-01-01", periods=40, freq="D")
    frame = pd.DataFrame({"flat_a": 1.0, "flat_b": 2.0, "fwd_return_1d": range(40)}, index=idx)
    panel = FeaturePanel(
        frame=frame, signals=("flat_a", "flat_b"), targets=("fwd_return_1d",), lag_days=1
    )
    splitter = PurgedWalkForward(train=20, test=5, embargo=1)
    recorder = _SignalRecordingForecaster()

    evaluate(lambda: recorder, panel, splitter, screen=True)

    assert recorder.seen_signals, "fixture must produce at least one fold"
    assert all(set(signals) == {"flat_a", "flat_b"} for signals in recorder.seen_signals)


def test_select_best_candidate_picks_the_higher_rank_ic_candidate():
    # A bigger panel than the summarize()/evaluate() tests above — PBO's CSCV
    # splits the combined strategy-return series into n_blocks pieces, so it
    # needs enough total rows across folds to be meaningful.
    panel = _panel_with_varying_folds(n=80)
    splitter = PurgedWalkForward(train=30, test=8, embargo=2)
    strong = evaluate(_EchoForecaster, panel, splitter)
    weak = evaluate(_InverseForecaster, panel, splitter)

    best_name, pbo_value = select_best_candidate({"strong": strong, "weak": weak}, n_blocks=4)

    assert best_name == "strong"
    assert 0.0 <= pbo_value <= 1.0


# --- per-fold screening, recorded on the fold and summarised for display ----


def _screening_fixture(signals: dict[str, object], n: int = 60) -> FeaturePanel:
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    frame = pd.DataFrame({**signals, "fwd_return_1d": list(range(n))}, index=idx)
    return FeaturePanel(
        frame=frame, signals=tuple(signals), targets=("fwd_return_1d",), lag_days=1
    )


def _strong_and_flat() -> FeaturePanel:
    return _screening_fixture({"strong": [v * 3 for v in range(60)], "flat": 1.0})


def _screened_strong_and_flat():
    splitter = PurgedWalkForward(30, 5, 2)
    return evaluate(_SignalRecordingForecaster, _strong_and_flat(), splitter, screen=True)


def test_no_screening_is_recorded_when_screen_is_false():
    folds = evaluate(_SignalRecordingForecaster, _strong_and_flat(), PurgedWalkForward(30, 5, 2))
    assert folds
    assert all(f.screening is None for f in folds)


def test_each_fold_records_exactly_the_signals_it_was_fit_on():
    # The recorded value must be what fit() actually received, not a second
    # screening computation that could drift from it.
    recorder = _SignalRecordingForecaster()
    folds = evaluate(lambda: recorder, _strong_and_flat(), PurgedWalkForward(30, 5, 2), screen=True)

    assert [f.screening.fitted for f in folds] == recorder.seen_signals
    assert all(f.screening.fitted == ("strong",) for f in folds)
    assert all(f.screening.candidates == ("strong", "flat") for f in folds)


def test_a_fold_that_excluded_every_signal_records_that_it_fell_back():
    # Screening kept nothing, so the fold was fit on every signal. Recording
    # only what screening kept would claim this fold used no signals at all.
    panel = _screening_fixture({"flat_a": 1.0, "flat_b": 2.0}, n=40)
    recorder = _SignalRecordingForecaster()
    folds = evaluate(lambda: recorder, panel, PurgedWalkForward(20, 5, 1), screen=True)

    for fold, seen in zip(folds, recorder.seen_signals, strict=True):
        assert fold.screening.included == ()
        assert fold.screening.fell_back
        assert fold.screening.fitted == seen == ("flat_a", "flat_b")


def test_the_summary_counts_how_many_folds_fit_each_signal():
    folds = _screened_strong_and_flat()
    result, _ = summarize(folds)

    summary = result.screening
    assert summary.folds == len(folds)
    assert summary.fell_back == 0
    assert dict(summary.counts) == {"strong": len(folds), "flat": 0}


def test_a_signal_no_fold_used_is_still_listed_with_zero():
    # The point of the table is to show a signal was screened out everywhere.
    folds = _screened_strong_and_flat()
    result, _ = summarize(folds)
    assert ("flat", 0) in result.screening.counts


def test_the_most_used_signals_are_listed_first():
    folds = _screened_strong_and_flat()
    result, _ = summarize(folds)
    assert [name for name, _ in result.screening.counts] == ["strong", "flat"]


def test_fallback_folds_count_as_using_every_signal():
    panel = _screening_fixture({"flat_a": 1.0, "flat_b": 2.0}, n=40)
    folds = evaluate(_SignalRecordingForecaster, panel, PurgedWalkForward(20, 5, 1), screen=True)
    result, _ = summarize(folds)

    assert result.screening.fell_back == len(folds)
    assert dict(result.screening.counts) == {"flat_a": len(folds), "flat_b": len(folds)}


def test_there_is_no_screening_summary_without_screening():
    folds = evaluate(_SignalRecordingForecaster, _strong_and_flat(), PurgedWalkForward(30, 5, 2))
    result, _ = summarize(folds)
    assert result.screening is None


# --- how many folds actually fitted terms -----------------------------------


def _fold_with_description(i: int, description: ModelDescription) -> FoldResult:
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    train, test = idx[:4], idx[4:]
    return FoldResult(
        fold=i,
        train=train,
        test=test,
        predicted=pd.Series([0.1, 0.2], index=test),
        predicted_train=pd.Series(0.1, index=train),
        realised=pd.Series([0.1, 0.3], index=test),
        realised_train=pd.Series(0.2, index=train),
        description=description,
    )


def test_summarize_counts_the_folds_whose_fit_kept_any_term():
    # A regularized fit can zero every coefficient on one fold and keep five on
    # the next, and ``summarize`` reports only the most recent fold's equation.
    # The count is what stops that one equation being read as the whole run.
    folds = tuple(
        _fold_with_description(
            i, ModelDescription(name="DerivedPolynomial", terms=terms, coefficients=coefs)
        )
        for i, (terms, coefs) in enumerate([(("x",), (1.0,)), ((), ()), (("x",), (2.0,))])
    )

    result, _ = summarize(folds)

    assert result.terms is not None
    assert (result.terms.folds, result.terms.with_terms) == (3, 2)


def test_a_run_where_every_fold_kept_terms_says_so():
    folds = tuple(
        _fold_with_description(
            i, ModelDescription(name="FamaFrench5", terms=("MKT",), coefficients=(0.5,))
        )
        for i in range(4)
    )

    result, _ = summarize(folds)

    assert (result.terms.folds, result.terms.with_terms) == (4, 4)


def test_screening_summary_records_each_folds_ic_and_the_latest_folds_choice():
    panel = _strong_and_flat()
    splitter = PurgedWalkForward(train=30, test=5, embargo=2)
    folds = evaluate(_SignalRecordingForecaster, panel, splitter, screen=True)

    result, _ = summarize(folds)

    screening = result.screening
    assert len(screening.fold_ics) == len(folds)
    assert screening.latest_ics["strong"] == 1.0
    assert screening.latest_included == ("strong",)


def test_rows_scored_counts_test_rows_with_a_prediction_and_a_realised_value():
    panel = _panel(n=20)
    splitter = PurgedWalkForward(train=8, test=4, embargo=1)
    last_tested = list(splitter.split(panel))[-1][1][-1]
    panel.frame.loc[last_tested, "signal_a"] = float("nan")
    folds = evaluate(_EchoForecaster, panel, splitter)

    result, _ = summarize(folds)

    tested = sum(len(f.test) for f in folds)
    assert result.rows_scored == tested - 1
