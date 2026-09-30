import numpy as np
import pandas as pd
import pytest

from forecasting_engine.ingest.align import FeaturePanel
from forecasting_engine.models.naive import NaiveDataError, NaiveMean, run_naive
from forecasting_engine.reporting.model_metrics import NO_CONFIG_SEARCH, build_metrics_rows
from forecasting_engine.validation.splitters import PurgedWalkForward


def _panel(n: int = 80) -> FeaturePanel:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    frame = pd.DataFrame(
        {"sig": rng.normal(size=n), "target": rng.normal(loc=0.01, size=n)}, index=idx
    )
    return FeaturePanel(frame=frame, signals=("sig",), targets=("target",))


def test_each_fold_forecasts_its_own_training_mean():
    panel = _panel()
    train, test = panel.frame.index[:30], panel.frame.index[35:40]
    model = NaiveMean()
    model.fit(panel, train)

    predicted = model.predict(panel, test)

    assert (predicted == panel.frame.loc[train, "target"].mean()).all()
    assert model.describe().intercept == pytest.approx(panel.frame.loc[train, "target"].mean())


def test_it_forecasts_only_the_rows_a_signal_model_can():
    panel = _panel()
    panel.frame.loc[panel.frame.index[36], "sig"] = np.nan
    model = NaiveMean()
    model.fit(panel, panel.frame.index[:30])

    predicted = model.predict(panel, panel.frame.index[35:40])

    assert predicted.isna().tolist() == [False, True, False, False, False]


def test_run_naive_is_scored_like_any_model_with_no_pbo():
    result, _description = run_naive(_panel(), PurgedWalkForward(train=20, test=5, embargo=2))

    assert result.pbo is None
    assert result.rows_scored > 0
    assert result.oos_rank_ic_se == result.oos_rank_ic_se  # a real number
    assert result.crash.recall == 0.0  # a constant forecast flags nothing
    row = build_metrics_rows({"Naive (training mean)": result})[0]
    assert row["PBO"].text == NO_CONFIG_SEARCH


def test_run_naive_says_why_when_there_are_no_folds():
    with pytest.raises(NaiveDataError, match="too few for one"):
        run_naive(_panel(n=10), PurgedWalkForward(train=20, test=5, embargo=2))
