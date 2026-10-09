import numpy as np
import pandas as pd
import pytest

from forecasting_engine.ingest.align import FeaturePanel
from forecasting_engine.models.polynomial import (
    DerivedPolynomial,
    run_derived_polynomial,
)
from forecasting_engine.models.shrinkage import ShrinkageDataError, ShrinkagePolynomial
from forecasting_engine.validation.splitters import PurgedWalkForward

SIGNALS = ("a", "b", "c")


def _panel(n: int = 400, *, weights=(1.0, 1.0, 1.0), seed: int = 5) -> FeaturePanel:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    x = rng.normal(size=(n, len(SIGNALS)))
    target = 0.002 * x @ np.asarray(weights) + rng.normal(scale=0.01, size=n)
    frame = pd.DataFrame(x, index=idx, columns=list(SIGNALS)).assign(target=target)
    return FeaturePanel(frame=frame, signals=SIGNALS, targets=("target",), lag_days=1)


def _fitted(panel: FeaturePanel) -> ShrinkagePolynomial:
    model = ShrinkagePolynomial()
    model.fit(panel, panel.frame.index)
    return model


def test_each_signals_direction_is_learned_from_the_training_rows():
    # Nothing tells the model that "a" predicts downward: it reads that from the data.
    description = _fitted(_panel(weights=(-1.0, 1.0, 1.0))).describe()
    assert description.coefficients[0] < 0
    assert all(c > 0 for c in description.coefficients[1:])


def test_the_forecast_level_is_the_training_mean():
    panel = _panel()
    model = _fitted(panel)
    at_mean = panel.frame[list(SIGNALS)].mean().to_frame().T.set_axis(panel.frame.index[:1])
    frame = pd.concat([at_mean.assign(target=np.nan), panel.frame.iloc[1:]])
    centred = FeaturePanel(frame=frame, signals=SIGNALS, targets=("target",), lag_days=1)
    forecast = model.predict(centred, frame.index[:1]).iloc[0]
    assert forecast == pytest.approx(panel.frame["target"].mean())


def test_the_same_seed_gives_the_same_forecast():
    panel = _panel()
    a = _fitted(panel).predict(panel, panel.frame.index)
    b = _fitted(panel).predict(panel, panel.frame.index)
    pd.testing.assert_series_equal(a, b)


def test_the_description_reproduces_the_forecast():
    panel = _panel(weights=(-1.0, 1.0, 0.5))
    model = _fitted(panel)
    d = model.describe()
    x = panel.frame[list(SIGNALS)]
    z = pd.DataFrame(
        {
            s: (x[s].clip(*d.input_bounds[s]) - d.standardisation[s][0]) / d.standardisation[s][1]
            for s in SIGNALS
        }
    )
    by_hand = d.intercept + z.to_numpy() @ np.asarray(d.coefficients)
    assert np.allclose(model.predict(panel, panel.frame.index).to_numpy(), by_hand)


def test_too_few_training_rows_is_refused_with_a_clear_message():
    with pytest.raises(ShrinkageDataError, match="training rows"):
        _fitted(_panel(n=20))


def test_the_derived_grid_includes_the_shrinkage_fit_and_can_choose_it():
    # A purely linear signal set: a shrinkage fit alone, and the grid with it, run cleanly.
    panel = _panel(n=300)
    splitter = PurgedWalkForward(train=100, test=20, embargo=2, tuning_rows=100)

    alone, description = run_derived_polynomial(panel, splitter, candidates=())
    assert alone.pbo is None
    assert description.name == "ShrinkagePolynomial"

    with_lasso, _ = run_derived_polynomial(
        panel, splitter, candidates=(DerivedPolynomial(degree=1),), n_blocks=4
    )
    assert with_lasso.pbo is not None
