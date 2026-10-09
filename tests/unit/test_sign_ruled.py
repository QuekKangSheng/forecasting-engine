import numpy as np
import pandas as pd
import pytest

from forecasting_engine.extraction.bloomberg_csv import ColumnSource
from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.ingest.align import FeaturePanel, Transform
from forecasting_engine.models.polynomial import PolynomialConfigError
from forecasting_engine.models.sign_ruled import (
    ECONOMIC_INPUTS,
    LEVEL_ROWS,
    SignRuledPolynomial,
    economic_signs,
    expanding,
    overlap,
    run_sign_ruled_polynomial,
)
from forecasting_engine.validation.splitters import PurgedWalkForward

SIGNALS = ("hy", "ig", "vix", "slope")


def _panel(n: int = 400, *, weights=(1.0, 1.0, 1.0, 1.0), seed: int = 3) -> FeaturePanel:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    x = rng.normal(size=(n, len(SIGNALS)))
    target = 0.002 * x @ np.asarray(weights) + rng.normal(scale=0.01, size=n)
    frame = pd.DataFrame(x, index=idx, columns=list(SIGNALS)).assign(target=target)
    return FeaturePanel(frame=frame, signals=SIGNALS, targets=("target",), lag_days=1)


def _fitted(panel: FeaturePanel, signs=None) -> SignRuledPolynomial:
    model = SignRuledPolynomial(dict(signs or dict.fromkeys(SIGNALS, 1)))
    model.fit(panel, panel.frame.index)
    return model


def test_every_slope_comes_out_positive_when_every_signal_predicts_upward():
    description = _fitted(_panel()).describe()
    assert description.terms == SIGNALS
    assert all(c > 0 for c in description.coefficients)


def test_the_slopes_are_shrunk_toward_each_other():
    # One signal three times as strong as the rest: the shared prior pulls the
    # four slopes together, so the strong one ends up less than three times the others.
    description = _fitted(_panel(weights=(3.0, 1.0, 1.0, 1.0))).describe()
    strong, *rest = description.coefficients
    assert strong > max(rest)
    assert strong < 3 * np.mean(rest)


def test_a_negative_sign_turns_its_signal_around():
    panel = _panel(weights=(-1.0, 1.0, 1.0, 1.0))
    description = _fitted(panel, signs={"hy": -1, "ig": 1, "vix": 1, "slope": 1}).describe()
    assert description.coefficients[0] < 0
    assert all(c > 0 for c in description.coefficients[1:])


def test_the_forecast_level_is_the_mean_of_the_first_training_labels():
    panel = _panel()
    flat = panel.frame.assign(**dict.fromkeys(SIGNALS, 1.0))
    flat_panel = FeaturePanel(frame=flat, signals=SIGNALS, targets=("target",), lag_days=1)
    model = _fitted(flat_panel)
    forecast = model.predict(flat_panel, flat.index)
    expected = flat["target"].iloc[:LEVEL_ROWS].mean()
    assert np.allclose(forecast, expected)


def test_the_same_seed_gives_the_same_forecast():
    panel = _panel()
    a = _fitted(panel).predict(panel, panel.frame.index)
    b = _fitted(panel).predict(panel, panel.frame.index)
    pd.testing.assert_series_equal(a, b)


@pytest.mark.parametrize("hy_sign", [1, -1])
def test_the_description_reproduces_the_forecast(hy_sign):
    panel = _panel(weights=(float(hy_sign), 1.0, 1.0, 1.0))
    model = _fitted(panel, signs={"hy": hy_sign, "ig": 1, "vix": 1, "slope": 1})
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


def test_rows_missing_a_signal_get_no_forecast():
    panel = _panel()
    model = _fitted(panel)
    holed = panel.frame.copy()
    holed.iloc[5, 0] = np.nan
    forecast = model.predict(
        FeaturePanel(frame=holed, signals=SIGNALS, targets=("target",), lag_days=1), holed.index
    )
    assert np.isnan(forecast.iloc[5])
    assert forecast.drop(holed.index[5]).notna().all()


def test_too_few_training_rows_is_refused_with_a_clear_message():
    panel = _panel(n=20)
    with pytest.raises(PolynomialConfigError, match="training rows"):
        _fitted(panel)


def test_overlap_reads_how_many_rows_a_label_shares():
    rng = np.random.default_rng(0)
    daily = pd.Series(rng.normal(size=3000))
    assert overlap(daily) == pytest.approx(1.0, abs=0.1)
    monthly = daily.rolling(21).sum().dropna()
    assert 10 < overlap(monthly) <= 21


def _sources(tickers):
    return {f"{t}_PX_LAST": ColumnSource(security=f"{t} Index", field="PX_LAST") for t in tickers}


def test_economic_signs_finds_the_equity_inputs_by_ticker():
    sources = _sources(["LF98OAS", "LUACOAS", "VIX", "USYC2Y10", "DXY"])
    signs = economic_signs(list(sources), sources, TargetRole.EQUITY)
    assert signs == {f"{t}_PX_LAST": 1 for t in ECONOMIC_INPUTS[TargetRole.EQUITY]}


def test_economic_signs_names_any_missing_input():
    sources = _sources(["LF98OAS", "VIX"])
    with pytest.raises(PolynomialConfigError, match="LUACOAS"):
        economic_signs(list(sources), sources, TargetRole.EQUITY)


def test_economic_signs_refuses_two_columns_for_one_input():
    sources = _sources(["LF98OAS", "LUACOAS", "VIX", "USYC2Y10"])
    sources["VIX_Curncy_PX_LAST"] = ColumnSource(security="VIX Curncy", field="PX_LAST")
    with pytest.raises(PolynomialConfigError, match="more than one column"):
        economic_signs(list(sources), sources, TargetRole.EQUITY)


def test_economic_signs_is_equity_only():
    sources = _sources(["LF98OAS", "LUACOAS", "VIX", "USYC2Y10"])
    with pytest.raises(PolynomialConfigError, match="equity"):
        economic_signs(list(sources), sources, TargetRole.BOND)


def _levels(n: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    walk = np.cumsum(rng.normal(size=(n, 4)), axis=0)
    frame = pd.DataFrame(walk, index=idx, columns=list(SIGNALS))
    frame["price"] = 100 * np.exp(np.cumsum(rng.normal(scale=0.01, size=n)))
    return frame


def test_the_runner_reads_every_input_as_a_level_and_trains_on_all_history():
    frame = _levels()
    splitter = PurgedWalkForward(train=60, test=20, embargo=5, tuning_rows=300)
    result, description, panel = run_sign_ruled_polynomial(
        frame, dict.fromkeys(SIGNALS, 1), "price", horizon=5, splitter=splitter
    )
    assert all(a.transform == Transform.LEVEL for a in panel.alignment.values())
    assert result.pbo is None
    assert description.terms == SIGNALS
    assert result.terms.folds > 1


def test_the_runner_keeps_the_test_windows_but_trains_on_every_earlier_row():
    splitter = PurgedWalkForward(train=60, test=20, embargo=5, tuning_rows=300)
    grown = expanding(splitter)
    assert grown.train is None
    assert (grown.test, grown.embargo, grown.tuning_rows) == (20, 5, 300)


def test_the_runner_opens_its_first_test_window_where_the_page_does():
    # A train window longer than the tuning period delays the page's first test
    # window; the expanding window waits for it too, so every model scores the same dates.
    frame = _levels()
    panel = run_sign_ruled_polynomial(
        frame,
        dict.fromkeys(SIGNALS, 1),
        "price",
        5,
        PurgedWalkForward(train=400, test=20, embargo=5, tuning_rows=300),
    )[2]
    page = PurgedWalkForward(train=400, test=20, embargo=5, tuning_rows=300)
    grown = expanding(page)
    assert [t[0] for _, t in grown.split(panel)] == [t[0] for _, t in page.split(panel)]


def test_the_runner_refuses_data_too_short_for_one_fold():
    frame = _levels(n=200)
    splitter = PurgedWalkForward(train=60, test=20, embargo=5, tuning_rows=300)
    with pytest.raises(PolynomialConfigError):
        run_sign_ruled_polynomial(frame, dict.fromkeys(SIGNALS, 1), "price", 5, splitter)
