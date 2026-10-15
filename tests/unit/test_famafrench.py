import numpy as np
import pandas as pd
import pytest

from forecasting_engine.ingest.align import FeaturePanel, align_and_lag
from forecasting_engine.models.famafrench import (
    FACTOR_COLUMNS,
    FamaFrench5,
    FamaFrenchDataError,
    factor_coverage,
    merge_factors,
)

_COEFFICIENTS = {"Mkt-RF": 1.2, "SMB": -0.5, "HML": 0.3, "RMW": 0.1, "CMA": -0.2}
_INTERCEPT = 0.01


def _exact_linear_panel(n: int = 60) -> FeaturePanel:
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    factors = {c: rng.normal(size=n) for c in FACTOR_COLUMNS}
    target = _INTERCEPT + sum(_COEFFICIENTS[c] * factors[c] for c in FACTOR_COLUMNS)
    frame = pd.DataFrame({**factors, "target": target}, index=idx)
    return FeaturePanel(frame=frame, signals=FACTOR_COLUMNS, targets=("target",), lag_days=1)


# --- FamaFrench5: FYP-116, known input produces expected coefficients ------


def test_famafrench5_recovers_exact_coefficients_on_noiseless_data():
    # Plain OLS, no regularization — unlike DerivedPolynomial, exact
    # coefficient recovery on noiseless data is a fair assertion here.
    panel = _exact_linear_panel()
    model = FamaFrench5()

    model.fit(panel, panel.frame.index)
    description = model.describe()

    assert description.terms == FACTOR_COLUMNS
    for term, coefficient in zip(description.terms, description.coefficients, strict=True):
        assert coefficient == pytest.approx(_COEFFICIENTS[term], abs=1e-8)
    assert description.intercept == pytest.approx(_INTERCEPT, abs=1e-8)


def test_famafrench5_predict_matches_the_fitted_relationship():
    panel = _exact_linear_panel()
    model = FamaFrench5()
    model.fit(panel, panel.frame.index)

    predicted = model.predict(panel, panel.frame.index)

    assert predicted.to_numpy() == pytest.approx(panel.frame["target"].to_numpy(), abs=1e-8)


def test_famafrench5_predict_masks_rows_with_a_missing_factor():
    panel = _exact_linear_panel()
    model = FamaFrench5()
    model.fit(panel, panel.frame.index)
    panel.frame.loc[panel.frame.index[0], "Mkt-RF"] = float("nan")

    predicted = model.predict(panel, panel.frame.index)

    assert pd.isna(predicted.iloc[0])
    assert not pd.isna(predicted.iloc[1])


def test_famafrench5_rejects_too_few_training_rows():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    frame = pd.DataFrame({**{c: range(5) for c in FACTOR_COLUMNS}, "target": range(5)}, index=idx)
    panel = FeaturePanel(frame=frame, signals=FACTOR_COLUMNS, targets=("target",), lag_days=1)

    with pytest.raises(FamaFrenchDataError):
        FamaFrench5().fit(panel, panel.frame.index)


def test_famafrench5_predict_before_fit_raises():
    panel = _exact_linear_panel()
    with pytest.raises(RuntimeError):
        FamaFrench5().predict(panel, panel.frame.index)


def test_famafrench5_describe_before_fit_raises():
    with pytest.raises(RuntimeError):
        FamaFrench5().describe()


# --- merge_factors: FYP-111 -------------------------------------------------


def test_merge_factors_keeps_every_bloomberg_date_and_leaves_uncovered_ones_blank():
    bloomberg = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "price": [100.0, 101.0, 102.0],
        }
    )
    factors = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2023-12-31", "2024-01-01", "2024-01-02", "2024-01-05"]),
            **{c: [0.0, 0.1, 0.2, 0.3] for c in FACTOR_COLUMNS},
        }
    )

    merged = merge_factors(bloomberg, factors)

    assert list(merged["Date"]) == list(bloomberg["Date"])
    assert merged["Mkt-RF"].tolist()[:2] == [0.1, 0.2]
    assert pd.isna(merged["Mkt-RF"].iloc[2])


def test_factor_coverage_counts_rows_used_and_target_dates_without_factors():
    dates = pd.bdate_range("2024-01-01", periods=30)
    bloomberg = pd.DataFrame({"Date": dates, "price": 100 + np.arange(30.0)})
    factors = pd.DataFrame({"Date": dates[:25], **{c: 0.1 for c in FACTOR_COLUMNS}})
    merged = merge_factors(bloomberg, factors).set_index("Date")
    panel = align_and_lag(merged, list(FACTOR_COLUMNS), "price", horizon=1, exact=FACTOR_COLUMNS)

    coverage = factor_coverage(factors, panel)

    assert coverage.first == dates[0]
    assert coverage.last == dates[24]
    assert coverage.missing_dates == 5
    # Lagged a row, factors reach rows 1-25 of 30; row 25's label is row 26's price.
    assert coverage.rows_used == 25


def test_merge_factors_rejects_an_empty_bloomberg_frame():
    empty = pd.DataFrame(columns=["Date", "price"])
    factors = pd.DataFrame(columns=["Date", *FACTOR_COLUMNS])

    with pytest.raises(FamaFrenchDataError):
        merge_factors(empty, factors)
