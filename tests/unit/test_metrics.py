import numpy as np
import pandas as pd
import pytest

from forecasting_engine.validation.metrics import ic, rank_ic, rank_ic_se, rmse


def test_rank_ic_perfect_positive_correlation():
    signal = pd.Series([1, 2, 3, 4, 5])
    target = pd.Series([10, 20, 30, 40, 50])
    assert rank_ic(signal, target) == 1.0


def test_rank_ic_perfect_negative_correlation():
    signal = pd.Series([1, 2, 3, 4, 5])
    target = pd.Series([50, 40, 30, 20, 10])
    assert rank_ic(signal, target) == -1.0


def test_rank_ic_drops_nan_pairs():
    signal = pd.Series([1, 2, np.nan, 4, 5])
    target = pd.Series([10, 20, 30, np.nan, 50])
    assert rank_ic(signal, target) == 1.0


def test_rank_ic_nan_when_fewer_than_two_pairs():
    signal = pd.Series([1, np.nan])
    target = pd.Series([np.nan, 20])
    assert np.isnan(rank_ic(signal, target))


def test_ic_perfect_positive_correlation():
    signal = pd.Series([1, 2, 3, 4, 5])
    target = pd.Series([10, 20, 30, 40, 50])
    assert ic(signal, target) == 1.0


def test_ic_sensitive_to_scale_unlike_rank_ic():
    # A single outlier breaks the linear relationship without breaking the
    # rank order — ic and rank_ic should disagree here.
    signal = pd.Series([1, 2, 3, 4, 100])
    target = pd.Series([10, 20, 30, 40, 41])
    assert rank_ic(signal, target) == 1.0
    assert ic(signal, target) < 1.0


def test_ic_drops_nan_pairs():
    signal = pd.Series([1, 2, np.nan, 4, 5])
    target = pd.Series([10, 20, 30, np.nan, 50])
    assert ic(signal, target) == 1.0


def test_ic_nan_when_fewer_than_two_pairs():
    signal = pd.Series([1, np.nan])
    target = pd.Series([np.nan, 20])
    assert np.isnan(ic(signal, target))


def test_rmse_zero_for_perfect_predictions():
    predicted = pd.Series([1.0, 2.0, 3.0])
    actual = pd.Series([1.0, 2.0, 3.0])
    assert rmse(predicted, actual) == 0.0


def test_rmse_matches_hand_computed_value():
    predicted = pd.Series([0.0, 0.0])
    actual = pd.Series([3.0, 4.0])
    # errors = [-3, -4]; mean squared = (9 + 16) / 2 = 12.5; sqrt(12.5)
    assert rmse(predicted, actual) == pytest.approx(12.5**0.5)


def test_rmse_drops_nan_pairs():
    predicted = pd.Series([1.0, np.nan, 3.0])
    actual = pd.Series([1.0, 2.0, np.nan])
    assert rmse(predicted, actual) == 0.0


def test_rmse_nan_when_no_valid_pairs():
    predicted = pd.Series([np.nan])
    actual = pd.Series([1.0])
    assert np.isnan(rmse(predicted, actual))


# --- Newey-West standard error of the rank IC ---------------------------------


def _noisy_pair(n: int, seed: int = 0) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    predicted = pd.Series(rng.normal(size=n))
    return predicted, predicted * 0.1 + pd.Series(rng.normal(size=n))


def test_rank_ic_se_on_independent_data_is_about_one_over_root_n():
    predicted, realised = _noisy_pair(2_000)
    assert rank_ic_se(predicted, realised, lags=0) == pytest.approx(1 / np.sqrt(2_000), rel=0.1)


def test_rank_ic_se_widens_when_neighbouring_errors_overlap():
    # Five-day sums of daily shocks overlap, like h = 5 labels do, so
    # neighbouring rows share most of their information.
    rng = np.random.default_rng(1)
    realised = pd.Series(rng.normal(size=4_004)).rolling(5).sum().dropna()
    predicted = pd.Series(rng.normal(size=4_004)).rolling(5).sum().dropna()

    overlapping = rank_ic_se(predicted, realised, lags=4)
    assert overlapping > 1.3 * rank_ic_se(predicted, realised, lags=0)


def test_rank_ic_se_is_nan_without_enough_pairs():
    assert np.isnan(rank_ic_se(pd.Series([1.0, 2.0]), pd.Series([2.0, 1.0]), lags=0))
