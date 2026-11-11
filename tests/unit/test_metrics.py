import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import weights_uniform

from forecasting_engine.validation.metrics import ic, overlap_fit, rank_ic, rank_ic_se, rmse


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


# --- the standard error for overlapping labels ---------------------------------


def _overlapping(n: int, h: int, seed: int) -> tuple[pd.Series, pd.Series]:
    """h-day forward returns built from independent daily returns, and a slow-moving
    signal that knows nothing about them: no predictability at all."""
    rng = np.random.default_rng(seed)
    path = np.concatenate([[0.0], np.cumsum(rng.normal(scale=0.01, size=n + h))])
    forward = pd.Series(path[h : h + n] - path[:n])
    shocks = rng.normal(size=n)
    signal = np.empty(n)
    signal[0] = shocks[0]
    for t in range(1, n):
        signal[t] = 0.98 * signal[t - 1] + shocks[t]
    return pd.Series(signal), forward


def test_the_overlap_standard_error_counts_every_overlapping_lag_in_full():
    # Hansen and Hodrick's estimator written out: every autocovariance of the
    # regression's scores up to lag h - 1, none of them weighted down.
    predicted, realised = _overlapping(600, 20, seed=0)
    ranks = pd.concat([predicted, realised], axis=1).rank()
    z = ((ranks - ranks.mean()) / ranks.std(ddof=0)).to_numpy()
    design = np.column_stack([np.ones(len(z)), z[:, 0]])
    slope = np.linalg.lstsq(design, z[:, 1], rcond=None)[0]
    scores = design * (z[:, 1] - design @ slope)[:, None]
    middle = scores.T @ scores
    for lag in range(1, 20):
        cross = scores[lag:].T @ scores[:-lag]
        middle += cross + cross.T
    bread = np.linalg.inv(design.T @ design)
    by_hand = float(np.sqrt((bread @ middle @ bread)[1, 1]))

    overlap = rank_ic_se(predicted, realised, lags=19, overlap=True)

    assert overlap == pytest.approx(by_hand, rel=1e-9)
    assert overlap != pytest.approx(rank_ic_se(predicted, realised, lags=19), rel=0.01)


def test_the_overlap_standard_error_matches_how_far_a_skill_less_rank_ic_strays():
    # With nothing to find, a standard error should be about as large as the spread of
    # the Rank IC itself. Newey-West's declining weights leave it too small on
    # overlapping labels, so "t above 2" happened too often.
    pairs = [_overlapping(1_000, 20, seed=seed) for seed in range(150)]
    spread = float(np.std([rank_ic(predicted, realised) for predicted, realised in pairs]))
    newey_west = float(np.mean([rank_ic_se(p, r, lags=19) for p, r in pairs]))
    overlap = float(np.mean([rank_ic_se(p, r, lags=19, overlap=True) for p, r in pairs]))

    assert newey_west < 0.9 * spread
    assert overlap == pytest.approx(spread, rel=0.1)


def test_the_overlap_fit_is_never_surer_than_newey_west():
    # On a sample not much longer than the lags, counting them in full can give a
    # standard error far too small: over every lag a sample has, it is exactly zero.
    seen = set()
    for seed in range(20):
        predicted, realised = _noisy_pair(40, seed=seed)
        x, y = predicted.to_numpy(), realised.to_numpy()
        model = sm.OLS(y, sm.add_constant(x))
        newey_west = model.fit(cov_type="HAC", cov_kwds={"maxlags": 19}).bse[1]
        with np.errstate(invalid="ignore"):
            in_full = model.fit(
                cov_type="HAC", cov_kwds={"maxlags": 19, "weights_func": weights_uniform}
            ).bse[1]
        wider = bool(in_full > newey_west)
        seen.add(wider)

        fit = overlap_fit(y, x, lags=19)

        assert fit.bse[1] == pytest.approx(in_full if wider else newey_west)
    assert seen == {True, False}  # both cases met


def test_the_overlap_fit_falls_back_to_newey_west_where_its_variance_is_not_positive():
    # Equal weights do not guarantee a positive variance. Scores that flip sign every
    # row have a lag-1 autocovariance of nearly minus their variance; counted in full
    # on both sides, it outweighs the variance.
    x = np.tile([1.0, 1.0, -1.0, -1.0], 10)
    y = 0.5 * x + np.tile([1.0, -1.0, -1.0, 1.0], 10)
    model = sm.OLS(y, sm.add_constant(x))
    with np.errstate(invalid="ignore"):
        equal = model.fit(cov_type="HAC", cov_kwds={"maxlags": 1, "weights_func": weights_uniform})
    assert not equal.bse[1] > 0
    newey_west = model.fit(cov_type="HAC", cov_kwds={"maxlags": 1})

    fit = overlap_fit(y, x, lags=1)

    assert fit.bse[1] > 0
    assert fit.bse[1] == pytest.approx(newey_west.bse[1])
    assert fit.params[1] == pytest.approx(0.5)


def test_with_no_lags_the_overlap_standard_error_is_the_plain_one():
    predicted, realised = _noisy_pair(500)

    assert rank_ic_se(predicted, realised, lags=0, overlap=True) == pytest.approx(
        rank_ic_se(predicted, realised, lags=0)
    )
