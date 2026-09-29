import numpy as np
import pandas as pd
import pytest

from forecasting_engine.validation.pbo import compute_pbo


def _realised(n: int, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(size=n), index=pd.date_range("2024-01-01", periods=n, freq="D"))


def _noise(like: pd.Series, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(size=len(like)), index=like.index)


def test_one_genuine_setup_among_noise_gives_low_pbo():
    realised = _realised(800)
    genuine = realised + _noise(realised, 1) * 2  # rank IC about 0.45
    configs = {"genuine": (genuine, realised)} | {
        f"noise_{i}": (_noise(realised, 10 + i), realised) for i in range(3)
    }

    assert compute_pbo(configs, n_blocks=8).pbo < 0.1


def test_all_noise_setups_give_pbo_near_one_half():
    # An even number of setups: with an odd number, "at or below the median"
    # covers a different share of ranks and pure noise lands away from 0.5.
    pbos = []
    for seed in range(20):
        realised = _realised(400, seed=seed)
        configs = {f"noise_{i}": (_noise(realised, 100 * seed + i), realised) for i in range(4)}
        pbos.append(compute_pbo(configs, n_blocks=8).pbo)

    assert np.mean(pbos) == pytest.approx(0.5, abs=0.1)


def test_pbo_is_one_when_the_is_winner_never_repeats_out_of_sample():
    # "overfit" ranks perfectly in the first half and perfectly backwards in the
    # second; "steady" is modestly right throughout. Whichever wins in-sample is
    # the one that falls behind out-of-sample, in both ways to split two blocks.
    realised = _realised(200)
    flip = pd.Series(np.r_[np.ones(100), -np.ones(100)], index=realised.index)
    overfit = realised * flip
    steady = realised + _noise(realised, 1) * 3

    result = compute_pbo({"overfit": (overfit, realised), "steady": (steady, realised)}, n_blocks=2)

    assert result.pbo == 1.0
    assert result.n_combinations_tested == 2
    assert result.configurations_tested == ("overfit", "steady")


def test_pbo_is_zero_when_one_config_dominates_every_block():
    realised = _realised(200)
    genuine = realised + _noise(realised, 1) * 0.1

    result = compute_pbo(
        {"genuine": (genuine, realised), "noise": (_noise(realised, 2), realised)}, n_blocks=2
    )

    assert result.pbo == 0.0


def test_a_setup_with_constant_predictions_ranks_last_rather_than_winning():
    realised = _realised(200)
    constant = pd.Series(0.0, index=realised.index)
    genuine = realised + _noise(realised, 1) * 0.5

    result = compute_pbo(
        {"constant": (constant, realised), "genuine": (genuine, realised)}, n_blocks=4
    )

    assert result.pbo == 0.0


def test_requires_at_least_two_configurations():
    realised = _realised(8)
    with pytest.raises(ValueError):
        compute_pbo({"solo": (realised, realised)}, n_blocks=2)


def test_requires_even_n_blocks():
    realised = _realised(12)
    configs = {"a": (realised, realised), "b": (_noise(realised, 1), realised)}
    with pytest.raises(ValueError):
        compute_pbo(configs, n_blocks=3)


def test_n_combinations_tested_matches_choose_n_blocks_half():
    realised = _realised(40)
    configs = {"a": (realised, realised), "b": (_noise(realised, 1), realised)}
    assert compute_pbo(configs, n_blocks=4).n_combinations_tested == 6  # C(4, 2)
