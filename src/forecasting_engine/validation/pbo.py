"""PBO (Probability of Backtest Overfitting) via Combinatorially Symmetric
Cross-Validation (CSCV).

Setups are ranked by Rank IC on each half, the same metric that selects the
winner and gates it, so PBO asks exactly the question selection raises: how
often does the setup with the best Rank IC in-sample fall to or below the
median out-of-sample?
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

N_BLOCKS: int = 16
"""Number of contiguous blocks to split history into. Must be even; 16 is the
standard CSCV choice."""


@dataclass(frozen=True)
class PBOResult:
    """``configurations_tested`` is the CTQ-required traceability log —
    every configuration compared, not just the resulting score."""

    pbo: float
    configurations_tested: tuple[str, ...]
    n_combinations_tested: int


def compute_pbo(
    configs: Mapping[str, tuple[pd.Series, pd.Series]], n_blocks: int = N_BLOCKS
) -> PBOResult:
    """Probability that the best in-sample configuration in ``configs`` is
    just noise rather than a real edge.

    ``configs`` maps each setup to its ``(predicted, realised)`` out-of-sample
    series. Rows where any setup lacks a prediction are dropped so every setup
    is scored on the same rows. History is split into ``n_blocks`` contiguous
    blocks and every way to call half of them in-sample is tried. PBO is the
    share of those splits where the in-sample winner (by Rank IC) falls at or
    below the out-of-sample median. A setup whose Rank IC is undefined on a
    half (say, constant predictions) ranks last there.
    """
    if len(configs) < 2:
        raise ValueError("compute_pbo needs at least two configurations")
    if n_blocks % 2 != 0:
        raise ValueError(f"n_blocks must be even, got {n_blocks}")

    names = tuple(configs.keys())
    predicted = pd.concat([p for p, _ in configs.values()], axis=1, keys=names)
    realised = next(iter(configs.values()))[1].rename("realised")
    frame = pd.concat([predicted, realised], axis=1).dropna().sort_index().to_numpy(dtype=float)
    blocks = np.array_split(np.arange(len(frame)), n_blocks)
    half = n_blocks // 2

    below_median = 0
    n_combinations = 0
    for is_blocks in combinations(range(n_blocks), half):
        oos_blocks = [b for b in range(n_blocks) if b not in is_blocks]
        is_rows = np.concatenate([blocks[b] for b in is_blocks])
        oos_rows = np.concatenate([blocks[b] for b in oos_blocks])

        best_config = int(np.argmax(_rank_ics(frame[is_rows])))
        oos = _rank_ics(frame[oos_rows])
        if _relative_rank(oos, best_config) <= 0.5:
            below_median += 1
        n_combinations += 1

    return PBOResult(
        pbo=below_median / n_combinations,
        configurations_tested=names,
        n_combinations_tested=n_combinations,
    )


def _rank_ics(rows: np.ndarray) -> np.ndarray:
    """Rank IC of each prediction column against the last (realised) column,
    ``-inf`` where it is undefined."""
    ranks = np.column_stack([_average_ranks(rows[:, j]) for j in range(rows.shape[1])])
    centred = ranks - ranks.mean(axis=0)
    spread = np.sqrt((centred**2).sum(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        ics = (centred[:, :-1] * centred[:, -1:]).sum(axis=0) / (spread[:-1] * spread[-1])
    return np.where(np.isfinite(ics), ics, -np.inf)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks from 1, ties sharing their average — what ``Series.rank()`` gives."""
    order = values.argsort(kind="mergesort")
    ordered = values[order]
    edges = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1], True])
    ranks = np.empty(len(values))
    ranks[order] = np.repeat((edges[:-1] + edges[1:] + 1) / 2, np.diff(edges))
    return ranks


def _relative_rank(scores: np.ndarray, position: int) -> float:
    """``scores[position]``'s percentile rank among ``scores``, ties averaged —
    ``Series.rank(pct=True)``."""
    value = scores[position]
    below = (scores < value).sum()
    tied = (scores == value).sum()
    return (below + (tied + 1) / 2) / len(scores)
