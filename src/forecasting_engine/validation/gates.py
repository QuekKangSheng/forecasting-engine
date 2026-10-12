"""Promotion-gate policy for a candidate model, kept separate from the
statistics that produce the numbers (rank_ic, rank_ic_se) per Validation
Metrics v2 Section 3. Crash recall/precision/F1 never appear here — they are
a diagnostic with no threshold, so they cannot fail a gate.
"""

from __future__ import annotations

from dataclasses import dataclass

SIGNAL_RANK_IC_GATE: float = 0.02
"""First gate: the Signal Rank IC — pooled OOS Rank IC of each forecast less its
fold's training-mean target (``validation.harness.signal_forecasts``) — must
exceed this (strictly)."""

SIGNIFICANCE_T_GATE: float = 2.0
"""Second gate: t, the Signal Rank IC divided by the larger of its two Newey-West
standard errors, must exceed this (strictly). At a standard error near 0.07, a
model with no skill clears the first gate alone about 39% of the time; the sponsor
asked for a t-statistic above 2."""


@dataclass(frozen=True)
class ValidationOutcome:
    promoted: bool
    failed_gates: tuple[str, ...]


def evaluate_candidate(signal_rank_ic: float, t: float) -> ValidationOutcome:
    """Apply the Signal Rank IC and t gates. A NaN input (e.g. a candidate whose
    IC or standard error couldn't be computed) fails its gate rather than passing
    or raising."""
    failed = []
    if not (signal_rank_ic > SIGNAL_RANK_IC_GATE):
        failed.append("signal_rank_ic")
    if not (t > SIGNIFICANCE_T_GATE):
        failed.append("t")
    return ValidationOutcome(promoted=not failed, failed_gates=tuple(failed))


def is_high_risk(outcome: ValidationOutcome) -> bool:
    """Working default, not sponsor-confirmed: high risk means both gates failed."""
    return len(outcome.failed_gates) == 2
