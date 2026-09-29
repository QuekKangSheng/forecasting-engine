"""Promotion-gate policy for a candidate model, kept separate from the
statistics that produce the numbers (compute_pbo, rank_ic) per Validation
Metrics v2 Section 3. Crash recall/precision/F1 never appear here — they are
a diagnostic with no threshold, so they cannot fail a gate.
"""

from __future__ import annotations

from dataclasses import dataclass

OOS_RANK_IC_GATE: float = 0.02
"""Tier 1 gate: pooled OOS Rank IC must exceed this (strictly) to validate a
candidate."""

PBO_GATE: float = 0.5
"""Tier 2 gate: PBO must be at or below this."""


@dataclass(frozen=True)
class ValidationOutcome:
    promoted: bool
    failed_gates: tuple[str, ...]


def evaluate_candidate(oos_rank_ic: float, pbo: float | None) -> ValidationOutcome:
    """Apply the OOS Rank IC and PBO promotion gates. A NaN input (e.g. a
    candidate whose IC couldn't be computed) fails its gate rather than
    passing or raising. ``pbo=None`` means no configuration search happened
    (a benchmark with no hyperparameters, or a user-supplied function) — the
    same "no configuration search" case ``reporting.model_metrics`` already
    renders as N/A rather than a pass/fail badge, so that gate is skipped
    here too rather than raising on the comparison ``None <= PBO_GATE``."""
    failed = []
    if not (oos_rank_ic > OOS_RANK_IC_GATE):
        failed.append("oos_rank_ic")
    if pbo is not None and not (pbo <= PBO_GATE):
        failed.append("pbo")
    return ValidationOutcome(promoted=not failed, failed_gates=tuple(failed))
