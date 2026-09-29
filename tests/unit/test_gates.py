import math

from forecasting_engine.validation.gates import evaluate_candidate


def test_promoted_when_both_gates_pass():
    outcome = evaluate_candidate(oos_rank_ic=0.03, pbo=0.4)
    assert outcome.promoted is True
    assert outcome.failed_gates == ()


def test_not_promoted_when_oos_rank_ic_gate_fails():
    outcome = evaluate_candidate(oos_rank_ic=0.01, pbo=0.4)
    assert outcome.promoted is False
    assert outcome.failed_gates == ("oos_rank_ic",)


def test_not_promoted_when_pbo_gate_fails():
    outcome = evaluate_candidate(oos_rank_ic=0.03, pbo=0.6)
    assert outcome.promoted is False
    assert outcome.failed_gates == ("pbo",)


def test_not_promoted_when_both_gates_fail():
    outcome = evaluate_candidate(oos_rank_ic=0.01, pbo=0.6)
    assert outcome.promoted is False
    assert outcome.failed_gates == ("oos_rank_ic", "pbo")


def test_oos_rank_ic_gate_is_strict_at_the_boundary():
    outcome = evaluate_candidate(oos_rank_ic=0.02, pbo=0.4)
    assert outcome.promoted is False
    assert outcome.failed_gates == ("oos_rank_ic",)


def test_pbo_gate_is_inclusive_at_the_boundary():
    outcome = evaluate_candidate(oos_rank_ic=0.03, pbo=0.5)
    assert outcome.promoted is True
    assert outcome.failed_gates == ()


def test_nan_inputs_fail_closed_rather_than_pass_or_raise():
    outcome = evaluate_candidate(oos_rank_ic=math.nan, pbo=math.nan)
    assert outcome.promoted is False
    assert outcome.failed_gates == ("oos_rank_ic", "pbo")


def test_pbo_none_skips_that_gate_rather_than_raising():
    # pbo=None means no configuration search happened (e.g. FF5 or a
    # user-supplied function) — unlike NaN, this isn't a failure to compute
    # something, so the PBO gate is skipped rather than failed closed.
    outcome = evaluate_candidate(oos_rank_ic=0.03, pbo=None)
    assert outcome.promoted is True
    assert outcome.failed_gates == ()


def test_pbo_none_still_lets_the_ic_gate_fail():
    outcome = evaluate_candidate(oos_rank_ic=0.01, pbo=None)
    assert outcome.promoted is False
    assert outcome.failed_gates == ("oos_rank_ic",)
