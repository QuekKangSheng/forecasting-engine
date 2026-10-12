import math

from forecasting_engine.validation.gates import evaluate_candidate, is_high_risk


def test_promoted_when_both_gates_pass():
    outcome = evaluate_candidate(signal_rank_ic=0.03, t=2.5)
    assert outcome.promoted is True
    assert outcome.failed_gates == ()


def test_not_promoted_when_signal_rank_ic_gate_fails():
    outcome = evaluate_candidate(signal_rank_ic=0.01, t=2.5)
    assert outcome.promoted is False
    assert outcome.failed_gates == ("signal_rank_ic",)


def test_not_promoted_when_t_gate_fails():
    outcome = evaluate_candidate(signal_rank_ic=0.03, t=1.5)
    assert outcome.promoted is False
    assert outcome.failed_gates == ("t",)


def test_not_promoted_when_both_gates_fail():
    outcome = evaluate_candidate(signal_rank_ic=0.01, t=1.5)
    assert outcome.promoted is False
    assert outcome.failed_gates == ("signal_rank_ic", "t")


def test_signal_rank_ic_gate_is_strict_at_the_boundary():
    outcome = evaluate_candidate(signal_rank_ic=0.02, t=2.5)
    assert outcome.promoted is False
    assert outcome.failed_gates == ("signal_rank_ic",)


def test_t_gate_is_strict_at_the_boundary():
    outcome = evaluate_candidate(signal_rank_ic=0.03, t=2.0)
    assert outcome.promoted is False
    assert outcome.failed_gates == ("t",)


def test_nan_inputs_fail_closed_rather_than_pass_or_raise():
    outcome = evaluate_candidate(signal_rank_ic=math.nan, t=math.nan)
    assert outcome.promoted is False
    assert outcome.failed_gates == ("signal_rank_ic", "t")


def test_high_risk_when_both_gates_fail():
    assert is_high_risk(evaluate_candidate(signal_rank_ic=0.01, t=1.5)) is True


def test_not_high_risk_when_only_one_gate_fails():
    assert is_high_risk(evaluate_candidate(signal_rank_ic=0.01, t=2.5)) is False
    assert is_high_risk(evaluate_candidate(signal_rank_ic=0.03, t=1.5)) is False


def test_not_high_risk_when_promoted():
    assert is_high_risk(evaluate_candidate(signal_rank_ic=0.03, t=2.5)) is False
