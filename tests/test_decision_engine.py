"""Decision-engine behaviour. Findings here are hand-written test inputs, not model output."""

import pytest

from pcb_inspection.decision_engine import DecisionConfig, Finding, Rule, StageResult, Verdict, decide

STAGES = ("image_quality", "alignment", "inspection")
OK_STAGES = [StageResult(name, ok=True) for name in STAGES]


def config(*rules):
    return DecisionConfig(rules=tuple(rules), required_stages=STAGES)


MISSING = Rule("missing_component", fail_threshold=0.8, review_threshold=0.4)


def test_shipped_profile_can_never_pass(active_profile):
    shipped = DecisionConfig.from_mapping(active_profile["decision_engine"])
    assert decide([], OK_STAGES, shipped).verdict is Verdict.REVIEW
    finding = Finding("missing_component", 0.99, "test", component_critical=True)
    assert decide([finding], OK_STAGES, shipped).verdict is Verdict.REVIEW


def test_pass_requires_all_required_stages():
    assert decide([], OK_STAGES, config(MISSING)).verdict is Verdict.PASS
    assert decide([], OK_STAGES[:2], config(MISSING)).verdict is Verdict.REVIEW
    failed = [*OK_STAGES[:2], StageResult("inspection", ok=False, detail="timeout")]
    decision = decide([], failed, config(MISSING))
    assert decision.verdict is Verdict.REVIEW
    assert "timeout" in decision.reasons[0].message


@pytest.mark.parametrize("confidence, expected", [(0.9, Verdict.FAIL), (0.8, Verdict.FAIL), (0.5, Verdict.REVIEW), (0.1, Verdict.PASS)])
def test_thresholds(confidence, expected):
    finding = Finding("missing_component", confidence, "test", component_refdes="C12")
    decision = decide([finding], OK_STAGES, config(MISSING))
    assert decision.verdict is expected
    assert any("C12" in r.message for r in decision.reasons)


def test_unknown_defect_type_goes_to_review():
    finding = Finding("never_seen_type", 0.01, "test")
    assert decide([finding], OK_STAGES, config(MISSING)).verdict is Verdict.REVIEW


def test_unconfigured_rule_goes_to_review():
    rule = Rule("broken_trace", fail_threshold=None, review_threshold=None)
    finding = Finding("broken_trace", 0.01, "test")
    assert decide([finding], OK_STAGES, config(rule)).verdict is Verdict.REVIEW


def test_critical_component_rule_takes_precedence():
    critical = Rule("missing_component", fail_threshold=0.5, review_threshold=0.2, applies_to="critical")
    rules = config(critical, MISSING)
    assert decide([Finding("missing_component", 0.6, "t", component_critical=True)], OK_STAGES, rules).verdict is Verdict.FAIL
    assert decide([Finding("missing_component", 0.6, "t", component_critical=False)], OK_STAGES, rules).verdict is Verdict.REVIEW
    assert decide([Finding("missing_component", 0.6, "t")], OK_STAGES, rules).verdict is Verdict.REVIEW


def test_fail_dominates_review():
    findings = [Finding("missing_component", 0.5, "t"), Finding("missing_component", 0.95, "t")]
    assert decide(findings, OK_STAGES, config(MISSING)).verdict is Verdict.FAIL


def test_invalid_rules_and_findings_are_rejected():
    with pytest.raises(ValueError):
        Rule("x", fail_threshold=0.3, review_threshold=0.6)
    with pytest.raises(ValueError):
        Rule("x", fail_threshold=1.5, review_threshold=0.1)
    with pytest.raises(ValueError):
        Rule("x", fail_threshold=0.9, review_threshold=0.1, applies_to="sometimes")
    with pytest.raises(ValueError):
        Finding("x", 1.2, "t")
