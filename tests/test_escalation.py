"""
Tests for the escalation module (app/escalation.py).

Run with:
    pytest tests/test_escalation.py -v
"""

from app.escalation import (
    REASON_LOW_CONFIDENCE,
    REASON_UNKNOWN_CATEGORY,
    check_escalation,
)


def test_high_confidence_does_not_escalate():
    result = check_escalation("Billing", 0.96)
    assert result.should_escalate is False
    assert result.reason == ""


def test_low_confidence_escalates():
    result = check_escalation("Technical", 0.36)
    assert result.should_escalate is True
    assert result.reason == REASON_LOW_CONFIDENCE


def test_confidence_exactly_at_threshold_does_not_escalate():
    # Threshold default is 0.70; 0.70 itself should count as confident
    # enough (the escalation rule is strictly "< threshold", not "<=").
    result = check_escalation("Account Access", 0.70)
    assert result.should_escalate is False
    assert result.reason == ""


def test_unknown_category_escalates_regardless_of_confidence():
    result = check_escalation("Unknown", 0.0)
    assert result.should_escalate is True
    assert result.reason == REASON_UNKNOWN_CATEGORY


def test_unknown_category_escalates_even_with_high_confidence_value():
    # Category "Unknown" always escalates, even if a stray confidence
    # value happened to be high — the category check takes priority.
    result = check_escalation("Unknown", 0.99)
    assert result.should_escalate is True
    assert result.reason == REASON_UNKNOWN_CATEGORY


def test_confidence_value_is_preserved_in_result():
    result = check_escalation("Billing", 0.83)
    assert result.confidence == 0.83
    assert result.category == "Billing"


def test_reason_is_present_and_human_readable_when_escalating():
    result = check_escalation("Technical", 0.20)
    assert result.should_escalate is True
    assert isinstance(result.reason, str)
    assert len(result.reason) > 0
    # Basic sanity check that it's a real sentence, not a code/enum value.
    assert result.reason[0].isupper()
    assert result.reason.endswith(".")


def test_reason_is_empty_string_when_not_escalating():
    result = check_escalation("Account Access", 0.91)
    assert result.should_escalate is False
    assert result.reason == ""