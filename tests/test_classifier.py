"""
Tests for the ticket classifier (app/classifier.py).

Run with:
    pytest tests/test_classifier.py -v
"""

from app.classifier import SUPPORTED_CATEGORIES, ClassificationResult, classify_ticket


def test_billing_message_classified_correctly():
    result = classify_ticket("I was charged twice for my subscription.")
    assert result.category == "Billing"


def test_technical_message_classified_correctly():
    result = classify_ticket("I am getting a 500 error when I try to log in.")
    assert result.category == "Technical"


def test_account_access_message_classified_correctly():
    result = classify_ticket("I forgot my password. How can I reset it?")
    assert result.category == "Account Access"


def test_confidence_score_is_valid_probability():
    result = classify_ticket("My credit card was declined.")
    assert isinstance(result, ClassificationResult)
    assert 0.0 <= result.confidence <= 1.0


def test_predicted_category_is_a_supported_category():
    result = classify_ticket("How do I reset my password?")
    assert result.category in SUPPORTED_CATEGORIES


def test_empty_input_returns_unknown_with_zero_confidence():
    result = classify_ticket("")
    assert result.category == "Unknown"
    assert result.confidence == 0.0


def test_blank_input_returns_unknown_with_zero_confidence():
    result = classify_ticket("   ")
    assert result.category == "Unknown"
    assert result.confidence == 0.0