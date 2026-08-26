"""
Escalation module: decides whether a classified ticket should be
escalated to a human, with a visible, human-readable reason.

This module implements the confidence-check step of the pipeline
(see Sec. 4 / Sec. 9 of the spec):

    Ticket Classifier (app/classifier.py)
            |
    category + confidence
            |
            v
    check_escalation(category, confidence)   <-- this module
            |
    +-------+-------+
    |               |
High Confidence  Low Confidence / Unknown
    |               |
    v               v
  Continue      Escalate to Human
  to RAG        (visible reason)

It deliberately does ONE job: turn (category, confidence) into an
escalation decision. It doesn't call the classifier, RAG, or the LLM
itself — those are wired together in a later integration phase.
"""

import logging
from dataclasses import dataclass

from app.config import settings

logger = logging.getLogger(__name__)

# Human-readable reasons. Kept as named constants so the exact wording
# is defined in exactly one place and reused consistently everywhere
# escalation happens (module code, tests, and later the API/UI layer).
REASON_LOW_CONFIDENCE = "Low classification confidence."
REASON_UNKNOWN_CATEGORY = "Unable to classify the request."


@dataclass
class EscalationResult:
    """The outcome of an escalation check for a single ticket."""

    should_escalate: bool
    reason: str
    category: str
    confidence: float


def check_escalation(category: str, confidence: float) -> EscalationResult:
    """
    Decide whether a ticket should be escalated to a human, based on the
    classifier's predicted category and confidence score.

    Escalation rules (checked in order):
      1. category == "Unknown" (e.g. empty/blank input, or any input the
         classifier itself could not meaningfully categorize)
         -> always escalate, reason: "Unable to classify the request."
      2. confidence < settings.confidence_threshold (default 0.70)
         -> escalate, reason: "Low classification confidence."
      3. otherwise (confidence >= threshold, category is known)
         -> do not escalate.

    Args:
        category: the predicted category, e.g. "Billing", "Technical",
            "Account Access", or "Unknown".
        confidence: the classifier's confidence score for that category,
            in the range [0.0, 1.0].

    Returns:
        EscalationResult with should_escalate, a visible human-readable
        reason (empty string if not escalating), and the original
        category/confidence passed through unchanged so callers don't
        need to track them separately.
    """
    if category == "Unknown":
        logger.info(
            "Escalating: category is Unknown (confidence=%.2f)", confidence
        )
        return EscalationResult(
            should_escalate=True,
            reason=REASON_UNKNOWN_CATEGORY,
            category=category,
            confidence=confidence,
        )

    if confidence < settings.confidence_threshold:
        logger.info(
            "Escalating: confidence %.2f is below threshold %.2f (category=%s)",
            confidence,
            settings.confidence_threshold,
            category,
        )
        return EscalationResult(
            should_escalate=True,
            reason=REASON_LOW_CONFIDENCE,
            category=category,
            confidence=confidence,
        )

    logger.info(
        "Not escalating: confidence %.2f meets threshold %.2f (category=%s)",
        confidence,
        settings.confidence_threshold,
        category,
    )
    return EscalationResult(
        should_escalate=False,
        reason="",
        category=category,
        confidence=confidence,
    )


if __name__ == "__main__":
    # Manual test harness. Run with:
    #     python -m app.escalation
    logging.basicConfig(level=logging.INFO)

    test_cases = [
        ("Billing", 0.96),
        ("Technical", 0.36),
        ("Account Access", 0.70),
        ("Unknown", 0.0),
    ]

    for category, confidence in test_cases:
        result = check_escalation(category, confidence)
        print(f"\ncategory={category}, confidence={confidence}")
        print(f"  should_escalate: {result.should_escalate}")
        print(f"  reason:          {result.reason!r}")