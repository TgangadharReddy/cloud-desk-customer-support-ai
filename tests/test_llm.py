"""
Tests for the LLM response generation module (app/llm.py).

Run with:
    pytest tests/test_llm.py -v

All Gemini API calls are mocked — no real network requests or API key
are needed to run these tests.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app import llm
from app.llm import (
    LLMGenerationError,
    NO_CONTEXT_RESPONSE,
    build_prompt,
    generate_response,
)


class FakeFAQ:
    """
    A minimal stand-in for app.rag.RetrievedFAQ.
    """

    def __init__(self, id, category, question, answer, score):
        self.id = id
        self.category = category
        self.question = question
        self.answer = answer
        self.score = score


@pytest.fixture
def sample_faqs():
    return [
        FakeFAQ(
            id="billing_001",
            category="Billing",
            question="I was charged twice for my subscription this month.",
            answer=(
                "Please provide the transaction IDs for both charges so "
                "our billing team can verify the duplicate and issue a refund."
            ),
            score=0.81,
        )
    ]


@pytest.fixture(autouse=True)
def fake_settings(monkeypatch):
    """
    Use a fake API key and model so tests never depend on real credentials.
    """
    monkeypatch.setattr(
        llm,
        "settings",
        SimpleNamespace(
            gemini_api_key="fake-test-key",
            gemini_model="gemini-2.5-flash",
        ),
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def test_build_prompt_includes_customer_query(sample_faqs):
    prompt = build_prompt("I was charged twice.", sample_faqs)

    assert "I was charged twice." in prompt


def test_build_prompt_includes_retrieved_faq_content(sample_faqs):
    prompt = build_prompt("I was charged twice.", sample_faqs)

    assert sample_faqs[0].question in prompt
    assert sample_faqs[0].answer in prompt


def test_build_prompt_instructs_model_to_avoid_hallucination(sample_faqs):
    prompt = build_prompt("I was charged twice.", sample_faqs)

    assert "only" in prompt.lower()
    assert "do not invent" in prompt.lower()


def test_build_prompt_handles_empty_retrieval_without_error():
    prompt = build_prompt(
        "What will the weather be tomorrow?",
        [],
    )

    assert "no relevant knowledge base entries" in prompt.lower()


# ---------------------------------------------------------------------------
# generate_response: empty retrieval
# ---------------------------------------------------------------------------


def test_generate_response_returns_fallback_when_no_faqs_retrieved():
    result = generate_response(
        "What will the weather be tomorrow?",
        [],
    )

    assert result == NO_CONTEXT_RESPONSE


def test_generate_response_does_not_call_gemini_when_no_faqs_retrieved():
    with patch("app.llm.genai.Client") as mock_client:
        generate_response(
            "What will the weather be tomorrow?",
            [],
        )

        mock_client.assert_not_called()


# ---------------------------------------------------------------------------
# generate_response: successful call
# ---------------------------------------------------------------------------


def test_generate_response_returns_gemini_text_on_success(sample_faqs):
    mock_client = MagicMock()

    mock_client.models.generate_content.return_value = SimpleNamespace(
        text=(
            "It looks like a duplicate charge. Please share both "
            "transaction IDs so billing can verify and refund it."
        )
    )

    with patch(
        "app.llm.genai.Client",
        return_value=mock_client,
    ) as mock_client_cls:

        result = generate_response(
            "I was charged twice.",
            sample_faqs,
        )

    assert result == (
        "It looks like a duplicate charge. Please share both "
        "transaction IDs so billing can verify and refund it."
    )

    mock_client_cls.assert_called_once_with(
        api_key="fake-test-key"
    )

    mock_client.models.generate_content.assert_called_once()


# ---------------------------------------------------------------------------
# generate_response: failure handling
# ---------------------------------------------------------------------------


def test_generate_response_raises_on_api_error(sample_faqs):
    mock_client = MagicMock()

    mock_client.models.generate_content.side_effect = RuntimeError(
        "network error"
    )

    with patch(
        "app.llm.genai.Client",
        return_value=mock_client,
    ):

        with pytest.raises(LLMGenerationError):
            generate_response(
                "I was charged twice.",
                sample_faqs,
            )


def test_generate_response_raises_on_empty_gemini_response(sample_faqs):
    mock_client = MagicMock()

    mock_client.models.generate_content.return_value = SimpleNamespace(
        text=""
    )

    with patch(
        "app.llm.genai.Client",
        return_value=mock_client,
    ):

        with pytest.raises(LLMGenerationError):
            generate_response(
                "I was charged twice.",
                sample_faqs,
            )


def test_generate_response_raises_value_error_on_empty_query(sample_faqs):
    with pytest.raises(ValueError):
        generate_response(
            "",
            sample_faqs,
        )