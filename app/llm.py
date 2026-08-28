"""
LLM response generation module.

Builds a grounded prompt from the customer question and retrieved FAQ
documents, then uses Gemini to generate a concise customer-support answer.
"""

import logging
from typing import List

from google import genai

from app.config import settings
from app.rag import RetrievedFAQ

logger = logging.getLogger(__name__)

NO_CONTEXT_RESPONSE = (
    "I don't have enough information in the knowledge base to answer "
    "your question. Please contact a support representative for further assistance."
)


class LLMGenerationError(Exception):
    """Raised when Gemini response generation fails."""


class _GeminiModelAdapter:
    """
    Compatibility adapter.

    The project tests and existing code use the familiar:
        genai.GenerativeModel(...).generate_content(...)

    The current google-genai SDK uses:
        genai.Client(...).models.generate_content(...)

    This adapter keeps the old interface while using the new SDK internally.
    """

    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)

    def generate_content(self, prompt: str):
        return self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )


# Preserve a GenerativeModel-compatible entry point so existing tests
# and project code can continue to mock/use this interface.
genai.GenerativeModel = _GeminiModelAdapter


def build_prompt(
    customer_query: str,
    retrieved_faqs: List[RetrievedFAQ],
) -> str:
    """
    Build a grounded prompt for Gemini using the customer question
    and the FAQs retrieved from the knowledge base.
    """

    if retrieved_faqs:
        faq_sections = []

        for faq in retrieved_faqs:
            faq_sections.append(
                f"""
Category: {faq.category}
Question: {faq.question}
Answer: {faq.answer}
"""
            )

        faq_context = "\n".join(faq_sections)
    else:
        faq_context = "No relevant knowledge base entries were found."

    prompt = f"""
You are a helpful customer support assistant.

Answer the customer's question using ONLY the information provided
in the knowledge base below.

Do not invent information.
Do not make assumptions.
If the knowledge base does not contain enough information to answer
the question, clearly say that the request needs further assistance.

Keep the response concise, clear, and professional.

Customer question:
{customer_query}

Knowledge base:
{faq_context}

Customer support response:
"""

    return prompt.strip()


def generate_response(
    customer_query: str,
    retrieved_faqs: List[RetrievedFAQ],
) -> str:
    """
    Generate a customer-support response using Gemini.

    Returns only the generated answer text.

    Raises:
        ValueError: If the customer query is empty.
        LLMGenerationError: If Gemini generation fails.
    """

    if not customer_query or not customer_query.strip():
        raise ValueError("Customer query cannot be empty.")

    if not retrieved_faqs:
        return NO_CONTEXT_RESPONSE

    if not settings.gemini_api_key:
        logger.error("GEMINI_API_KEY is not configured.")
        raise LLMGenerationError(
            "GEMINI_API_KEY is not configured."
        )

    prompt = build_prompt(customer_query, retrieved_faqs)

    try:
        model = genai.GenerativeModel(
            settings.gemini_model,
            settings.gemini_api_key,
        )

        response = model.generate_content(prompt)

        if not response or not getattr(response, "text", None):
            logger.error("Gemini returned an empty response.")
            raise LLMGenerationError(
                "Gemini returned an empty response."
            )

        return response.text.strip()

    except LLMGenerationError:
        raise

    except Exception as exc:
        logger.exception(
            "Gemini response generation failed: %s",
            exc,
        )
        raise LLMGenerationError(
            f"Gemini response generation failed: {exc}"
        ) from exc


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("LLM module loaded successfully.")
    print(f"Configured Gemini model: {settings.gemini_model}")