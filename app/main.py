"""
FastAPI application entrypoint — Phase 7 full pipeline integration.

Wires together the four independently-built and independently-tested
modules from Phases 3-6 into one end-to-end request/response flow. This
file does orchestration ONLY — none of the underlying logic (how to
classify, how to decide escalation, how to retrieve, how to prompt
Gemini) lives here; it all stays in its own module from its own phase.

    Customer Message
            |
            v
    classify_ticket()              (app/classifier.py)
            |
    category + confidence
            |
            v
    check_escalation()              (app/escalation.py)
            |
    +-------+-------+
    |               |
 escalate        continue
    |               |
    v               v
 (stop here,   retrieve_relevant_faqs()   (app/rag.py)
  no Gemini          |
  call made)   +-----+-----+
               |           |
          no FAQs      FAQs found
               |           |
               v           v
           escalate   generate_response()   (app/llm.py)
               |           |
               +-----+-----+
                     |
                     v
              SupportResponse
                     |
                     v
              FastAPI JSON response
"""

import logging
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.classifier import classify_ticket
from app.config import settings
from app.escalation import check_escalation
from app.llm import LLMGenerationError, generate_response
from app.rag import retrieve_relevant_faqs
from fastapi.middleware.cors import CORSMiddleware
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CloudDesk Customer Support AI Employee",
    description="Tier-1 support triage: classification + RAG + escalation.",
    version="0.2.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request / response models ------------------------------------------------


class SupportRequest(BaseModel):
    """
    Incoming customer message. `message` intentionally has no min_length
    constraint: an empty/blank message is a VALID input that should flow
    through the pipeline and come back as a normal 200 escalation
    response (category "Unknown"), not a 422 validation error — that's
    how app.classifier.classify_ticket() already handles blank text.
    """

    message: str = Field(..., description="The customer's support message.")


class RetrievedFAQOut(BaseModel):
    """A single retrieved FAQ, shaped for the API response."""

    id: str
    category: str
    question: str
    answer: str
    score: float


class SupportResponse(BaseModel):
    """
    Full triage result for one customer message: what it was classified
    as, whether it was escalated (and why), what knowledge-base context
    was used (if any), and the final customer-facing response text.
    """

    customer_message: str
    category: str
    confidence: float
    escalated: bool
    escalation_reason: Optional[str] = None
    retrieved_faqs: List[RetrievedFAQOut] = Field(default_factory=list)
    response: str


# --- Pipeline orchestration ----------------------------------------------------

# One consistent template for every escalation-triggered response, so the
# customer-facing wording is defined in exactly one place.
ESCALATION_RESPONSE_TEMPLATE = (
    "This request needs to be reviewed by our human support team. Reason: {reason}"
)

REASON_NO_RELEVANT_FAQ = "No relevant knowledge-base information was found for this request."
REASON_LLM_FAILURE = "The AI assistant was unable to generate a response."


def _to_retrieved_faq_out(faqs) -> List[RetrievedFAQOut]:
    """Convert app.rag.RetrievedFAQ dataclasses into API response models."""
    return [
        RetrievedFAQOut(id=f.id, category=f.category, question=f.question, answer=f.answer, score=f.score)
        for f in faqs
    ]


def process_customer_message(message: str) -> SupportResponse:
    """
    Run the full Tier-1 triage pipeline for a single customer message.

    This is the one function that ties together classify_ticket(),
    check_escalation(), retrieve_relevant_faqs(), and generate_response().
    Each of those keeps doing exactly the one job it was built and tested
    for in its own phase; this function is purely the wiring between them.
    """
    # Step 1: classify the ticket.
    classification = classify_ticket(message)

    # Step 2: confidence / Unknown-category escalation check.
    # If this escalates, we stop immediately -- no RAG call, no Gemini
    # call. There is no point retrieving context or spending an LLM call
    # on a ticket we already know we can't confidently handle.
    escalation = check_escalation(classification.category, classification.confidence)
    if escalation.should_escalate:
        logger.info("Escalating at classification step: %s", escalation.reason)
        return SupportResponse(
            customer_message=message,
            category=classification.category,
            confidence=classification.confidence,
            escalated=True,
            escalation_reason=escalation.reason,
            retrieved_faqs=[],
            response=ESCALATION_RESPONSE_TEMPLATE.format(reason=escalation.reason),
        )

    # Step 3: RAG retrieval.
    retrieved_faqs = retrieve_relevant_faqs(message)
    if not retrieved_faqs:
        # The classifier was confident about a category, but the
        # knowledge base has nothing relevant to actually answer from.
        # Escalate rather than let Gemini guess (Sec. 11 requirement).
        logger.info("Escalating at retrieval step: %s", REASON_NO_RELEVANT_FAQ)
        return SupportResponse(
            customer_message=message,
            category=classification.category,
            confidence=classification.confidence,
            escalated=True,
            escalation_reason=REASON_NO_RELEVANT_FAQ,
            retrieved_faqs=[],
            response=ESCALATION_RESPONSE_TEMPLATE.format(reason=REASON_NO_RELEVANT_FAQ),
        )

    # Step 4: grounded LLM response generation.
    try:
        answer = generate_response(message, retrieved_faqs)
    except LLMGenerationError as exc:
        # A failed Gemini call (bad/missing key, network error, empty
        # response) becomes a human escalation too -- the customer never
        # sees a raw exception, and support still gets the FAQ context
        # RAG already found, which is useful even without an AI answer.
        logger.error("LLM generation failed, escalating instead: %s", exc)
        return SupportResponse(
            customer_message=message,
            category=classification.category,
            confidence=classification.confidence,
            escalated=True,
            escalation_reason=REASON_LLM_FAILURE,
            retrieved_faqs=_to_retrieved_faq_out(retrieved_faqs),
            response=ESCALATION_RESPONSE_TEMPLATE.format(reason=REASON_LLM_FAILURE),
        )

    return SupportResponse(
        customer_message=message,
        category=classification.category,
        confidence=classification.confidence,
        escalated=False,
        escalation_reason=None,
        retrieved_faqs=_to_retrieved_faq_out(retrieved_faqs),
        response=answer,
    )


# --- API endpoints ---------------------------------------------------------------


@app.get("/health")
def health_check() -> dict:
    """Basic liveness check + confirms config loaded correctly. Unchanged since Phase 1."""
    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        "confidence_threshold": settings.confidence_threshold,
    }


@app.post("/chat", response_model=SupportResponse)
def chat(request: SupportRequest) -> SupportResponse:
    """
    Submit a customer support message and receive the full triage result:
    predicted category, confidence, escalation status/reason (if any),
    retrieved FAQ context (if any), and the final customer-facing response.
    """
    return process_customer_message(request.message)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)