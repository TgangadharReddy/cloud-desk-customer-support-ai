"""
Tests for the RAG retrieval module (app/rag.py).

The production application uses Gemini embeddings.

These tests deliberately mock the embedding layer so the test suite:
- does not consume Gemini API quota
- does not require an API key
- does not depend on network availability
- remains deterministic and fast

The tests still exercise the real FAISS indexing and retrieval logic.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

import app.rag as rag


def _fake_embed_texts(texts):
    """
    Deterministic local replacement for Gemini embeddings.

    Creates a small semantic-style vector:
        [billing, technical, account_access]

    This is only for testing FAISS retrieval mechanics. The real
    application continues to use Gemini embeddings.
    """
    vectors = []

    for text in texts:
        text_lower = text.lower()

        billing = any(
            word in text_lower
            for word in (
                "charged",
                "charge",
                "billing",
                "bill",
                "subscription",
                "payment",
                "refund",
                "invoice",
            )
        )

        technical = any(
            word in text_lower
            for word in (
                "500",
                "error",
                "technical",
                "server",
                "application",
                "crash",
                "bug",
                "slow",
            )
        )

        account = any(
            word in text_lower
            for word in (
                "password",
                "login",
                "log in",
                "sign in",
                "account",
                "access",
                "reset",
            )
        )

        vector = np.array(
            [
                1.0 if billing else 0.0,
                1.0 if technical else 0.0,
                1.0 if account else 0.0,
            ],
            dtype="float32",
        )

        # Give unrelated text a small deterministic vector so FAISS
        # can still process it, while keeping its similarity below
        # the configured retrieval threshold.
        if not np.any(vector):
            vector = np.array(
                [0.0, 0.0, 0.0],
                dtype="float32",
            )

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        vectors.append(vector)

    return np.asarray(vectors, dtype="float32")


def _fake_embedding_dimension():
    """Return the dimension used by the deterministic test vectors."""
    return 3


def _prepare_test_rag(monkeypatch, tmp_path):
    """
    Configure app.rag to use:
    - deterministic local embeddings
    - temporary FAISS files
    - the real FAQ dataset

    This prevents tests from touching the production vector store.
    """
    test_settings = SimpleNamespace(
        faq_path=Path("data/faq.json"),
        vector_index_path=tmp_path / "faq.index",
        vector_metadata_path=tmp_path / "faq_metadata.pkl",
        top_k_results=3,
        min_retrieval_score=0.40,
    )

    monkeypatch.setattr(rag, "settings", test_settings)
    monkeypatch.setattr(rag, "embed_texts", _fake_embed_texts)
    monkeypatch.setattr(rag, "embedding_dimension", _fake_embedding_dimension)

    # Reset module-level FAISS cache between tests.
    monkeypatch.setattr(rag, "_faiss_index", None)
    monkeypatch.setattr(rag, "_faq_metadata", None)

    return test_settings


def test_build_index_creates_expected_files(monkeypatch, tmp_path):
    settings = _prepare_test_rag(monkeypatch, tmp_path)

    rag.build_index()

    assert Path(settings.vector_index_path).exists()
    assert Path(settings.vector_metadata_path).exists()


def test_retrieve_billing_query_returns_billing_faq(monkeypatch, tmp_path):
    _prepare_test_rag(monkeypatch, tmp_path)

    results = rag.retrieve_relevant_faqs(
        "I was charged twice for my subscription."
    )

    assert len(results) > 0
    assert results[0].category == "Billing"
    assert results[0].score > 0.5


def test_retrieve_technical_query_returns_technical_faq(monkeypatch, tmp_path):
    _prepare_test_rag(monkeypatch, tmp_path)

    results = rag.retrieve_relevant_faqs(
        "I'm getting a 500 error when I try to log in."
    )

    assert len(results) > 0
    assert results[0].category == "Technical"


def test_retrieve_account_access_query_returns_account_faq(
    monkeypatch, tmp_path
):
    _prepare_test_rag(monkeypatch, tmp_path)

    results = rag.retrieve_relevant_faqs(
        "I forgot my password."
    )

    assert len(results) > 0
    assert results[0].category == "Account Access"


def test_retrieve_out_of_scope_query_returns_no_results(
    monkeypatch, tmp_path
):
    _prepare_test_rag(monkeypatch, tmp_path)

    results = rag.retrieve_relevant_faqs(
        "What will the weather be tomorrow?"
    )

    assert len(results) == 0


def test_retrieve_respects_top_k(monkeypatch, tmp_path):
    _prepare_test_rag(monkeypatch, tmp_path)

    results = rag.retrieve_relevant_faqs(
        "How do I reset my password?",
        top_k=1,
    )

    assert len(results) <= 1


def test_retrieve_results_are_sorted_by_score_descending(
    monkeypatch, tmp_path
):
    _prepare_test_rag(monkeypatch, tmp_path)

    results = rag.retrieve_relevant_faqs(
        "password reset issue",
        top_k=5,
    )

    scores = [result.score for result in results]

    assert scores == sorted(scores, reverse=True)


def test_retrieve_empty_query_returns_empty_list(
    monkeypatch, tmp_path
):
    _prepare_test_rag(monkeypatch, tmp_path)

    assert rag.retrieve_relevant_faqs("") == []
    assert rag.retrieve_relevant_faqs("   ") == []


def test_index_is_detected_as_stale_when_faq_is_newer(
    monkeypatch, tmp_path
):
    """The FAISS index is stale when faq.json is newer."""

    import os
    import time

    faq_path = tmp_path / "faq.json"
    index_path = tmp_path / "faq.index"
    metadata_path = tmp_path / "faq_metadata.pkl"

    faq_path.write_text("[]", encoding="utf-8")
    index_path.write_text("index", encoding="utf-8")
    metadata_path.write_text("metadata", encoding="utf-8")

    old_time = time.time() - 100
    os.utime(index_path, (old_time, old_time))
    os.utime(metadata_path, (old_time, old_time))

    new_time = time.time()
    os.utime(faq_path, (new_time, new_time))

    test_settings = SimpleNamespace(
        faq_path=faq_path,
        vector_index_path=index_path,
        vector_metadata_path=metadata_path,
    )

    monkeypatch.setattr(rag, "settings", test_settings)

    assert rag._index_files_are_current() is False