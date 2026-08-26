"""
Tests for the RAG retrieval module (app/rag.py).

Run with:
    pytest tests/test_rag.py -v

Note: the first test run builds the FAISS index from data/faq.json,
which downloads the Sentence Transformer model on first use if it isn't
already cached locally, so it requires internet access the first time.
"""

from pathlib import Path

from app.config import settings
from app.rag import build_index, retrieve_relevant_faqs


def test_build_index_creates_expected_files():
    build_index()
    assert Path(settings.vector_index_path).exists()
    assert Path(settings.vector_metadata_path).exists()


def test_retrieve_billing_query_returns_billing_faq():
    results = retrieve_relevant_faqs("I was charged twice for my subscription.")
    assert len(results) > 0
    assert results[0].category == "Billing"
    assert results[0].score > 0.5


def test_retrieve_technical_query_returns_technical_faq():
    results = retrieve_relevant_faqs("I'm getting a 500 error when I try to log in.")
    assert len(results) > 0
    assert results[0].category == "Technical"


def test_retrieve_account_access_query_returns_account_faq():
    results = retrieve_relevant_faqs("I forgot my password.")
    assert len(results) > 0
    assert results[0].category == "Account Access"


def test_retrieve_out_of_scope_query_returns_no_results():
    # An unrelated query should not match any FAQ above min_score.
    results = retrieve_relevant_faqs("What will the weather be tomorrow?")
    assert len(results) == 0


def test_retrieve_respects_top_k():
    results = retrieve_relevant_faqs("How do I reset my password?", top_k=1)
    assert len(results) <= 1


def test_retrieve_results_are_sorted_by_score_descending():
    results = retrieve_relevant_faqs("password reset issue", top_k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_empty_query_returns_empty_list():
    assert retrieve_relevant_faqs("") == []
    assert retrieve_relevant_faqs("   ") == []
def test_index_is_detected_as_stale_when_faq_is_newer(monkeypatch, tmp_path):
    """The FAISS index is stale when faq.json is newer."""

    import os
    import time
    import app.rag as rag

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

    from types import SimpleNamespace
    test_settings = SimpleNamespace(
        faq_path=faq_path,
        vector_index_path=index_path,
        vector_metadata_path=metadata_path,
        )
    monkeypatch.setattr(rag, "settings", test_settings)