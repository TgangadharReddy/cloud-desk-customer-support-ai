"""
RAG retrieval module: FAISS indexing + similarity search over the FAQ
knowledge base.

This module does exactly two jobs:

1. build_index()
   Reads data/faq.json, embeds every FAQ question, and writes a FAISS
   index + a metadata file to vector_store/.

2. retrieve_relevant_faqs(query)
   Embeds a customer query and searches the FAISS index for the top-K
   most similar FAQs, returning each with its similarity score.

Pipeline this module implements (see Sec. 6 of the project spec):

    Indexing (build_index):

        FAQ Documents (data/faq.json)
                |
                v
        Sentence Transformer   (app/embeddings.py)
                |
                v
        Embeddings
                |
                v
        FAISS IndexFlatIP  --> saved to vector_store/faq.index
                |
                v
        FAQ metadata        --> saved to vector_store/faq_metadata.pkl

    Querying (retrieve_relevant_faqs):

        Customer Question
                |
                v
        Sentence Transformer
                |
                v
        Question Embedding
                |
                v
        FAISS Similarity Search
                |
                v
        Top-K Relevant FAQ Documents (id, category, question, answer, score)
"""

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import faiss

from app.config import settings
from app.embeddings import embed_texts, embedding_dimension

logger = logging.getLogger(__name__)


@dataclass
class RetrievedFAQ:
    """A single FAQ retrieved from the vector store, with its similarity score."""

    id: str
    category: str
    question: str
    answer: str
    score: float


def _load_faq_documents() -> List[dict]:
    """Load the raw FAQ knowledge base from data/faq.json."""
    faq_path = Path(settings.faq_path)
    if not faq_path.exists():
        raise FileNotFoundError(
            f"Knowledge base not found at {faq_path}. "
            "Make sure data/faq.json exists (see Phase 2)."
        )
    with open(faq_path, "r", encoding="utf-8") as f:
        documents = json.load(f)
    logger.info("Loaded %d FAQ documents from %s", len(documents), faq_path)
    return documents


def build_index() -> None:
    """
    Build the FAISS index from data/faq.json and persist it to disk.

    We embed only the FAQ *question* text, not the answer. Customer
    queries are also questions, so question-to-question matching gives
    the closest semantic match. The answer text is never embedded — it's
    stored as metadata and returned alongside the matched question, used
    only after retrieval (by the LLM in a later phase), not during
    similarity search itself.
    """
    documents = _load_faq_documents()
    questions = [doc["question"] for doc in documents]

    logger.info("Embedding %d FAQ questions...", len(questions))
    embeddings = embed_texts(questions)

    dimension = embedding_dimension()

    # IndexFlatIP = exact (brute-force) inner-product search.
    # Combined with normalized embeddings (see app/embeddings.py), inner
    # product equals cosine similarity. This is intentionally the
    # simplest FAISS index type available: with only ~30 FAQs, an
    # approximate index (IVF, HNSW, etc.) would add complexity for zero
    # benefit — brute-force search over 30 vectors is effectively instant.
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    vector_store_dir = Path(settings.vector_index_path).parent
    vector_store_dir.mkdir(parents=True, exist_ok=True)

    # faiss's Python bindings (SWIG) require a plain str filename, not a
    # pathlib.Path — passing a Path directly raises:
    #   TypeError: Wrong number or type of arguments for overloaded function 'write_index'
    faiss.write_index(index, str(settings.vector_index_path))

    # Metadata list order MUST match the order vectors were added to the
    # index: FAISS returns positional indices (0, 1, 2, ...) from search(),
    # and we use those positions to look back up into this list to recover
    # the original id / category / question / answer.
    with open(settings.vector_metadata_path, "wb") as f:
        pickle.dump(documents, f)

    logger.info(
        "FAISS index built: %d vectors, dimension %d. Saved to %s and %s",
        index.ntotal,
        dimension,
        settings.vector_index_path,
        settings.vector_metadata_path,
    )


def _index_files_are_current() -> bool:
    """
    Return True when both FAISS files exist and were created/updated
    after the FAQ knowledge base was last modified.

    If data/faq.json is newer than either vector-store file, the index
    is considered stale and will be rebuilt.
    """
    faq_path = Path(settings.faq_path)
    index_path = Path(settings.vector_index_path)
    metadata_path = Path(settings.vector_metadata_path)

    if not faq_path.exists():
        raise FileNotFoundError(
            f"Knowledge base not found at {faq_path}. "
            "Make sure data/faq.json exists."
        )

    if not index_path.exists() or not metadata_path.exists():
        return False

    faq_mtime = faq_path.stat().st_mtime
    index_mtime = index_path.stat().st_mtime
    metadata_mtime = metadata_path.stat().st_mtime

    return index_mtime >= faq_mtime and metadata_mtime >= faq_mtime


# Module-level cache so we only load the FAISS index and metadata from
# disk once per process, not on every single retrieve() call.
_faiss_index: Optional[faiss.Index] = None
_faq_metadata: Optional[List[dict]] = None


def _ensure_index_loaded() -> None:
    """
    Load the FAISS index and FAQ metadata into memory, building them
    first from data/faq.json if they don't exist on disk yet. This means
    a fresh checkout of the project "just works" on the first query,
    without a separate manual build step.
    """
    global _faiss_index, _faq_metadata

    if _faiss_index is not None and _faq_metadata is not None:
        return


    if not _index_files_are_current():
        logger.info(
            "FAISS index is missing or stale relative to %s. Rebuilding...",
            settings.faq_path,
            )
        build_index()

    # Same str() conversion needed here as in write_index above.
    _faiss_index = faiss.read_index(str(settings.vector_index_path))
    with open(settings.vector_metadata_path, "rb") as f:
        _faq_metadata = pickle.load(f)

    logger.info("FAISS index loaded into memory: %d vectors", _faiss_index.ntotal)


def retrieve_relevant_faqs(
    query: str,
    top_k: Optional[int] = None,
    min_score: Optional[float] = None,
) -> List[RetrievedFAQ]:
    """
    Retrieve the top-K most relevant FAQs for a customer query.

    Args:
        query: the customer's raw question text.
        top_k: how many results to return at most. Defaults to
            settings.top_k_results (configurable via .env, per Sec. 9).
        min_score: results with cosine similarity below this value are
            dropped entirely. Defaults to settings.min_retrieval_score.
            This threshold is what lets a later phase (escalation)
            detect "the knowledge base has nothing relevant to this
            question" — if filtering leaves zero results, that's a
            strong signal to escalate rather than let the LLM guess.

    Returns:
        A list of RetrievedFAQ, ordered from most to least similar,
        already filtered by min_score. Empty list if nothing relevant
        was found (including for empty/blank queries).
    """
    if not query or not query.strip():
        return []

    top_k = top_k if top_k is not None else settings.top_k_results
    min_score = min_score if min_score is not None else settings.min_retrieval_score

    _ensure_index_loaded()

    query_embedding = embed_texts([query])  # shape: (1, dimension)
    scores, indices = _faiss_index.search(query_embedding, top_k)
    # scores: shape (1, top_k) -- cosine similarity, since embeddings are normalized
    # indices: shape (1, top_k) -- positional indices into _faq_metadata

    results: List[RetrievedFAQ] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            # FAISS pads with -1 when there are fewer than top_k vectors
            # in the index in total (e.g. asking for top 5 out of 3).
            continue
        if score < min_score:
            continue
        doc = _faq_metadata[idx]
        results.append(
            RetrievedFAQ(
                id=doc["id"],
                category=doc["category"],
                question=doc["question"],
                answer=doc.get("answer"),
                score=float(score),
            )
        )

    return results


if __name__ == "__main__":
    # Manual test harness. Run with:
    #     python -m app.rag
    # Rebuilds the index from data/faq.json and runs a few sample queries
    # covering each category plus an out-of-scope case.
    logging.basicConfig(level=logging.INFO)

    build_index()

    test_queries = [
        "I was charged twice for my subscription.",
        "I'm getting a 500 error when I try to log in.",
        "I forgot my password. How can I reset it?",
        "What will the weather be tomorrow?",  # expect no matches above threshold
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        results = retrieve_relevant_faqs(q)
        if not results:
            print("  No relevant FAQs found above min_score threshold.")
        for r in results:
            print(f"  [{r.score:.3f}] ({r.category}) {r.question}")