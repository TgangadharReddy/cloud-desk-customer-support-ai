"""
Embedding model wrapper.

Turns FAQ documents and customer queries into dense vectors using a
Sentence Transformer model. Both FAQs (at index-build time) and customer
queries (at search time) MUST use this same function, or their vectors
would live in different spaces and similarity scores would be meaningless.

Where this sits in the RAG pipeline:

    FAQ text / Customer query
            |
            v
    SentenceTransformer.encode()   <-- this module
            |
            v
    Normalized embedding vector (compared via FAISS in app/rag.py)
"""

import logging
from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """
    Load the Sentence Transformer model once and cache it.

    Loading a transformer model is slow (seconds) and memory-heavy, so we
    only want to do it once per process, not on every request. lru_cache
    with maxsize=1 gives us a simple singleton without extra boilerplate
    (no need for a class or global variable management).
    """
    logger.info("Loading embedding model: %s", settings.embedding_model)
    model = SentenceTransformer(settings.embedding_model)
    logger.info(
        "Embedding model loaded. Vector dimension: %d",
        model.get_embedding_dimension()
    )
    return model


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Encode a list of texts into normalized embedding vectors.

    normalize_embeddings=True makes every vector unit length. This matters
    because we use a FAISS inner-product index (IndexFlatIP): the inner
    product of two unit vectors equals their cosine similarity. Without
    normalization we'd get raw dot products, which are harder to reason
    about and don't have a fixed, comparable range.

    Returns a float32 numpy array of shape (len(texts), embedding_dim),
    since FAISS requires float32 input.
    """
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return embeddings.astype("float32")


def embedding_dimension() -> int:
    """Return the vector dimension produced by the current embedding model."""
    return get_embedding_model().get_embedding_dimension()