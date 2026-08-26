"""
Gemini embedding wrapper.

Uses Google's Gemini Embedding API instead of loading a local
Sentence Transformer model. This keeps the application lightweight
enough for low-memory deployment environments such as Render Free.
"""

import logging
from functools import lru_cache
from typing import List

import numpy as np
from google import genai
from google.genai import types

from app.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedding_client() -> genai.Client:
    """Create and cache the Gemini API client."""
    if not settings.gemini_api_key:
        raise ValueError(
            "GEMINI_API_KEY is not configured. "
            "Set it in your .env file locally or in Render environment variables."
        )

    logger.info("Creating Gemini embedding client")
    return genai.Client(api_key=settings.gemini_api_key)


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Convert text into normalized Gemini embedding vectors.

    The same embedding model and dimensionality are used for both:
    - FAQ documents
    - customer queries

    This is required so FAISS similarity search operates in the
    same vector space.
    """
    if not texts:
        return np.empty((0, embedding_dimension()), dtype="float32")

    client = get_embedding_client()

    result = client.models.embed_content(
        model=settings.embedding_model,
        contents=texts,
        config=types.EmbedContentConfig(
            output_dimensionality=settings.embedding_dimension,
        ),
    )

    embeddings = np.array(
        [embedding.values for embedding in result.embeddings],
        dtype="float32",
    )

    # Normalize vectors so FAISS IndexFlatIP behaves like cosine similarity.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-12)

    return embeddings


def embedding_dimension() -> int:
    """Return the configured Gemini embedding dimension."""
    return settings.embedding_dimension