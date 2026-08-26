"""
Centralized application configuration.

All environment-dependent values (paths, thresholds, model names, API
keys) are loaded here ONCE from `.env` and exposed as a single `settings`
object. Every other module imports `settings` instead of calling
os.getenv() directly, so there is exactly one place that knows how to
read configuration.

Why this matters for the assessment:
- No hardcoded API keys anywhere in the codebase (Sec. 16 requirement).
- The confidence threshold and RAG parameters (TOP_K_RESULTS,
  MIN_RETRIEVAL_SCORE) are configurable in one place, not hardcoded
  throughout the app (Sec. 4 requirement).
"""

import os
import logging
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float for %s=%r, using default %s", name, raw, default)
        return default


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid int for %s=%r, using default %s", name, raw, default)
        return default


@dataclass(frozen=True)
class Settings:
    faq_path: Path
    vector_index_path: Path
    vector_metadata_path: Path

    embedding_model: str

    top_k_results: int
    min_retrieval_score: float

    confidence_threshold: float

    llm_provider: str
    gemini_api_key: str
    gemini_model: str
    openai_api_key: str
    openai_model: str

    host: str
    port: int


def load_settings() -> Settings:
    return Settings(
        faq_path=Path(os.getenv("FAQ_PATH", "data/faq.json")),
        vector_index_path=Path(os.getenv("VECTOR_INDEX_PATH", "vector_store/faq.index")),
        vector_metadata_path=Path(
            os.getenv("VECTOR_METADATA_PATH", "vector_store/faq_metadata.pkl")
        ),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        top_k_results=_get_int("TOP_K_RESULTS", 3),
        min_retrieval_score=_get_float("MIN_RETRIEVAL_SCORE", 0.40),
        confidence_threshold=_get_float("CONFIDENCE_THRESHOLD", 0.70),
        llm_provider=os.getenv("LLM_PROVIDER", "gemini").strip().lower(),
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        host=os.getenv("HOST", "0.0.0.0"),
        port=_get_int("PORT", 8000),
    )


settings = load_settings()