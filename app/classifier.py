"""
Ticket classification module: TF-IDF + Logistic Regression.

Determines which of the three supported categories a customer ticket
belongs to, along with a confidence score, using a simple, explainable
classical ML pipeline (per Sec. 11: no deep-learning classifier needed
for this kind of short-text categorization).

Pipeline this module implements (see Sec. 4 / Sec. 11 of the spec):

    Training (train_classifier):

        data/faq.json
                |
                v
        X = FAQ questions, y = FAQ categories
                |
                v
        TfidfVectorizer.fit_transform(X)
                |
                v
        LogisticRegression.fit(tfidf_vectors, y)

    Inference (classify_ticket):

        Customer ticket text
                |
                v
        TfidfVectorizer.transform([text])   (same fitted vectorizer)
                |
                v
        LogisticRegression.predict_proba()
                |
                v
        (predicted category, confidence score)
"""

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.config import settings

logger = logging.getLogger(__name__)

# The three categories this classifier is trained to recognize. Kept as
# a named constant (rather than scattering the literal strings around)
# so later phases (escalation, tests) can import and reuse this list.
SUPPORTED_CATEGORIES = ["Billing", "Technical", "Account Access"]


@dataclass
class ClassificationResult:
    """The classifier's prediction for a single ticket."""

    category: str
    confidence: float


def _load_training_data() -> Tuple[List[str], List[str]]:
    """
    Load (question, category) pairs from data/faq.json to use as the
    classifier's training set.

    X = FAQ questions (the raw text the classifier learns from)
    y = FAQ categories (the label the classifier learns to predict)
    """
    faq_path = Path(settings.faq_path)
    if not faq_path.exists():
        raise FileNotFoundError(
            f"Knowledge base not found at {faq_path}. "
            "Make sure data/faq.json exists (see Phase 2)."
        )
    with open(faq_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    X = [doc["question"] for doc in documents]
    y = [doc["category"] for doc in documents]

    logger.info(
        "Loaded %d training examples across categories: %s",
        len(X),
        sorted(set(y)),
    )
    return X, y


def train_classifier() -> Pipeline:
    """
    Train a TF-IDF + Logistic Regression ticket classifier on data/faq.json.

    We use an sklearn Pipeline to bundle the vectorizer and the classifier
    into a single object, so the exact same fitted TF-IDF vocabulary is
    used consistently at both training time and prediction time — you
    can't accidentally call predict() with a mismatched vectorizer.

    Returns:
        A fitted sklearn Pipeline: TfidfVectorizer -> LogisticRegression.
    """
    X, y = _load_training_data()

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english")),
            # max_iter=1000: with only ~30 short training examples,
            # LogisticRegression's default solver can need more than the
            # default 100 iterations to fully converge; this avoids a
            # ConvergenceWarning without changing anything else about
            # the model.
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    logger.info(
        "Training TF-IDF + Logistic Regression classifier on %d examples...",
        len(X),
    )
    pipeline.fit(X, y)
    logger.info("Classifier trained. Categories: %s", list(pipeline.classes_))

    return pipeline


@lru_cache(maxsize=1)
def _get_classifier() -> Pipeline:
    """
    Train the classifier once and cache it for the lifetime of the
    process, using the same lru_cache singleton pattern as the embedding
    model in app/embeddings.py.

    Training on ~30 short examples takes a fraction of a second, so we
    deliberately keep this in-memory rather than persisting a model file
    to disk: it satisfies "don't retrain on every request" (Sec. 11)
    without introducing extra file I/O or a new configurable path.
    """
    return train_classifier()


def classify_ticket(text: str) -> ClassificationResult:
    """
    Classify a customer support ticket into Billing, Technical, or
    Account Access, with a confidence score.

    Args:
        text: the raw customer ticket/question text.

    Returns:
        ClassificationResult(category, confidence).
        For empty or blank input, returns category="Unknown" with
        confidence=0.0 instead of running the model on meaningless text
        — an empty string carries no signal for TF-IDF to work with.
    """
    if not text or not text.strip():
        return ClassificationResult(category="Unknown", confidence=0.0)

    pipeline = _get_classifier()

    probabilities = pipeline.predict_proba([text])[0]  # shape: (n_categories,)
    classes = pipeline.classes_

    best_index = probabilities.argmax()
    predicted_category = classes[best_index]
    confidence = float(probabilities[best_index])

    return ClassificationResult(category=predicted_category, confidence=confidence)


if __name__ == "__main__":
    # Manual test harness. Run with:
    #     python -m app.classifier
    logging.basicConfig(level=logging.INFO)

    test_messages = [
        "I was charged twice for my subscription.",
        "I am getting a 500 error when I try to log in.",
        "I forgot my password. How can I reset it?",
        "What will the weather be tomorrow?",
    ]

    for msg in test_messages:
        result = classify_ticket(msg)
        print(f"\nMessage: {msg}")
        print(f"  Category:   {result.category}")
        print(f"  Confidence: {result.confidence:.2%}")