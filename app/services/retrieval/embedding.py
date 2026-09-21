"""
KnowledgeMesh Embedding Service
================================

Canonical embedding service for private RAG retrieval and ingestion.

Current production embedding:
    BAAI/bge-small-en-v1.5

Expected dimension:
    384

Important invariants:
    - No Gemini embedding fallback.
    - No BGE-M3 fallback.
    - The configured model must match the canonical model.
    - The loaded embedding dimension must match 384.
    - Query and document embeddings therefore remain compatible
      with the existing Qdrant `enterprise_rag` collection.

The Qdrant collection currently contains 384-dimensional vectors,
so changing this model/dimension requires an explicit collection
migration and re-indexing. This module intentionally fails closed
instead of silently changing embedding models.
"""

from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import List, Optional

import logfire


# ============================================================
# Canonical embedding configuration
# ============================================================

CANONICAL_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CANONICAL_EMBEDDING_DIMENSION = 384

# Read environment configuration.
#
# The fallback is deliberately the same canonical model so that a
# missing environment variable cannot silently switch to another
# embedding family.
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    CANONICAL_EMBEDDING_MODEL,
).strip()

_EXPECTED_DIMENSION_RAW = os.getenv(
    "EMBEDDING_DIMENSION",
    str(CANONICAL_EMBEDDING_DIMENSION),
).strip()


# ============================================================
# Configuration validation
# ============================================================

def _parse_expected_dimension(value: str) -> int:
    """Parse and validate the configured embedding dimension."""

    try:
        dimension = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"EMBEDDING_DIMENSION must be an integer, got {value!r}"
        ) from exc

    if dimension <= 0:
        raise ValueError(
            f"EMBEDDING_DIMENSION must be positive, got {dimension}"
        )

    return dimension


EXPECTED_EMBEDDING_DIMENSION = _parse_expected_dimension(
    _EXPECTED_DIMENSION_RAW
)


if EMBEDDING_MODEL_NAME != CANONICAL_EMBEDDING_MODEL:
    raise RuntimeError(
        "Embedding configuration mismatch.\n"
        f"Expected model: {CANONICAL_EMBEDDING_MODEL}\n"
        f"Configured model: {EMBEDDING_MODEL_NAME}\n\n"
        "KnowledgeMesh is intentionally fail-closed here because "
        "changing the embedding model without rebuilding the Qdrant "
        "collection can corrupt retrieval compatibility."
    )


if EXPECTED_EMBEDDING_DIMENSION != CANONICAL_EMBEDDING_DIMENSION:
    raise RuntimeError(
        "Embedding dimension configuration mismatch.\n"
        f"Expected dimension: {CANONICAL_EMBEDDING_DIMENSION}\n"
        f"Configured dimension: {EXPECTED_EMBEDDING_DIMENSION}\n\n"
        "The current Qdrant enterprise_rag collection uses "
        f"{CANONICAL_EMBEDDING_DIMENSION}-dimensional vectors."
    )


# ============================================================
# Runtime model
# ============================================================

_model = None
_embedding_dimension: Optional[int] = None


def _load_model():
    """
    Lazily load the SentenceTransformer model.

    Lazy loading prevents expensive model initialization during
    unrelated application imports.
    """

    global _model
    global _embedding_dimension

    if _model is not None:
        return _model

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for the KnowledgeMesh "
            "embedding service. Install it with:\n"
            "uv pip install sentence-transformers"
        ) from exc

    logfire.info(
        f"Loading embedding model: {CANONICAL_EMBEDDING_MODEL}"
    )

    model = SentenceTransformer(
        CANONICAL_EMBEDDING_MODEL,
        device=os.getenv("EMBEDDING_DEVICE") or None,
    )

    # SentenceTransformer exposes get_sentence_embedding_dimension().
    detected_dimension = model.get_sentence_embedding_dimension()

    if detected_dimension is None:
        raise RuntimeError(
            "Unable to determine embedding dimension for "
            f"{CANONICAL_EMBEDDING_MODEL}"
        )

    detected_dimension = int(detected_dimension)

    if detected_dimension != CANONICAL_EMBEDDING_DIMENSION:
        raise RuntimeError(
            "Loaded embedding model has an unexpected dimension.\n"
            f"Model: {CANONICAL_EMBEDDING_MODEL}\n"
            f"Expected: {CANONICAL_EMBEDDING_DIMENSION}\n"
            f"Loaded: {detected_dimension}"
        )

    if detected_dimension != EXPECTED_EMBEDDING_DIMENSION:
        raise RuntimeError(
            "Loaded embedding dimension does not match configuration.\n"
            f"Configured: {EXPECTED_EMBEDDING_DIMENSION}\n"
            f"Loaded: {detected_dimension}"
        )

    _model = model
    _embedding_dimension = detected_dimension

    print(
        f"🧠 Embedding model ready: "
        f"{CANONICAL_EMBEDDING_MODEL}"
    )
    print(
        f"📐 Embedding dimension: "
        f"{detected_dimension}"
    )

    logfire.info(
        "Embedding model ready",
        model=CANONICAL_EMBEDDING_MODEL,
        dimension=detected_dimension,
    )

    return _model


# ============================================================
# Public model metadata
# ============================================================

def get_active_model_type() -> str:
    """
    Return a stable identifier describing the active embedding model.
    """

    return (
        "huggingface:"
        "sentence-transformers:"
        f"{CANONICAL_EMBEDDING_MODEL}"
    )


def get_embedding_dim() -> int:
    """
    Return the active embedding dimension.

    Loads the model if necessary so the returned value reflects the
    actual loaded model rather than only configuration.
    """

    global _embedding_dimension

    if _embedding_dimension is None:
        _load_model()

    assert _embedding_dimension is not None

    return _embedding_dimension


# ============================================================
# Embedding helpers
# ============================================================

def _validate_embedding(vector) -> List[float]:
    """
    Validate and normalize a single embedding vector.
    """

    values = vector.tolist() if hasattr(vector, "tolist") else list(vector)

    values = [float(value) for value in values]

    expected = CANONICAL_EMBEDDING_DIMENSION

    if len(values) != expected:
        raise RuntimeError(
            "Embedding dimension mismatch.\n"
            f"Expected: {expected}\n"
            f"Actual: {len(values)}"
        )

    return values


# ============================================================
# Query embedding
# ============================================================

def embed_query(query: str) -> List[float]:
    """
    Generate one embedding for a query.

    Returns:
        List[float] with exactly 384 values.
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError(
            "query must be a non-empty string"
        )

    model = _load_model()

    vector = model.encode(
        query.strip(),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    return _validate_embedding(vector)


# ============================================================
# Batch document embedding
# ============================================================

def embed_documents(
    texts: List[str],
    batch_size: int = 16,
    max_retries: int = 3,
) -> List[List[float]]:
    """
    Generate embeddings for multiple documents.

    Args:
        texts:
            List of document/chunk strings.

        batch_size:
            SentenceTransformer batch size.

        max_retries:
            Number of attempts if model inference fails.

    Returns:
        List of 384-dimensional embedding vectors.
    """

    if not isinstance(texts, list):
        raise TypeError("texts must be a list")

    if not texts:
        return []

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive"
        )

    if max_retries <= 0:
        raise ValueError(
            "max_retries must be positive"
        )

    cleaned_texts = []

    for index, text in enumerate(texts):
        if not isinstance(text, str):
            raise TypeError(
                f"texts[{index}] must be a string"
            )

        cleaned_texts.append(text)

    model = _load_model()

    last_exception: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            vectors = model.encode(
                cleaned_texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            result = [
                _validate_embedding(vector)
                for vector in vectors
            ]

            if len(result) != len(cleaned_texts):
                raise RuntimeError(
                    "Embedding count mismatch.\n"
                    f"Expected: {len(cleaned_texts)}\n"
                    f"Received: {len(result)}"
                )

            return result

        except Exception as exc:  # noqa: BLE001
            last_exception = exc

            if attempt == max_retries:
                break

            delay = 0.5 * (2 ** (attempt - 1))

            logfire.warning(
                f"Document embedding failed "
                f"(attempt {attempt}/{max_retries}): {exc}. "
                f"Retrying in {delay:.1f}s."
            )

            time.sleep(delay)

    raise RuntimeError(
        f"Document embedding failed after {max_retries} attempts"
    ) from last_exception


# ============================================================
# Convenience aliases / metadata
# ============================================================

def get_embedding_model_name() -> str:
    """Return the canonical active model name."""

    return CANONICAL_EMBEDDING_MODEL


def get_embedding_dimension() -> int:
    """Return the active embedding dimension."""

    return get_embedding_dim()
