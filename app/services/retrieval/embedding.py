"""
KnowledgeMesh local embedding service using Hugging Face
Sentence Transformers.

Default model:
    BAAI/bge-m3

IMPORTANT:
    The embedding dimension is detected dynamically from the loaded model.

    Do NOT hardcode the dimension in this module.

    The active Qdrant collection is currently configured for:
        384 dimensions

Supported configuration through .env:
    EMBEDDING_MODEL
    EMBEDDING_DEVICE
    EMBEDDING_BATCH_SIZE
    EMBEDDING_NORMALIZE
    EMBEDDING_ENCODE_RETRIES
    EMBEDDING_MODEL_LOAD_RETRIES
    EXPECTED_EMBEDDING_DIMENSION

Public API:
    get_embedding_dim()
    get_active_model_type()
    embed_query(query)
    embed_texts(texts)
    safe_summary()
"""

from __future__ import annotations

import os
import threading
import time
from typing import List, Optional

import logfire


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


class EmbeddingConfigError(RuntimeError):
    """Raised when embedding-service configuration is invalid."""


def _int_env(
    var_name: str,
    default: str,
) -> int:
    """
    Read a positive integer environment variable.
    """

    raw = os.getenv(
        var_name,
        default,
    )

    try:
        value = int(raw)

    except ValueError as exc:
        raise EmbeddingConfigError(
            f"Environment variable {var_name}={raw!r} must be an integer."
        ) from exc

    if value <= 0:
        raise EmbeddingConfigError(
            f"{var_name} must be a positive integer, got {value}."
        )

    return value


# ---------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------


EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-m3",
).strip()


if not EMBEDDING_MODEL_NAME:
    raise EmbeddingConfigError("EMBEDDING_MODEL is set but empty.")


BATCH_SIZE = _int_env(
    "EMBEDDING_BATCH_SIZE",
    "4",
)


EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE") or None


NORMALIZE_EMBEDDINGS = os.getenv(
    "EMBEDDING_NORMALIZE",
    "true",
).lower() in {
    "1",
    "true",
    "yes",
    "on",
}


# ---------------------------------------------------------------------
# Optional dimension cross-check
# ---------------------------------------------------------------------
#
# This should normally be:
#
#     EXPECTED_EMBEDDING_DIMENSION=384
#
# in your .env because your current Qdrant collection is 384-dimensional.
#
# However, the actual dimension is always detected from the loaded model.
# ---------------------------------------------------------------------


_expected_dimension_raw = os.getenv("EXPECTED_EMBEDDING_DIMENSION")


EXPECTED_EMBEDDING_DIMENSION: Optional[int] = None


if _expected_dimension_raw:
    try:
        EXPECTED_EMBEDDING_DIMENSION = int(_expected_dimension_raw)

    except ValueError as exc:
        raise EmbeddingConfigError(
            "EXPECTED_EMBEDDING_DIMENSION must be an integer."
        ) from exc

    if EXPECTED_EMBEDDING_DIMENSION <= 0:
        raise EmbeddingConfigError(
            "EXPECTED_EMBEDDING_DIMENSION must be greater than zero."
        )


# ---------------------------------------------------------------------
# Retry configuration
# ---------------------------------------------------------------------


MAX_RETRIES = _int_env(
    "EMBEDDING_ENCODE_RETRIES",
    "2",
)


MODEL_LOAD_RETRIES = _int_env(
    "EMBEDDING_MODEL_LOAD_RETRIES",
    "2",
)


RETRY_BASE_DELAY_SECONDS = 1.0


# ---------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------


_active_model = None

_model_type: Optional[str] = None

_embedding_dim: Optional[int] = None

_init_lock = threading.Lock()


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------


def _load_model():
    """
    Load the configured Hugging Face Sentence Transformer model.

    The embedding dimension is detected dynamically from the
    loaded model.

    This avoids hardcoding dimensions and makes the service
    compatible with models having different output sizes.
    """

    from sentence_transformers import SentenceTransformer

    last_exc: Optional[Exception] = None

    for attempt in range(
        1,
        MODEL_LOAD_RETRIES + 1,
    ):
        try:
            logfire.info(
                "Loading local Hugging Face Sentence Transformer",
                model=EMBEDDING_MODEL_NAME,
                device=EMBEDDING_DEVICE or "auto",
                attempt=attempt,
            )

            kwargs = {}

            if EMBEDDING_DEVICE:
                kwargs["device"] = EMBEDDING_DEVICE

            # ---------------------------------------------------------
            # Load model
            # ---------------------------------------------------------

            model = SentenceTransformer(
                EMBEDDING_MODEL_NAME,
                **kwargs,
            )

            # ---------------------------------------------------------
            # Detect dimension
            # ---------------------------------------------------------
            #
            # get_sentence_embedding_dimension()
            # is deprecated.
            #
            # Use get_embedding_dimension().
            # ---------------------------------------------------------

            dim = model.get_embedding_dimension()

            if not dim or dim <= 0:
                raise ValueError(
                    f"Invalid embedding dimension for {EMBEDDING_MODEL_NAME}: {dim}"
                )

            # ---------------------------------------------------------
            # Optional configured dimension check
            # ---------------------------------------------------------

            if (
                EXPECTED_EMBEDDING_DIMENSION is not None
                and dim != EXPECTED_EMBEDDING_DIMENSION
            ):
                raise EmbeddingConfigError(
                    f"Loaded model "
                    f"{EMBEDDING_MODEL_NAME} produces "
                    f"{dim}-dim vectors, but "
                    f"EXPECTED_EMBEDDING_DIMENSION="
                    f"{EXPECTED_EMBEDDING_DIMENSION}."
                )

            # ---------------------------------------------------------
            # Success
            # ---------------------------------------------------------

            logfire.info(
                "Embedding model ready",
                model=EMBEDDING_MODEL_NAME,
                dimension=dim,
                normalized=NORMALIZE_EMBEDDINGS,
            )

            print(f"🧠 Embedding model ready: {EMBEDDING_MODEL_NAME}")

            print(f"📐 Embedding dimension: {dim}")

            return model, dim

        # -------------------------------------------------------------
        # Configuration errors should NOT retry
        # -------------------------------------------------------------

        except (
            ValueError,
            EmbeddingConfigError,
        ):
            raise

        # -------------------------------------------------------------
        # Transient model-loading errors
        # -------------------------------------------------------------

        except Exception as exc:
            last_exc = exc

            if attempt == MODEL_LOAD_RETRIES:
                logfire.exception(
                    "Failed to load embedding model",
                    model=EMBEDDING_MODEL_NAME,
                    attempts=MODEL_LOAD_RETRIES,
                )

                break

            delay = RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))

            logfire.warning(
                "Embedding model load failed; retrying",
                model=EMBEDDING_MODEL_NAME,
                attempt=attempt,
                max_retries=MODEL_LOAD_RETRIES,
                error=str(exc),
                retry_delay_seconds=delay,
            )

            time.sleep(delay)

    raise RuntimeError(
        f"Failed to load embedding model "
        f"{EMBEDDING_MODEL_NAME} after "
        f"{MODEL_LOAD_RETRIES} attempts."
    ) from last_exc


# ---------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------


def _init() -> None:
    """
    Initialize the embedding model exactly once per process.
    """

    global _active_model
    global _model_type
    global _embedding_dim

    if _active_model is not None:
        return

    with _init_lock:
        if _active_model is not None:
            return

        model, dim = _load_model()

        _active_model = model

        _model_type = f"huggingface:sentence-transformers:{EMBEDDING_MODEL_NAME}"

        _embedding_dim = dim


# ---------------------------------------------------------------------
# Public metadata API
# ---------------------------------------------------------------------


def get_embedding_dim() -> int:
    """
    Return the active model's embedding dimension.
    """

    _init()

    assert _embedding_dim is not None

    return _embedding_dim


def get_active_model_type() -> str:
    """
    Return the active embedding backend and model.
    """

    _init()

    assert _model_type is not None

    return _model_type


def safe_summary() -> dict:
    """
    Return a non-secret snapshot of embedding configuration.

    Safe for health checks and startup diagnostics.
    """

    return {
        "model": EMBEDDING_MODEL_NAME,
        "device": EMBEDDING_DEVICE or "auto",
        "batch_size": BATCH_SIZE,
        "normalize": NORMALIZE_EMBEDDINGS,
        "expected_dimension": (EXPECTED_EMBEDDING_DIMENSION),
        "loaded": _active_model is not None,
        "active_dimension": _embedding_dim,
    }


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------


def _validate_texts(
    texts: List[str],
) -> None:
    """
    Validate a list of text inputs.
    """

    if not isinstance(
        texts,
        list,
    ):
        raise TypeError(f"texts must be a list[str], got {type(texts).__name__}")

    for text in texts:
        if not isinstance(
            text,
            str,
        ):
            raise TypeError(
                f"every item in texts must be str, found {type(text).__name__}"
            )


def _check_dim(
    vectors: List[List[float]],
    what: str,
) -> None:
    """
    Verify that every vector has the expected dimension.
    """

    expected = get_embedding_dim()

    for i, vector in enumerate(vectors):
        actual = len(vector)

        if actual != expected:
            raise ValueError(
                f"{what}: embedding {i} has dim {actual}, expected {expected}."
            )


# ---------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------


def _encode(
    batch: List[str],
) -> List[List[float]]:
    """
    Encode a batch using the active Sentence Transformer model.
    """

    if _active_model is None:
        raise RuntimeError("Embedding model is not initialized.")

    last_exc: Optional[Exception] = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):
        try:
            vectors = _active_model.encode(
                batch,
                batch_size=BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=(NORMALIZE_EMBEDDINGS),
            )

            result = vectors.tolist()

            _check_dim(
                result,
                "Sentence Transformer batch",
            )

            return result

        except Exception as exc:
            last_exc = exc

            if attempt == MAX_RETRIES:
                logfire.exception(
                    "Sentence Transformer embedding failed",
                    model=EMBEDDING_MODEL_NAME,
                    attempts=MAX_RETRIES,
                )

                break

            delay = RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))

            logfire.warning(
                "Sentence Transformer embedding failed; retrying",
                model=EMBEDDING_MODEL_NAME,
                attempt=attempt,
                max_retries=MAX_RETRIES,
                error=str(exc),
                retry_delay_seconds=delay,
            )

            time.sleep(delay)

    raise RuntimeError(
        "Embedding failed for model "
        f"{EMBEDDING_MODEL_NAME} after "
        f"{MAX_RETRIES} attempts."
    ) from last_exc


# ---------------------------------------------------------------------
# Public embedding API
# ---------------------------------------------------------------------


def embed_query(
    query: str,
) -> List[float]:
    """
    Embed one query into a vector.

    Returns:
        List[float]
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    _init()

    vector = _encode([query])[0]

    _check_dim(
        [vector],
        "embed_query",
    )

    return vector


def embed_texts(
    texts: List[str],
) -> List[List[float]]:
    """
    Embed document chunks in batches.

    Returns:
        List[List[float]]
    """

    _validate_texts(texts)

    if not texts:
        return []

    _init()

    all_embeddings: List[List[float]] = []

    for start in range(
        0,
        len(texts),
        BATCH_SIZE,
    ):
        batch = texts[start : start + BATCH_SIZE]

        with logfire.span(
            "Sentence Transformer embedding batch",
            model=EMBEDDING_MODEL_NAME,
            start=start,
            size=len(batch),
        ):
            batch_embeddings = _encode(batch)

            all_embeddings.extend(batch_embeddings)

    # -------------------------------------------------------------
    # Final count validation
    # -------------------------------------------------------------

    if len(all_embeddings) != len(texts):
        raise RuntimeError(
            "Embedding count mismatch: generated "
            f"{len(all_embeddings)} vectors for "
            f"{len(texts)} texts."
        )

    # -------------------------------------------------------------
    # Final dimension validation
    # -------------------------------------------------------------

    _check_dim(
        all_embeddings,
        "embed_texts",
    )

    return all_embeddings
