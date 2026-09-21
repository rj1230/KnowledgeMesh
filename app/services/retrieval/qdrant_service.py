"""
Qdrant vector search over the enterprise knowledge base.

Responsibilities:
    - Embed the incoming query using the canonical embedding service.
    - Search Qdrant with optional source_type and score filtering.
    - Retry transient embedding/Qdrant failures with exponential backoff.
    - Return stable retrieval metadata required by reranking, citations,
      evaluation, tracing, and downstream agents.

Performance instrumentation:
    - embedding_ms
    - qdrant_query_ms
    - payload_normalization_ms
    - total_retrieval_ms

The instrumentation is intentionally non-invasive:
    retrieval behavior, embedding model, Qdrant collection, filters,
    limits, and returned metadata remain unchanged.
"""

from __future__ import annotations

import time
from typing import List, Optional, TypedDict

import logfire
from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import settings
from app.services.retrieval.embedding import embed_query


# ============================================================
# Retry configuration
# ============================================================

MAX_RETRIES = 3
RETRY_BASE_DELAY_SECONDS = 0.5


# ============================================================
# Qdrant client
# ============================================================

client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
)


# ============================================================
# Stable retrieval contract
# ============================================================


class SearchResult(TypedDict):
    """
    Canonical result returned by enterprise retrieval.
    """

    id: str
    document_id: str
    chunk_id: int
    total_chunks: int
    content: str
    source: str
    source_type: str
    score: float


# ============================================================
# Retry helper
# ============================================================


def _retry(fn, *args, what: str, **kwargs):
    """
    Execute a callable with bounded exponential-backoff retries.
    """

    last_exc: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)

        except Exception as exc:  # noqa: BLE001
            last_exc = exc

            if attempt == MAX_RETRIES:
                break

            delay = RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))

            logfire.warning(
                f"{what} failed "
                f"(attempt {attempt}/{MAX_RETRIES}): {exc}. "
                f"Retrying in {delay:.1f}s."
            )

            time.sleep(delay)

    raise RuntimeError(f"{what} failed after {MAX_RETRIES} attempts") from last_exc


# ============================================================
# Payload normalization
# ============================================================


def _safe_int(value, default: int = 0) -> int:
    """
    Safely convert a Qdrant payload value to int.
    """

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_str(value, default: str = "") -> str:
    """
    Safely convert a Qdrant payload value to string.
    """

    if value is None:
        return default

    return str(value)


# ============================================================
# Enterprise retrieval
# ============================================================


def search_enterprise_knowledge(
    query: str,
    limit: int = 8,
    source_type: Optional[str] = None,
    score_threshold: Optional[float] = None,
) -> List[SearchResult]:
    """
    Search the enterprise Qdrant knowledge base.

    Returns:
        List of SearchResult objects.

    Performance instrumentation does not change retrieval behavior.
    """

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    if limit <= 0:
        raise ValueError("limit must be positive")

    if score_threshold is not None and not isinstance(
        score_threshold,
        (int, float),
    ):
        raise ValueError("score_threshold must be numeric or None")

    # --------------------------------------------------------
    # Overall retrieval timer
    # --------------------------------------------------------

    total_start = time.perf_counter()

    with logfire.span(
        "Enterprise knowledge search",
        query=query,
        limit=limit,
        source_type=source_type,
        score_threshold=score_threshold,
    ):
        try:
            # =================================================
            # 1. Embed query
            # =================================================

            embedding_start = time.perf_counter()

            with logfire.span(
                "Query embedding",
                model="BAAI/bge-small-en-v1.5",
                dimension=384,
            ):
                query_vector = _retry(
                    embed_query,
                    query,
                    what="Query embedding",
                )

            embedding_ms = (time.perf_counter() - embedding_start) * 1000

            # =================================================
            # 2. Build optional source filter
            # =================================================

            filter_start = time.perf_counter()

            query_filter = None

            if source_type:
                query_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_type",
                            match=models.MatchValue(value=source_type),
                        )
                    ]
                )

            filter_ms = (time.perf_counter() - filter_start) * 1000

            # =================================================
            # 3. Query Qdrant
            # =================================================

            qdrant_start = time.perf_counter()

            with logfire.span(
                "Qdrant query_points",
                collection=settings.QDRANT_COLLECTION,
                limit=limit,
                with_payload=True,
            ):
                response = _retry(
                    client.query_points,
                    collection_name=settings.QDRANT_COLLECTION,
                    query=query_vector,
                    query_filter=query_filter,
                    score_threshold=score_threshold,
                    limit=limit,
                    with_payload=True,
                    what="Qdrant query_points",
                )

            qdrant_ms = (time.perf_counter() - qdrant_start) * 1000

            # =================================================
            # 4. Normalize Qdrant response
            # =================================================

            normalization_start = time.perf_counter()

            results: List[SearchResult] = []

            for point in response.points:
                payload = point.payload or {}

                result: SearchResult = {
                    # Qdrant vector identity
                    "id": str(point.id),
                    # Corpus/document identity
                    "document_id": _safe_str(
                        payload.get("document_id"),
                        default="",
                    ),
                    # Chunk identity
                    "chunk_id": _safe_int(
                        payload.get("chunk_id"),
                        default=-1,
                    ),
                    # Document chunk count
                    "total_chunks": _safe_int(
                        payload.get("total_chunks"),
                        default=0,
                    ),
                    # Actual text
                    "content": _safe_str(
                        payload.get("text"),
                        default="",
                    ),
                    # Source metadata
                    "source": _safe_str(
                        payload.get("source"),
                        default="Unknown",
                    ),
                    "source_type": _safe_str(
                        payload.get("source_type"),
                        default="Unknown",
                    ),
                    # Similarity
                    "score": float(point.score),
                }

                results.append(result)

            normalization_ms = (time.perf_counter() - normalization_start) * 1000

            # =================================================
            # 5. Total timing
            # =================================================

            total_ms = (time.perf_counter() - total_start) * 1000

            # =================================================
            # 6. Structured observability
            # =================================================

            logfire.info(
                "📊 Enterprise retrieval timing",
                embedding_ms=round(embedding_ms, 2),
                qdrant_query_ms=round(qdrant_ms, 2),
                filter_ms=round(filter_ms, 2),
                normalization_ms=round(
                    normalization_ms,
                    2,
                ),
                total_retrieval_ms=round(
                    total_ms,
                    2,
                ),
                candidates=len(results),
                limit=limit,
            )

            logfire.info(f"Search returned {len(results)} result(s) for query.")

            return results

        except Exception as exc:  # noqa: BLE001
            total_ms = (time.perf_counter() - total_start) * 1000

            logfire.exception(f"Qdrant search failed after {total_ms:.2f} ms: {exc}")

            return []
