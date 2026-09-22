"""
Enterprise knowledge retrieval backed by Qdrant.

Retrieval contract:

    query
      ↓
    canonical embedding
      ↓
    optional metadata filter
      ↓
    Qdrant vector search
      ↓
    canonical SearchResult objects

The function intentionally distinguishes two states:

    1. Successful retrieval with zero matches
       → returns []

    2. Retrieval infrastructure failure
       → raises the underlying failure after bounded retries

This distinction is required by downstream agents. An empty result set
means retrieval completed but found no usable matches; an exception means
retrieval was unavailable and must be handled explicitly by the graph.

Observability records:

    - embedding_ms
    - qdrant_query_ms
    - filter_ms
    - normalization_ms
    - total_retrieval_ms

The instrumentation does not modify retrieval parameters, ranking,
embedding behavior, filters, limits, or returned metadata.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, TypedDict

import logfire
from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import settings
from app.services.retrieval.embedding import embed_query


logger = logging.getLogger("knowledgemesh")


MAX_RETRIES = 3
RETRY_BASE_DELAY_SECONDS = 0.5


client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
)


class SearchResult(TypedDict):
    """Canonical retrieval result consumed by downstream agents."""

    id: str
    document_id: str
    chunk_id: int
    total_chunks: int
    content: str
    source: str
    source_type: str
    score: float


def _retry(fn, *args, what: str, **kwargs):
    """Execute an operation with bounded exponential-backoff retries."""

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

    raise RuntimeError(
        f"{what} failed after {MAX_RETRIES} attempts"
    ) from last_exc


def _safe_int(value, default: int = 0) -> int:
    """Convert a payload value to int without failing normalization."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_str(value, default: str = "") -> str:
    """Convert a payload value to string without failing normalization."""

    if value is None:
        return default

    return str(value)


def search_enterprise_knowledge(
    query: str,
    limit: int = 8,
    source_type: Optional[str] = None,
    score_threshold: Optional[float] = None,
) -> List[SearchResult]:
    """
    Search the enterprise Qdrant knowledge base.

    Args:
        query:
            Natural-language retrieval query.

        limit:
            Maximum number of Qdrant points to return.

        source_type:
            Optional payload filter restricting results to a source type.

        score_threshold:
            Optional minimum Qdrant similarity score.

    Returns:
        Canonical SearchResult objects.

        An empty list means retrieval succeeded but Qdrant returned no
        matching points.

    Raises:
        ValueError:
            If the query or retrieval parameters are invalid.

        RuntimeError:
            If embedding or Qdrant retrieval fails after all retries.

        Exception:
            Any unexpected failure during retrieval is re-raised after
            being logged. Retrieval failures must remain distinguishable
            from an empty successful result set.
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    if limit <= 0:
        raise ValueError("limit must be positive")

    if score_threshold is not None and not isinstance(
        score_threshold,
        (int, float),
    ):
        raise ValueError(
            "score_threshold must be numeric or None"
        )

    total_start = time.perf_counter()

    with logfire.span(
        "Enterprise knowledge search",
        query=query,
        limit=limit,
        source_type=source_type,
        score_threshold=score_threshold,
    ):
        try:
            # ---------------------------------------------------------
            # 1. Embed query
            # ---------------------------------------------------------
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

            embedding_ms = (
                time.perf_counter() - embedding_start
            ) * 1000

            # ---------------------------------------------------------
            # 2. Build optional Qdrant filter
            # ---------------------------------------------------------
            filter_start = time.perf_counter()

            query_filter = None

            if source_type:
                query_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_type",
                            match=models.MatchValue(
                                value=source_type
                            ),
                        )
                    ]
                )

            filter_ms = (
                time.perf_counter() - filter_start
            ) * 1000

            # ---------------------------------------------------------
            # 3. Execute Qdrant search
            # ---------------------------------------------------------
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

            qdrant_ms = (
                time.perf_counter() - qdrant_start
            ) * 1000

            points = response.points

            logger.info(
                "Qdrant retrieval completed | "
                "query=%r | collection=%s | "
                "points=%d | limit=%d | "
                "score_threshold=%s",
                query,
                settings.QDRANT_COLLECTION,
                len(points),
                limit,
                score_threshold,
            )

            if points:
                logger.debug(
                    "Qdrant scores | scores=%s",
                    [
                        round(float(point.score), 6)
                        for point in points
                    ],
                )
            else:
                logger.info(
                    "Qdrant retrieval returned zero matches | "
                    "query=%r | collection=%s",
                    query,
                    settings.QDRANT_COLLECTION,
                )

            # ---------------------------------------------------------
            # 4. Normalize Qdrant payloads
            # ---------------------------------------------------------
            normalization_start = time.perf_counter()

            results: List[SearchResult] = []

            for point in points:
                payload = point.payload or {}

                results.append(
                    {
                        "id": str(point.id),
                        "document_id": _safe_str(
                            payload.get("document_id")
                        ),
                        "chunk_id": _safe_int(
                            payload.get("chunk_id"),
                            default=-1,
                        ),
                        "total_chunks": _safe_int(
                            payload.get("total_chunks"),
                            default=0,
                        ),
                        "content": _safe_str(
                            payload.get("text")
                        ),
                        "source": _safe_str(
                            payload.get("source"),
                            default="Unknown",
                        ),
                        "source_type": _safe_str(
                            payload.get("source_type"),
                            default="Unknown",
                        ),
                        "score": float(point.score),
                    }
                )

            normalization_ms = (
                time.perf_counter() - normalization_start
            ) * 1000

            # ---------------------------------------------------------
            # 5. Record retrieval metrics
            # ---------------------------------------------------------
            total_ms = (
                time.perf_counter() - total_start
            ) * 1000

            logfire.info(
                "Enterprise retrieval completed",
                embedding_ms=round(embedding_ms, 2),
                qdrant_query_ms=round(qdrant_ms, 2),
                filter_ms=round(filter_ms, 2),
                normalization_ms=round(normalization_ms, 2),
                total_retrieval_ms=round(total_ms, 2),
                candidates=len(results),
                limit=limit,
            )

            return results

        except Exception as exc:  # noqa: BLE001
            total_ms = (
                time.perf_counter() - total_start
            ) * 1000

            logger.exception(
                "Qdrant retrieval failed | "
                "query=%r | collection=%s | "
                "limit=%d | score_threshold=%s | "
                "elapsed_ms=%.2f",
                query,
                settings.QDRANT_COLLECTION,
                limit,
                score_threshold,
                total_ms,
            )

            try:
                logfire.exception(
                    "Enterprise retrieval failed",
                    query=query,
                    collection=settings.QDRANT_COLLECTION,
                    limit=limit,
                    score_threshold=score_threshold,
                    elapsed_ms=round(total_ms, 2),
                )
            except Exception:
                pass

            raise
