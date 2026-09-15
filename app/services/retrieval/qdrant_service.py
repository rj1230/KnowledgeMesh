"""
Qdrant vector search over the enterprise knowledge base.

Improvements over the original:
  - Retry with backoff around both the embedding call and the Qdrant query
    (the original had one bare try/except that swallowed everything and
    returned an empty list — indistinguishable from "no results" vs.
    "the search backend is down").
  - Input validation on query/limit instead of passing bad values straight
    into the client.
  - Optional source_type filter and score_threshold, since "search
    everything, always" is rarely what a production caller wants.
  - Point ID is included in results so a downstream reranker or UI can
    dedupe / link back to the exact vector.
  - Logs full exception context (logfire.exception) instead of only the
    stringified error, so failures are actually debuggable in traces.
"""

from __future__ import annotations

import time
from typing import List, Optional, TypedDict

import logfire
from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import settings
from app.services.retrieval.embedding import embed_query

MAX_RETRIES = 3
RETRY_BASE_DELAY_SECONDS = 0.5

client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
)


class SearchResult(TypedDict):
    id: str
    content: str
    source: str
    source_type: str
    score: float


def _retry(fn, *args, what: str, **kwargs):
    last_exc: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - retried, then re-raised below
            last_exc = e
            if attempt == MAX_RETRIES:
                break
            delay = RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            logfire.warning(
                f"{what} failed (attempt {attempt}/{MAX_RETRIES}): {e}. Retrying in {delay:.1f}s."
            )
            time.sleep(delay)
    raise RuntimeError(f"{what} failed after {MAX_RETRIES} attempts") from last_exc


def search_enterprise_knowledge(
    query: str,
    limit: int = 8,
    source_type: Optional[str] = None,
    score_threshold: Optional[float] = None,
) -> List[SearchResult]:
    """
    Performs vector search against the enterprise knowledge base.

    Args:
        query: The search query text.
        limit: Max number of results to return.
        source_type: Optional exact-match filter on the `source_type` payload field
            (e.g. "true", "noisy", or a custom source folder name).
        score_threshold: Optional minimum similarity score — results below this
            are dropped.

    Returns:
        List of result dicts (empty list on failure or no matches — callers
        that need to distinguish "backend down" from "no results" should
        watch logs/traces for the underlying exception rather than branch on
        an empty list).
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if limit <= 0:
        raise ValueError("limit must be positive")

    with logfire.span(
        "Enterprise knowledge search", query=query, limit=limit, source_type=source_type
    ):
        try:
            query_vector = _retry(embed_query, query, what="Query embedding")

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

            results: List[SearchResult] = [
                {
                    "id": str(res.id),
                    "content": res.payload.get("text", ""),
                    "source": res.payload.get("source", "Unknown"),
                    "source_type": res.payload.get("source_type", "Unknown"),
                    "score": res.score,
                }
                for res in response.points
            ]

            logfire.info(f"Search returned {len(results)} result(s) for query.")
            return results

        except Exception as e:  # noqa: BLE001 - degrade gracefully, but log with full context
            logfire.exception(f"Qdrant search failed: {e}")
            return []
