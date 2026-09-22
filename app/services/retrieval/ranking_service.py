"""
FlashRank cross-encoder reranking over retrieved passages.

Features
--------
- Accepts structured retrieval dictionaries and plain strings.
- Preserves source/source_type/original retrieval score.
- Adds `rerank_score` from FlashRank.
- Thread-safe lazy initialization of FlashRank.
- Validates query and top_n.
- Filters empty passages.
- Gracefully falls back to original retrieval order if reranking fails.
- Keeps `rerank_texts()` for backward compatibility.

Architecture
------------
Qdrant dense retrieval
        ↓
normalized documents
        ↓
FlashRank cross-encoder
        ↓
result["score"]
        ↓
document["rerank_score"]
        ↓
retriever evidence selector

Important
---------
interfere with the FlashRank execution path.
"""

from __future__ import annotations

import threading
import time
from typing import Any, List, Optional, TypedDict

import logfire
from flashrank import Ranker, RerankRequest


# ============================================================
# FLASHRANK SINGLETON
# ============================================================

_ranker: Optional[Ranker] = None
_ranker_lock = threading.Lock()


# ============================================================
# RESULT TYPE
# ============================================================


class RankedResult(TypedDict, total=False):
    id: str
    content: str
    source: str
    source_type: str

    # Original vector-search score.
    score: float

    # FlashRank cross-encoder score.
    rerank_score: float


# ============================================================
# FLASHRANK INITIALIZATION
# ============================================================


def _get_ranker() -> Ranker:
    """
    Thread-safe lazy initialization of FlashRank.

    FlashRank loads a local ONNX cross-encoder.
    """

    global _ranker

    if _ranker is not None:
        return _ranker

    with _ranker_lock:
        if _ranker is not None:
            return _ranker

        logfire.info("Initializing FlashRank model locally...")

        try:
            _ranker = Ranker(cache_dir="/tmp/flashrank")

        except Exception as exc:
            logfire.warning(
                "FlashRank cache_dir initialization failed "
                f"({exc}); retrying with default cache."
            )

            _ranker = Ranker()

        logfire.info("FlashRank model initialized")

        return _ranker


# ============================================================
# DOCUMENT NORMALIZATION
# ============================================================


def _normalize_document(
    document: Any,
    text_key: str = "content",
) -> dict:
    """
    Normalize a retrieved document into a dictionary.

    Supported inputs:

        "plain text"

    or:

        {
            "content": "...",
            "source": "...",
            "score": 0.82
        }

    The original document metadata is preserved.
    """

    # --------------------------------------------------------
    # Plain string
    # --------------------------------------------------------

    if isinstance(document, str):
        return {
            text_key: document,
        }

    # --------------------------------------------------------
    # Dictionary
    # --------------------------------------------------------

    if isinstance(document, dict):
        normalized = dict(document)

        # Requested text key already exists.
        value = normalized.get(text_key)

        if value is not None and str(value).strip():
            normalized[text_key] = str(value)
            return normalized

        # ----------------------------------------------------
        # Common alternative text fields
        # ----------------------------------------------------

        alternative_keys = (
            "content",
            "text",
            "page_content",
            "chunk",
            "document",
            "body",
        )

        for key in alternative_keys:
            value = normalized.get(key)

            if value is not None and str(value).strip():
                normalized[text_key] = str(value)
                return normalized

        # No usable text.
        normalized[text_key] = ""

        return normalized

    # --------------------------------------------------------
    # Unknown object
    # --------------------------------------------------------

    return {
        text_key: str(document),
    }


# ============================================================
# FLASHRANK RESULT HELPERS
# ============================================================


def _extract_result_id(result: Any) -> str:
    """
    Extract the passage identifier from a FlashRank result.
    """

    if isinstance(result, dict):
        value = result.get("id")
    else:
        value = getattr(result, "id", None)

    if value is None:
        return ""

    return str(value)


def _extract_result_score(result: Any) -> Optional[float]:
    """
    Extract the FlashRank score.

    FlashRank normally returns:

        {
            "id": "...",
            "text": "...",
            "score": ...
        }

    The score is explicitly converted to float so numpy scalar
    values such as np.float32 are safely propagated.
    """

    if isinstance(result, dict):
        value = result.get("score")
    else:
        value = getattr(result, "score", None)

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ============================================================
# RERANK DOCUMENTS
# ============================================================


def rerank_documents(
    query: str,
    documents: List[Any],
    top_n: int = 5,
    text_key: str = "content",
) -> List[RankedResult]:
    """
    Re-score retrieved documents against the query with FlashRank.

    Parameters
    ----------
    query:
        User query.

    documents:
        Retrieved documents.

        Supports:

            List[str]

        and:

            List[dict]

    top_n:
        Maximum number of results returned.

    text_key:
        Field containing the document text.

    Returns
    -------
    List[RankedResult]

        Documents sorted by FlashRank score.

    Failure behavior
    ----------------
    If FlashRank fails, the original retrieval order is returned.

    Important
    ---------
    A missing FlashRank score is never converted to 0.0.
    """

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    if not documents:
        logfire.info("No documents available for reranking.")
        return []

    if top_n <= 0:
        raise ValueError("top_n must be positive")

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    start_time = time.perf_counter()

    logfire.info(
        "Sending documents to FlashRank cross-encoder",
        input_documents=len(documents),
        top_n=top_n,
    )

    # ========================================================
    # NORMALIZE INPUT
    # ========================================================

    normalized_documents = [
        _normalize_document(
            document,
            text_key=text_key,
        )
        for document in documents
    ]

    # ========================================================
    # FILTER EMPTY DOCUMENTS
    # ========================================================

    indexed_docs = [
        (index, document)
        for index, document in enumerate(normalized_documents)
        if str(
            document.get(
                text_key,
                "",
            )
        ).strip()
    ]

    if not indexed_docs:
        logfire.warning(f"No document had non-empty '{text_key}' — skipping rerank.")

        return [dict(document) for document in normalized_documents[:top_n]]

    try:
        # ====================================================
        # GET FLASHRANK MODEL
        # ====================================================

        ranker = _get_ranker()

        # ====================================================
        # BUILD FLASHRANK PASSAGES
        # ====================================================

        passages = [
            {
                "id": str(index),
                "text": str(
                    document.get(
                        text_key,
                        "",
                    )
                ),
            }
            for index, document in indexed_docs
        ]

        # ====================================================
        # RERANK
        # ====================================================

        request = RerankRequest(
            query=query,
            passages=passages,
        )

        results = ranker.rerank(request)

        if results is None:
            results = []

        # ====================================================
        # MAP FLASHRANK RESULTS BACK TO DOCUMENTS
        # ====================================================

        by_id = {str(index): document for index, document in indexed_docs}

        reranked: List[RankedResult] = []

        mapping_misses = 0
        missing_scores = 0

        for result in results[:top_n]:
            result_id = _extract_result_id(result)
            rerank_score = _extract_result_score(result)

            original = by_id.get(result_id)

            if original is None:
                mapping_misses += 1
                continue

            enriched = dict(original)

            # ------------------------------------------------
            # Critical score propagation
            # ------------------------------------------------

            if rerank_score is not None:
                enriched["rerank_score"] = rerank_score
            else:
                missing_scores += 1

            reranked.append(enriched)

        # ====================================================
        # OBSERVABILITY
        # ====================================================

        duration = time.perf_counter() - start_time

        top_score = reranked[0].get("rerank_score") if reranked else None

        logfire.info(
            "FlashRank reranking completed",
            input_documents=len(normalized_documents),
            non_empty_documents=len(indexed_docs),
            flashrank_results=len(results),
            mapped_results=len(reranked),
            mapping_misses=mapping_misses,
            missing_scores=missing_scores,
            top_rerank_score=top_score,
            duration_ms=round(
                duration * 1000,
                2,
            ),
        )

        # ====================================================
        # DEFENSIVE FALLBACK
        # ====================================================

        if not reranked:
            logfire.warning(
                "FlashRank returned no usable mapped results. "
                "Falling back to original retrieval order."
            )

            return [dict(document) for document in normalized_documents[:top_n]]

        # ====================================================
        # IMPORTANT SAFETY CHECK
        # ====================================================

        # A successful reranking result must carry a semantic
        # score. Missing scores must never become 0.0 because
        # downstream retrieval interprets the score as semantic
        # evidence quality.

        valid_scored_results = [
            document
            for document in reranked
            if document.get("rerank_score") is not None
        ]

        if not valid_scored_results:
            logfire.warning(
                "FlashRank returned mapped documents but no "
                "valid rerank scores. Falling back to original "
                "retrieval order."
            )

            return [dict(document) for document in normalized_documents[:top_n]]

        # ====================================================
        # FINAL ORDER
        # ====================================================

        valid_scored_results.sort(
            key=lambda document: float(document["rerank_score"]),
            reverse=True,
        )

        return valid_scored_results[:top_n]

    # ========================================================
    # GRACEFUL FAILURE
    # ========================================================

    except Exception as exc:
        duration = time.perf_counter() - start_time

        logfire.exception(
            "FlashRank reranking failed",
            duration_ms=round(
                duration * 1000,
                2,
            ),
        )

        logfire.warning(
            f"Falling back to original retrieval order because reranking failed: {exc}"
        )

        return [dict(document) for document in normalized_documents[:top_n]]


# ============================================================
# BACKWARD-COMPATIBLE TEXT API
# ============================================================


def rerank_texts(
    query: str,
    documents: List[str],
    top_n: int = 5,
) -> List[str]:
    """
    Backward-compatible list[str] API.

    Input:
        List[str]

    Output:
        List[str]

    Prefer `rerank_documents()` when source metadata
    needs to be preserved.
    """

    if not documents:
        return []

    wrapped = [
        {
            "content": document,
        }
        for document in documents
    ]

    reranked = rerank_documents(
        query=query,
        documents=wrapped,
        top_n=top_n,
        text_key="content",
    )

    return [
        str(
            result.get(
                "content",
                "",
            )
        )
        for result in reranked
    ]
