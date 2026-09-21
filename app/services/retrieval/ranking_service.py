"""
FlashRank cross-encoder reranking over retrieved passages.

Features
--------
- Accepts both structured retrieval dictionaries and plain strings.
- Preserves source/source_type/original retrieval score.
- Adds `rerank_score` after FlashRank.
- Thread-safe lazy initialization of FlashRank.
- Validates query and top_n.
- Filters empty passages.
- Gracefully falls back to original retrieval order if reranking fails.
- Keeps `rerank_texts()` for backward compatibility.
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

    # Original vector-search score
    score: float

    # FlashRank cross-encoder score
    rerank_score: float



# ============================================================
# FLASHRANK INITIALIZATION
# ============================================================


def _get_ranker() -> Ranker:
    """
    Thread-safe lazy initialization of FlashRank.

    FlashRank loads a local ONNX cross-encoder:
        ms-marco-MiniLM-L-6-v2
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
                f"FlashRank cache_dir initialization failed "
                f"({exc}); retrying with default cache."
            )

            _ranker = Ranker()

        logfire.info("✅ FlashRank model initialized")

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

    This is the key compatibility fix for the current
    KnowledgeMesh retrieval pipeline.
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

        # If requested text key already exists, keep it.
        value = normalized.get(text_key)

        if value is not None and str(value).strip():
            normalized[text_key] = str(value)
            return normalized

        # ----------------------------------------------------
        # Try common alternative text fields
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
# GP005 DIAGNOSTIC TRACE (temporary)
# ============================================================


def _log_gp005_trace(query: str, results: list, indexed_docs: list) -> None:
    """
    Print a debug trace for one specific diagnostic query.

    Isolated in its own function so it can never affect (or be
    accidentally nested around) the real result-mapping logic in
    `rerank_documents`. Safe to delete once GP005 is resolved.
    """

    if query != _GP005_TRACE_QUERY:
        return

    for result in results:
        result_id = str(
            result.get("id") if isinstance(result, dict) else getattr(result, "id", "")
        )

        original = next(
            (document for index, document in indexed_docs if str(index) == result_id),
            None,
        )

        if original is None:
            continue

        chunk_id = original.get("chunk_id")

        if (
            str(original.get("document_id")) == _GP005_TARGET_DOCUMENT_ID
            and int(chunk_id or -1) in _GP005_TARGET_CHUNK_IDS
        ):
            score = (
                result.get("score")
                if isinstance(result, dict)
                else getattr(result, "score", None)
            )
            print(
                "GP005 FLASHRANK:",
                f"chunk_id={chunk_id}",
                f"score={score}",
                flush=True,
            )


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

        Supports both:

            List[str]

        and:

            List[dict]

        Structured dictionaries should preferably contain:

            content
            source
            source_type
            score

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

    start_time = time.time()

    logfire.info(f"Sending {len(documents)} doc(s) to FlashRank cross-encoder...")

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

        passages = []

        for index, document in indexed_docs:
            passages.append(
                {
                    "id": str(index),
                    "text": str(
                        document.get(
                            text_key,
                            "",
                        )
                    ),
                }
            )

        # ====================================================
        # RERANK
        # ====================================================

        request = RerankRequest(
            query=query,
            passages=passages,
        )

        results = ranker.rerank(request)

        # GP005 diagnostic trace never gates the logic below it.
        _log_gp005_trace(query, results, indexed_docs)

        # ====================================================
        # MAP RESULTS BACK TO ORIGINAL DOCUMENTS
        # ====================================================

        by_id = {str(index): document for index, document in indexed_docs}

        reranked: List[RankedResult] = []

        for result in results[:top_n]:
            # FlashRank normally returns dictionaries.
            # Handle object-style results defensively too.

            if isinstance(result, dict):
                result_id = str(result.get("id"))

                rerank_score = result.get("score")

            else:
                result_id = str(
                    getattr(
                        result,
                        "id",
                        "",
                    )
                )

                rerank_score = getattr(
                    result,
                    "score",
                    None,
                )

            original = by_id.get(result_id)

            if original is None:
                continue

            enriched = dict(original)

            if rerank_score is not None:
                enriched["rerank_score"] = float(rerank_score)

            reranked.append(enriched)

        # ====================================================
        # LOG RESULT
        # ====================================================

        duration = time.time() - start_time

        top_score = reranked[0].get("rerank_score") if reranked else "N/A"

        logfire.info(
            f"Rerank done in {duration:.2f}s. "
            f"Kept {len(reranked)} document(s). "
            f"Top score: {top_score}."
        )

        # ----------------------------------------------------
        # Defensive fallback if FlashRank returned nothing
        # ----------------------------------------------------

        if not reranked:
            logfire.warning(
                "⚠️ FlashRank returned no usable results. "
                "Falling back to original retrieval order."
            )

            return [dict(document) for document in normalized_documents[:top_n]]

        return reranked

    # ========================================================
    # GRACEFUL FAILURE
    # ========================================================

    except Exception as exc:
        logfire.exception("❌ FlashRank reranking failed")

        logfire.warning(
            f"⚠️ Falling back to original retrieval "
            f"order because reranking failed: {exc}"
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

    wrapped = [{"content": document} for document in documents]

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

