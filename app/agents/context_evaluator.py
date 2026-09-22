"""Context quality evaluation for the agentic RAG pipeline."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("knowledgemesh")


CURRENT_INFORMATION_TERMS = {
    "latest",
    "current",
    "today",
    "recent",
    "news",
    "updated",
    "new version",
    "release",
    "pricing",
    "price",
    "2026",
}


def requires_current_web_information(query: str) -> bool:
    query_lower = query.lower()

    return any(
        term in query_lower
        for term in CURRENT_INFORMATION_TERMS
    )


def evaluate_context(
    query: str,
    documents: list[dict[str, Any]],
    generation_documents: list[dict[str, Any]] | None = None,
    graded_documents: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Decide whether internal documents are sufficient.

    Evaluation priority:

    1. Current/time-sensitive query -> web.
    2. No retrieved documents -> web.
    3. Grader-selected generation documents -> strong internal context.
    4. Graded documents exist but none are suitable for generation -> web.
    5. Fall back to dense retrieval confidence.

    FlashRank reranker scores are intentionally NOT used as a
    0-1 relevance threshold because their score scale is not
    compatible with the dense cosine similarity scale.
    """

    generation_documents = generation_documents or []
    graded_documents = graded_documents or []

    # ---------------------------------------------------------
    # 1. Current / time-sensitive information
    # ---------------------------------------------------------

    if requires_current_web_information(query):
        return {
            "quality": "needs_web",
            "reason": (
                "The query requests current or time-sensitive information."
            ),
            "should_search_web": True,
        }

    # ---------------------------------------------------------
    # 2. No internal retrieval
    # ---------------------------------------------------------

    if not documents:
        return {
            "quality": "empty",
            "reason": "No internal documents were retrieved.",
            "should_search_web": True,
        }

    # ---------------------------------------------------------
    # 3. Grader found usable generation evidence
    # ---------------------------------------------------------

    if generation_documents:
        logger.info(
            "🧪 CONTEXT EVALUATOR | "
            "usable_generation_documents=%s",
            len(generation_documents),
        )

        return {
            "quality": "strong",
            "reason": (
                f"{len(generation_documents)} relevant internal "
                "documents are available for generation."
            ),
            "should_search_web": False,
        }

    # ---------------------------------------------------------
    # 4. Documents were graded, but none passed generation
    # ---------------------------------------------------------

    if graded_documents:
        logger.info(
            "🧪 CONTEXT EVALUATOR | "
            "graded_documents=%s | generation_documents=0",
            len(graded_documents),
        )

        return {
            "quality": "insufficient",
            "reason": (
                "Internal documents were retrieved and graded, "
                "but none met the relevance threshold required "
                "for generation."
            ),
            "should_search_web": True,
        }

    # ---------------------------------------------------------
    # 5. Dense retrieval fallback
    # ---------------------------------------------------------

    vector_scores = [
        doc.get("score")
        for doc in documents
        if isinstance(doc.get("score"), (int, float))
    ]

    rerank_scores = [
        doc.get("rerank_score")
        for doc in documents
        if isinstance(doc.get("rerank_score"), (int, float))
    ]

    best_vector_score = max(vector_scores, default=0.0)
    best_rerank_score = max(rerank_scores, default=0.0)

    logger.info(
        "🧪 CONTEXT EVALUATOR SCORES | "
        "documents=%s | vector_scores=%s | rerank_scores=%s | "
        "best_vector=%.6f | best_rerank=%.6f",
        len(documents),
        vector_scores,
        rerank_scores,
        best_vector_score,
        best_rerank_score,
    )

    # ---------------------------------------------------------
    # IMPORTANT:
    # Do NOT compare FlashRank scores against 0.70.
    #
    # The current FlashRank scores are on a different scale.
    # Keep the existing dense-vector threshold unchanged.
    # ---------------------------------------------------------

    if best_vector_score >= 0.65:
        return {
            "quality": "moderate",
            "reason": (
                "Internal context is available, but it has not "
                "yet been confirmed as sufficient by the grader."
            ),
            "should_search_web": True,
        }

    return {
        "quality": "weak",
        "reason": "Internal retrieval confidence is low.",
        "should_search_web": True,
    }