"""
Evaluates whether internal retrieval is strong enough
to answer the user's question.
"""

from __future__ import annotations

import re
from typing import Any


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

    return any(term in query_lower for term in CURRENT_INFORMATION_TERMS)


def evaluate_context(
    query: str,
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Decide whether internal documents are sufficient.

    This is intentionally deterministic and inexpensive.
    """

    if requires_current_web_information(query):
        return {
            "quality": "needs_web",
            "reason": "The query requests current or time-sensitive information.",
            "should_search_web": True,
        }

    if not documents:
        return {
            "quality": "empty",
            "reason": "No internal documents were retrieved.",
            "should_search_web": True,
        }

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

    # Tune these thresholds using your evaluation dataset.
    if best_rerank_score >= 0.70:
        return {
            "quality": "strong",
            "reason": "Relevant internal context was retrieved.",
            "should_search_web": False,
        }

    if best_vector_score >= 0.65:
        return {
            "quality": "moderate",
            "reason": "Internal context is available but may be incomplete.",
            "should_search_web": True,
        }

    return {
        "quality": "weak",
        "reason": "Internal retrieval confidence is low.",
        "should_search_web": True,
    }
