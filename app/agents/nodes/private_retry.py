"""
KnowledgeMesh — Private Knowledge Base Retry

Runs Qdrant + reranking again using the rewritten query.
"""

from __future__ import annotations

import logging

from app.agents.nodes.retriever import retrieve_node

logger = logging.getLogger(__name__)


def retry_private_retrieval_node(state):
    """
    Reuse the existing retrieval implementation with the rewritten query.
    """

    rewritten_query = (
        state.get("rewritten_query")
        or state.get("current_query")
        or state.get("original_query")
        or ""
    ).strip()

    if not rewritten_query:
        return {
            "documents": [],
            "context_quality": "empty",
            "status": "Private KB retry skipped: empty query.",
            "plan": state.get("plan", [])
            + [
                "Private KB Retry: skipped",
            ],
        }

    retry_state = dict(state)

    retry_state["current_query"] = rewritten_query

    try:
        result = retrieve_node(retry_state)

        documents = result.get("documents") or []

        plan = state.get("plan", [])

        plan = plan + [
            f"Private KB Retry: {len(documents)} documents",
        ]

        return {
            **result,
            "current_query": rewritten_query,
            "documents": documents,
            "status": (f"Private KB retry returned {len(documents)} documents."),
            "plan": plan,
        }

    except Exception as exc:
        logger.exception("Private KB retry failed")

        return {
            "documents": [],
            "context_quality": "empty",
            "status": (f"Private KB retry failed: {type(exc).__name__}"),
            "plan": state.get("plan", [])
            + [
                f"Private KB Retry: failed ({type(exc).__name__})",
            ],
        }
