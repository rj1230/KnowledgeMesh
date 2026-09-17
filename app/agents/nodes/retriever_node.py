from __future__ import annotations

import time

import logfire

from app.agents.state import AgentState
from app.services.retrieval.qdrant_service import (
    search_enterprise_knowledge,
)
from app.services.retrieval.ranking_service import (
    rerank_documents,
)


def retrieve_node(state: AgentState):
    """
    Retrieve enterprise documents from Qdrant and rerank them.

    Contract:
        documents -> List[RetrievedDocument]

    The output is intentionally structured so downstream
    grading/evaluation nodes can consume document metadata,
    scores, and content consistently.
    """

    query = state["current_query"]

    total_start = time.perf_counter()

    # ============================================================
    # QDRANT RETRIEVAL
    # ============================================================

    retrieval_start = time.perf_counter()

    with logfire.span(
        "🔍 Internal Knowledge Retrieval",
        query=query,
    ):
        raw_results = search_enterprise_knowledge(
            query=query,
            limit=15,
        )

    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

    # ============================================================
    # FLASHRANK RERANKING
    # ============================================================

    rerank_start = time.perf_counter()

    with logfire.span(
        "⚖️ Internal Semantic Reranking",
        candidate_count=len(raw_results),
    ):
        reranked_documents = rerank_documents(
            query=query,
            documents=raw_results,
            top_n=5,
            text_key="content",
        )

    rerank_ms = (time.perf_counter() - rerank_start) * 1000

    total_ms = (time.perf_counter() - total_start) * 1000

    # ============================================================
    # NORMALIZE DOCUMENTS
    # ============================================================

    documents = []

    for doc in reranked_documents:
        documents.append(
            {
                "id": str(doc.get("id", "")),
                "content": str(doc.get("content", "")),
                "source": str(doc.get("source", "Unknown")),
                "source_type": str(doc.get("source_type", "internal")),
                "score": float(doc.get("score", 0.0) or 0.0),
                "rerank_score": (
                    float(doc["rerank_score"])
                    if doc.get("rerank_score") is not None
                    else None
                ),
                "url": doc.get("url"),
            }
        )

    # ============================================================
    # OBSERVABILITY
    # ============================================================

    logfire.info(
        "📊 Retrieval pipeline timing",
        retrieval_ms=round(retrieval_ms, 2),
        rerank_ms=round(rerank_ms, 2),
        total_ms=round(total_ms, 2),
        candidates=len(raw_results),
        final_documents=len(documents),
    )

    # ============================================================
    # RETURN
    # ============================================================

    return {
        "documents": documents,
        "search_query": query,
        "retrieval_latency_ms": retrieval_ms,
        "rerank_latency_ms": rerank_ms,
        "status": (f"Retrieved and reranked {len(documents)} internal sources."),
        "plan": list(state.get("plan", []))
        + [
            f"Internal Retrieval: {len(documents)} documents",
            f"Retrieval Time: {retrieval_ms:.0f} ms",
            f"Rerank Time: {rerank_ms:.0f} ms",
        ],
    }
