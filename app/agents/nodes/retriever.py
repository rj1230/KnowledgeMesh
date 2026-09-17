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
    # TIMING OBSERVABILITY
    # ============================================================

    logfire.info(
        "📊 Retrieval pipeline timing",
        retrieval_ms=round(retrieval_ms, 2),
        rerank_ms=round(rerank_ms, 2),
        total_ms=round(total_ms, 2),
        candidates=len(raw_results),
        final_documents=len(reranked_documents),
    )

    # ============================================================
    # NORMALIZE DOCUMENTS
    # ============================================================

    documents = [
        {
            "id": doc.get("id"),
            "content": doc.get("content", ""),
            "source": doc.get("source", "Unknown"),
            "source_type": doc.get("source_type", "internal"),
            "score": doc.get("score"),
            "rerank_score": doc.get("rerank_score"),
            "url": doc.get("url"),
        }
        for doc in reranked_documents
    ]

    return {
        "documents": documents,
        "all_documents": documents,
        "search_query": query,
        "retrieval_latency_ms": retrieval_ms,
        "rerank_latency_ms": rerank_ms,
        "status": (f"Retrieved and reranked {len(documents)} internal sources."),
        "plan": state.get("plan", [])
        + [
            f"Internal Retrieval: {len(documents)} documents",
            f"Retrieval Time: {retrieval_ms:.0f} ms",
            f"Rerank Time: {rerank_ms:.0f} ms",
        ],
    }
