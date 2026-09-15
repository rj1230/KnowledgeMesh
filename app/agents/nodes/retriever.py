from __future__ import annotations

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

    with logfire.span(
        "🔍 Internal Knowledge Retrieval",
        query=query,
    ):
        raw_results = search_enterprise_knowledge(
            query=query,
            limit=15,
        )

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
        "status": (f"Retrieved and reranked {len(documents)} internal sources."),
        "plan": state.get("plan", [])
        + [f"Internal Retrieval: {len(documents)} documents"],
    }
