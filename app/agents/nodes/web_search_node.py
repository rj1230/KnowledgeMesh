from __future__ import annotations

import logfire

from app.agents.state import AgentState
from app.services.web_search import search_web


def web_search_node(state: AgentState):
    query = state.get("search_query") or state["current_query"]

    with logfire.span(
        "🌐 Web Search Node",
        query=query,
    ):
        web_documents = search_web(
            query=query,
            max_results=5,
        )

    existing_documents = state.get("documents") or []

    all_documents = [
        *existing_documents,
        *web_documents,
    ]

    return {
        "web_documents": web_documents,
        "all_documents": all_documents,
        "web_search_used": bool(web_documents),
        "status": (f"External search returned {len(web_documents)} sources."),
        "plan": state.get("plan", [])
        + [f"Web Search: {len(web_documents)} external sources"],
    }
