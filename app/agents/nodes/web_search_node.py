from __future__ import annotations

import logfire

from app.agents.state import AgentState
from app.services.web_search import search_web


def web_search_node(state: AgentState):
    """
    Web fallback for insufficient private evidence.

    Routing contract
    ----------------
    Private retrieval is attempted first.

    This node is entered only when the private context evaluator
    determines that external evidence is required.

    The node:
        1. Preserves private documents.
        2. Searches the web.
        3. Stores web evidence separately.
        4. Builds all_documents for downstream generation.
        5. Marks web_search_used.
        6. Clears web_search_required because the web-search
           decision has now been executed.
    """

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

    web_search_used = bool(web_documents)

    if web_search_used:
        context_quality = "web_sufficient"
        context_reason = (
            f"Private retrieval was insufficient; external search "
            f"returned {len(web_documents)} sources."
        )
    else:
        context_quality = "web_empty"
        context_reason = (
            "Private retrieval was insufficient and external search "
            "returned no usable sources."
        )

    return {
        # --------------------------------------------------------
        # Web evidence
        # --------------------------------------------------------
        "web_documents": web_documents,
        # --------------------------------------------------------
        # Combined evidence for downstream generation
        # --------------------------------------------------------
        "all_documents": all_documents,
        # --------------------------------------------------------
        # Web execution state
        # --------------------------------------------------------
        "web_search_used": web_search_used,
        # Web search has already been executed.
        "web_search_required": False,
        # Preserve the legacy/API-facing field.
        "should_search_web": False,
        # --------------------------------------------------------
        # Context status
        # --------------------------------------------------------
        "context_quality": context_quality,
        "context_reason": context_reason,
        "status": context_reason,
        # --------------------------------------------------------
        # Observability / planning
        # --------------------------------------------------------
        "plan": state.get("plan", [])
        + [f"Web Search: {len(web_documents)} external sources"],
    }
