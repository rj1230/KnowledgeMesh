from __future__ import annotations

import logfire

from app.agents.context_evaluator import evaluate_context
from app.agents.state import AgentState


def context_evaluator_node(state: AgentState):
    """
    Evaluate whether privately retrieved evidence is sufficient.

    State contract
    --------------
    should_search_web:
        Preserves the evaluator's original web-search decision.

    web_search_required:
        Canonical routing flag consumed by graph.py.

    Keeping both fields prevents breaking existing API/observability
    code while ensuring the LangGraph router receives the correct
    decision.
    """

    query = state["current_query"]
    documents = state.get("documents") or []

    evaluation = evaluate_context(
        query=query,
        documents=documents,
    )

    should_search_web = bool(evaluation["should_search_web"])

    with logfire.span(
        "🧪 Context Evaluation",
        quality=evaluation["quality"],
        should_search_web=should_search_web,
        web_search_required=should_search_web,
    ):
        logfire.info(
            "Context evaluation completed",
            reason=evaluation["reason"],
            quality=evaluation["quality"],
            web_search_required=should_search_web,
        )

    return {
        # ============================================================
        # CONTEXT EVALUATION
        # ============================================================
        "context_quality": evaluation["quality"],
        "context_reason": evaluation["reason"],
        # ============================================================
        # WEB ROUTING
        #
        # Keep the original evaluator field for compatibility.
        # graph.py uses web_search_required as the canonical
        # routing signal.
        # ============================================================
        "should_search_web": should_search_web,
        "web_search_required": should_search_web,
        # ============================================================
        # STATUS / OBSERVABILITY
        # ============================================================
        "status": evaluation["reason"],
        "plan": state.get("plan", [])
        + [
            f"Context Quality: {evaluation['quality']}",
            f"Web Search Required: {should_search_web}",
        ],
    }
