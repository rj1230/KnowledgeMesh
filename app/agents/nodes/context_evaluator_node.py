from __future__ import annotations

import logfire

from app.agents.state import AgentState
from app.agents.context_evaluator import evaluate_context


def context_evaluator_node(state: AgentState):
    query = state["current_query"]
    documents = state.get("documents") or []

    evaluation = evaluate_context(
        query=query,
        documents=documents,
    )

    with logfire.span(
        "🧪 Context Evaluation",
        quality=evaluation["quality"],
        should_search_web=evaluation["should_search_web"],
    ):
        logfire.info(
            "Context evaluation completed",
            reason=evaluation["reason"],
        )

    return {
        "context_quality": evaluation["quality"],
        "context_reason": evaluation["reason"],
        "should_search_web": evaluation["should_search_web"],
        "status": evaluation["reason"],
        "plan": state.get("plan", []) + [f"Context Quality: {evaluation['quality']}"],
    }
