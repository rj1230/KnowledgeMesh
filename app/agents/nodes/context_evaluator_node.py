from __future__ import annotations

import logging

import logfire

from app.agents.context_evaluator import evaluate_context
from app.agents.state import AgentState


logger = logging.getLogger("knowledgemesh")


def context_evaluator_node(state: AgentState):
    """
    Evaluate whether privately retrieved evidence is sufficient.

    The evaluator uses the document-grading outcome as the primary
    signal for deciding whether private context is sufficient.

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
    graded_documents = state.get("graded_documents") or []
    generation_documents = state.get("generation_documents") or []

    logger.info(
        "🧪 CONTEXT EVALUATOR INPUT | "
        "documents=%s | scores=%s | rerank_scores=%s | "
        "graded_documents=%s | generation_documents=%s",
        len(documents),
        [doc.get("score") for doc in documents],
        [doc.get("rerank_score") for doc in documents],
        len(graded_documents),
        len(generation_documents),
    )

    logger.info(
        "🧪 CONTEXT EVALUATOR STATE | "
        "documents=%s | graded_documents=%s | "
        "generation_documents=%s",
        len(documents),
        len(graded_documents),
        len(generation_documents),
    )

    evaluation = evaluate_context(
        query=query,
        documents=documents,
        generation_documents=generation_documents,
        graded_documents=graded_documents,
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