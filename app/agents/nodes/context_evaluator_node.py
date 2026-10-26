from __future__ import annotations

import logging
import re

import logfire

from app.agents.context_evaluator import evaluate_context
from app.agents.state import AgentState
from app.evaluation.trace import append_trace_event


logger = logging.getLogger("knowledgemesh")

# "according to the loop paper", "in the survey", "from this report" ...
# Generic words such as "documentation" or "docs" are deliberately left out:
# they mean the whole knowledge base, not one named document.
_DOCUMENT_REF = re.compile(
    r"\b(?:according to|as (?:stated|described|reported|defined|shown|discussed) in|"
    r"in|from|per|based on)\s+(?:the|this|that)\s+(?:[\w\-\.]+\s+){0,4}?"
    r"(?:paper|survey|report|article|study|whitepaper|pdf|manual)\b",
    re.IGNORECASE,
)


# A literal file name ("loop_engineering.pdf", "Q3-report.docx") is an
# unambiguous reference to one document, with or without "the".
_FILENAME_REF = re.compile(
    r"\b[\w\-]+(?:\.[\w\-]+)*\.(?:pdf|docx?|pptx?|md|txt|html?|csv|xlsx?)\b",
    re.IGNORECASE,
)


def _is_document_scoped(query: str) -> bool:
    query = query or ""
    return bool(_DOCUMENT_REF.search(query) or _FILENAME_REF.search(query))


def context_evaluator_node(state: AgentState):
    """
    Evaluate whether privately retrieved evidence is sufficient.

    State contract
    --------------
    should_search_web / web_search_required:
        The evaluator's web-search decision (both kept for compatibility).

    document_scoped:
        True when the user's question names a specific document. The graph
        uses it to avoid answering such a question from the web.
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

    evaluation = evaluate_context(
        query=query,
        documents=documents,
        generation_documents=generation_documents,
        graded_documents=graded_documents,
    )

    should_search_web = bool(evaluation["should_search_web"])

    document_scoped = _is_document_scoped(
        str(state.get("original_query") or state.get("current_query") or "")
    )

    with logfire.span(
        "🧪 Context Evaluation",
        quality=evaluation["quality"],
        should_search_web=should_search_web,
        web_search_required=should_search_web,
        document_scoped=document_scoped,
    ):
        logfire.info(
            "Context evaluation completed",
            reason=evaluation["reason"],
            quality=evaluation["quality"],
            web_search_required=should_search_web,
            document_scoped=document_scoped,
        )

    trace_update = append_trace_event(
        state,
        step="context_evaluator",
        status=evaluation["quality"],
        should_search_web=should_search_web,
        reason=evaluation["reason"],
        graded_documents=len(graded_documents),
        generation_documents=len(generation_documents),
        document_scoped=document_scoped,
    )

    return {
        **trace_update,
        "context_quality": evaluation["quality"],
        "context_reason": evaluation["reason"],
        "should_search_web": should_search_web,
        "web_search_required": should_search_web,
        "document_scoped": document_scoped,
        "status": evaluation["reason"],
        "plan": state.get("plan", [])
        + [
            f"Context Quality: {evaluation['quality']}",
            f"Web Search Required: {should_search_web}",
        ],
    }
