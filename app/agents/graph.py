"""
KnowledgeMesh · Agentic RAG Graph
=================================

LangGraph orchestration for the Agentic RAG pipeline.

Pipeline
--------

START
  ↓
planner
  ↓
retriever
  ↓
grader
  ↓
context_evaluator
  ├── strong → responder
  ├── insufficient → query_rewriter → private_retry → grader
  └── web fallback → web_search → grader
                                      ↓
                                   responder
                                      ↓
                                citation_check
                                  ├── PASS → grounding_critic
                                  │              ├── PASS → END
                                  │              └── FAIL → prepare_revision
                                  │                              ↓
                                  │                          revision
                                  │                              ↓
                                  │                          responder
                                  │
                                  └── FAIL → prepare_revision
                                                 ↓
                                             revision
                                                 ↓
                                             responder


Revision safety
---------------

answer_revision_count is the canonical revision counter.

MAX_ANSWER_REVISIONS controls the maximum number of answer
regenerations after the initial response.

The graph explicitly prepares a revision before entering the
revision responder.

prepare_revision_node:

    current count
        ↓
    +1
        ↓
    revision_requested = True
        ↓
    revision_prompt
        ↓
    revision responder

After regeneration, the responder is responsible for clearing
revision_requested.

Grounding compatibility
-----------------------

The current grounding critic returns:

    is_grounded
    answer_supported
    support_score
    grounding_scores
    grounding_feedback
    unsupported_atomic_claims
    revision_prompt

The graph therefore treats:

    is_grounded

as the canonical grounding routing signal, while also accepting:

    grounding_valid
    answer_supported

for backward compatibility.

This prevents a state-field mismatch from turning every successful
grounding evaluation into a revision.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import logfire

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.agents.nodes.citation_check import (
    citation_check_node,
)
from app.agents.nodes.context_evaluator_node import (
    context_evaluator_node,
)
from app.agents.nodes.grader import (
    grade_documents_node,
)
from app.agents.nodes.grounding_critic import (
    grounding_critic_node,
)
from app.agents.nodes.planner import (
    planner_node,
)
from app.agents.nodes.private_retry import (
    retry_private_retrieval_node,
)
from app.agents.nodes.query_rewriter import (
    rewrite_query_node,
)
from app.agents.nodes.responder import (
    generate_node,
)
from app.agents.nodes.retriever import (
    retrieve_node,
)
from app.agents.nodes.web_search_node import (
    web_search_node,
)
from app.agents.state import AgentState


# ============================================================
# Configuration
# ============================================================

MAX_RETRIEVAL_REWRITES = 2

MAX_ANSWER_REVISIONS = 2


logger = logging.getLogger(__name__)


# ============================================================
# Helpers
# ============================================================


def _safe_int(
    value: Any,
    default: int = 0,
) -> int:
    """
    Safely convert a state value to int.
    """

    try:
        return int(value or default)
    except (
        TypeError,
        ValueError,
    ):
        return default


def _revision_count(
    state: AgentState,
) -> int:
    """
    Return the canonical answer revision count.

    answer_revision_count is preferred.

    revision_count is retained as a compatibility fallback for
    older nodes/state.
    """

    value = state.get("answer_revision_count")

    if value is None:
        value = state.get(
            "revision_count",
            0,
        )

    return max(
        0,
        _safe_int(value),
    )


def _normalise_feedback(
    value: Any,
) -> list[str]:
    """
    Normalize grounding/citation feedback into a list of strings.

    Important:

        grounding_feedback may be either:

            "Grounding validation failed: 4 unsupported atomic claim(s)."

        or:

            [
                "Grounding validation failed...",
                "Remove unsupported claims.",
            ]

    Never iterate directly over a string because that would produce
    individual characters.
    """

    if value is None:
        return []

    if isinstance(value, str):
        cleaned = value.strip()

        if not cleaned:
            return []

        return [cleaned]

    if isinstance(value, (list, tuple, set)):
        result: list[str] = []

        for item in value:
            cleaned = str(item).strip()

            if cleaned:
                result.append(cleaned)

        return result

    cleaned = str(value).strip()

    if not cleaned:
        return []

    return [cleaned]


# ============================================================
# Planner routing
# ============================================================


def route_after_planner(
    state: AgentState,
) -> str:
    """
    Decide whether the planner sends the request directly to the
    responder or through retrieval.
    """

    route = str(state.get("route") or "").strip().lower()

    retrieval_required = bool(
        state.get(
            "retrieval_required",
            True,
        )
    )

    if (
        route
        in {
            "responder",
            "direct",
            "conversation",
            "chat",
        }
        and not retrieval_required
    ):
        logger.info("Planner route: responder")

        return "responder"

    logger.info("Planner route: retriever")

    return "retriever"


# ============================================================
# Context evaluator routing
# ============================================================


def route_after_context_evaluator(
    state: AgentState,
) -> str:
    """
    Decide whether context is sufficient, should be rewritten,
    or should escalate to web search.
    """

    quality = str(state.get("context_quality") or "").strip().lower()

    web_search_attempted = bool(
        state.get(
            "web_search_attempted",
            False,
        )
    )

    web_search_required = bool(
        state.get(
            "web_search_required",
            False,
        )
    )

    should_search_web = bool(
        state.get(
            "should_search_web",
            False,
        )
    )

    grader_status = str(state.get("grader_status") or "").strip().lower()

    grading_mode = str(state.get("grading_mode") or "").strip().lower()

    rewrite_count = _safe_int(
        state.get(
            "retrieval_rewrite_count",
            0,
        )
    )

    # --------------------------------------------------------
    # 1. Strong validated context
    # --------------------------------------------------------

    if quality == "strong":
        logger.info("Context route: responder | strong context")

        return "responder"

    # --------------------------------------------------------
    # 2. Web was already attempted
    #
    # Prevent repeated web-search loops.
    # --------------------------------------------------------

    if web_search_attempted:
        logger.warning(
            "Web search already attempted; preventing another "
            "web-search loop. Routing to responder."
        )

        return "responder"

    # --------------------------------------------------------
    # 3. Explicit web requirement
    # --------------------------------------------------------

    if web_search_required or should_search_web:
        logger.info("Context route: web_search | explicit web requirement")

        return "web_search"

    # --------------------------------------------------------
    # 4. Grader failure
    #
    # Private evidence could not be validated.
    # Escalate to web before generation.
    # --------------------------------------------------------

    if grader_status == "failed" or grading_mode == "unavailable":
        logger.warning("Context route: web_search | grader unavailable")

        return "web_search"

    # --------------------------------------------------------
    # 5. Retrieval rewrite budget
    # --------------------------------------------------------

    if rewrite_count >= MAX_RETRIEVAL_REWRITES:
        logger.info("Retrieval rewrite budget exhausted; routing to web_search.")

        return "web_search"

    # --------------------------------------------------------
    # 6. Default
    # --------------------------------------------------------

    logger.info("Context route: query_rewriter | insufficient private context")

    return "query_rewriter"


# ============================================================
# Responder routing
# ============================================================


def route_after_responder(
    state: AgentState,
) -> str:
    """
    Route a successfully generated answer into citation validation.

    If generation failed, terminate instead of entering validation
    with an empty/invalid answer.
    """

    generation_failed = bool(
        state.get(
            "generation_failed",
            False,
        )
    )

    if generation_failed:
        logger.warning("Responder generation failed; ending graph.")

        return "end"

    logger.info("Responder route: citation_check | answer generated")

    return "citation_check"


# ============================================================
# Citation routing
# ============================================================


def route_after_citation_check(
    state: AgentState,
) -> str:
    """
    Route based on citation validation.

    Citation PASS:
        grounding_critic

    Citation FAIL:
        bounded revision

    If the revision budget is exhausted:
        END
    """

    citation_valid = bool(
        state.get(
            "citation_valid",
            False,
        )
    )

    revision_count = _revision_count(state)

    # --------------------------------------------------------
    # Citation PASS
    # --------------------------------------------------------

    if citation_valid:
        logger.info("Citation route: grounding_critic | citation validation passed")

        return "grounding_critic"

    # --------------------------------------------------------
    # Citation FAIL + budget exhausted
    # --------------------------------------------------------

    if revision_count >= MAX_ANSWER_REVISIONS:
        logger.warning(
            "Citation validation failed but answer revision "
            "budget is exhausted. Ending graph.",
            extra={
                "revision_count": revision_count,
                "max_answer_revisions": MAX_ANSWER_REVISIONS,
            },
        )

        return "end"

    # --------------------------------------------------------
    # Citation FAIL → revision
    # --------------------------------------------------------

    logger.warning(
        "Citation route: revision | citation validation failed | revisions=%s/%s",
        revision_count,
        MAX_ANSWER_REVISIONS,
    )

    return "revision"


# ============================================================
# Grounding routing
# ============================================================


def route_after_grounding_critic(
    state: AgentState,
) -> str:
    """
    Route based on strict grounding validation.

    Canonical signal:
        is_grounded

    Compatibility signals:
        grounding_valid
        answer_supported

    This is important because the current grounding critic returns
    `is_grounded` and `answer_supported`, while older graph versions
    expected `grounding_valid`.
    """

    # --------------------------------------------------------
    # Canonical grounding result
    # --------------------------------------------------------

    is_grounded_value = state.get("is_grounded")

    # --------------------------------------------------------
    # Compatibility fallback
    # --------------------------------------------------------

    if is_grounded_value is None:
        grounding_valid_value = state.get("grounding_valid")

        if grounding_valid_value is not None:
            is_grounded_value = grounding_valid_value

    # --------------------------------------------------------
    # Final compatibility fallback
    # --------------------------------------------------------

    if is_grounded_value is None:
        answer_supported_value = state.get("answer_supported")

        if answer_supported_value is not None:
            is_grounded_value = answer_supported_value

    grounding_valid = bool(is_grounded_value)

    revision_count = _revision_count(state)

    # --------------------------------------------------------
    # Observability
    # --------------------------------------------------------

    logger.info(
        "Grounding routing | is_grounded=%s | "
        "answer_supported=%s | grounding_valid=%s | "
        "revision_count=%s/%s",
        state.get("is_grounded"),
        state.get("answer_supported"),
        state.get("grounding_valid"),
        revision_count,
        MAX_ANSWER_REVISIONS,
    )

    # --------------------------------------------------------
    # Grounding PASS
    # --------------------------------------------------------

    if grounding_valid:
        logger.info("Grounding route: end | grounding validation passed")

        return "end"

    # --------------------------------------------------------
    # Grounding FAIL + budget exhausted
    # --------------------------------------------------------

    if revision_count >= MAX_ANSWER_REVISIONS:
        logger.warning(
            "Grounding validation failed but answer revision "
            "budget is exhausted. Ending graph.",
            extra={
                "revision_count": revision_count,
                "max_answer_revisions": MAX_ANSWER_REVISIONS,
            },
        )

        return "end"

    # --------------------------------------------------------
    # Grounding FAIL → revision
    # --------------------------------------------------------

    logger.warning(
        "Grounding route: revision | grounding validation failed | revisions=%s/%s",
        revision_count,
        MAX_ANSWER_REVISIONS,
    )

    return "revision"


# ============================================================
# Revision preparation
# ============================================================


def prepare_revision_node(
    state: AgentState,
) -> Dict[str, Any]:
    """
    Prepare exactly one bounded answer revision.

    Responsibilities:

        1. Read current revision count.
        2. Increment it exactly once.
        3. Preserve citation feedback.
        4. Preserve grounding feedback.
        5. Preserve grounding-generated revision_prompt.
        6. Set revision_requested=True.

    The actual regenerated answer is produced by generate_node.
    """

    current_count = _revision_count(state)

    next_count = current_count + 1

    # --------------------------------------------------------
    # Hard revision budget
    # --------------------------------------------------------

    if next_count > MAX_ANSWER_REVISIONS:
        logger.warning(
            "Revision request blocked because revision budget is exhausted.",
            extra={
                "revision_count": current_count,
                "max_answer_revisions": MAX_ANSWER_REVISIONS,
            },
        )

        return {
            "revision_requested": False,
            "answer_revision_count": current_count,
            "revision_count": current_count,
            "status": ("Revision blocked: maximum answer revisions reached."),
        }

    # --------------------------------------------------------
    # Citation feedback
    # --------------------------------------------------------

    citation_feedback = str(state.get("citation_feedback") or "").strip()

    # --------------------------------------------------------
    # Grounding feedback
    #
    # IMPORTANT:
    #
    # The grounding critic currently returns a string.
    # Normalize it before processing.
    # --------------------------------------------------------

    grounding_feedback = _normalise_feedback(state.get("grounding_feedback"))

    # --------------------------------------------------------
    # Combine feedback
    # --------------------------------------------------------

    feedback_parts: list[str] = []

    if citation_feedback:
        feedback_parts.append(citation_feedback)

    feedback_parts.extend(grounding_feedback)

    combined_feedback = "\n\n".join(feedback_parts)

    # --------------------------------------------------------
    # Grounding-generated revision prompt
    # --------------------------------------------------------

    revision_prompt = str(state.get("revision_prompt") or "").strip()

    if not revision_prompt:
        revision_prompt = (
            "Revise the previous answer using only the supplied "
            "evidence. Remove unsupported claims and ensure every "
            "substantive factual statement has a valid citation."
        )

    # --------------------------------------------------------
    # If feedback exists but the critic prompt is generic,
    # append the concrete validation feedback.
    # --------------------------------------------------------

    if (
        combined_feedback
        and "Grounding validation failed" not in revision_prompt
        and "unsupported" not in revision_prompt.lower()
    ):
        revision_prompt = (
            f"{revision_prompt}\n\nValidation feedback:\n{combined_feedback}"
        )

    # --------------------------------------------------------
    # Observability
    # --------------------------------------------------------

    logger.info(
        "Preparing answer revision %s/%s",
        next_count,
        MAX_ANSWER_REVISIONS,
    )

    with logfire.span(
        "🔄 Prepare Answer Revision",
        current_revision=current_count,
        next_revision=next_count,
        max_answer_revisions=MAX_ANSWER_REVISIONS,
        has_citation_feedback=bool(citation_feedback),
        grounding_feedback_count=len(grounding_feedback),
        has_revision_prompt=bool(revision_prompt),
    ):
        return {
            "revision_requested": True,
            # Canonical counter.
            "answer_revision_count": next_count,
            # Legacy compatibility counter.
            "revision_count": next_count,
            # Explicit revision instructions.
            "revision_prompt": revision_prompt,
            # Keep normalized feedback.
            "grounding_feedback": grounding_feedback,
            "citation_feedback": citation_feedback,
            "status": (
                f"Preparing answer revision {next_count}/{MAX_ANSWER_REVISIONS}."
            ),
        }


# ============================================================
# Graph construction
# ============================================================


def build_graph():
    """
    Build and compile the KnowledgeMesh Agentic RAG graph.
    """

    workflow = StateGraph(AgentState)

    # ========================================================
    # Nodes
    # ========================================================

    workflow.add_node(
        "planner",
        planner_node,
    )

    workflow.add_node(
        "retriever",
        retrieve_node,
    )

    workflow.add_node(
        "grader",
        grade_documents_node,
    )

    workflow.add_node(
        "context_evaluator",
        context_evaluator_node,
    )

    workflow.add_node(
        "query_rewriter",
        rewrite_query_node,
    )

    workflow.add_node(
        "private_retry",
        retry_private_retrieval_node,
    )

    workflow.add_node(
        "web_search",
        web_search_node,
    )

    workflow.add_node(
        "responder",
        generate_node,
    )

    workflow.add_node(
        "citation_check",
        citation_check_node,
    )

    workflow.add_node(
        "grounding_critic",
        grounding_critic_node,
    )

    # ========================================================
    # Revision preparation
    #
    # IMPORTANT:
    #
    # Do not point "revision" directly to generate_node.
    #
    # prepare_revision_node increments the bounded revision
    # counter and prepares the revision prompt first.
    # ========================================================

    workflow.add_node(
        "prepare_revision",
        prepare_revision_node,
    )

    # Logical revision node.
    #
    # This is intentionally the same generation implementation
    # used by the normal responder. The state tells responder
    # whether this invocation is a revision.
    workflow.add_node(
        "revision",
        generate_node,
    )

    # ========================================================
    # Entry
    # ========================================================

    workflow.add_edge(
        START,
        "planner",
    )

    # ========================================================
    # Planner
    # ========================================================

    workflow.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "retriever": "retriever",
            "responder": "responder",
        },
    )

    # ========================================================
    # Private retrieval
    # ========================================================

    workflow.add_edge(
        "retriever",
        "grader",
    )

    # ========================================================
    # Grading
    # ========================================================

    workflow.add_edge(
        "grader",
        "context_evaluator",
    )

    # ========================================================
    # Context evaluation
    # ========================================================

    workflow.add_conditional_edges(
        "context_evaluator",
        route_after_context_evaluator,
        {
            "responder": "responder",
            "query_rewriter": "query_rewriter",
            "web_search": "web_search",
        },
    )

    # ========================================================
    # Query rewrite
    # ========================================================

    workflow.add_edge(
        "query_rewriter",
        "private_retry",
    )

    workflow.add_edge(
        "private_retry",
        "grader",
    )

    # ========================================================
    # Web fallback
    #
    # Web results return to grader so they are validated before
    # reaching generation.
    # ========================================================

    workflow.add_edge(
        "web_search",
        "grader",
    )

    # ========================================================
    # Responder
    # ========================================================

    workflow.add_conditional_edges(
        "responder",
        route_after_responder,
        {
            "citation_check": "citation_check",
            "end": END,
        },
    )

    # ========================================================
    # Citation validation
    # ========================================================

    workflow.add_conditional_edges(
        "citation_check",
        route_after_citation_check,
        {
            "grounding_critic": "grounding_critic",
            "revision": "prepare_revision",
            "end": END,
        },
    )

    # ========================================================
    # Grounding validation
    # ========================================================

    workflow.add_conditional_edges(
        "grounding_critic",
        route_after_grounding_critic,
        {
            "revision": "prepare_revision",
            "end": END,
        },
    )

    # ========================================================
    # Revision preparation
    # ========================================================

    workflow.add_edge(
        "prepare_revision",
        "revision",
    )

    # ========================================================
    # Revision generation
    # ========================================================

    workflow.add_edge(
        "revision",
        "responder",
    )

    # ========================================================
    # Compile
    # ========================================================

    return workflow.compile(checkpointer=MemorySaver())


# ============================================================
# Compiled graph
# ============================================================

rag_agent = build_graph()
