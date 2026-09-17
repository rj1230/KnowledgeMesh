"""
KnowledgeMesh — LangGraph Self-RAG Orchestrator

Architecture:

    Planner
       ↓
    Private Retrieval
       ↓
    Document Grader
       ↓
    Context Evaluator
       ├── strong → Responder
       ├── needs_web → Web Search → Responder
       └── weak/moderate/empty
                ↓
          Query Rewriter
                ↓
          Private Retry
                ↓
             Grader
                ↓
        Context Evaluator

    Responder
       ↓
    Grounding Critic
       ↓
    Citation Check
       ├── PASS → END
       └── FAIL
            ↓
         Revision
            ↓
        Responder

Bounded loops:

    Private retrieval rewrites ≤ 2
    Answer revisions           ≤ 2
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.agents.state import AgentState

from app.agents.nodes.planner import planner_node
from app.agents.nodes.retriever import retrieve_node
from app.agents.nodes.grader import grade_documents_node
from app.agents.nodes.query_rewriter import rewrite_query_node
from app.agents.nodes.private_retry import (
    retry_private_retrieval_node,
)
from app.agents.nodes.context_evaluator_node import (
    context_evaluator_node,
)
from app.agents.nodes.web_search_node import (
    web_search_node,
)
from app.agents.nodes.responder import (
    generate_node,
)
from app.agents.nodes.grounding_critic import (
    grounding_critic_node,
)
from app.agents.nodes.citation_check import (
    citation_check_node,
)


# ============================================================
# CONSTANTS
# ============================================================

MAX_RETRIEVAL_REWRITES = 2
MAX_ANSWER_REVISIONS = 2


# ============================================================
# GRAPH
# ============================================================

workflow = StateGraph(AgentState)


# ============================================================
# NODES
# ============================================================

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
    "grounding_critic",
    grounding_critic_node,
)

workflow.add_node(
    "citation_check",
    citation_check_node,
)


# ============================================================
# PLANNER ROUTING
# ============================================================


def route_after_planner(
    state: AgentState,
) -> str:
    """
    Conversational requests skip retrieval.

    Technical requests enter private retrieval.
    """

    query = (state.get("current_query") or "").strip()

    if query == "CONVERSATIONAL":
        return "responder"

    return "retriever"


# ============================================================
# CONTEXT ROUTING
# ============================================================


def route_after_context_evaluator(
    state: AgentState,
) -> str:
    """
    Route based on deterministic context quality.
    """

    quality = (state.get("context_quality") or "empty").strip().lower()

    # --------------------------------------------------------
    # Strong private evidence
    # --------------------------------------------------------

    if quality == "strong":
        return "responder"

    # --------------------------------------------------------
    # Explicit current-information requirement
    # --------------------------------------------------------

    if quality == "needs_web":
        return "web_search"

    # --------------------------------------------------------
    # Private retrieval retry
    # --------------------------------------------------------

    rewrite_count = int(
        state.get(
            "retrieval_rewrite_count",
            0,
        )
    )

    if rewrite_count >= MAX_RETRIEVAL_REWRITES:
        return "web_search"

    return "query_rewriter"


# ============================================================
# ANSWER VALIDATION ROUTING
# ============================================================


def route_after_citation_check(
    state: AgentState,
) -> str:
    """
    Return END when both citation and semantic grounding checks pass.

    Otherwise perform a bounded answer revision.
    """

    route = (state.get("route") or "").strip().lower()

    # --------------------------------------------------------
    # Conversational route
    # --------------------------------------------------------

    if route == "simple":
        return "end"

    # --------------------------------------------------------
    # Validation results
    # --------------------------------------------------------

    citation_valid = bool(
        state.get(
            "citation_valid",
            False,
        )
    )

    is_grounded = bool(
        state.get(
            "is_grounded",
            False,
        )
    )

    # --------------------------------------------------------
    # Fully validated
    # --------------------------------------------------------

    if citation_valid and is_grounded:
        return "end"

    # --------------------------------------------------------
    # Revision budget
    # --------------------------------------------------------

    revision_count = int(
        state.get(
            "revision_count",
            0,
        )
    )

    if revision_count >= MAX_ANSWER_REVISIONS:
        return "end"

    return "revise"


# ============================================================
# REVISION NODE
# ============================================================


def revision_node(
    state: AgentState,
):
    """
    Prepare state for another response-generation attempt.

    The node itself does not generate an answer.

    It only:
        - increments revision_count
        - records why revision was required
        - preserves grounding feedback
        - records support score
        - records feedback count
    """

    current_revision = int(
        state.get(
            "revision_count",
            0,
        )
    )

    next_revision = current_revision + 1

    citation_valid = bool(
        state.get(
            "citation_valid",
            False,
        )
    )

    is_grounded = bool(
        state.get(
            "is_grounded",
            False,
        )
    )

    support_score = float(
        state.get(
            "support_score",
            0.0,
        )
        or 0.0
    )

    feedback = (
        state.get(
            "grounding_feedback",
        )
        or []
    )

    reasons: list[str] = []

    if not citation_valid:
        reasons.append("citation validation failed")

    if not is_grounded:
        reasons.append("grounding validation failed")

    reason_text = "; ".join(reasons) if reasons else "answer requires revision"

    status = (
        f"Answer revision "
        f"{next_revision}/"
        f"{MAX_ANSWER_REVISIONS}: "
        f"{reason_text}. "
        f"Support score="
        f"{support_score:.3f}; "
        f"feedback_items="
        f"{len(feedback)}."
    )

    plan = list(
        state.get(
            "plan",
            [],
        )
    )

    plan.append(f"Answer Revision: {next_revision}/{MAX_ANSWER_REVISIONS}")

    return {
        "revision_count": next_revision,
        "status": status,
        "plan": plan,
    }


# ============================================================
# ENTRY POINT
# ============================================================

workflow.set_entry_point("planner")


# ============================================================
# PLANNER ROUTING
# ============================================================

workflow.add_conditional_edges(
    "planner",
    route_after_planner,
    {
        "retriever": "retriever",
        "responder": "responder",
    },
)


# ============================================================
# PRIVATE RETRIEVAL
# ============================================================

workflow.add_edge(
    "retriever",
    "grader",
)

workflow.add_edge(
    "grader",
    "context_evaluator",
)


# ============================================================
# CONTEXT EVALUATION
# ============================================================

workflow.add_conditional_edges(
    "context_evaluator",
    route_after_context_evaluator,
    {
        "responder": "responder",
        "query_rewriter": "query_rewriter",
        "web_search": "web_search",
    },
)


# ============================================================
# PRIVATE RETRIEVAL RETRY
# ============================================================

workflow.add_edge(
    "query_rewriter",
    "private_retry",
)

workflow.add_edge(
    "private_retry",
    "grader",
)


# ============================================================
# WEB FALLBACK
# ============================================================

workflow.add_edge(
    "web_search",
    "responder",
)


# ============================================================
# RESPONSE GENERATION
# ============================================================

workflow.add_edge(
    "responder",
    "grounding_critic",
)


# ============================================================
# GROUNDING VALIDATION
# ============================================================

workflow.add_edge(
    "grounding_critic",
    "citation_check",
)


# ============================================================
# CITATION VALIDATION
# ============================================================

workflow.add_conditional_edges(
    "citation_check",
    route_after_citation_check,
    {
        "end": END,
        "revise": "revision",
    },
)


# ============================================================
# ANSWER REVISION
# ============================================================

workflow.add_node(
    "revision",
    revision_node,
)

workflow.add_edge(
    "revision",
    "responder",
)


# ============================================================
# CHECKPOINTING
# ============================================================

checkpointer = MemorySaver()


rag_agent = workflow.compile(
    checkpointer=checkpointer,
)
