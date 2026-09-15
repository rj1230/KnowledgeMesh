"""
KnowledgeMesh — LangGraph Self-RAG Orchestrator
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.agents.state import AgentState

from app.agents.nodes.planner import planner_node
from app.agents.nodes.retriever import retrieve_node
from app.agents.nodes.grader import grade_documents_node
from app.agents.nodes.query_rewriter import rewrite_query_node
from app.agents.nodes.private_retry import retry_private_retrieval_node
from app.agents.nodes.responder import generate_node


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
    "query_rewriter",
    rewrite_query_node,
)

workflow.add_node(
    "private_retry",
    retry_private_retrieval_node,
)

workflow.add_node(
    "responder",
    generate_node,
)


# ============================================================
# ROUTING
# ============================================================


def route_after_planner(state: AgentState):
    """
    Decide whether the request needs enterprise retrieval.
    """

    query = state.get("current_query", "")

    if query == "CONVERSATIONAL":
        return "responder"

    return "retriever"


def route_after_grader(state: AgentState):
    """
    Strong private context → answer.

    Weak/empty context → query rewrite.
    """

    context_quality = state.get(
        "context_quality",
        "empty",
    )

    rewrite_count = int(
        state.get(
            "retrieval_rewrite_count",
            0,
        )
    )

    # Strong context is sufficient.
    if context_quality == "strong":
        return "responder"

    # Prevent infinite retrieval loops.
    if rewrite_count >= 2:
        return "responder"

    return "query_rewriter"


# ============================================================
# ENTRY
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
# INITIAL PRIVATE RETRIEVAL
# ============================================================

workflow.add_edge(
    "retriever",
    "grader",
)


# ============================================================
# DOCUMENT GRADING
# ============================================================

workflow.add_conditional_edges(
    "grader",
    route_after_grader,
    {
        "responder": "responder",
        "query_rewriter": "query_rewriter",
    },
)


# ============================================================
# QUERY REWRITE
# ============================================================

workflow.add_edge(
    "query_rewriter",
    "private_retry",
)


# ============================================================
# PRIVATE KB RETRY
# ============================================================

workflow.add_edge(
    "private_retry",
    "grader",
)


# ============================================================
# RESPONSE
# ============================================================

workflow.add_edge(
    "responder",
    END,
)


# ============================================================
# CHECKPOINTING
# ============================================================

checkpointer = MemorySaver()


# ============================================================
# COMPILE
# ============================================================

rag_agent = workflow.compile(
    checkpointer=checkpointer,
)
