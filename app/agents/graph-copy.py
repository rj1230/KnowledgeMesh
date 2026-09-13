from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.agents.state import AgentState
from app.agents.nodes.planner import planner_node
from app.agents.nodes.retriever import retrieve_node
from app.agents.nodes.grader import grade_kb_node, grade_web_node
from app.agents.nodes.web_search import web_search_node
from app.agents.nodes.rewriter import rewrite_node
from app.agents.nodes.responder import generate_node
from app.agents.nodes.grounding_critic import grounding_critic_node
from app.agents.nodes.citation_check import citation_check_node
from app.config import settings


workflow = StateGraph(AgentState)

workflow.add_node("planner", planner_node)
workflow.add_node("retriever", retrieve_node)
workflow.add_node("grade_kb", grade_kb_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("grade_web", grade_web_node)
workflow.add_node("rewriter", rewrite_node)
workflow.add_node("responder", generate_node)
workflow.add_node("grounding_critic", grounding_critic_node)
workflow.add_node("citation_check", citation_check_node)


def route_planner(state: AgentState):
    if state["route"] == "simple":
        return "responder"
    return "retriever"


workflow.set_entry_point("planner")
workflow.add_conditional_edges(
    "planner", route_planner, {"retriever": "retriever", "responder": "responder"}
)

workflow.add_edge("retriever", "grade_kb")


def route_kb_grade(state: AgentState):
    """Good KB context -> answer directly. Weak -> fall back to web search."""
    if state["kb_grade"] == "good":
        return "responder"
    return "web_search"


workflow.add_conditional_edges(
    "grade_kb", route_kb_grade, {"responder": "responder", "web_search": "web_search"}
)

workflow.add_edge("web_search", "grade_web")


def route_web_grade(state: AgentState):
    """
    Good web context -> answer from it. Still weak -> rewrite the query and
    retry KB retrieval, capped by MAX_RETRIEVAL_LOOPS (incremented in rewrite_node).
    """
    if (
        state["web_grade"] == "good"
        or state.get("retrieval_loops", 0) >= settings.MAX_RETRIEVAL_LOOPS
    ):
        return "responder"
    return "rewriter"


workflow.add_conditional_edges(
    "grade_web", route_web_grade, {"responder": "responder", "rewriter": "rewriter"}
)

workflow.add_edge("rewriter", "retriever")

workflow.add_edge("responder", "grounding_critic")
workflow.add_edge("grounding_critic", "citation_check")


def route_verification(state: AgentState):
    """
    Grounded + citations check out -> done. Otherwise loop back to the
    responder to regenerate, capped by MAX_REVISIONS (incremented in citation_check_node).
    """
    if (state["is_grounded"] and state["citation_valid"]) or state.get(
        "revision_count", 0
    ) >= settings.MAX_REVISIONS:
        return END
    return "responder"


workflow.add_conditional_edges(
    "citation_check", route_verification, {"responder": "responder", END: END}
)

checkpointer = MemorySaver()
rag_agent = workflow.compile(checkpointer=checkpointer)
