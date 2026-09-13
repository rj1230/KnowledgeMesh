import logfire
from app.agents.state import AgentState
from app.gateway import get_langchain_llm

llm = get_langchain_llm(feature="rewriter")

REWRITE_PROMPT = """The following search query did not return sufficient results from
either our internal knowledge base or the web. Rewrite it to be more specific and
more likely to retrieve relevant technical documentation.

ORIGINAL QUERY:
{query}

Output ONLY the rewritten query, nothing else.
"""


def rewrite_node(state: AgentState):
    """
    Rewrites the query when both KB and web evidence were graded weak,
    then loops back to the retriever. Bounded by MAX_RETRIEVAL_LOOPS in graph.py.
    """
    with logfire.span("🔄 Query Rewrite"):
        new_query = llm.invoke(
            REWRITE_PROMPT.format(query=state["current_query"])
        ).content.strip()
        logfire.info(f"Rewritten query: {new_query}")

    return {
        "current_query": new_query,
        "status": f"Refining search: {new_query}",
        "plan": [f"Rewritten Query: {new_query}"],
        "retrieval_loops": state.get("retrieval_loops", 0) + 1,
    }
