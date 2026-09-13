import logfire
from app.agents.state import AgentState
from app.tools.tavily_tool import tavily_search


def web_search_node(state: AgentState):
    """
    Falls back to live web results via Tavily when the KB grade is weak.
    """
    query = state["current_query"]

    with logfire.span("🌐 Web Search Fallback"):
        logfire.info(f"Searching Tavily for: {query}")
        results = tavily_search(query, max_results=5)
        logfire.info(f"Retrieved {len(results)} web results")

    formatted_results = [
        f"SOURCE: {r['url']}\nCONTENT: {r['content']}" for r in results
    ]

    return {
        "web_results": formatted_results,
        "status": "Searching the web for fresh context...",
        "plan": ["Web Search: Tavily"],
    }
