import logfire
from app.agents.state import AgentState
from app.gateway import get_langchain_llm

llm = get_langchain_llm(feature="grader")

GRADE_PROMPT = """You are grading whether retrieved context is sufficient to answer a question.

QUESTION:
{query}

RETRIEVED CONTEXT:
{context}

Does the context contain enough specific information to fully answer the question?
Reply with exactly one word: GOOD or WEAK.
"""


def _grade(query: str, context: str) -> str:
    if not context.strip():
        return "weak"
    verdict = (
        llm.invoke(GRADE_PROMPT.format(query=query, context=context))
        .content.strip()
        .upper()
    )
    return "good" if verdict.startswith("GOOD") else "weak"


def grade_kb_node(state: AgentState):
    """
    Grades the KB (Qdrant) retrieval. Good -> answer directly.
    Weak -> falls back to web search (see graph.py routing).
    """
    context = "\n\n".join(state.get("documents", []))
    with logfire.span("📊 Grading KB Evidence"):
        grade = _grade(state["current_query"], context)
        logfire.info(f"KB grade: {grade}")

    return {
        "kb_grade": grade,
        "context_source": "kb" if grade == "good" else None,
        "plan": [f"KB Grade: {grade}"],
    }


def grade_web_node(state: AgentState):
    """
    Grades the Tavily web search results. Good -> answer from web.
    Weak (and under the retry cap) -> query gets rewritten and KB is retried.
    """
    context = "\n\n".join(state.get("web_results", []))
    with logfire.span("📊 Grading Web Evidence"):
        grade = _grade(state["current_query"], context)
        logfire.info(f"Web grade: {grade}")

    return {
        "web_grade": grade,
        "context_source": "web" if grade == "good" else None,
        "plan": [f"Web Grade: {grade}"],
    }
