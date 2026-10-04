import logfire
from app.agents.state import AgentState
from app.gateway import get_langchain_llm

llm = get_langchain_llm(feature="planner")


def planner_node(state: AgentState):
    """
    The Planner determines if a search is needed based on the ENTIRE conversation.
    """
    history = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"

    user_message = state["messages"][-1]["content"] if state["messages"] else ""

    prompt = f"""
    You are an intelligent Assistant Planner.
    Analyze the conversation history and the latest user message.

    CONVERSATION HISTORY:
    {history}

    LATEST MESSAGE:
    "{user_message}"

    Task:
    1. If the latest message is a greeting (hi, hello) or a question that can be answered using ONLY the conversation history above (e.g., "what is my name"), respond with 'CONVERSATIONAL'.
    2. For a technical knowledge request, output a concise refined search query.

    Technical requests may include AI, machine learning, LLMs,
    RAG, retrieval, agents, evaluation, software engineering,
    programming, APIs, infrastructure, networking, cloud,
    architecture, system design, or other technical subjects
    covered by the KnowledgeMesh knowledge base.

    The pre-RAG guard has already handled clearly non-technical
    and ambiguous requests. Do not add scope restrictions based
    on a short fixed topic list.

    Output ONLY 'CONVERSATIONAL' or the refined search query.
    """

    with logfire.span("🧠 Planner Decision"):
        decision = llm.invoke(prompt).content.strip()
        logfire.info(f"Intent identified: {decision}")

    if decision == "CONVERSATIONAL":
        return {
            "current_query": user_message,
            "original_query": user_message,
            "route": "simple",
            "status": "Handling conversationally (using memory)...",
            "plan": ["Intent: Conversational/Memory", "Retrieval: Skipped"],
            "retrieval_loops": 0,
            "revision_count": 0,
        }

    return {
        "current_query": decision,
        "original_query": user_message,
        "route": "knowledge",
        "status": f"Technical research needed. Searching for: {decision}",
        "plan": ["Intent: Technical", f"Search Term: {decision}"],
        "retrieval_loops": 0,
        "revision_count": 0,
    }
