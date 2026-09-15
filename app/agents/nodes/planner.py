from app.agents.state import AgentState
from app.gateway import get_langchain_llm
import logfire

llm = get_langchain_llm(feature="planner")


def planner_node(state: AgentState):
    history = ""

    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = str(msg.get("content", "")).strip()

        if content:
            history += f"{role}: {content}\n"

    user_message = state["messages"][-1]["content"] if state["messages"] else ""

    prompt = f"""
You are an Enterprise RAG Planner.

Your job is to decide whether the user's LATEST message should use:

A) CONVERSATIONAL
B) TECHNICAL RETRIEVAL

CONVERSATION HISTORY:
{history or "(No previous conversation.)"}

LATEST USER MESSAGE:
"{user_message}"

IMPORTANT DECISION RULES:

1. Use CONVERSATIONAL ONLY for:
   - Greetings such as "hi", "hello", "thanks"
   - Casual conversation
   - Questions explicitly about the conversation itself
   - Questions asking what the user previously said
   - Questions whose answer depends only on personal information
     already present in the conversation

2. Use TECHNICAL RETRIEVAL for:
   - Technical questions
   - Definitions
   - How/why questions about technology
   - Documentation questions
   - Engineering questions
   - Architecture questions
   - Programming questions
   - AI/ML questions
   - Networking questions
   - Operating-system questions
   - Hardware questions
   - Scientific/technical factual questions
   - Any question where enterprise documents could provide useful evidence

CRITICAL RULE:

The fact that a previous assistant response contains an answer
does NOT mean a new technical question should be treated as
CONVERSATIONAL.

For example:

Previous:
"What is loop engineering?"

New:
"Explain the verification step."

The new message is still TECHNICAL RETRIEVAL.

Another example:

Previous:
"What is loop engineering?"

New:
"What is Intel EPT?"

This MUST be TECHNICAL RETRIEVAL.

Another example:

Previous:
"My name is Raj."

New:
"What is my name?"

This can be CONVERSATIONAL.

Another example:

Previous:
"What is loop engineering?"

New:
"What did I ask you previously?"

This is CONVERSATIONAL.

For technical retrieval, return a concise standalone search query.

Examples:

"What is loop engineering?"
→ What is loop engineering

"Explain the verification step."
→ loop engineering verification step

"What is Intel EPT?"
→ Intel EPT explanation

"How does Kubernetes networking work?"
→ Kubernetes networking architecture

"What is my name?"
→ CONVERSATIONAL

"Thanks!"
→ CONVERSATIONAL

"Hello"
→ CONVERSATIONAL

OUTPUT RULE:

Return ONLY:

CONVERSATIONAL

OR

a concise technical search query.

Never return explanations.
"""

    with logfire.span("🧠 Planner Decision"):
        decision = llm.invoke(prompt).content.strip()

        if decision.upper().rstrip(".") == "CONVERSATIONAL":
            decision = "CONVERSATIONAL"

        logfire.info(
            "Intent identified",
            decision=decision,
        )

    if decision == "CONVERSATIONAL":
        return {
            "current_query": "CONVERSATIONAL",
            "status": "Handling conversationally using memory...",
            "plan": [
                "Intent: Conversational/Memory",
                "Retrieval: Skipped",
            ],
        }

    return {
        "current_query": decision,
        "status": f"Technical research needed. Searching for: {decision}",
        "plan": [
            "Intent: Technical",
            f"Search Term: {decision}",
        ],
    }
