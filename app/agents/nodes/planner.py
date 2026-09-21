from __future__ import annotations

import logfire

from app.agents.state import AgentState
from app.gateway import get_langchain_llm


llm = get_langchain_llm(feature="planner")


def _build_history(state: AgentState) -> str:
    """
    Build a compact conversation history for planner context.

    The history is used only to understand references and intent.
    The latest user message remains the primary retrieval signal.
    """
    history_parts: list[str] = []

    for msg in state.get("messages", [])[:-1]:
        role = str(msg.get("role", "")).strip().lower()
        content = str(msg.get("content", "")).strip()

        if not content:
            continue

        if role == "user":
            label = "User"
        elif role == "assistant":
            label = "Assistant"
        else:
            label = role.title() or "Message"

        history_parts.append(f"{label}: {content}")

    return "\n".join(history_parts)


def _extract_decision(response: object) -> str:
    """
    Normalize the LLM planner response.
    """
    decision = str(getattr(response, "content", response)).strip()

    if not decision:
        return "CONVERSATIONAL"

    # Remove accidental surrounding quotes.
    decision = decision.strip("\"'")

    if decision.upper().rstrip(".") == "CONVERSATIONAL":
        return "CONVERSATIONAL"

    return decision


def planner_node(state: AgentState):
    """
    Enterprise retrieval planner.

    Responsibilities:
      1. Determine conversational vs technical retrieval.
      2. Preserve the exact user question as original_query.
      3. Produce a retrieval-optimized current_query for technical questions.

    Important design decision:
        original_query != current_query

    original_query:
        Exact user wording, preserved for answer generation/evaluation.

    current_query:
        Search-oriented representation optimized for semantic retrieval.
    """

    messages = state.get("messages", [])

    if not messages:
        return {
            "current_query": "CONVERSATIONAL",
            "original_query": "",
            "route": "simple",
            "status": "No user message provided.",
            "plan": [
                "Intent: Conversational/Memory",
                "Retrieval: Skipped",
            ],
            "retrieval_loops": 0,
            "revision_count": 0,
        }

    latest_message = messages[-1]
    user_message = str(latest_message.get("content", "")).strip()

    history = _build_history(state)

    prompt = f"""
You are the retrieval planner for an enterprise Agentic RAG system.

Your job has TWO responsibilities:

1. Decide whether the latest user message is conversational or
   requires technical knowledge retrieval.

2. If retrieval is required, transform the user's question into
   ONE HIGH-QUALITY SEMANTIC SEARCH QUERY for a vector database.

The search query is an information-retrieval representation of the
question. It must help retrieval find ALL important evidence needed
to answer the question, including complementary passages that may
describe different parts of the same mechanism.

============================================================
CONVERSATION HISTORY
============================================================

{history or "(No previous conversation.)"}

============================================================
LATEST USER MESSAGE
============================================================

"{user_message}"

============================================================
ROUTING RULES
============================================================

Return CONVERSATIONAL only when the latest message is:

- a greeting
- casual conversation
- thanks/acknowledgement
- explicitly asking about the conversation itself
- asking what the user previously said
- asking for personal information already present in the conversation

Technical questions MUST use retrieval.

Technical retrieval includes:

- definitions
- explanations
- how/why questions
- architecture
- components
- mechanisms
- workflows
- algorithms
- AI/ML
- LLMs
- RAG
- agents
- memory
- programming
- infrastructure
- networking
- operating systems
- hardware
- scientific/technical concepts
- documentation questions

A previous assistant answer does NOT make a new technical question
conversational.

============================================================
SEMANTIC SEARCH QUERY RULES
============================================================

For technical retrieval, produce ONE concise standalone search query.

The query is NOT the answer.

The query MUST:

1. Preserve the user's actual information need.
2. Include the central technical entity or system.
3. Include important mechanisms, relationships, components, or
   operations implied by the question.
4. Include useful technical terminology that source documents are
   likely to use.
5. Represent complementary evidence when the answer may be distributed
   across multiple passages.
6. Remain concise and retrieval-oriented.
7. Avoid inventing facts or technologies.

Do NOT merely convert the question into a bag of generic keywords.

Do NOT optimize for one expected passage.

Prefer semantic concepts and technical relationships over grammatical
phrasing.

============================================================
MECHANISM QUESTIONS
============================================================

For questions asking:

- how something works
- how something obtains, produces, processes, or retrieves something
- why a mechanism works
- what happens during a process
- how components interact

represent BOTH:

- the main entity being asked about
- the mechanism or operations responsible for the behavior

When appropriate, include related terms for:

- inputs
- encoders
- indexes
- retrieval/search operations
- scoring
- ranking
- selection
- intermediate representations
- outputs
- conditioning
- aggregation
- memory
- generation

Do not add mechanisms merely because they are common in the field.
Only add concepts reasonably implied by the question or useful as
technical synonyms.

Example:

User:
"How does the RAG retriever obtain documents for a query?"

Weak:
RAG retriever dense retrieval top-k documents input query

Better:
RAG retriever query encoder document index MIPS top-K document retrieval

The better query captures both the retriever and the mechanism used
to obtain the documents. This helps retrieve complementary passages
that describe different parts of the same process.

============================================================
DEFINITION QUESTIONS
============================================================

For questions asking what something is, include:

- the entity
- important technical synonyms
- the defining mechanism or purpose when directly implied

Example:

User:
"What is retrieval-augmented generation?"

Good:
retrieval-augmented generation RAG definition retrieval non-parametric memory generation

============================================================
COMPONENT / ARCHITECTURE QUESTIONS
============================================================

For questions asking for:

- components
- parts
- modules
- architecture
- structure

include the major conceptual entities that describe the structure.

Example:

User:
"What are the main components of an LLM-based agent according to the survey?"

Weak:
components of LLM-based agents

Better:
LLM-based agent architecture brain perception action memory reasoning

Do not invent components that are not reasonably implied by the
question or known terminology.

============================================================
SPECIFIC TECHNICAL CONCEPT QUESTIONS
============================================================

Example:

User:
"What is the role of the brain in an LLM-based agent?"

Good:
LLM-based agent brain memory information processing decision-making reasoning planning

============================================================
MEMORY QUESTIONS
============================================================

Example:

User:
"What memory capabilities does AutoGPT provide?"

Good:
AutoGPT memory long-term memory short-term memory memory capabilities

============================================================
QUERY EXPANSION RULES
============================================================

You MAY add:

- important technical synonyms
- domain-specific terminology implied by the question
- named components implied by the question
- mechanisms explicitly suggested by the wording
- closely related retrieval terminology
- concepts likely to appear in technical source documents

You MUST NOT add:

- facts unrelated to the user's question
- speculative information
- invented entities
- arbitrary technologies
- unsupported implementation details
- answers to the question
- citations
- long explanations
- punctuation-heavy prose

The query should normally contain approximately 5-14 meaningful
technical terms.

Prefer a small number of high-value concepts over many generic terms.

============================================================
COMPLEMENTARY EVIDENCE
============================================================

Some technical answers are distributed across multiple passages.

For example, one passage may describe:

- the system or component

while another passage describes:

- the mechanism
- algorithm
- scoring method
- indexing method
- retrieval operation
- downstream interaction

When the user's question asks HOW or WHY something works, create a
query that can retrieve both the entity-level explanation and the
mechanism-level explanation.

Do NOT deliberately optimize the query toward only one passage.

============================================================
CONVERSATION CONTEXT
============================================================

Use conversation history only when it helps resolve references.

Example:

Previous:
"What is loop engineering?"

Latest:
"Explain the verification step."

Good retrieval query:
loop engineering verification step

Example:

Previous:
"What is LLM memory?"

Latest:
"What are its main types?"

Good retrieval query:
LLM memory types short-term long-term memory

The query should resolve pronouns such as "it", "its", "they", or
"this" using the previous context and produce a standalone query.

============================================================
OUTPUT FORMAT
============================================================

Return ONLY one of:

CONVERSATIONAL

OR

a single standalone semantic retrieval query.

Never explain your decision.
Never use bullets.
Never prefix the query with "Query:".
Never include quotation marks.
"""

    with logfire.span("🧠 Planner Decision"):
        response = llm.invoke(prompt)
        decision = _extract_decision(response)

        logfire.info(
            "Intent identified",
            decision=decision,
            original_query=user_message,
        )

    if decision == "CONVERSATIONAL":
        return {
            "current_query": "CONVERSATIONAL",
            "original_query": user_message,
            "route": "simple",
            "status": "Handling conversationally using memory...",
            "plan": [
                "Intent: Conversational/Memory",
                "Retrieval: Skipped",
            ],
            "retrieval_loops": 0,
            "revision_count": 0,
        }

    return {
        # Retrieval-optimized query.
        "current_query": decision,
        # Exact user wording must remain untouched.
        "original_query": user_message,
        "route": "knowledge",
        "status": (f"Technical research needed. Searching for: {decision}"),
        "plan": [
            "Intent: Technical",
            f"Search Term: {decision}",
        ],
        "retrieval_loops": 0,
        "revision_count": 0,
    }
