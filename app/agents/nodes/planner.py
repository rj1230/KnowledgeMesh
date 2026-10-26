from __future__ import annotations

import logging
import re
import time

import logfire

from app.agents.state import AgentState
from app.evaluation.trace import append_trace_event
from app.gateway import get_langchain_llm

logger = logging.getLogger(__name__)

llm = get_langchain_llm(feature="planner")


# ============================================================
# Configuration
# ============================================================

# Safety net on top of the prompt: at most this many words in the
# planner's query may be NEW (not present in the user's question or the
# conversation). Stops runaway expansion from steering retrieval.
MAX_NOVEL_TERMS = 5

# Hard cap on the final query length (terms).
MAX_QUERY_TERMS = 18

# The planner is the first node of every request, so it is the natural
# place to clear per-request state. This matters when a thread_id is
# reused across turns (MemorySaver keeps state): without it, counters
# such as answer_revision_count or relevance_retrieval_count carry over
# and silently disable revisions / re-retrieval for later questions.
# Harmless when every request starts from a fresh state.
RESET_PER_REQUEST_STATE = True


# ============================================================
# Per-request reset
# ============================================================


def _per_request_reset() -> dict:
    """Fields that must start clean for every new user turn."""

    return {
        # Revision / re-retrieval budgets.
        "answer_revision_count": 0,
        "revision_count": 0,
        "revision_requested": False,
        "revision_prompt": "",
        "relevance_retrieval_count": 0,
        "expand_retrieval": False,
        "final_answer": "",
        "candidate_answer": "",
        "document_scoped": False,
        # Retrieval / web fallback control.
        "retrieval_rewrite_count": 0,
        "retrieval_failed": False,
        "web_search_attempted": False,
        "web_search_used": False,
        # Generation / salvage / abstention.
        "generation_failed": False,
        "answer_pruned": False,
        "pruned_answer": "",
        "pruned_claims": [],
        "prunable_claims": [],
        "answer_before_pruning": "",
        "answer_abstained": False,
        "abstention_reason": "",
        # HHEM score cache is per request.
        "entailment_cache": [],
    }


# ============================================================
# History / response helpers
# ============================================================


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

    Takes the first non-empty line, strips quotes and a stray "Query:"
    prefix, and maps any casing of CONVERSATIONAL to the sentinel.
    """
    raw = str(getattr(response, "content", response)).strip()

    if not raw:
        return "CONVERSATIONAL"

    first_line = next((line.strip() for line in raw.splitlines() if line.strip()), "")

    decision = first_line.strip("\"'` ")
    decision = re.sub(r"^(?:search\s+)?query\s*:\s*", "", decision, flags=re.IGNORECASE)
    decision = decision.strip("\"'` ")

    if not decision:
        return "CONVERSATIONAL"

    if decision.upper().rstrip(".") == "CONVERSATIONAL":
        return "CONVERSATIONAL"

    return decision


# ============================================================
# Query guard
# ============================================================

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+\-]*")

_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "of",
    "to",
    "in",
    "on",
    "for",
    "by",
    "with",
    "from",
    "as",
    "at",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "do",
    "does",
    "did",
    "how",
    "what",
    "which",
    "who",
    "why",
    "when",
    "where",
    "it",
    "its",
    "they",
    "their",
    "this",
    "that",
    "these",
    "those",
    "can",
    "could",
    "should",
    "would",
    "will",
    "me",
    "my",
    "you",
    "your",
    "about",
    "into",
    "than",
    "then",
    "also",
    "vs",
    "versus",
    "according",
}


def _norm(word: str) -> str:
    """Lowercase and lightly de-pluralise so 'component' matches 'components'."""
    word = word.lower().strip("-_+")

    if len(word) > 3 and word.endswith("s"):
        word = word[:-1]

    return word


def _content_words(text: str) -> list[str]:
    return [
        word
        for word in _WORD.findall(text or "")
        if _norm(word) and _norm(word) not in _STOPWORDS
    ]


def _guard_query(
    query: str,
    user_message: str,
    history: str,
) -> tuple[str, list[str], list[str]]:
    """
    Deterministic safety net around the planner's query.

    1. Limit NEW words (not in the question or conversation) to
       MAX_NOVEL_TERMS, so expansion cannot take over the query.
    2. Re-add any distinctive word from the user's question that the
       planner dropped ("five", "according to the survey", entity names),
       so the user's own wording is never lost.
    3. Cap total length.

    Returns (query, dropped_novel_terms, restored_question_terms).
    """

    vocabulary = {_norm(w) for w in _content_words(f"{user_message} {history}")}

    kept: list[str] = []
    dropped: list[str] = []
    novel_seen = 0

    for term in query.split():
        bare = term.strip(",.;:()[]{}\"'")
        normalised = _norm(bare)

        if not normalised:
            continue

        is_known = normalised in _STOPWORDS or normalised in vocabulary

        if not is_known:
            if novel_seen >= MAX_NOVEL_TERMS:
                dropped.append(bare)
                continue

            novel_seen += 1

        kept.append(bare)

    present = {_norm(t) for t in kept}
    restored: list[str] = []

    for word in _content_words(user_message):
        if _norm(word) not in present:
            kept.append(word)
            present.add(_norm(word))
            restored.append(word)

    return " ".join(kept[:MAX_QUERY_TERMS]), dropped, restored


# ============================================================
# Prompt
# ============================================================


def _build_prompt(user_message: str, history: str) -> str:
    return f"""
You are the retrieval planner for an enterprise Agentic RAG system.
You do NOT know what is inside the knowledge base.

Decide whether the latest user message is CONVERSATIONAL or needs
retrieval, and if it needs retrieval, write ONE search query for a
vector database.

CONVERSATION HISTORY
{history or "(none)"}

LATEST USER MESSAGE
"{user_message}"

ROUTING
Return CONVERSATIONAL only for: greetings, thanks or acknowledgements,
small talk, questions about the conversation itself, or personal
information already stated in the conversation.
Every technical or factual question needs retrieval (definitions,
how/why, architecture, components, mechanisms, workflows, AI/ML, LLMs,
RAG, agents, memory, programming, infrastructure, documentation). An
earlier answer does not make a new technical question conversational.

QUERY RULES (retrieval only)
- One standalone query of about 4-14 terms, no sentence, no quotes.
- Keep every distinctive word of the user's question: entities, numbers,
  and qualifiers such as "five", "main", "according to the survey".
- Resolve references (it, its, they, this) using the history.
- You may ADD a term only if it is
    (a) a standard synonym, abbreviation or expansion of a term already
        in the question or history (RAG <-> retrieval-augmented
        generation), or
    (b) the obvious mechanism noun implied by "how does X get/produce Y"
        (retrieval, search, scoring, ranking).
- NEVER add the parts, steps, components, types, or typical contents of
  the thing being asked about. The question is asking for them, you do
  not know what the documents call them, and guessing pulls retrieval
  toward unrelated passages.
- Never add facts, invented entities, technologies, answers or citations.

EXAMPLES
User: "How does a B-tree index speed up database lookups?"
Query: B-tree index database lookup speed search keys

User: "What are the main components of a transformer according to the paper?"
Query: transformer main components according to the paper

Previous: "What is LLM memory?"  Latest: "What are its main types?"
Query: LLM memory main types

OUTPUT
Return ONLY the word CONVERSATIONAL, or the query on a single line.
Never explain. Never use bullets. Never prefix with "Query:".
"""


# ============================================================
# Node
# ============================================================


def planner_node(state: AgentState):
    """
    Enterprise retrieval planner.

    Responsibilities:
      1. Determine conversational vs technical retrieval.
      2. Preserve the exact user question as original_query.
      3. Produce a retrieval-optimized current_query for technical questions.
      4. Start every request from clean per-request state.

    original_query != current_query

    original_query:
        Exact user wording, preserved for answer generation/evaluation.

    current_query:
        Search-oriented representation optimized for semantic retrieval.
        Guarded so the planner cannot drop the user's own words or flood
        the query with terms the user never mentioned.
    """

    reset = _per_request_reset() if RESET_PER_REQUEST_STATE else {}

    # A reused thread would otherwise accumulate trace events from earlier
    # turns and mix them into this request's trajectory evaluation.
    trace_state = (
        {**state, "evaluation_trace": []} if RESET_PER_REQUEST_STATE else state
    )

    messages = state.get("messages", [])

    if not messages:
        trace_update = append_trace_event(
            trace_state,
            step="planner",
            status="empty_input",
            route="simple",
            retrieval_required=False,
        )

        return {
            **reset,
            **trace_update,
            "current_query": "CONVERSATIONAL",
            "original_query": "",
            "route": "simple",
            "retrieval_required": False,
            "status": "No user message provided.",
            "plan": [
                "Intent: Conversational/Memory",
                "Retrieval: Skipped",
            ],
            "revision_count": 0,
        }

    latest_message = messages[-1]
    user_message = str(latest_message.get("content", "")).strip()

    history = _build_history(state)

    prompt = _build_prompt(user_message, history)

    llm_failed = False

    with logfire.span("🧠 Planner Decision"):
        planner_started = time.perf_counter()

        try:
            response = llm.invoke(prompt)
            decision = _extract_decision(response)
        except Exception as exc:
            # Planner outage must not kill the request. Fall back to
            # retrieving with the user's own question.
            llm_failed = True
            decision = user_message
            logger.warning(
                "Planner LLM failed (%s); falling back to the user question.",
                type(exc).__name__,
            )

        planner_ms = round((time.perf_counter() - planner_started) * 1000, 2)
        logfire.info("Planner timing", planner_ms=planner_ms)

        logfire.info(
            "Intent identified",
            decision=decision,
            original_query=user_message,
            llm_failed=llm_failed,
        )

    if decision == "CONVERSATIONAL":
        trace_update = append_trace_event(
            trace_state,
            step="planner",
            status="conversational",
            route="simple",
            retrieval_required=False,
        )

        return {
            **reset,
            **trace_update,
            "current_query": "CONVERSATIONAL",
            "original_query": user_message,
            "route": "simple",
            "retrieval_required": False,
            "status": "Handling conversationally using memory...",
            "plan": [
                "Intent: Conversational/Memory",
                "Retrieval: Skipped",
            ],
            "revision_count": 0,
        }

    raw_query = decision

    query, dropped_terms, restored_terms = _guard_query(
        raw_query,
        user_message,
        history,
    )

    if not query:
        query = user_message

    if dropped_terms or restored_terms or query != raw_query:
        logfire.info(
            "Planner query guard",
            raw_query=raw_query,
            final_query=query,
            dropped_novel_terms=dropped_terms,
            restored_question_terms=restored_terms,
        )

    trace_update = append_trace_event(
        trace_state,
        step="planner",
        status="technical",
        route="knowledge",
        retrieval_required=True,
        query=query,
        raw_query=raw_query,
        dropped_novel_terms=dropped_terms,
        restored_question_terms=restored_terms,
        llm_failed=llm_failed,
    )

    return {
        **reset,
        **trace_update,
        # Retrieval-optimized, guarded query.
        "current_query": query,
        # Exact user wording must remain untouched.
        "original_query": user_message,
        "route": "knowledge",
        "retrieval_required": True,
        "status": f"Technical research needed. Searching for: {query}",
        "plan": [
            "Intent: Technical",
            f"Search Term: {query}",
        ],
        "revision_count": 0,
    }
