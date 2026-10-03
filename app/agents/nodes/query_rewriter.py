"""
KnowledgeMesh — Self-RAG Query Rewriter

Rewrites a failed/weak retrieval query into a concise standalone
search query suitable for the private enterprise knowledge base.
"""

from __future__ import annotations

import logging
import re

from app.gateway import create_chat_completion
from app.evaluation.trace import append_trace_event

logger = logging.getLogger(__name__)


def _clean_query(text: str) -> str:
    """Normalize the LLM-generated search query."""

    text = (text or "").strip()

    # Remove accidental markdown/code fences.
    text = re.sub(r"```(?:text|json)?", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")

    # Remove common prefixes.
    text = re.sub(
        r"^(search query|rewritten query|query)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    # Remove surrounding quotes.
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()

    return text


def rewrite_query_node(state):
    """
    Rewrite the current retrieval query after weak/empty private context.

    This node does NOT perform retrieval itself.
    It only creates the improved query.

    The following node performs the Qdrant retry.
    """

    original_query = (
        state.get("original_query") or state.get("current_query") or ""
    ).strip()

    current_query = (state.get("current_query") or original_query).strip()

    revision_count = int(state.get("retrieval_rewrite_count", 0))

    prompt = f"""
You are an expert enterprise RAG query-rewriting agent.

The private knowledge-base retrieval for the current query produced
weak or insufficient context.

ORIGINAL USER QUERY:
{original_query}

CURRENT SEARCH QUERY:
{current_query}

Your task is to create ONE improved standalone search query for
retrieving relevant technical documentation from a private enterprise
knowledge base.

Rules:

    1. Preserve the user's actual intent.
    2. Do not answer the question.
    3. Do not add unsupported facts, entities, or specific implementation details.
    4. Preserve important technical entities, acronyms, model names, algorithm names,
       component names, and domain-specific terminology from the original query.
    5. Expand the query with technical terminology that is directly implied by the
       subject and would plausibly appear in technical documentation.
       For technical systems and mechanisms, include closely related concepts such
       as the relevant component type, representation, operation, retrieval method,
       algorithm family, or input/output terminology when strongly implied.
    6. Preserve the original technical concept instead of replacing it with generic
       words such as process, system, information, or method.
    7. Remove conversational wording.
    8. Make the query specific enough for semantic retrieval.
    9. Keep the query concise and focused on the original intent.
    10. Return ONLY the rewritten search query.
    11. Do not include explanations.
    12. Do not include quotation marks.

Example:

User:
"What is EPT?"

Good:
"Intel Extended Page Tables EPT virtualization memory translation"

User:
"How does this work?"

Good:
"mechanism and operation of the previously discussed system"

User:
"Explain loop engineering"

Good:
"loop engineering definition architecture verification stopping rules memory"

Return only the search query.
"""

    try:
        response = create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise enterprise search-query "
                        "rewriter. Return only the rewritten query."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.0,
        )

        rewritten_query = _clean_query(response.choices[0].message.content or "")

    except Exception as exc:
        logger.exception("Query rewrite failed")

        # Safe fallback: retain the current query.
        rewritten_query = current_query

    if not rewritten_query:
        rewritten_query = current_query

    plan = state.get("plan", [])

    plan = plan + [
        f"Query Rewrite: {rewritten_query}",
    ]

    trace_update = append_trace_event(
        state,
        step="query_rewriter",
        status=(
            "rewritten"
            if rewritten_query != current_query
            else "retry_same_query"
        ),
        rewrite_count=revision_count + 1,
        query_changed=rewritten_query != current_query,
    )

    return {
        **trace_update,
        "original_query": original_query,
        "rewritten_query": rewritten_query,
        "current_query": rewritten_query,
        "retrieval_rewrite_count": revision_count + 1,
        "status": f"Retrying private KB with rewritten query: {rewritten_query}",
        "plan": plan,
    }

