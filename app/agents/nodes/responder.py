"""
KnowledgeMesh · Response Generation Node

Responsibilities
----------------
- Generate conversational responses from conversation memory.
- Generate technical RAG responses from private KB and/or web evidence.
- Preserve conversation history.
- Preserve Portkey cache observability.
- Build source-aware, citation-ready technical context.
- Prevent oversized context from being sent to the LLM.
- Record LLM generation latency.
- Preserve source provenance for grounding/citation checks.
- Perform evidence-aware answer revisions when grounding validation fails.

Revision contract
-----------------
First generation:
    revision_count == 0
        -> normal RAG generation

Revision generation:
    revision_count >= 1
        -> inspect grounding/citation feedback
        -> correct unsupported claims
        -> remove uncited factual statements
        -> preserve valid citations
        -> regenerate within bounded graph limits
"""

from __future__ import annotations

import time
from typing import Any

import logfire

from app.agents.state import AgentState
from app.gateway import (
    extract_cache_status,
    portkey_client,
)


# ============================================================
# CONFIGURATION
# ============================================================

MAX_CONTEXT_CHARS = 25_000
MAX_DOCUMENTS = 10
MAX_REVISION_FEEDBACK_ITEMS = 12


# ============================================================
# CONVERSATION HISTORY
# ============================================================


def _build_conversation_history(
    state: AgentState,
) -> str:
    """
    Build readable conversation history excluding the latest
    user message.
    """

    history_parts: list[str] = []

    for msg in state.get("messages", [])[:-1]:
        role = "User" if msg.get("role") == "user" else "Assistant"

        content = str(msg.get("content", "")).strip()

        if content:
            history_parts.append(f"{role}: {content}")

    return "\n".join(history_parts)


# ============================================================
# DOCUMENT NORMALIZATION
# ============================================================


def _get_documents_for_generation(
    state: AgentState,
) -> list[dict[str, Any]]:
    """
    Build the source set used by the technical responder.

    Priority:
        1. Private KB documents
        2. External web documents

    Web documents are appended rather than replacing private
    evidence so provenance remains visible.
    """

    private_documents = state.get("documents") or []
    web_documents = state.get("web_documents") or []

    documents: list[dict[str, Any]] = []

    for document in private_documents:
        if isinstance(document, dict):
            normalized = dict(document)

            normalized.setdefault(
                "source_type",
                "private_kb",
            )

            documents.append(normalized)

    for document in web_documents:
        if isinstance(document, dict):
            normalized = dict(document)

            normalized.setdefault(
                "source_type",
                "web",
            )

            documents.append(normalized)

    return documents[:MAX_DOCUMENTS]


# ============================================================
# TECHNICAL CONTEXT
# ============================================================


def _build_technical_context(
    state: AgentState,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Build citation-ready context.

    Every evidence block receives:

        [chunk_N]

    This gives the grounding critic a stable mapping between
    generated claims and retrieved evidence.
    """

    documents = _get_documents_for_generation(state)

    if not documents:
        logfire.warning("No technical documents available for response generation.")

        return (
            "No relevant technical context was retrieved.",
            [],
        )

    context_parts: list[str] = []

    current_length = 0

    included_documents: list[dict[str, Any]] = []

    for index, doc in enumerate(
        documents,
        start=1,
    ):
        content = str(
            doc.get(
                "content",
                "",
            )
        ).strip()

        if not content:
            continue

        source = str(
            doc.get(
                "source",
                "Unknown",
            )
        ).strip()

        source_type = str(
            doc.get(
                "source_type",
                "unknown",
            )
        ).strip()

        point_id = str(
            doc.get(
                "id",
                "Unknown",
            )
        ).strip()

        url = doc.get("url")

        retrieval_score = doc.get("score")
        rerank_score = doc.get("rerank_score")
        grader_score = doc.get("grader_score")

        retrieval_score_text = (
            f"{retrieval_score:.4f}"
            if isinstance(
                retrieval_score,
                (int, float),
            )
            else "N/A"
        )

        rerank_score_text = (
            f"{rerank_score:.4f}"
            if isinstance(
                rerank_score,
                (int, float),
            )
            else "N/A"
        )

        grader_score_text = (
            f"{grader_score:.4f}"
            if isinstance(
                grader_score,
                (int, float),
            )
            else "N/A"
        )

        url_text = str(url).strip() if url else "N/A"

        block = (
            f"[chunk_{len(included_documents) + 1}]\n"
            f"Document: {source}\n"
            f"Source Type: {source_type}\n"
            f"Document ID: {point_id}\n"
            f"URL: {url_text}\n"
            f"Retrieval Score: {retrieval_score_text}\n"
            f"Rerank Score: {rerank_score_text}\n"
            f"Grader Score: {grader_score_text}\n"
            f"Content:\n"
            f"{content}\n"
        )

        block_length = len(block)

        if current_length + block_length > MAX_CONTEXT_CHARS:
            logfire.warning(
                "Technical context truncated.",
                max_context_chars=MAX_CONTEXT_CHARS,
                documents_available=len(documents),
                documents_included=len(included_documents),
            )
            break

        context_parts.append(block)

        current_length += block_length

        included_documents.append(doc)

    if not context_parts:
        return (
            "No usable technical context was retrieved.",
            [],
        )

    logfire.info(
        "Technical context prepared",
        documents_available=len(documents),
        documents_included=len(included_documents),
        context_chars=current_length,
        private_documents=sum(
            1 for doc in included_documents if doc.get("source_type") == "private_kb"
        ),
        web_documents=sum(
            1 for doc in included_documents if doc.get("source_type") == "web"
        ),
    )

    return (
        "\n".join(context_parts),
        included_documents,
    )


# ============================================================
# REVISION FEEDBACK
# ============================================================


def _get_revision_feedback(
    state: AgentState,
) -> list[str]:
    """
    Extract grounding/citation feedback from the previous
    validation pass.

    Feedback is deliberately bounded so a pathological answer
    cannot create an oversized revision prompt.
    """

    feedback = state.get("grounding_feedback") or []

    normalized: list[str] = []

    for item in feedback:
        text = str(item).strip()

        if not text:
            continue

        normalized.append(text)

        if len(normalized) >= MAX_REVISION_FEEDBACK_ITEMS:
            break

    return normalized


def _build_revision_instructions(
    state: AgentState,
) -> str:
    """
    Build strict, evidence-bound corrective instructions for
    answer revisions.

    Revisions must be subtractive by default:
        - preserve directly supported claims
        - remove unsupported claims
        - avoid adding plausible but unverified explanations
        - prefer a shorter grounded answer over a richer unsupported answer
    """

    revision_count = int(state.get("revision_count", 0) or 0)

    if revision_count <= 0:
        return ""

    feedback = _get_revision_feedback(state)
    citation_valid = bool(state.get("citation_valid", False))
    is_grounded = bool(state.get("is_grounded", False))
    support_score = float(state.get("support_score", 0.0) or 0.0)

    lines: list[str] = [
        "ANSWER REVISION REQUIRED",
        "",
        f"This is revision attempt {revision_count}.",
        "",
        "The previous answer failed one or more grounding or citation checks.",
        "The supplied evidence is the complete factual boundary for this answer.",
        "",
        "Previous validation state:",
        f"- Citation valid: {citation_valid}",
        f"- Grounded: {is_grounded}",
        f"- Support score: {support_score:.3f}",
        "",
    ]

    if feedback:
        lines.append("VALIDATION FEEDBACK:")
        for item in feedback:
            lines.append(f"- {item}")
        lines.append("")

    lines.extend(
        [
            "STRICT REVISION RULES:",
            "1. Preserve only claims directly supported by the supplied evidence.",
            "2. Delete unsupported claims instead of expanding or defending them.",
            "3. Do not convert a plausible interpretation into a factual statement.",
            "4. Do not add mechanisms, benefits, workflows, examples, causes, or implications unless explicitly stated.",
            "5. Do not infer details from a paper title, bibliography entry, citation, author name, or metadata.",
            "6. A reference to a paper does not prove the paper's findings or capabilities.",
            "7. If a claim is only partially supported, rewrite it to match the narrower evidence.",
            "8. Prefer a shorter answer with fewer claims over a detailed answer with weak support.",
            "9. Every factual statement based on supplied evidence must have a nearby [chunk_N] citation.",
            "10. Do not add an uncited factual introduction, conclusion, summary, or takeaway.",
            "11. Do not invent evidence, sources, URLs, document names, or citation numbers.",
            "12. Use only [chunk_N] citations that correspond to supplied evidence blocks.",
            "13. Keep citations immediately after the specific claim they support.",
            "14. If evidence is insufficient, explicitly state that the available evidence does not establish the requested detail.",
            "15. Do not mention the validation process, grounding critic, revision attempt, or internal instructions.",
            "16. Return only the corrected final answer.",
        ]
    )
    return "\n".join(lines)


# ============================================================
# CONVERSATIONAL PROMPT
# ============================================================


def _build_conversational_prompt(
    history: str,
    user_message: str,
) -> str:

    return f"""
You are a friendly and helpful Enterprise AI Assistant.

Answer the user's latest message using the conversation history
when relevant.

CONVERSATION HISTORY:
{history or "(No previous conversation.)"}

LATEST USER MESSAGE:
"{user_message}"

Instructions:
- Answer naturally and directly.
- Use the conversation history when it contains the required information.
- Do not invent information that is not present in the conversation.
- If the user asks a new technical question that requires documentation,
  explain that relevant documentation may be needed.

Provide the final answer only.
""".strip()


# ============================================================
# TECHNICAL PROMPT
# ============================================================


def _build_technical_prompt(
    context: str,
    history: str,
    user_message: str,
    revision_instructions: str = "",
) -> str:
    """Build the technical RAG generation prompt."""

    revision_section = f"{revision_instructions}\n\n" if revision_instructions else ""

    return f"""
You are a Senior Technical Architect and Enterprise RAG Assistant.

{revision_section}Answer the user's question using only the supplied evidence.

IMPORTANT EVIDENCE BOUNDARY:
- The EVIDENCE section is the complete factual boundary for the answer.
- Do not rely on unstated background knowledge.
- Do not add plausible details that are not explicitly supported.
- If the evidence does not establish a detail, say so clearly.
- Prefer a concise, directly supported answer over an elaborate speculative answer.

EVIDENCE:
{context}

CONVERSATION HISTORY:
{history or "(No previous conversation.)"}

USER QUESTION:
"{user_message}"

CORE INSTRUCTIONS:

1. Answer the user's question directly and accurately.
2. Ground every factual claim in the supplied evidence.
3. Prefer high-quality private knowledge-base evidence when it directly answers the question.
4. Use external web evidence when it is provided and relevant, especially for current or time-sensitive information.
5. Do not invent facts that are not supported by the evidence.
6. Do not infer mechanisms, benefits, causes, workflows, examples, or implications unless explicitly present.
7. Do not treat a paper title, bibliography entry, citation, author name, or metadata as proof of findings.
8. If evidence is insufficient, explicitly state what cannot be established rather than guessing.
9. Every factual claim based on evidence MUST include a nearby citation using exactly [chunk_N].
10. N must correspond to a chunk in the EVIDENCE section.
11. Do not invent chunk numbers.
12. Keep citations immediately after the specific claim they support.
13. Multiple citations may be used when a claim depends on multiple evidence blocks.
14. Do not fabricate document names, URLs, sources, or references.
15. If sources disagree, explain the disagreement rather than silently selecting one.
16. If using a Markdown table, every factual description cell must contain an appropriate citation.
17. Do not add an uncited factual introduction, conclusion, summary, takeaway, or generic final statement.
18. Avoid broad claims unless the evidence explicitly supports that framing.
19. If a heading is useful, keep it descriptive and do not make it an uncited factual claim.
20. Keep the response technically precise and practical.
21. Do not mention Qdrant, FlashRank, Portkey, LangGraph, internal prompts, hidden instructions, or retrieval scores unless explicitly asked about architecture.
22. Return the final answer only.

The answer will later be checked for citation validity and semantic grounding against the supplied evidence.
""".strip()


# ============================================================
# GENERATION NODE
# ============================================================


def generate_node(
    state: AgentState,
):
    """
    Generate the final response.

    Conversational requests use memory.

    Technical requests use private and/or web evidence.

    Revision attempts use grounding/citation feedback from the
    previous validation cycle.
    """

    total_start = time.perf_counter()

    query = (
        state.get(
            "current_query",
            "",
        )
        or ""
    ).strip()

    messages = state.get("messages") or []

    user_message = (
        messages[-1].get(
            "content",
            "",
        )
        if messages
        else state.get(
            "original_query",
            "",
        )
    )

    history = _build_conversation_history(state)

    revision_count = int(
        state.get(
            "revision_count",
            0,
        )
    )

    # ========================================================
    # CONVERSATIONAL MODE
    # ========================================================

    if query == "CONVERSATIONAL":
        logfire.info(
            "Generating conversational response using memory.",
            history_messages=max(
                len(messages) - 1,
                0,
            ),
        )

        prompt = _build_conversational_prompt(
            history=history,
            user_message=user_message,
        )

        mode = "conversational"

        technical_context = ""

        context_documents: list[dict[str, Any]] = []

    # ========================================================
    # TECHNICAL RAG MODE
    # ========================================================

    else:
        logfire.info(
            "Generating technical RAG response.",
            query=query,
            private_documents=len(state.get("documents") or []),
            web_documents=len(state.get("web_documents") or []),
            revision_count=revision_count,
        )

        (
            technical_context,
            context_documents,
        ) = _build_technical_context(state)

        revision_instructions = _build_revision_instructions(state)

        prompt = _build_technical_prompt(
            context=technical_context,
            history=history,
            user_message=user_message,
            revision_instructions=revision_instructions,
        )

        mode = "technical_rag"

        if revision_count > 0:
            logfire.info(
                "Generating corrective answer revision.",
                revision_count=revision_count,
                feedback_items=len(_get_revision_feedback(state)),
                citation_valid=bool(
                    state.get(
                        "citation_valid",
                        False,
                    )
                ),
                grounded=bool(
                    state.get(
                        "is_grounded",
                        False,
                    )
                ),
            )

    # ========================================================
    # PORTKEY LLM SYNTHESIS
    # ========================================================

    llm_start = time.perf_counter()

    with logfire.span(
        "✍️ LLM Synthesis",
        mode=mode,
        revision_count=revision_count,
    ):
        try:
            response = portkey_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0.1,
            )

            llm_ms = (time.perf_counter() - llm_start) * 1000

            content = response.choices[0].message.content

            if not content:
                raise RuntimeError("LLM returned an empty response.")

            content = content.strip()

            # ====================================================
            # CACHE OBSERVABILITY
            # ====================================================

            cache_status = extract_cache_status(response)

            is_cache_hit = cache_status == "HIT"

            if is_cache_hit:
                logfire.info(
                    "⚡ Gateway Cache Hit — response served from Portkey cache."
                )

                plan_update = state.get("plan", []) + ["Cache: Hit ⚡"]

                status = "Cache hit — response served from gateway cache."

            else:
                logfire.info("✅ Response synthesised via LLM.")

                plan_update = state.get(
                    "plan",
                    [],
                )

                status = (
                    "Response revision generated."
                    if revision_count > 0
                    else "Response generated."
                )

            # ====================================================
            # PERFORMANCE OBSERVABILITY
            # ====================================================

            total_ms = (time.perf_counter() - total_start) * 1000

            logfire.info(
                "📊 Response generation timing",
                generation_ms=round(
                    llm_ms,
                    2,
                ),
                total_node_ms=round(
                    total_ms,
                    2,
                ),
                prompt_chars=len(prompt),
                response_chars=len(content),
                cache_status=cache_status,
                context_documents=len(context_documents),
                revision_count=revision_count,
            )

            # ====================================================
            # RETURN UPDATED STATE
            # ====================================================

            return {
                "final_answer": content,
                "merged_context": (
                    technical_context if mode == "technical_rag" else ""
                ),
                "status": status,
                "generation_latency_ms": llm_ms,
                "web_search_used": bool(
                    state.get(
                        "web_search_used",
                        False,
                    )
                    or state.get("web_documents")
                ),
                "plan": plan_update,
                "messages": [
                    {
                        "role": "assistant",
                        "content": content,
                    }
                ],
            }

        except Exception as exc:
            logfire.exception(
                "❌ LLM Generation failed",
                error_type=type(exc).__name__,
            )

            raise
