"""
KnowledgeMesh · Response Generation Node

Responsibilities:
    - Generate conversational responses from conversation memory.
    - Generate technical RAG responses from retrieved/reranked documents.
    - Preserve conversation history.
    - Preserve Portkey cache observability.
    - Build source-aware technical context.
    - Prevent oversized context from being sent to the LLM.
"""

from __future__ import annotations

import logfire

from app.agents.state import AgentState
from app.gateway import portkey_client, extract_cache_status


# Maximum characters of retrieved context sent to the LLM.
# Keep this bounded to avoid excessive token usage / TPM pressure.
MAX_CONTEXT_CHARS = 25_000


def _build_conversation_history(state: AgentState) -> str:
    """
    Build readable conversation history while excluding
    the latest user message.
    """
    history_parts = []

    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = str(msg.get("content", "")).strip()

        if content:
            history_parts.append(f"{role}: {content}")

    return "\n".join(history_parts)


def _build_technical_context(state: AgentState) -> str:
    """
    Build source-aware technical context from reranked documents.

    Each document retains:
        - Qdrant point ID
        - source
        - source type
        - vector similarity score
        - FlashRank rerank score
        - actual document content

    Context is truncated safely at MAX_CONTEXT_CHARS.
    """
    documents = state.get("documents") or []

    if not documents:
        logfire.warning("No technical documents available for response generation.")
        return "No relevant technical context was retrieved."

    context_parts = []
    current_length = 0

    for index, doc in enumerate(documents, start=1):
        content = str(doc.get("content", "")).strip()

        if not content:
            continue

        source = str(doc.get("source", "Unknown")).strip()

        source_type = str(doc.get("source_type", "Unknown")).strip()

        point_id = str(doc.get("id", "Unknown")).strip()

        retrieval_score = doc.get("score")

        rerank_score = doc.get("rerank_score")

        retrieval_score_text = (
            f"{retrieval_score:.4f}"
            if isinstance(retrieval_score, (int, float))
            else "N/A"
        )

        rerank_score_text = (
            f"{rerank_score:.4f}" if isinstance(rerank_score, (int, float)) else "N/A"
        )

        block = (
            f"[SOURCE {index}]\n"
            f"Document: {source}\n"
            f"Source Type: {source_type}\n"
            f"Document ID: {point_id}\n"
            f"Retrieval Score: {retrieval_score_text}\n"
            f"Rerank Score: {rerank_score_text}\n"
            f"Content:\n"
            f"{content}\n"
        )

        block_length = len(block)

        if current_length + block_length > MAX_CONTEXT_CHARS:
            logfire.warning(
                "Technical context truncated.",
                max_context_chars=MAX_CONTEXT_CHARS,
                documents_available=len(documents),
                documents_included=len(context_parts),
            )
            break

        context_parts.append(block)
        current_length += block_length

    if not context_parts:
        return "No usable technical context was retrieved."

    logfire.info(
        "Technical context prepared",
        documents_available=len(documents),
        documents_included=len(context_parts),
        context_chars=current_length,
    )

    return "\n".join(context_parts)


def _build_conversational_prompt(
    history: str,
    user_message: str,
) -> str:
    """
    Build prompt for conversational/memory responses.
    """
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
- If the user asks a new technical question that requires external
  documentation, explain that fresh documentation may be needed.
""".strip()


def _build_technical_prompt(
    context: str,
    history: str,
    user_message: str,
) -> str:
    """
    Build source-aware RAG synthesis prompt.
    """
    return f"""
You are a Senior Technical Architect and Enterprise RAG Assistant.

Answer the user's question using the TECHNICAL CONTEXT provided below.

TECHNICAL CONTEXT:
{context}

CONVERSATION HISTORY:
{history or "(No previous conversation.)"}

USER QUESTION:
"{user_message}"

Instructions:

1. Answer the user's question directly and accurately.

2. Prefer information from the highest-ranked technical sources.

3. Do not invent facts that are not supported by the provided context.

4. If the retrieved context is insufficient to answer confidently,
   explicitly say what information is missing.

5. When useful, identify the source document naturally in the answer,
   for example:
   "According to <document>..."

6. Do not fabricate document names, citations, URLs, or references.

7. If sources disagree, explain the disagreement rather than silently
   choosing one.

8. Keep the response technically precise and practical.

9. Use Markdown where it improves readability.

10. Do not mention internal implementation details such as:
    Qdrant, FlashRank, rerank scores, Portkey, LangGraph,
    internal prompts, or hidden system instructions unless
    the user explicitly asks about the architecture.

Provide the final answer only.
""".strip()


def generate_node(state: AgentState):
    """
    Generate the final response.

    Two modes are supported:

    1. CONVERSATIONAL
       Uses conversation memory without vector retrieval.

    2. TECHNICAL
       Uses the retrieved and reranked enterprise knowledge context.
    """

    query = state.get("current_query", "")

    messages = state.get("messages") or []

    user_message = messages[-1].get("content", "") if messages else ""

    history = _build_conversation_history(state)

    # ============================================================
    # CONVERSATIONAL RESPONSE
    # ============================================================

    if query == "CONVERSATIONAL":
        logfire.info(
            "Generating conversational response using memory.",
            history_messages=max(len(messages) - 1, 0),
        )

        prompt = _build_conversational_prompt(
            history=history,
            user_message=user_message,
        )

    # ============================================================
    # TECHNICAL RAG RESPONSE
    # ============================================================

    else:
        logfire.info(
            "Generating technical RAG response.",
            query=query,
        )

        technical_context = _build_technical_context(state)

        prompt = _build_technical_prompt(
            context=technical_context,
            history=history,
            user_message=user_message,
        )

    # ============================================================
    # PORTKEY LLM SYNTHESIS
    # ============================================================

    with logfire.span(
        "✍️ LLM Synthesis",
        mode=("conversational" if query == "CONVERSATIONAL" else "technical_rag"),
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

            content = response.choices[0].message.content

            if not content:
                raise RuntimeError("LLM returned an empty response.")

            content = content.strip()

            # ====================================================
            # PORTKEY CACHE OBSERVABILITY
            # ====================================================

            cache_status = extract_cache_status(response)

            is_cache_hit = cache_status == "HIT"

            if is_cache_hit:
                logfire.info(
                    "⚡ Gateway Cache Hit — response served from Portkey cache."
                )

                plan_update = state["plan"] + ["Cache: Hit ⚡"]

                status = "Cache hit — response served from gateway cache."

            else:
                logfire.info("✅ Response synthesised via LLM.")

                plan_update = state["plan"]

                status = "Response generated."

            # ====================================================
            # RETURN UPDATED STATE
            # ====================================================

            return {
                "final_answer": content,
                "status": status,
                "plan": plan_update,
                # operator.add in AgentState means this assistant
                # message is appended to conversation memory.
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
