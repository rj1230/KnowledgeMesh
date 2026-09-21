# ↓
# Qdrant / Reranker / Grader / Grounding / Citation
#
# NeMo MUST NOT:
#   - answer technical questions
#   - perform retrieval
#   - generate RAG responses
#   - perform answer synthesis
#   - decide whether exact evidence exists
#
# Exact answerability belongs to LangGraph.
# ============================================================

from __future__ import annotations

import asyncio
import re
from typing import Any

import logfire
from langchain_groq import ChatGroq
from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.actions import action

from app.config import settings
from app.guardrails.colang_rules import (
    COLANG_CONTENT,
    YAML_CONTENT,
)


# ============================================================
# SINGLETON STATE
# ============================================================

_rails: LLMRails | None = None
_check_topic_llm: ChatGroq | None = None


# ============================================================
# DETERMINISTIC JAILBREAK PATTERNS
# ============================================================

_JAILBREAK_PATTERNS = [
    r"\bignore\s+(all\s+)?previous\s+instructions\b",
    r"\bignore\s+(all\s+)?prior\s+instructions\b",
    r"\bignore\s+the\s+system\s+prompt\b",
    r"\bforget\s+(all\s+)?previous\s+instructions\b",
    r"\bforget\s+your\s+system\s+prompt\b",
    r"\byou\s+are\s+now\s+dan\b",
    r"\bdeveloper\s+mode\b",
    r"\bdisable\s+your\s+guardrails\b",
    r"\bbypass\s+(your\s+)?safety\b",
    r"\bbypass\s+(your\s+)?restrictions\b",
    r"\bshow\s+me\s+your\s+system\s+prompt\b",
    r"\breveal\s+your\s+system\s+prompt\b",
    r"\bprint\s+your\s+system\s+prompt\b",
    r"\breveal\s+your\s+hidden\s+instructions\b",
]


def _is_jailbreak_attempt(message: str) -> bool:
    """
    Deterministic jailbreak detector.

    This runs before the LLM scope classifier so obvious
    prompt override attempts do not consume an LLM call.
    """

    normalized = " ".join(message.lower().split())

    return any(
        re.search(
            pattern,
            normalized,
        )
        for pattern in _JAILBREAK_PATTERNS
    )


# ============================================================
# DETERMINISTIC SPECIAL-CASE RESPONSES
# ============================================================

_JAILBREAK_RESPONSE = (
    "I maintain consistent safety and retrieval guidelines "
    "regardless of prompt-level override attempts. "
    "I can help with questions covered by the KnowledgeMesh "
    "technical knowledge base."
)


# ============================================================
# KNOWLEDGEMESH TOPIC CLASSIFIER
# ============================================================


@action(name="check_topic")
async def check_topic(
    context: dict | None = None,
) -> bool:
    """
    Determine whether a user request is plausibly within the
    KnowledgeMesh technical knowledge domain.

    IMPORTANT:
    This function answers ONLY:
        "Could this reasonably belong to the KnowledgeMesh
         technical knowledge domain?"

    It does NOT answer:
        "Does the exact answer exist?"

    Exact answerability is determined later by:

        Planner
          ↓
        Qdrant
          ↓
        FlashRank
          ↓
        Grader
          ↓
        Context Evaluator
          ↓
        Grounding
          ↓
        Citation Validation
    """

    if _check_topic_llm is None:
        logfire.warning(
            "⚠️ check_topic called before classifier initialization — allowing request."
        )
        return True

    ctx = context or {}

    user_message = (
        ctx.get("user_message")
        or ctx.get("last_user_message")
        or ctx.get("user_input")
        or ""
    )

    user_message = str(user_message).strip()

    if not user_message:
        logfire.warning(
            "⚠️ check_topic received no user message "
            f"(keys={list(ctx.keys())}) — allowing."
        )
        return True

    # --------------------------------------------------------
    # Deterministic jailbreak protection
    # --------------------------------------------------------

    if _is_jailbreak_attempt(user_message):
        logfire.warning(f"🛡️ Jailbreak attempt detected | query='{user_message[:120]}'")
        return False

    # --------------------------------------------------------
    # Scope classifier
    # --------------------------------------------------------

    prompt = f"""
You are the BROAD SCOPE CLASSIFIER for KnowledgeMesh.

KnowledgeMesh is an enterprise technical knowledge assistant
connected to a private indexed document corpus.

Your ONLY task is to determine whether the user's request is
PLAUSIBLY a technical knowledge request that should be passed
to the downstream retrieval system.

You MUST NOT decide whether the exact answer exists.

The downstream LangGraph pipeline will determine that by using:
    - semantic retrieval
    - Qdrant
    - reranking
    - document grading
    - context evaluation
    - query rewriting
    - bounded retry
    - grounding validation
    - citation validation

============================================================
ALLOW
============================================================

Return YES for requests involving:
    - AI
    - machine learning
    - deep learning
    - LLMs
    - language models
    - RAG
    - retrieval
    - vector databases
    - embeddings
    - transformers
    - attention
    - agents
    - agent memory
    - AutoGPT
    - agent architectures
    - agent harnesses
    - reasoning systems
    - evaluation
    - observability
    - software engineering
    - programming
    - APIs
    - infrastructure
    - networking
    - cloud engineering
    - technical documentation
    - engineering documentation
    - architecture documentation
    - system design
    - technical concepts
    - technical explanations
    - summaries of technical documentation
    - summaries of the indexed knowledge base
    - questions referring to "our documentation"
    - questions referring to "the documentation"
    - questions asking what the indexed documents contain

IMPORTANT:
Requests such as:
    "Summarize the key points from our documentation."
must be classified as YES. The downstream retrieval system
decides whether sufficient documentation evidence exists.

============================================================
REJECT
============================================================

Return NO for clearly unrelated requests such as:
    - recipes
    - restaurant recommendations
    - weather
    - sports scores
    - entertainment recommendations
    - jokes
    - unrelated creative writing
    - unrelated mathematics
    - unrelated homework
    - unrelated celebrity information
    - unrelated politics
    - unrelated general-world questions

============================================================
FAIL OPEN
============================================================

If uncertain, return YES.

Do not reject a potentially technical request merely because
the exact topic is not explicitly listed above.

============================================================
USER REQUEST
============================================================

{user_message}

============================================================
OUTPUT
============================================================

Return exactly:
    YES
or:
    NO
"""

    try:
        response = await _check_topic_llm.ainvoke(prompt)

        answer = str(response.content).strip().upper()

    except Exception as exc:
        # ----------------------------------------------------
        # Fail open.
        #
        # LangGraph remains responsible for retrieval,
        # evidence quality, grounding, and final correctness.
        # ----------------------------------------------------

        logfire.warning(
            "⚠️ check_topic classifier failed "
            f"({type(exc).__name__}: {exc}) "
            "— allowing request."
        )
        return True

    if answer.startswith("YES"):
        logfire.info(f"🟢 KnowledgeMesh scope allowed | query='{user_message[:120]}'")
        return True

    if answer.startswith("NO"):
        logfire.info(f"🔴 KnowledgeMesh scope rejected | query='{user_message[:120]}'")
        return False

    # --------------------------------------------------------
    # Unknown model output -> fail open
    # --------------------------------------------------------

    logfire.warning(
        f"⚠️ check_topic returned unexpected output '{answer}' — allowing request."
    )
    return True


# ============================================================
# INITIALIZE NEMO
# ============================================================


def initialize_rails() -> None:
    """
    Initialize NeMo Guardrails infrastructure.

    IMPORTANT:
    The LLMRails object is retained for configuration and
    observability compatibility.

    The request path intentionally does NOT call:
        _rails.generate(...)

    because that API invokes NeMo's full conversational
    generation machinery.

    KnowledgeMesh already has LangGraph as its RAG
    orchestrator.
    """

    global _rails
    global _check_topic_llm

    # --------------------------------------------------------
    # Shared classifier LLM
    # --------------------------------------------------------

    guard_llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model="openai/gpt-oss-20b",
        temperature=0,
    )

    _check_topic_llm = guard_llm

    # --------------------------------------------------------
    # NeMo configuration
    # --------------------------------------------------------

    config = RailsConfig.from_content(
        colang_content=COLANG_CONTENT,
        yaml_content=YAML_CONTENT,
    )

    _rails = LLMRails(
        config,
        llm=guard_llm,
    )

    _rails.register_action(
        check_topic,
        "check_topic",
    )

    logfire.info(
        "🛡️ NeMo Guardrails initialized "
        "as pre-RAG scope/safety gate "
        "(openai/gpt-oss-20b)."
    )


# ============================================================
# SYNCHRONOUS CLASSIFIER BRIDGE
# ============================================================


def _run_topic_classifier(
    message: str,
) -> bool:
    """
    Execute the async scope classifier from the synchronous
    FastAPI request path.

    main.py currently exposes a synchronous /query endpoint,
    so asyncio.run() is safe here because FastAPI executes
    the synchronous endpoint outside the running event loop.
    """

    if _check_topic_llm is None:
        logfire.warning("⚠️ Topic classifier unavailable — allowing request.")
        return True

    try:
        return bool(
            asyncio.run(
                check_topic(
                    {
                        "user_message": message,
                    }
                )
            )
        )

    except Exception as exc:
        logfire.warning(
            "⚠️ Topic classifier bridge failed "
            f"({type(exc).__name__}: {exc}) "
            "— allowing request."
        )
        return True


# ============================================================
# GUARD ENTRY POINT
# ============================================================


def guard(
    message: str,
) -> tuple[bool, str | None]:
    """
    Execute the KnowledgeMesh PRE-RAG gate.

    Returns:
        (True, response)
            The request was blocked and a response should be
            returned directly to the user.

        (False, None)
            The request is allowed to continue into LangGraph.

    CRITICAL:
    This function intentionally does NOT call:
        _rails.generate()

    That call would activate NeMo's full conversational
    generation pipeline and can cause:
        generate_user_intent
        generate_next_steps
        retrieve_relevant_chunks
        generate_bot_message

    KnowledgeMesh already performs those responsibilities
    through its own LangGraph architecture.
    """

    message = str(message or "").strip()

    if not message:
        return (
            False,
            None,
        )

    with logfire.span("🛡️ NeMo Pre-RAG Gate"):
        # ====================================================
        # GATE A · JAILBREAK
        # ====================================================

        if _is_jailbreak_attempt(message):
            logfire.warning("🛡️ Request blocked by deterministic jailbreak gate.")
            return (
                True,
                _JAILBREAK_RESPONSE,
            )

        # ====================================================
        # GATE B · BROAD KNOWLEDGE SCOPE
        # ====================================================

        allowed = _run_topic_classifier(message)

        if not allowed:
            response = (
                "I'm Knowldemina, a KnowledgeMesh technical "
                "knowledge assistant. This request falls outside "
                "the configured technical knowledge scope, so I "
                "can't provide a grounded answer for it. Please "
                "ask an AI, machine-learning, retrieval, LLM, "
                "agent, infrastructure, software, or other "
                "technical knowledge question."
            )

            logfire.info(f"🔴 Pre-RAG guard blocked request | query='{message[:120]}'")

            return (
                True,
                response,
            )

        # ====================================================
        # ALLOW
        # ====================================================

        logfire.info(f"✅ Pre-RAG guard passed | query='{message[:120]}'")

        return (
            False,
            None,
        )
