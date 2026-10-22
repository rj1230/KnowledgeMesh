# ============================================================
# KnowledgeMesh · NeMo Guardrails
#
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
import logging
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


logger = logging.getLogger(__name__)


# ============================================================
# SINGLETON STATE
# ============================================================

_rails: LLMRails | None = None
_check_topic_llm: ChatGroq | None = None
_TOPIC_CLASSIFIER_DEADLINE_SECONDS = 12.0


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


def _build_topic_classifier_prompt(user_message: str) -> str:
    """Build the shared broad-scope classification prompt."""
    return f"""
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
CLASSIFICATION
============================================================

Return exactly ONE label:

TECHNICAL
NON_TECHNICAL

============================================================
TECHNICAL
============================================================

Classify as TECHNICAL when the request is clearly or plausibly
related to technical knowledge, engineering, software, AI/ML,
infrastructure, or the indexed technical knowledge base.

Examples include:

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

must be classified as TECHNICAL.

The downstream retrieval system decides whether sufficient
documentation evidence exists.

============================================================
NON_TECHNICAL
============================================================

Classify as NON_TECHNICAL when the request is clearly unrelated
to the configured technical scope.

Examples include:

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
IMPORTANT
============================================================

There is NO ambiguous category.

If the request could reasonably be interpreted as a technical
knowledge request, classify it as TECHNICAL.

Only clearly unrelated requests should be NON_TECHNICAL.

============================================================
USER REQUEST
============================================================

{user_message}

============================================================
OUTPUT
============================================================

Return exactly one label:

TECHNICAL
or
NON_TECHNICAL
"""


@action(name="check_topic")
async def check_topic(
    context: dict | None = None,
) -> str:
    """
    Determine whether a user request is within the broad
    KnowledgeMesh technical knowledge domain.

    Normal scope outcomes are TECHNICAL and NON_TECHNICAL.
    UNAVAILABLE indicates that a trustworthy decision could
    not be made; the synchronous PRE-RAG gate must block it.

    It does NOT determine whether the exact answer exists.

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

    # --------------------------------------------------------
    # Classifier unavailable: do not guess the scope.
    # The PRE-RAG gate will fail closed on this state.
    # --------------------------------------------------------

    if _check_topic_llm is None:
        logger.warning(
            "TOPIC_DIAG: classifier_state=UNINITIALIZED result=UNAVAILABLE"
        )
        return "UNAVAILABLE"

    ctx = context or {}

    user_message = (
        ctx.get("user_message")
        or ctx.get("last_user_message")
        or ctx.get("user_input")
        or ""
    )

    user_message = str(user_message).strip()

    logger.info(
        "TOPIC_DIAG: classifier_state=INITIALIZED message_present=%s message_length=%s",
        bool(user_message),
        len(user_message),
    )

    # --------------------------------------------------------
    # Empty request
    #
    # Keep this non-blocking. main.py already handles an empty
    # request separately.
    # --------------------------------------------------------

    if not user_message:
        logger.warning(
            "TOPIC_DIAG: empty_user_message result=TECHNICAL keys=%s",
            list(ctx.keys()),
        )
        return "TECHNICAL"

    # --------------------------------------------------------
    # Deterministic jailbreak protection
    # --------------------------------------------------------

    if _is_jailbreak_attempt(user_message):
        logger.warning(
            "TOPIC_DIAG: jailbreak_detected result=NON_TECHNICAL query=%r",
            user_message[:120],
        )
        return "NON_TECHNICAL"

    # --------------------------------------------------------
    # Scope classifier
    # --------------------------------------------------------

    prompt = _build_topic_classifier_prompt(user_message)

    try:
        logger.info("TOPIC_DIAG: invoking_classifier")

        response = await asyncio.wait_for(
            _check_topic_llm.ainvoke(prompt),
            timeout=_TOPIC_CLASSIFIER_DEADLINE_SECONDS,
        )

        answer = str(response.content).strip().upper()

        logger.info(
            "TOPIC_DIAG: classifier_returned=%s",
            answer[:80],
        )

    except Exception as exc:
        # An uncertain scope decision must not enter retrieval.
        logger.warning(
            "TOPIC_DIAG: classifier_failed error_type=%s error=%s result=UNAVAILABLE",
            type(exc).__name__,
            exc,
        )

        return "UNAVAILABLE"

    # --------------------------------------------------------
    # Explicit TECHNICAL
    # --------------------------------------------------------

    if answer.startswith("TECHNICAL"):
        logger.info(
            "TOPIC_DIAG: scope_result=TECHNICAL query=%r",
            user_message[:120],
        )
        return "TECHNICAL"

    # --------------------------------------------------------
    # Explicit NON_TECHNICAL
    # --------------------------------------------------------

    if answer.startswith("NON_TECHNICAL"):
        logger.info(
            "TOPIC_DIAG: scope_result=NON_TECHNICAL query=%r",
            user_message[:120],
        )
        return "NON_TECHNICAL"

    # --------------------------------------------------------
    # Backward compatibility with older classifier output.
    # --------------------------------------------------------

    if answer.startswith("YES"):
        logger.info(
            "TOPIC_DIAG: legacy_classifier_output=YES result=TECHNICAL query=%r",
            user_message[:120],
        )
        return "TECHNICAL"

    if answer.startswith("NO"):
        logger.info(
            "TOPIC_DIAG: legacy_classifier_output=NO result=NON_TECHNICAL query=%r",
            user_message[:120],
        )
        return "NON_TECHNICAL"

    # Unknown output is not evidence that the request is in scope.
    logger.warning(
        "TOPIC_DIAG: classifier_unexpected_output=%r result=UNAVAILABLE",
        answer[:120],
    )

    return "UNAVAILABLE"


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
        timeout=10.0,
        max_retries=0,
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

    logger.info("TOPIC_DIAG: classifier_initialized model=openai/gpt-oss-20b")

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
) -> str:
    """
    Classify a request synchronously for the FastAPI pre-RAG gate.

    Do not bridge through asyncio.run(): the synchronous API path
    should not reuse the shared async client across event loops.
    Uncertain results fail closed as UNAVAILABLE.
    """

    if _check_topic_llm is None:
        logger.warning(
            "TOPIC_DIAG: classifier_state=UNINITIALIZED bridge_result=UNAVAILABLE"
        )
        return "UNAVAILABLE"

    user_message = str(message or "").strip()

    if not user_message:
        return "TECHNICAL"

    if _is_jailbreak_attempt(user_message):
        logger.warning(
            "TOPIC_DIAG: jailbreak_detected result=NON_TECHNICAL query=%r",
            user_message[:120],
        )
        return "NON_TECHNICAL"

    try:
        logger.info("TOPIC_DIAG: invoking_classifier_sync")

        response = _check_topic_llm.invoke(
            _build_topic_classifier_prompt(user_message)
        )
        answer = str(response.content).strip().upper()

        logger.info(
            "TOPIC_DIAG: classifier_sync_returned=%s",
            answer[:80],
        )

        if answer.startswith("TECHNICAL"):
            return "TECHNICAL"
        if answer.startswith("NON_TECHNICAL"):
            return "NON_TECHNICAL"
        if answer.startswith("YES"):
            return "TECHNICAL"
        if answer.startswith("NO"):
            return "NON_TECHNICAL"

        logger.warning(
            "TOPIC_DIAG: classifier_unexpected_output=%r result=UNAVAILABLE",
            answer[:120],
        )
        return "UNAVAILABLE"

    except Exception as exc:
        logger.warning(
            "TOPIC_DIAG: synchronous_classifier_failed "
            "error_type=%s error=%s result=UNAVAILABLE",
            type(exc).__name__,
            exc,
        )
        return "UNAVAILABLE"


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
            Request was blocked.

        (False, None)
            Request is allowed to continue into LangGraph.

    A request is allowed only for an explicit TECHNICAL
    result. NON_TECHNICAL and UNAVAILABLE are both blocked.

    CRITICAL:
    This function intentionally does NOT call:
        _rails.generate()

    That call would activate NeMo's full conversational
    generation pipeline.

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
            logger.warning("TOPIC_DIAG: deterministic_jailbreak_block")

            return (
                True,
                _JAILBREAK_RESPONSE,
            )

        # ====================================================
        # GATE B · BROAD KNOWLEDGE SCOPE
        # ====================================================

        scope = _run_topic_classifier(message)

        # ----------------------------------------------------
        # Unavailable classification: fail closed.
        # ----------------------------------------------------

        if scope == "UNAVAILABLE":
            response = (
                "KnowledgeMesh temporarily couldn't verify whether "
                "this request is within its supported technical scope. "
                "No retrieval or answer generation was performed. "
                "Please retry shortly."
            )

            logger.warning(
                "TOPIC_DIAG: PRE_RAG_BLOCK scope=UNAVAILABLE query=%r",
                message[:120],
            )

            return (True, response)

        # ----------------------------------------------------
        # Explicit out-of-scope request.
        # ----------------------------------------------------

        if scope == "NON_TECHNICAL":
            response = (
                "I'm KnowledgeMesh, a KnowledgeMesh technical "
                "knowledge assistant. This request falls outside "
                "the configured technical knowledge scope, so I "
                "can't provide a grounded answer for it. Please "
                "ask an AI, machine-learning, retrieval, LLM, "
                "agent, infrastructure, software, or other "
                "technical knowledge question."
            )

            logger.info(
                "TOPIC_DIAG: PRE_RAG_BLOCK scope=NON_TECHNICAL query=%r",
                message[:120],
            )

            return (
                True,
                response,
            )

        # ====================================================
        # ALLOW
        # ====================================================

        logger.info(
            "TOPIC_DIAG: PRE_RAG_ALLOW scope=TECHNICAL query=%r",
            message[:120],
        )

        return (
            False,
            None,
        )
