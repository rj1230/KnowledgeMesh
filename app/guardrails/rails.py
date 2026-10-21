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
) -> str:
    """
    Determine whether a user request is within the broad
    KnowledgeMesh technical knowledge domain.

    Runtime outcomes:

        TECHNICAL
        NON_TECHNICAL
        AMBIGUOUS

    Only an explicit TECHNICAL result may pass the pre-RAG gate.

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
    # Classifier unavailable.
    #
    # Fail closed: without a trustworthy scope decision, do not
    # authorize retrieval. The request-level gate blocks AMBIGUOUS.
    # --------------------------------------------------------

    if _check_topic_llm is None:
        logger.warning(
            "TOPIC_DIAG: classifier_state=UNINITIALIZED result=AMBIGUOUS"
        )
        return "AMBIGUOUS"

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
    # Fail closed: missing input cannot authorize retrieval.
    # main.py also validates empty requests at the API boundary.
    # --------------------------------------------------------

    if not user_message:
        logger.warning(
            "TOPIC_DIAG: empty_user_message result=AMBIGUOUS keys=%s",
            list(ctx.keys()),
        )
        return "AMBIGUOUS"

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

    prompt = f"""
You are the BROAD SCOPE CLASSIFIER for KnowledgeMesh.

KnowledgeMesh is an enterprise technical knowledge assistant
connected to a private indexed document corpus.

Your ONLY task is to determine whether the user's actual intent
falls within KnowledgeMesh's configured technical knowledge scope.

Do not classify a request as TECHNICAL merely because it contains
a technical-sounding word. Classify the request by its overall intent
and subject domain, not by whether the answer could come from general
knowledge or whether the answer exists in the indexed document corpus.

A direct conceptual question about software engineering, systems
engineering, AI/ML, or another listed technical domain is TECHNICAL,
even when it is short or asks only "what is" or "how does" something
work. For example, "What is loop engineering?" is TECHNICAL.

A request for an ordinary recipe or cooking instructions remains
NON_TECHNICAL. A technical question about the engineering or control
systems of a cooking appliance can be TECHNICAL.

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
AMBIGUOUS

============================================================
TECHNICAL
============================================================

Classify as TECHNICAL when the user's actual request is within
the technical domains supported by KnowledgeMesh, such as AI/ML,
LLMs, retrieval, agent systems, software engineering, programming,
infrastructure, networking, cloud engineering, or technical
documentation.

The request need not use specialist terminology if its intent is
clearly within one of these domains.

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

Use NON_TECHNICAL for requests whose actual intent is outside
the configured technical scope, including ordinary recipes, cooking
instructions, greetings that ask no technical question, weather,
sports, entertainment, and unrelated general-world requests.

Examples:
- "How do I make coffee?" -> NON_TECHNICAL
- "What is loop engineering?" -> TECHNICAL
- "How does an espresso machine's temperature controller work?"
  -> TECHNICAL, because the intent is engineering.

Use AMBIGUOUS only when the request is genuinely unclear or you
cannot confidently determine its domain. Do not treat ambiguity
as permission to retrieve.

Do not decide whether an answer exists in the indexed documents;
that remains the responsibility of downstream retrieval and
evidence validation.

============================================================
USER REQUEST
============================================================

{user_message}

============================================================
OUTPUT
============================================================

Return exactly one label:

TECHNICAL
NON_TECHNICAL
or
AMBIGUOUS
"""

    try:
        logger.info("TOPIC_DIAG: invoking_classifier")

        response = await _check_topic_llm.ainvoke(prompt)

        answer = str(response.content).strip().upper()

        logger.info(
            "TOPIC_DIAG: classifier_returned=%s",
            answer[:80],
        )

    except Exception as exc:
        # ----------------------------------------------------
        # Classifier failure -> fail CLOSED for this request.
        # Do not authorize retrieval without a scope decision.
        # ----------------------------------------------------

        logger.warning(
            "TOPIC_DIAG: classifier_failed error_type=%s error=%s result=AMBIGUOUS",
            type(exc).__name__,
            exc,
        )

        return "AMBIGUOUS"

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
    # Explicit AMBIGUOUS: a valid fail-closed decision.
    # --------------------------------------------------------

    if answer.startswith("AMBIGUOUS"):
        logger.info(
            "TOPIC_DIAG: scope_result=AMBIGUOUS query=%r",
            user_message[:120],
        )
        return "AMBIGUOUS"

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

    # --------------------------------------------------------
    # Unknown model output is not a trustworthy scope decision.
    # --------------------------------------------------------

    logger.warning(
        "TOPIC_DIAG: classifier_unexpected_output=%r result=AMBIGUOUS",
        answer[:120],
    )

    return "AMBIGUOUS"


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
    Execute the async scope classifier from the synchronous
    FastAPI request path.

    main.py currently exposes a synchronous /query endpoint,
    so asyncio.run() is safe here because FastAPI executes
    the synchronous endpoint outside the running event loop.

    Runtime output is normalized to one of three states:

        TECHNICAL
        NON_TECHNICAL
        AMBIGUOUS
    """

    if _check_topic_llm is None:
        logger.warning(
            "TOPIC_DIAG: classifier_state=UNINITIALIZED bridge_result=AMBIGUOUS"
        )
        return "AMBIGUOUS"

    try:
        result = (
            str(
                asyncio.run(
                    check_topic(
                        {
                            "user_message": message,
                        }
                    )
                )
            )
            .strip()
            .upper()
        )

        # ----------------------------------------------------
        # Accept only the three documented scope states.
        # ----------------------------------------------------

        if result == "TECHNICAL":
            return "TECHNICAL"

        if result == "NON_TECHNICAL":
            return "NON_TECHNICAL"

        if result == "AMBIGUOUS":
            return "AMBIGUOUS"

        logger.warning(
            "TOPIC_DIAG: invalid_classifier_result=%r result=AMBIGUOUS",
            result[:80],
        )
        return "AMBIGUOUS"

    except Exception as exc:
        logger.warning(
            "TOPIC_DIAG: classifier_bridge_failed "
            "error_type=%s error=%s result=AMBIGUOUS",
            type(exc).__name__,
            exc,
        )

        return "AMBIGUOUS"


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

    Only an explicit TECHNICAL result is allowed through.
    NON_TECHNICAL and AMBIGUOUS results are blocked.

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
        # Explicitly out-of-scope requests are blocked.
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
        # GATE C · UNCERTAIN OR UNAVAILABLE CLASSIFIER
        # ====================================================

        if scope != "TECHNICAL":
            response = (
                "I couldn't confidently verify that request's technical "
                "scope right now. Please provide more technical context "
                "or retry shortly. No retrieval or generation was run."
            )

            logger.warning(
                "TOPIC_DIAG: PRE_RAG_BLOCK scope=AMBIGUOUS query=%r",
                message[:120],
            )

            return (
                True,
                response,
            )

        # ====================================================
        # ALLOW ONLY EXPLICIT TECHNICAL
        # ====================================================

        logger.info(
            "TOPIC_DIAG: PRE_RAG_ALLOW scope=TECHNICAL query=%r",
            message[:120],
        )

        return (
            False,
            None,
        )
