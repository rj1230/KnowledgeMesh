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
#
# This gate also answers plain small talk (greetings, thanks, "what can
# you do") deterministically, so those messages get a friendly reply
# instead of an out-of-scope refusal, with no LLM call.
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

# Messages that obviously concern the knowledge base skip the scope-
# classifier LLM call (about 1 to 2 seconds). The jailbreak check still
# runs first, and LangGraph still decides whether evidence exists. Set to
# False to send every request through the classifier.
FAST_PATH_TECHNICAL = True


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

_SCOPE_HINT = (
    "Ask me an AI, machine-learning, retrieval, LLM, agent, "
    "infrastructure or software question, or about the documents "
    "in the knowledge base."
)

_NON_TECHNICAL_RESPONSE = (
    "I'm KnowledgeMesh, a technical knowledge assistant. This request "
    "falls outside the configured technical knowledge scope, so I can't "
    "provide a grounded answer for it. " + _SCOPE_HINT
)


# ============================================================
# SMALL TALK (answered here, no LLM call, no retrieval)
#
# Each pattern must match the WHOLE message, so a technical question
# that merely starts with "hi" or "help" is never treated as small talk.
# ============================================================

_SMALL_TALK: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"^(?:hi|hello|hey|hiya|howdy|yo|greetings|"
            r"good\s+(?:morning|afternoon|evening))"
            r"(?:\s+(?:there|team|everyone|knowledgemesh))?[\s!.,?]*$",
            re.IGNORECASE,
        ),
        "Hello! I'm KnowledgeMesh, a technical knowledge assistant. " + _SCOPE_HINT,
    ),
    (
        re.compile(
            r"^how\s+are\s+you(?:\s+doing)?(?:\s+today)?[\s!.,?]*$",
            re.IGNORECASE,
        ),
        "I'm doing well, thanks for asking. " + _SCOPE_HINT,
    ),
    (
        re.compile(
            r"^(?:thanks|thank\s+you|thx|ty|cheers)"
            r"(?:\s+(?:a\s+lot|so\s+much|very\s+much))?[\s!.,]*$",
            re.IGNORECASE,
        ),
        "You're welcome! Ask me another question whenever you like.",
    ),
    (
        re.compile(
            r"^(?:bye|goodbye|see\s+you|see\s+ya|cya|good\s*night)[\s!.,]*$",
            re.IGNORECASE,
        ),
        "Goodbye! Come back any time you have a technical question.",
    ),
    (
        re.compile(
            r"^(?:who\s+are\s+you|what\s+are\s+you|what\s+do\s+you\s+do|"
            r"what\s+can\s+you\s+do|how\s+can\s+you\s+help(?:\s+me)?|help)"
            r"[\s!.,?]*$",
            re.IGNORECASE,
        ),
        "I'm KnowledgeMesh, a technical knowledge assistant. I answer "
        "questions from the documents in the knowledge base, cite the "
        "passage behind every claim, and tell you when the documents "
        "don't contain an answer. " + _SCOPE_HINT,
    ),
]


def _small_talk_reply(message: str) -> str | None:
    """Return a canned reply for plain small talk, otherwise None."""

    normalized = " ".join(message.split())

    for pattern, reply in _SMALL_TALK:
        if pattern.match(normalized):
            return reply

    return None


# ============================================================
# TECHNICAL FAST PATH (skips the scope-classifier LLM call)
#
# Deliberately limited to strong, unambiguous terms. Anything else still
# goes to the classifier, and LangGraph still decides whether evidence
# exists, so a false positive costs only a retrieval.
# ============================================================

_TECHNICAL_HINTS = re.compile(
    r"\b(?:"
    r"llms?|rag|retrieval|embeddings?|vector|transformers?|attention|"
    r"agentic|agents?|harness|loop\s+(?:engineering|specification)|"
    r"prompt(?:s|ing)?|fine-?tun\w*|hallucinat\w+|grounding|"
    r"neural|machine\s+learning|deep\s+learning|"
    r"api|sdk|kubernetes|docker|database|latency|infrastructure|"
    r"algorithm|architecture|framework|pipeline|"
    r"according\s+to\s+the\s+\w+(?:\s+\w+)?\s+(?:paper|survey|report)|"
    r"documentation|knowledge\s+base"
    r")\b",
    re.IGNORECASE,
)


def _looks_clearly_technical(message: str) -> bool:
    return bool(_TECHNICAL_HINTS.search(message or ""))


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
AMBIGUOUS

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

If the request could reasonably be interpreted as a technical
knowledge request, classify it as TECHNICAL.

Only clearly unrelated requests should be NON_TECHNICAL.

Use AMBIGUOUS only when the request is genuinely unclear or you
cannot confidently determine its domain. Do not treat ambiguity
as permission to retrieve.

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


@action(name="check_topic")
async def check_topic(
    context: dict | None = None,
) -> str:
    """
    Determine whether a user request is within the broad
    KnowledgeMesh technical knowledge domain.

    Normal scope outcomes are TECHNICAL and NON_TECHNICAL.
    AMBIGUOUS indicates that a trustworthy decision could
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
    # Classifier AMBIGUOUS: do not guess the scope.
    # The PRE-RAG gate will fail closed on this state.
    # --------------------------------------------------------

    if _check_topic_llm is None:
        logger.warning("TOPIC_DIAG: classifier_state=UNINITIALIZED result=AMBIGUOUS")
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
    Uncertain results fail closed as AMBIGUOUS.
    """

    if _check_topic_llm is None:
        logger.warning(
            "TOPIC_DIAG: classifier_state=UNINITIALIZED bridge_result=AMBIGUOUS"
        )
        return "AMBIGUOUS"

    user_message = str(message or "").strip()

    if not user_message:
        logger.warning("TOPIC_DIAG: empty_user_message result=AMBIGUOUS")
        return "AMBIGUOUS"

    if _is_jailbreak_attempt(user_message):
        logger.warning(
            "TOPIC_DIAG: jailbreak_detected result=NON_TECHNICAL query=%r",
            user_message[:120],
        )
        return "NON_TECHNICAL"

    # Clearly technical wording: skip the classifier LLM call.
    if FAST_PATH_TECHNICAL and _looks_clearly_technical(user_message):
        logger.info(
            "TOPIC_DIAG: fast_path_technical query=%r",
            user_message[:120],
        )
        return "TECHNICAL"

    try:
        logger.info("TOPIC_DIAG: invoking_classifier_sync")

        response = _check_topic_llm.invoke(_build_topic_classifier_prompt(user_message))
        answer = str(response.content).strip().upper()

        logger.info(
            "TOPIC_DIAG: classifier_sync_returned=%s",
            answer[:80],
        )

        if answer.startswith("TECHNICAL"):
            return "TECHNICAL"
        if answer.startswith("NON_TECHNICAL"):
            return "NON_TECHNICAL"
        if answer.startswith("AMBIGUOUS"):
            return "AMBIGUOUS"
        if answer.startswith("YES"):
            return "TECHNICAL"
        if answer.startswith("NO"):
            return "NON_TECHNICAL"

        logger.warning(
            "TOPIC_DIAG: classifier_unexpected_output=%r result=AMBIGUOUS",
            answer[:120],
        )
        return "AMBIGUOUS"

    except Exception as exc:
        logger.warning(
            "TOPIC_DIAG: synchronous_classifier_failed "
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
            The gate answered the request itself (a jailbreak block,
            an out-of-scope refusal, or a small-talk reply). No
            retrieval or generation runs.

        (False, None)
            Request is allowed to continue into LangGraph.

    A request is allowed only for an explicit TECHNICAL
    result. NON_TECHNICAL and AMBIGUOUS are both blocked.

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
        # GATE A2 · SMALL TALK
        #
        # Greetings, thanks and "what can you do" get a friendly
        # reply instead of an out-of-scope refusal. No LLM call.
        # ====================================================

        small_talk = _small_talk_reply(message)

        if small_talk is not None:
            logger.info(
                "TOPIC_DIAG: small_talk_reply query=%r",
                message[:120],
            )

            return (
                True,
                small_talk,
            )

        # ====================================================
        # GATE B · BROAD KNOWLEDGE SCOPE
        # ====================================================

        scope = _run_topic_classifier(message)

        # ----------------------------------------------------
        # AMBIGUOUS classification: fail closed.
        # ----------------------------------------------------

        if scope == "AMBIGUOUS":
            response = (
                "I couldn't confidently verify that request's technical "
                "scope right now. Please provide more technical context "
                "or retry shortly. No retrieval or generation was run."
            )

            logger.warning(
                "TOPIC_DIAG: PRE_RAG_BLOCK scope=AMBIGUOUS query=%r",
                message[:120],
            )

            return (True, response)

        # ----------------------------------------------------
        # Explicit out-of-scope request.
        # ----------------------------------------------------

        if scope == "NON_TECHNICAL":
            logger.info(
                "TOPIC_DIAG: PRE_RAG_BLOCK scope=NON_TECHNICAL query=%r",
                message[:120],
            )

            return (
                True,
                _NON_TECHNICAL_RESPONSE,
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
