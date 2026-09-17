# ============================================================
# KnowledgeMesh · Enterprise Agentic RAG API
#
# IMPORTANT:
# Logfire MUST be configured before importing application
# modules so spans from all modules are captured.
# ============================================================

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("knowledgemesh")


# ============================================================
# ENVIRONMENT
# ============================================================

# app/main.py -> project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ENV_PATH = os.path.join(
    BASE_DIR,
    ".env",
)

from dotenv import load_dotenv


# Load project-root .env first.
load_dotenv(
    dotenv_path=ENV_PATH,
    override=False,
)


# Optional fallback if .env is located one directory above.
if not os.path.exists(ENV_PATH):
    parent_env_path = os.path.join(
        os.path.dirname(BASE_DIR),
        ".env",
    )

    if os.path.exists(parent_env_path):
        load_dotenv(
            dotenv_path=parent_env_path,
            override=False,
        )


# ============================================================
# LOGFIRE
# ============================================================

import logfire


LOGFIRE_TOKEN = os.getenv("LOGFIRE_TOKEN")

if LOGFIRE_TOKEN:
    try:
        logfire.configure(
            token=LOGFIRE_TOKEN,
        )

        logger.info("Logfire cloud telemetry configured.")

    except Exception as exc:
        logger.warning(
            "Logfire configuration failed. Continuing without cloud telemetry: %s",
            exc,
        )

else:
    logger.warning(
        "LOGFIRE_TOKEN is not configured. Continuing without Logfire cloud telemetry."
    )


# ============================================================
# APPLICATION IMPORTS
# ============================================================

from fastapi import FastAPI, Response
from pydantic import BaseModel, Field

from app.agents.graph import rag_agent
from app.guardrails import (
    guard,
    initialize_rails,
)


# ============================================================
# APPLICATION METADATA
# ============================================================

APP_NAME = "KnowledgeMesh"

APP_TITLE = "KnowledgeMesh · Enterprise Agentic RAG API"

APP_VERSION = "2.2.0"


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title=APP_TITLE,
    description=(
        "Enterprise Agentic RAG using LangGraph, "
        "Qdrant Cloud, FlashRank, Portkey, "
        "NeMo Guardrails, document grading, "
        "query rewriting, grounding validation, "
        "citation validation, bounded answer revision, "
        "and conversational memory."
    ),
    version=APP_VERSION,
)


# ============================================================
# STARTUP STATE
# ============================================================

_guardrails_initialized = False


# ============================================================
# STARTUP
# ============================================================


@app.on_event("startup")
def startup_event() -> None:
    """
    Initialize application components before serving requests.
    """

    global _guardrails_initialized

    logfire.info(
        "🚀 KnowledgeMesh API startup",
        version=APP_VERSION,
        base_dir=BASE_DIR,
        env_path=ENV_PATH,
    )

    try:
        # ----------------------------------------------------
        # NeMo Guardrails
        # ----------------------------------------------------

        initialize_rails()

        _guardrails_initialized = True

        logfire.info(
            "✅ NeMo Guardrails initialized successfully",
        )

    except Exception:
        _guardrails_initialized = False

        logfire.exception(
            "❌ Failed to initialize NeMo Guardrails",
        )

        raise


# ============================================================
# REQUEST MODELS
# ============================================================


class QueryRequest(BaseModel):
    """
    Incoming RAG request.
    """

    q: str = Field(
        ...,
        min_length=1,
        description="User question",
    )

    thread_id: Optional[str] = Field(
        default="default_user",
        description="Conversation memory thread ID",
    )


# ============================================================
# RESPONSE HELPERS
# ============================================================


def _safe_float(
    value: Any,
    default: Optional[float] = None,
) -> Optional[float]:
    """
    Safely convert a value to float.
    """

    if value is None:
        return default

    try:
        return round(
            float(value),
            4,
        )

    except (
        TypeError,
        ValueError,
    ):
        return default


def _safe_int(
    value: Any,
    default: int = 0,
) -> int:
    """
    Safely convert a value to int.
    """

    if value is None:
        return default

    try:
        return int(value)

    except (
        TypeError,
        ValueError,
    ):
        return default


def _safe_bool(
    value: Any,
    default: Optional[bool] = None,
) -> Optional[bool]:
    """
    Safely normalize boolean-like values.
    """

    if value is None:
        return default

    if isinstance(
        value,
        bool,
    ):
        return value

    if isinstance(
        value,
        str,
    ):
        normalized = value.strip().lower()

        if normalized in {
            "true",
            "1",
            "yes",
            "y",
            "supported",
            "useful",
            "grounded",
            "valid",
        }:
            return True

        if normalized in {
            "false",
            "0",
            "no",
            "n",
            "unsupported",
            "not useful",
            "not_useful",
            "ungrounded",
            "invalid",
        }:
            return False

    return bool(value)


# ============================================================
# DOCUMENT NORMALIZATION
# ============================================================


def _normalize_document(
    document: Any,
    source_type: str,
) -> Dict[str, Any]:
    """
    Normalize a private or web document into a stable API format.

    This prevents the frontend from breaking when individual
    retrieval services return slightly different structures.
    """

    if isinstance(
        document,
        dict,
    ):
        normalized = dict(document)

        content = (
            normalized.get("content")
            or normalized.get("text")
            or normalized.get("page_content")
            or ""
        )

        source = (
            normalized.get("source")
            or normalized.get("title")
            or normalized.get("name")
            or normalized.get("url")
            or "Unknown source"
        )

        normalized["content"] = str(content)

        normalized["source"] = str(source)

        normalized.setdefault(
            "source_type",
            source_type,
        )

        normalized.setdefault(
            "id",
            None,
        )

        normalized.setdefault(
            "url",
            None,
        )

        normalized.setdefault(
            "score",
            None,
        )

        normalized.setdefault(
            "rerank_score",
            None,
        )

        normalized.setdefault(
            "grader_score",
            None,
        )

        normalized.setdefault(
            "grader_relevant",
            None,
        )

        normalized.setdefault(
            "grader_reason",
            "",
        )

        return normalized

    return {
        "id": None,
        "content": str(document),
        "source": "Unknown source",
        "source_type": source_type,
        "url": None,
        "score": None,
        "rerank_score": None,
        "grader_score": None,
        "grader_relevant": None,
        "grader_reason": "",
    }


def _normalize_documents(
    documents: Any,
    source_type: str,
) -> List[Dict[str, Any]]:
    """
    Normalize a collection of documents.
    """

    if not documents:
        return []

    if isinstance(
        documents,
        tuple,
    ):
        documents = list(documents)

    if not isinstance(
        documents,
        list,
    ):
        documents = [documents]

    return [
        _normalize_document(
            document=document,
            source_type=source_type,
        )
        for document in documents
    ]


# ============================================================
# GROUNDING / REFLECTION NORMALIZATION
# ============================================================


def _normalize_claims(
    claims: Any,
) -> List[Dict[str, Any]]:
    """
    Normalize grounding claims for API consumers.
    """

    if not claims:
        return []

    if not isinstance(
        claims,
        list,
    ):
        claims = [claims]

    normalized: List[Dict[str, Any]] = []

    for claim in claims:
        if isinstance(
            claim,
            dict,
        ):
            item = dict(claim)

            item["text"] = str(
                item.get(
                    "text",
                    "",
                )
            )

            chunk_id = item.get("cited_chunk_id")

            if chunk_id is not None:
                try:
                    item["cited_chunk_id"] = int(chunk_id)
                except (
                    TypeError,
                    ValueError,
                ):
                    pass

            normalized.append(item)

        else:
            normalized.append(
                {
                    "text": str(claim),
                }
            )

    return normalized


def _normalize_grounding_scores(
    scores: Any,
) -> List[Dict[str, Any]]:
    """
    Normalize semantic grounding evaluation results.
    """

    if not scores:
        return []

    if not isinstance(
        scores,
        list,
    ):
        scores = [scores]

    normalized: List[Dict[str, Any]] = []

    for score in scores:
        if not isinstance(
            score,
            dict,
        ):
            normalized.append(
                {
                    "claim": str(score),
                    "score": None,
                    "supported": None,
                }
            )
            continue

        item = dict(score)

        item["claim"] = str(
            item.get(
                "claim",
                "",
            )
        )

        if "score" in item:
            item["score"] = _safe_float(
                item.get("score"),
                default=None,
            )

        if "cited_chunk_id" in item:
            try:
                item["cited_chunk_id"] = int(item["cited_chunk_id"])
            except (
                TypeError,
                ValueError,
            ):
                pass

        if "supported" in item:
            item["supported"] = _safe_bool(
                item.get("supported"),
                default=None,
            )

        normalized.append(item)

    return normalized


def _normalize_grounding_feedback(
    feedback: Any,
) -> List[str]:
    """
    Normalize revision feedback generated by the grounding critic.
    """

    if not feedback:
        return []

    if not isinstance(
        feedback,
        list,
    ):
        feedback = [feedback]

    return [str(item) for item in feedback if str(item).strip()]


# ============================================================
# LATENCY
# ============================================================


def _get_latency(
    output: Dict[str, Any],
    key: str,
) -> Optional[float]:
    """
    Read latency metadata from LangGraph output.
    """

    return _safe_float(
        output.get(key),
        default=None,
    )


# ============================================================
# CONTEXT QUALITY
# ============================================================


def _normalize_context_quality(
    value: Any,
) -> str:
    """
    Normalize context quality into a stable string.
    """

    if value is None:
        return "unknown"

    normalized = str(value).strip().lower()

    allowed_values = {
        "strong",
        "moderate",
        "weak",
        "empty",
        "needs_web",
        "blocked",
        "error",
        "unknown",
    }

    if normalized in allowed_values:
        return normalized

    return "unknown"


# ============================================================
# INVALID RESPONSE
# ============================================================


def _build_invalid_response() -> Dict[str, Any]:
    """
    Stable response for blank questions.
    """

    return {
        "question": "",
        "answer": "Please provide a question.",
        "thought_process": [],
        "status": "invalid_request",
        "sources": [],
        "private_sources": [],
        "web_sources": [],
        "search_query": None,
        "retrieval_used": False,
        "web_search_used": False,
        "context_quality": "empty",
        "citation_valid": None,
        "is_grounded": None,
        "answer_supported": None,
        "answer_useful": None,
        "support_score": None,
        "usefulness_score": None,
        "claims": [],
        "grounding_scores": [],
        "grounding_feedback": [],
        "revision_count": 0,
        "retrieval_rewrite_count": 0,
        "web_rewrite_count": 0,
        "support_retry_count": 0,
        "retrieval_latency_ms": None,
        "rerank_latency_ms": None,
        "grader_latency_ms": None,
        "generation_latency_ms": None,
        "latency_ms": 0.0,
    }


# ============================================================
# BLOCKED RESPONSE
# ============================================================


def _build_blocked_response(
    question: str,
    answer: Any,
    elapsed_ms: float,
) -> Dict[str, Any]:
    """
    Stable response for guardrail-blocked requests.
    """

    safe_answer = (
        str(answer) if answer is not None else "This request cannot be processed."
    )

    return {
        "question": question,
        "answer": safe_answer,
        "thought_process": [
            "Intent: Guardrails Fired",
            "Retrieval: Skipped",
        ],
        "status": "Blocked by guardrails.",
        "sources": [],
        "private_sources": [],
        "web_sources": [],
        "search_query": None,
        "retrieval_used": False,
        "web_search_used": False,
        "context_quality": "blocked",
        "citation_valid": None,
        "is_grounded": None,
        "answer_supported": None,
        "answer_useful": None,
        "support_score": None,
        "usefulness_score": None,
        "claims": [],
        "grounding_scores": [],
        "grounding_feedback": [],
        "revision_count": 0,
        "retrieval_rewrite_count": 0,
        "web_rewrite_count": 0,
        "support_retry_count": 0,
        "retrieval_latency_ms": None,
        "rerank_latency_ms": None,
        "grader_latency_ms": None,
        "generation_latency_ms": None,
        "latency_ms": elapsed_ms,
    }


# ============================================================
# ERROR RESPONSE
# ============================================================


def _build_error_response(
    question: str,
    elapsed_ms: float,
) -> Dict[str, Any]:
    """
    Stable error response.

    Detailed exception information is logged internally and is
    not returned to the user.
    """

    return {
        "question": question,
        "answer": (
            "I apologize, but I encountered an internal error "
            "while processing your request. Please try again later."
        ),
        "thought_process": [
            "Error encountered during execution.",
        ],
        "status": "error",
        "sources": [],
        "private_sources": [],
        "web_sources": [],
        "search_query": question,
        "retrieval_used": False,
        "web_search_used": False,
        "context_quality": "error",
        "citation_valid": None,
        "is_grounded": None,
        "answer_supported": None,
        "answer_useful": None,
        "support_score": None,
        "usefulness_score": None,
        "claims": [],
        "grounding_scores": [],
        "grounding_feedback": [],
        "revision_count": 0,
        "retrieval_rewrite_count": 0,
        "web_rewrite_count": 0,
        "support_retry_count": 0,
        "retrieval_latency_ms": None,
        "rerank_latency_ms": None,
        "grader_latency_ms": None,
        "generation_latency_ms": None,
        "latency_ms": elapsed_ms,
    }


# ============================================================
# ROOT
# ============================================================


@app.get("/")
def home() -> Dict[str, Any]:
    """
    Basic service information.
    """

    return {
        "message": ("KnowledgeMesh Enterprise Agentic RAG API is live."),
        "service": APP_NAME,
        "status": "ok",
        "version": APP_VERSION,
        "architecture": [
            "NeMo Guardrails",
            "LangGraph",
            "Qdrant Cloud",
            "FlashRank",
            "Document Grader",
            "Query Rewriter",
            "Web Search Fallback",
            "Grounding Critic",
            "Citation Check",
            "Bounded Answer Revision",
            "Portkey",
            "Conversational Memory",
        ],
    }


# ============================================================
# HEALTH
# ============================================================


@app.get("/health")
def health() -> Dict[str, Any]:
    """
    Lightweight liveness check.

    This endpoint does not call Qdrant, the LLM, or LangGraph.
    """

    return {
        "status": "ok",
        "service": APP_NAME,
        "version": APP_VERSION,
    }


# ============================================================
# READINESS
# ============================================================


@app.get("/ready")
def ready() -> Dict[str, Any]:
    """
    Readiness check for deployment platforms.
    """

    checks = {
        "api": "ok",
        "rag_agent": ("ok" if rag_agent is not None else "error"),
        "guardrails": ("ok" if _guardrails_initialized else "not_initialized"),
    }

    overall_status = (
        "ready" if all(value == "ok" for value in checks.values()) else "not_ready"
    )

    return {
        "status": overall_status,
        "service": APP_NAME,
        "version": APP_VERSION,
        "checks": checks,
    }


# ============================================================
# GRAPH
# ============================================================


@app.get(
    "/graph",
    response_model=None,
)
def graph():
    """
    Return the current LangGraph as a Mermaid-generated PNG.
    """

    try:
        png_bytes = rag_agent.get_graph().draw_mermaid_png()

        return Response(
            content=png_bytes,
            media_type="image/png",
            headers={
                "Content-Disposition": ('inline; filename="knowledgemesh-graph.png"')
            },
        )

    except Exception:
        logfire.exception(
            "❌ Could not generate graph image",
        )

        return {
            "error": ("Could not generate graph image."),
            "status": "error",
        }


# ============================================================
# QUERY
# ============================================================


@app.post("/query")
def query(
    request: QueryRequest,
) -> Dict[str, Any]:
    """
    Execute the KnowledgeMesh Agentic RAG pipeline.

    High-level flow:

        User Query
             ↓
        NeMo Guardrails
             ↓
        LangGraph Planner
             ↓
        Private Retrieval
             ↓
        FlashRank
             ↓
        Document Grader
             ↓
        Context Evaluator
             ├── strong → Responder
             ├── weak → Query Rewrite → Private Retry
             └── needs_web → Web Search
                              ↓
                           Responder
                              ↓
                       Grounding Critic
                              ↓
                        Citation Check
                              ├── PASS → END
                              └── FAIL → Revision
                                           ↓
                                       Responder
    """

    # ========================================================
    # REQUEST PREPARATION
    # ========================================================

    q = (request.q or "").strip()

    thread_id = request.thread_id.strip() if request.thread_id else "default_user"

    if not thread_id:
        thread_id = "default_user"

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    if not q:
        return _build_invalid_response()

    # ========================================================
    # REQUEST LOGGING
    # ========================================================

    start_time = time.perf_counter()

    logfire.info(
        "📥 RAG request received",
        thread_id=thread_id,
        query_length=len(q),
    )

    # ========================================================
    # INITIAL LANGGRAPH STATE
    # ========================================================

    initial_state = {
        "messages": [
            {
                "role": "user",
                "content": q,
            }
        ],
        "original_query": q,
        "current_query": q,
        "rewritten_query": "",
        "documents": [],
        "graded_documents": [],
        "web_documents": [],
        "all_documents": [],
        "merged_context": "",
        "plan": [
            "Start",
        ],
        "status": "Initializing Graph...",
        "search_query": q,
        "web_search_used": False,
        "retrieval_required": True,
        "web_search_required": False,
        "context_quality": "unknown",
        "citation_valid": None,
        "is_grounded": None,
        "answer_supported": None,
        "answer_useful": None,
        "support_score": None,
        "usefulness_score": None,
        "claims": [],
        "grounding_scores": [],
        "grounding_feedback": [],
        "retrieval_rewrite_count": 0,
        "web_rewrite_count": 0,
        "support_retry_count": 0,
        "revision_count": 0,
        "max_revisions": 2,
        "final_answer": "",
    }

    # ========================================================
    # LANGGRAPH THREAD CONFIGURATION
    # ========================================================

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    # ========================================================
    # PIPELINE EXECUTION
    # ========================================================

    try:
        # ====================================================
        # GATE 1 · NeMo Guardrails
        # ====================================================

        with logfire.span(
            "🛡️ Guardrails Check",
            thread_id=thread_id,
        ):
            rail_fired, rail_response = guard(q)

        # ----------------------------------------------------
        # BLOCKED REQUEST
        # ----------------------------------------------------

        if rail_fired:
            elapsed_ms = round(
                (time.perf_counter() - start_time) * 1000,
                2,
            )

            logfire.info(
                "🛡️ Request blocked by guardrails",
                thread_id=thread_id,
                latency_ms=elapsed_ms,
            )

            return _build_blocked_response(
                question=q,
                answer=rail_response,
                elapsed_ms=elapsed_ms,
            )

        logfire.info(
            "✅ Guardrails passed",
            thread_id=thread_id,
        )

        # ====================================================
        # GATE 2 · LANGGRAPH
        # ====================================================

        with logfire.span(
            "🧠 LangGraph Agentic RAG Pipeline",
            thread_id=thread_id,
        ):
            final_output = rag_agent.invoke(
                initial_state,
                config=config,
            )

        # ====================================================
        # DEFENSIVE EXTRACTION
        # ====================================================

        if not isinstance(
            final_output,
            dict,
        ):
            final_output = {}

        # ========================================================
        # ANSWER
        # ========================================================

        answer = (
            final_output.get("final_answer")
            or final_output.get("answer")
            or "I was unable to generate an answer."
        )

        if not isinstance(
            answer,
            str,
        ):
            answer = str(answer)

        # ========================================================
        # EXECUTION TRACE
        # ========================================================

        plan = final_output.get("plan") or []

        if not isinstance(
            plan,
            list,
        ):
            plan = [str(plan)]

        plan = [str(step) for step in plan]

        # ========================================================
        # STATUS
        # ========================================================

        status = final_output.get(
            "status",
            "completed",
        )

        if not isinstance(
            status,
            str,
        ):
            status = str(status)

        # ========================================================
        # PRIVATE DOCUMENTS
        # ========================================================

        documents = _normalize_documents(
            documents=final_output.get("documents"),
            source_type="private_kb",
        )

        # ========================================================
        # WEB DOCUMENTS
        # ========================================================

        web_documents = _normalize_documents(
            documents=final_output.get("web_documents"),
            source_type="web",
        )

        # ========================================================
        # QUERY METADATA
        # ========================================================

        search_query = (
            final_output.get("search_query") or final_output.get("current_query") or q
        )

        if not isinstance(
            search_query,
            str,
        ):
            search_query = str(search_query)

        # ========================================================
        # CONTEXT QUALITY
        # ========================================================

        context_quality = _normalize_context_quality(
            final_output.get(
                "context_quality",
                "unknown",
            )
        )

        # ========================================================
        # CITATION VALIDATION
        # ========================================================

        citation_valid = _safe_bool(
            final_output.get("citation_valid"),
            default=None,
        )

        # ========================================================
        # SEMANTIC GROUNDING
        # ========================================================

        is_grounded = _safe_bool(
            final_output.get("is_grounded"),
            default=None,
        )

        # ========================================================
        # ANSWER SUPPORT
        # ========================================================

        answer_supported = _safe_bool(
            final_output.get("answer_supported"),
            default=None,
        )

        answer_useful = _safe_bool(
            final_output.get("answer_useful"),
            default=None,
        )

        support_score = _safe_float(
            final_output.get("support_score"),
            default=None,
        )

        usefulness_score = _safe_float(
            final_output.get("usefulness_score"),
            default=None,
        )

        # ========================================================
        # CLAIMS
        # ========================================================

        claims = _normalize_claims(final_output.get("claims"))

        # ========================================================
        # GROUNDING SCORES
        # ========================================================

        grounding_scores = _normalize_grounding_scores(
            final_output.get("grounding_scores")
        )

        # ========================================================
        # GROUNDING FEEDBACK
        # ========================================================

        grounding_feedback = _normalize_grounding_feedback(
            final_output.get("grounding_feedback")
        )

        # ========================================================
        # REVISION / RETRY METADATA
        # ========================================================

        revision_count = _safe_int(
            final_output.get("revision_count"),
            default=0,
        )

        retrieval_rewrite_count = _safe_int(
            final_output.get("retrieval_rewrite_count"),
            default=0,
        )

        web_rewrite_count = _safe_int(
            final_output.get("web_rewrite_count"),
            default=0,
        )

        support_retry_count = _safe_int(
            final_output.get("support_retry_count"),
            default=0,
        )

        # ========================================================
        # WEB SEARCH
        # ========================================================

        web_search_used = bool(
            final_output.get(
                "web_search_used",
                False,
            )
            or len(web_documents) > 0
        )

        # ========================================================
        # RETRIEVAL
        # ========================================================

        retrieval_used = bool(
            len(documents) > 0
            or final_output.get(
                "retrieval_required",
                False,
            )
        )

        # ========================================================
        # PERFORMANCE
        # ========================================================

        retrieval_latency_ms = _get_latency(
            final_output,
            "retrieval_latency_ms",
        )

        rerank_latency_ms = _get_latency(
            final_output,
            "rerank_latency_ms",
        )

        grader_latency_ms = _get_latency(
            final_output,
            "grader_latency_ms",
        )

        generation_latency_ms = _get_latency(
            final_output,
            "generation_latency_ms",
        )

        elapsed_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2,
        )

        # ========================================================
        # COMBINED SOURCES
        # ========================================================

        all_sources: List[Dict[str, Any]] = [
            *documents,
            *web_documents,
        ]

        # ========================================================
        # SELF-RAG VALIDATION SUMMARY
        # ========================================================

        validation_passed = bool(citation_valid is True and is_grounded is True)

        # ========================================================
        # COMPLETION LOG
        # ========================================================

        logfire.info(
            "✅ RAG request completed",
            thread_id=thread_id,
            documents_retrieved=len(documents),
            web_documents_retrieved=len(web_documents),
            web_search_used=web_search_used,
            context_quality=context_quality,
            citation_valid=citation_valid,
            is_grounded=is_grounded,
            answer_supported=answer_supported,
            answer_useful=answer_useful,
            support_score=support_score,
            grounding_claims=len(claims),
            grounding_feedback_items=len(grounding_feedback),
            validation_passed=validation_passed,
            revision_count=revision_count,
            retrieval_rewrite_count=(retrieval_rewrite_count),
            web_rewrite_count=(web_rewrite_count),
            support_retry_count=(support_retry_count),
            retrieval_latency_ms=(retrieval_latency_ms),
            rerank_latency_ms=(rerank_latency_ms),
            grader_latency_ms=(grader_latency_ms),
            generation_latency_ms=(generation_latency_ms),
            latency_ms=elapsed_ms,
        )

        # ========================================================
        # API RESPONSE
        # ========================================================

        return {
            # ----------------------------------------------------
            # Request / answer
            # ----------------------------------------------------
            "question": q,
            "answer": answer,
            "status": status,
            # ----------------------------------------------------
            # Execution trace
            # ----------------------------------------------------
            "thought_process": plan,
            # ----------------------------------------------------
            # Sources
            # ----------------------------------------------------
            "sources": all_sources,
            "private_sources": documents,
            "web_sources": web_documents,
            # ----------------------------------------------------
            # Query / retrieval
            # ----------------------------------------------------
            "search_query": search_query,
            "retrieval_used": retrieval_used,
            "web_search_used": web_search_used,
            # ----------------------------------------------------
            # Context evaluation
            # ----------------------------------------------------
            "context_quality": context_quality,
            # ----------------------------------------------------
            # Citation validation
            # ----------------------------------------------------
            "citation_valid": citation_valid,
            # ----------------------------------------------------
            # Semantic grounding
            # ----------------------------------------------------
            "is_grounded": is_grounded,
            "claims": claims,
            "grounding_scores": grounding_scores,
            "grounding_feedback": grounding_feedback,
            # ----------------------------------------------------
            # Final answer evaluation
            # ----------------------------------------------------
            "answer_supported": answer_supported,
            "answer_useful": answer_useful,
            "support_score": support_score,
            "usefulness_score": usefulness_score,
            "validation_passed": validation_passed,
            # ----------------------------------------------------
            # Retry / revision
            # ----------------------------------------------------
            "revision_count": revision_count,
            "retrieval_rewrite_count": (retrieval_rewrite_count),
            "web_rewrite_count": (web_rewrite_count),
            "support_retry_count": (support_retry_count),
            "max_revisions": _safe_int(
                final_output.get("max_revisions"),
                default=2,
            ),
            # ----------------------------------------------------
            # Performance
            # ----------------------------------------------------
            "retrieval_latency_ms": (retrieval_latency_ms),
            "rerank_latency_ms": (rerank_latency_ms),
            "grader_latency_ms": (grader_latency_ms),
            "generation_latency_ms": (generation_latency_ms),
            "latency_ms": elapsed_ms,
        }

    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as exc:
        elapsed_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2,
        )

        logfire.exception(
            "❌ Backend execution failed",
            thread_id=thread_id,
            error_type=type(exc).__name__,
            error_message=str(exc),
            latency_ms=elapsed_ms,
        )

        return _build_error_response(
            question=q,
            elapsed_ms=elapsed_ms,
        )
