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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ENV_PATH = os.path.join(
    BASE_DIR,
    ".env",
)

from dotenv import load_dotenv


load_dotenv(
    dotenv_path=ENV_PATH,
    override=False,
)


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

APP_VERSION = "2.3.2"


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
        "canonical citation provenance, "
        "candidate-answer preservation, "
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
# ANSWER NORMALIZATION
# ============================================================


def _normalize_answer(
    value: Any,
) -> str:
    """
    Normalize a LangGraph answer into a safe string.

    Empty strings are preserved as empty strings rather than
    being replaced with an error message.
    """

    if value is None:
        return ""

    if isinstance(
        value,
        str,
    ):
        return value.strip()

    return str(value).strip()


def _select_best_answer(
    final_output: Dict[str, Any],
) -> tuple[str, str]:
    """
    Select the best available generated answer.

    Priority:

        final_answer
            ↓
        answer
            ↓
        candidate_answer
            ↓
        previous_answer

    Returns:

        (answer, answer_source)

    The API must distinguish:

        1. no answer was generated
        2. an answer was generated but validation failed

    Validation failure must NOT erase the generated answer.
    """

    candidates = [
        (
            "final_answer",
            final_output.get("final_answer"),
        ),
        (
            "answer",
            final_output.get("answer"),
        ),
        (
            "candidate_answer",
            final_output.get("candidate_answer"),
        ),
        (
            "previous_answer",
            final_output.get("previous_answer"),
        ),
    ]

    for source, value in candidates:
        answer = _normalize_answer(value)

        if answer:
            return answer, source

    return "", "none"


# ============================================================
# DOCUMENT NORMALIZATION
# ============================================================


def _normalize_document(
    document: Any,
    source_type: str,
) -> Dict[str, Any]:
    """
    Normalize a private or web document into a stable API format.
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
            "document_id",
            None,
        )

        normalized.setdefault(
            "chunk_id",
            None,
        )

        normalized.setdefault(
            "total_chunks",
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
        "document_id": None,
        "chunk_id": None,
        "total_chunks": None,
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
# CITATION PROVENANCE NORMALIZATION
# ============================================================


def _normalize_citation_provenance(
    provenance: Any,
) -> Dict[str, Dict[str, Any]]:
    """
    Normalize canonical citation provenance.
    """

    if not provenance:
        return {}

    if not isinstance(
        provenance,
        dict,
    ):
        return {}

    normalized: Dict[str, Dict[str, Any]] = {}

    for citation_id, item in provenance.items():
        citation_key = str(citation_id).strip()

        if not citation_key:
            continue

        if not isinstance(
            item,
            dict,
        ):
            continue

        normalized_item = dict(item)

        normalized_item["citation_id"] = str(
            normalized_item.get(
                "citation_id",
                citation_key,
            )
        )

        point_id = normalized_item.get("point_id")

        normalized_item["point_id"] = (
            str(point_id).strip()
            if point_id is not None and str(point_id).strip()
            else None
        )

        document_id = normalized_item.get("document_id")

        normalized_item["document_id"] = (
            str(document_id).strip()
            if document_id is not None and str(document_id).strip()
            else None
        )

        chunk_id = normalized_item.get("chunk_id")

        normalized_item["chunk_id"] = (
            _safe_int(
                chunk_id,
                default=0,
            )
            if chunk_id is not None
            else None
        )

        total_chunks = normalized_item.get("total_chunks")

        normalized_item["total_chunks"] = (
            _safe_int(
                total_chunks,
                default=0,
            )
            if total_chunks is not None
            else None
        )

        normalized_item["source"] = str(
            normalized_item.get(
                "source",
                "Unknown source",
            )
        )

        normalized_item["source_type"] = str(
            normalized_item.get(
                "source_type",
                "unknown",
            )
        )

        normalized[citation_key] = normalized_item

    return normalized


# ============================================================
# GROUNDING / REFLECTION NORMALIZATION
# ============================================================


def _normalize_claims(
    claims: Any,
    *,
    citation_provenance: Any = None,
) -> list[dict[str, Any]]:
    """
    Normalize grounding claims into the public API contract.

    Citation namespaces are deliberately separated:

        citations
            LLM citation identities, e.g. ["chunk_1", "chunk_2"]

        cited_chunk_id
            Primary LLM citation identity, e.g. "chunk_1"

        cited_chunk_ids
            Canonical private evidence identities,
            e.g. ["document_id::3", "document_id::0"]

        cited_chunk_key
            Primary canonical private evidence identity,
            e.g. "document_id::3"

    IMPORTANT:

    The evaluator's citation metrics operate on the LLM citation
    namespace. Therefore `cited_chunk_id` MUST remain `chunk_N`
    and MUST NOT contain `document_id::chunk_id`.
    """

    if not isinstance(
        claims,
        list,
    ):
        return []

    provenance = (
        citation_provenance
        if isinstance(
            citation_provenance,
            dict,
        )
        else {}
    )

    normalized: list[dict[str, Any]] = []

    for raw_claim in claims:
        if not isinstance(
            raw_claim,
            dict,
        ):
            continue

        claim = dict(raw_claim)

        raw_citations = claim.get(
            "citations",
            [],
        )

        if isinstance(
            raw_citations,
            str,
        ):
            raw_citations = [raw_citations]

        if not isinstance(
            raw_citations,
            list,
        ):
            raw_citations = []

        citation_ids: list[str] = []

        cited_chunk_keys: list[str] = []

        for raw_citation in raw_citations:
            if raw_citation is None:
                continue

            citation_id = str(raw_citation).strip()

            if not citation_id:
                continue

            # ------------------------------------------------
            # LLM citation namespace
            # ------------------------------------------------

            if citation_id not in citation_ids:
                citation_ids.append(citation_id)

            # ------------------------------------------------
            # Canonical evidence namespace
            # ------------------------------------------------

            provenance_item = provenance.get(
                citation_id,
                {},
            )

            if not isinstance(
                provenance_item,
                dict,
            ):
                continue

            document_id = provenance_item.get("document_id")

            chunk_id = provenance_item.get("chunk_id")

            # Private / Qdrant evidence.
            if document_id is not None and chunk_id is not None:
                canonical_key = f"{document_id}::{chunk_id}"

                if canonical_key not in cited_chunk_keys:
                    cited_chunk_keys.append(canonical_key)

                continue

            # Web evidence intentionally does not receive a
            # fabricated Qdrant key.
            #
            # Its canonical identity remains the citation ID.
            provenance_citation_id = provenance_item.get("citation_id")

            web_key = str(provenance_citation_id or citation_id).strip()

            if web_key and web_key not in cited_chunk_keys:
                cited_chunk_keys.append(web_key)

        # ----------------------------------------------------
        # Public LLM citation namespace
        # ----------------------------------------------------

        claim["citations"] = citation_ids

        claim["cited_chunk_id"] = citation_ids[0] if citation_ids else None

        # ----------------------------------------------------
        # Canonical evidence namespace
        # ----------------------------------------------------

        claim["cited_chunk_ids"] = cited_chunk_keys

        claim["cited_chunk_key"] = cited_chunk_keys[0] if cited_chunk_keys else None

        normalized.append(claim)

    return normalized


def _normalize_grounding_scores(
    scores: Any,
) -> List[Dict[str, Any]]:
    """
    Normalize grounding results into a flat evaluator-compatible list.

    Canonical Grounding Critic schema:

        {
            "atomic_claims": [...],
            "claims": [...],
            "unsupported_claims": [...],
            "uncited_claims": [...],
            "invalid_citations": [...],
            "entailment_threshold": 0.50,
            ...
        }

    The evaluator expects:

        grounding_scores = [
            {
                "claim": "...",
                "score": 0.91,
                "supported": True,
                "citations": [...],
                ...
            }
        ]

    Therefore only the claim-level results are flattened here.

    The complete canonical object is preserved separately as
    `grounding_details`.
    """

    if not scores:
        return []

    # ------------------------------------------------------------
    # Canonical Grounding Critic dictionary
    # ------------------------------------------------------------

    if isinstance(
        scores,
        dict,
    ):
        claim_results = scores.get("claims")

        if isinstance(
            claim_results,
            list,
        ):
            scores = claim_results

        else:
            return []

    # ------------------------------------------------------------
    # Legacy list schema
    # ------------------------------------------------------------

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
                    "citations": [],
                    "atomic_claims": [],
                }
            )

            continue

        item = dict(score)

        # --------------------------------------------------------
        # Claim text
        # --------------------------------------------------------

        item["claim"] = str(
            item.get(
                "claim",
                item.get(
                    "text",
                    "",
                ),
            )
        )

        # --------------------------------------------------------
        # Score
        # --------------------------------------------------------

        if "score" in item:
            item["score"] = _safe_float(
                item.get("score"),
                default=None,
            )

        # --------------------------------------------------------
        # Support status
        # --------------------------------------------------------

        if "supported" in item:
            item["supported"] = _safe_bool(
                item.get("supported"),
                default=None,
            )

        # --------------------------------------------------------
        # Citations
        #
        # Keep LLM citation IDs unchanged.
        # --------------------------------------------------------

        citations = item.get(
            "citations",
            [],
        )

        if isinstance(
            citations,
            str,
        ):
            citations = [citations]

        if not isinstance(
            citations,
            list,
        ):
            citations = []

        normalized_citations: list[str] = []

        for citation in citations:
            if citation is None:
                continue

            citation_id = str(citation).strip()

            if citation_id and citation_id not in normalized_citations:
                normalized_citations.append(citation_id)

        item["citations"] = normalized_citations

        # --------------------------------------------------------
        # Atomic claims
        # --------------------------------------------------------

        atomic_claims = item.get("atomic_claims")

        if not isinstance(
            atomic_claims,
            list,
        ):
            atomic_claims = []

        item["atomic_claims"] = atomic_claims

        normalized.append(item)

    return normalized


def _extract_grounding_details(
    scores: Any,
) -> Dict[str, Any]:
    """
    Preserve the canonical Grounding Critic structure.

    This prevents the API normalization layer from destroying the
    richer grounding metadata required for debugging, UI inspection,
    and future evaluation analysis.
    """

    if not isinstance(
        scores,
        dict,
    ):
        return {
            "atomic_claims": [],
            "claims": [],
            "unsupported_claims": [],
            "uncited_claims": [],
            "invalid_citations": [],
            "entailment_threshold": None,
            "claim_count": 0,
            "atomic_claim_count": 0,
            "unsupported_atomic_count": 0,
            "available_citation_count": 0,
        }

    details = dict(scores)

    for key in (
        "atomic_claims",
        "claims",
        "unsupported_claims",
        "uncited_claims",
        "invalid_citations",
    ):
        value = details.get(key)

        if not isinstance(
            value,
            list,
        ):
            details[key] = []

    details["claim_count"] = _safe_int(
        details.get("claim_count"),
        default=len(details["claims"]),
    )

    details["atomic_claim_count"] = _safe_int(
        details.get("atomic_claim_count"),
        default=len(details["atomic_claims"]),
    )

    details["unsupported_atomic_count"] = _safe_int(
        details.get("unsupported_atomic_count"),
        default=len(details["unsupported_claims"]),
    )

    details["available_citation_count"] = _safe_int(
        details.get("available_citation_count"),
        default=0,
    )

    details["entailment_threshold"] = _safe_float(
        details.get("entailment_threshold"),
        default=None,
    )

    return details


def _normalize_grounding_feedback(
    feedback: Any,
) -> List[str]:
    """
    Normalize revision feedback.
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
    Read latency metadata.
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

    if value is None:
        return "unknown"

    normalized = str(value).strip().lower()

    allowed_values = {
        "strong",
        "moderate",
        "weak",
        "empty",
        "needs_web",
        "web_sufficient",
        "web_empty",
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

    return {
        "question": "",
        "answer": "Please provide a question.",
        "candidate_answer": "",
        "answer_available": False,
        "answer_source": "none",
        "validation_status": "invalid_request",
        "thought_process": [],
        "status": "invalid_request",
        "sources": [],
        "private_sources": [],
        "answer_sources": [],
        "retrieved_private_sources": [],
        "web_sources": [],
        "citations": {},
        "citation_provenance": {},
        "search_query": None,
        "retrieval_used": False,
        "web_search_used": False,
        "should_search_web": False,
        "context_quality": "empty",
        "context_reason": "No question was provided.",
        "citation_valid": None,
        "is_grounded": None,
        "answer_supported": None,
        "answer_useful": None,
        "support_score": None,
        "usefulness_score": None,
        "claims": [],
        "grounding_scores": [],
        "grounding_feedback": [],
        "grounding_details": {},
        "validation_passed": False,
        "revision_count": 0,
        "retrieval_rewrite_count": 0,
        "web_rewrite_count": 0,
        "support_retry_count": 0,
        "max_revisions": 0,
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

    safe_answer = (
        str(answer) if answer is not None else "This request cannot be processed."
    )

    return {
        "question": question,
        "answer": safe_answer,
        "candidate_answer": safe_answer,
        "answer_available": bool(safe_answer.strip()),
        "answer_source": "guardrails",
        "validation_status": "blocked",
        "thought_process": [
            "Intent: Guardrails Fired",
            "Retrieval: Skipped",
        ],
        "status": "Blocked by guardrails.",
        "sources": [],
        "private_sources": [],
        "answer_sources": [],
        "retrieved_private_sources": [],
        "web_sources": [],
        "citations": {},
        "citation_provenance": {},
        "search_query": None,
        "retrieval_used": False,
        "web_search_used": False,
        "should_search_web": False,
        "context_quality": "blocked",
        "context_reason": ("The request was blocked by NeMo Guardrails."),
        "citation_valid": None,
        "is_grounded": None,
        "answer_supported": None,
        "answer_useful": None,
        "support_score": None,
        "usefulness_score": None,
        "claims": [],
        "grounding_scores": [],
        "grounding_feedback": [],
        "grounding_details": {},
        "validation_passed": False,
        "revision_count": 0,
        "retrieval_rewrite_count": 0,
        "web_rewrite_count": 0,
        "support_retry_count": 0,
        "max_revisions": 0,
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

    answer = (
        "I apologize, but I encountered an internal error "
        "while processing your request. Please try again later."
    )

    return {
        "question": question,
        "answer": answer,
        "candidate_answer": "",
        "answer_available": False,
        "answer_source": "error",
        "validation_status": "error",
        "thought_process": [
            "Error encountered during execution.",
        ],
        "status": "error",
        "sources": [],
        "private_sources": [],
        "answer_sources": [],
        "retrieved_private_sources": [],
        "web_sources": [],
        "citations": {},
        "citation_provenance": {},
        "search_query": question,
        "retrieval_used": False,
        "web_search_used": False,
        "should_search_web": False,
        "context_quality": "error",
        "context_reason": ("An internal backend error occurred."),
        "citation_valid": None,
        "is_grounded": None,
        "answer_supported": None,
        "answer_useful": None,
        "support_score": None,
        "usefulness_score": None,
        "claims": [],
        "grounding_scores": [],
        "grounding_details": {},
        "grounding_feedback": [],
        "validation_passed": False,
        "revision_count": 0,
        "retrieval_rewrite_count": 0,
        "web_rewrite_count": 0,
        "support_retry_count": 0,
        "max_revisions": 0,
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
            "Citation Check",
            "Grounding Critic",
            "Canonical Citation Provenance",
            "Bounded Answer Revision",
            "Candidate Answer Preservation",
            "Portkey",
            "Conversational Memory",
        ],
    }


# ============================================================
# HEALTH
# ============================================================


@app.get("/health")
def health() -> Dict[str, Any]:

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

    start_time = time.perf_counter()

    logfire.info(
        "📥 RAG request received",
        thread_id=thread_id,
        query_length=len(q),
    )

    # ========================================================
    # INITIAL STATE
    # ========================================================

    initial_state = {
        # ----------------------------------------------------
        # Conversation
        # ----------------------------------------------------
        "messages": [
            {
                "role": "user",
                "content": q,
            }
        ],
        # ----------------------------------------------------
        # Query state
        # ----------------------------------------------------
        "original_query": q,
        "current_query": q,
        "rewritten_query": "",
        "search_query": q,
        # ----------------------------------------------------
        # Retrieval state
        # ----------------------------------------------------
        "retrieval_required": True,
        "documents": [],
        "private_documents": [],
        "graded_documents": [],
        "generation_documents": [],
        # ----------------------------------------------------
        # Web state
        # ----------------------------------------------------
        "web_documents": [],
        "all_documents": [],
        "web_search_required": False,
        "web_search_attempted": False,
        "web_search_used": False,
        "should_search_web": False,
        # ----------------------------------------------------
        # Context
        # ----------------------------------------------------
        "merged_context": "",
        "context_quality": "unknown",
        "context_reason": "",
        # ----------------------------------------------------
        # Citation state
        # ----------------------------------------------------
        "citation_provenance": {},
        "citation_valid": None,
        "citation_feedback": [],
        "citation_errors": [],
        # ----------------------------------------------------
        # Grounding / evaluation
        # ----------------------------------------------------
        "is_grounded": None,
        "answer_supported": None,
        "answer_useful": None,
        "support_score": None,
        "usefulness_score": None,
        "claims": [],
        "grounding_scores": [],
        "grounding_details": {},
        "grounding_feedback": [],
        # ----------------------------------------------------
        # Revision / retry state
        # ----------------------------------------------------
        "revision_requested": False,
        "revision_count": 0,
        "max_revisions": 2,
        "retrieval_rewrite_count": 0,
        "web_rewrite_count": 0,
        "support_retry_count": 0,
        # ----------------------------------------------------
        # Answer state
        # ----------------------------------------------------
        "final_answer": "",
        "candidate_answer": "",
        # ----------------------------------------------------
        # Execution
        # ----------------------------------------------------
        "plan": ["Start"],
        "status": "Initializing Graph...",
    }

    # ========================================================
    # THREAD CONFIG
    # ========================================================

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    # ========================================================
    # PIPELINE
    # ========================================================

    try:
        # ====================================================
        # GATE 1 · GUARDRAILS
        # ====================================================

        with logfire.span(
            "🛡️ Guardrails Check",
            thread_id=thread_id,
        ):
            rail_fired, rail_response = guard(q)

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

        if not isinstance(
            final_output,
            dict,
        ):
            final_output = {}

        # ====================================================
        # GROUNDING DEBUG LOGGING
        # ====================================================

        logger.info(
            "GROUNDING DEBUG | answer_supported=%s | "
            "answer_useful=%s | support_score=%s | usefulness_score=%s",
            final_output.get("answer_supported"),
            final_output.get("answer_useful"),
            final_output.get("support_score"),
            final_output.get("usefulness_score"),
        )

        logger.info(
            "GROUNDING SCORES | %s",
            final_output.get("grounding_scores", []),
        )

        logger.info(
            "GROUNDING DETAILS | %s",
            final_output.get("grounding_details", {}),
        )

        logger.info(
            "GROUNDING FEEDBACK | %s",
            final_output.get("grounding_feedback", []),
        )

        # ====================================================
        # FINAL STATE RETRIEVAL DIAGNOSTICS
        # ====================================================

        raw_documents = final_output.get("documents") or []

        raw_all_documents = final_output.get("all_documents") or []

        raw_graded_documents = final_output.get("graded_documents") or []

        logger.info(
            (
                "Final retrieval state | "
                "documents=%d | "
                "all_documents=%d | "
                "graded_documents=%d"
            ),
            len(raw_documents),
            len(raw_all_documents),
            len(raw_graded_documents),
        )

        logfire.info(
            "🔎 Final retrieval state",
            documents=len(raw_documents),
            all_documents=len(raw_all_documents),
            graded_documents=len(raw_graded_documents),
        )

        # ====================================================
        # ANSWER SELECTION
        # ====================================================

        answer, answer_source = _select_best_answer(final_output)

        candidate_answer = _normalize_answer(final_output.get("candidate_answer"))

        if not candidate_answer:
            candidate_answer = answer

        answer_available = bool(answer.strip())

        # ====================================================
        # EXECUTION TRACE
        # ====================================================

        plan = final_output.get("plan") or []

        if not isinstance(
            plan,
            list,
        ):
            plan = [str(plan)]

        plan = [str(step) for step in plan]

        # ====================================================
        # STATUS
        # ====================================================

        status = final_output.get(
            "status",
            "completed",
        )

        if not isinstance(
            status,
            str,
        ):
            status = str(status)

        # ====================================================
        # ANSWER / GENERATION DOCUMENTS
        # ====================================================

        answer_sources = _normalize_documents(
            documents=raw_documents,
            source_type="private_kb",
        )

        # ====================================================
        # RETRIEVED PRIVATE DOCUMENTS
        # ====================================================

        retrieval_documents_raw = (
            raw_all_documents if raw_all_documents else raw_documents
        )

        private_documents_only = [
            doc
            for doc in retrieval_documents_raw
            if isinstance(doc, dict) and (doc.get("source_type") or "").lower() != "web"
        ]

        private_sources = _normalize_documents(
            documents=private_documents_only,
            source_type="private_kb",
        )

        # ====================================================
        # WEB DOCUMENTS
        # ====================================================

        web_documents = _normalize_documents(
            documents=final_output.get("web_documents"),
            source_type="web",
        )

        # ====================================================
        # CITATION PROVENANCE
        # ====================================================

        citation_provenance = _normalize_citation_provenance(
            final_output.get("citation_provenance")
        )

        citations = dict(citation_provenance)

        # ====================================================
        # QUERY METADATA
        # ====================================================

        search_query = (
            final_output.get("search_query") or final_output.get("current_query") or q
        )

        if not isinstance(
            search_query,
            str,
        ):
            search_query = str(search_query)

        # ====================================================
        # CONTEXT QUALITY
        # ====================================================

        context_quality = _normalize_context_quality(
            final_output.get(
                "context_quality",
                "unknown",
            )
        )

        context_reason = str(
            final_output.get(
                "context_reason",
                "",
            )
            or ""
        )

        # ====================================================
        # WEB SEARCH
        # ====================================================

        should_search_web = bool(
            final_output.get(
                "should_search_web",
                False,
            )
        )

        web_search_used = bool(
            final_output.get(
                "web_search_used",
                False,
            )
            or len(web_documents) > 0
        )

        # ====================================================
        # CITATION VALIDATION
        # ====================================================

        citation_valid = _safe_bool(
            final_output.get("citation_valid"),
            default=None,
        )

        # ====================================================
        # GROUNDING
        # ====================================================

        is_grounded = _safe_bool(
            final_output.get("is_grounded"),
            default=None,
        )

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

        # ====================================================
        # GROUNDING DETAILS
        # ====================================================

        raw_grounding_scores = final_output.get("grounding_scores")

        grounding_details = _extract_grounding_details(raw_grounding_scores)

        # ====================================================
        # CLAIMS
        # ====================================================

        canonical_claims = grounding_details.get(
            "claims",
            [],
        )

        if canonical_claims:
            claims = _normalize_claims(
                canonical_claims,
                citation_provenance=citation_provenance,
            )

        else:
            claims = _normalize_claims(
                final_output.get("claims"),
                citation_provenance=citation_provenance,
            )

        # ====================================================
        # GROUNDING SCORES
        # ====================================================

        grounding_scores = _normalize_grounding_scores(raw_grounding_scores)

        # ====================================================
        # GROUNDING FEEDBACK
        # ====================================================

        grounding_feedback = _normalize_grounding_feedback(
            final_output.get("grounding_feedback")
        )

        # ====================================================
        # REVISION METADATA
        # ====================================================

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

        max_revisions = _safe_int(
            final_output.get("max_revisions"),
            default=2,
        )

        # ====================================================
        # RETRIEVAL
        # ====================================================

        retrieval_used = bool(
            len(private_sources) > 0
            or final_output.get(
                "retrieval_required",
                False,
            )
        )

        # ====================================================
        # PERFORMANCE
        # ====================================================

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

        # ====================================================
        # SOURCES
        # ====================================================

        all_sources: List[Dict[str, Any]] = [
            *private_sources,
            *web_documents,
        ]

        # ====================================================
        # VALIDATION
        # ====================================================

        validation_passed = bool(citation_valid is True and is_grounded is True)

        # ====================================================
        # VALIDATION STATUS
        # ====================================================

        if not answer_available:
            validation_status = "generation_failed"

        elif validation_passed:
            validation_status = "validated"

        elif citation_valid is False:
            validation_status = "citation_validation_failed"

        elif is_grounded is False:
            validation_status = "grounding_validation_failed"

        else:
            validation_status = "validation_incomplete"

        # ====================================================
        # FINAL STATUS
        # ====================================================

        if answer_available and not validation_passed and status == "completed":
            status = "completed_with_validation_failure"

        # ====================================================
        # COMPLETION LOG
        # ====================================================

        logfire.info(
            "✅ RAG request completed",
            thread_id=thread_id,
            answer_available=answer_available,
            answer_source=answer_source,
            retrieved_private_documents=len(private_sources),
            answer_documents=len(answer_sources),
            web_documents_retrieved=len(web_documents),
            citations=len(citation_provenance),
            web_search_used=web_search_used,
            should_search_web=should_search_web,
            context_quality=context_quality,
            context_reason=context_reason,
            citation_valid=citation_valid,
            is_grounded=is_grounded,
            answer_supported=answer_supported,
            answer_useful=answer_useful,
            support_score=support_score,
            grounding_claims=len(claims),
            grounding_atomic_claims=_safe_int(
                grounding_details.get(
                    "atomic_claim_count",
                    0,
                ),
                default=0,
            ),
            grounding_unsupported_atomic_claims=_safe_int(
                grounding_details.get(
                    "unsupported_atomic_count",
                    0,
                ),
                default=0,
            ),
            grounding_feedback_items=len(grounding_feedback),
            validation_passed=validation_passed,
            validation_status=validation_status,
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

        # ====================================================
        # API RESPONSE
        # ====================================================

        return {
            # ------------------------------------------------
            # Request / answer
            # ------------------------------------------------
            "question": q,
            "answer": answer,
            "candidate_answer": candidate_answer,
            "answer_available": answer_available,
            "answer_source": answer_source,
            "status": status,
            "validation_status": validation_status,
            # ------------------------------------------------
            # Execution trace
            # ------------------------------------------------
            "thought_process": plan,
            # ------------------------------------------------
            # Sources
            # ------------------------------------------------
            "sources": all_sources,
            "private_sources": private_sources,
            "answer_sources": answer_sources,
            "retrieved_private_sources": private_sources,
            "web_sources": web_documents,
            "generation_documents": final_output.get("generation_documents") or [],
            "graded_documents": raw_graded_documents,
            "grader_status": final_output.get("grader_status"),
            "grading_mode": final_output.get("grading_mode"),
            "grader_error_type": final_output.get("grader_error_type"),
            # ------------------------------------------------
            # Citation provenance
            # ------------------------------------------------
            "citations": citations,
            "citation_provenance": citation_provenance,
            # ------------------------------------------------
            # Query / retrieval
            # ------------------------------------------------
            "search_query": search_query,
            "retrieval_used": retrieval_used,
            "web_search_used": web_search_used,
            "should_search_web": should_search_web,
            # ------------------------------------------------
            # Context
            # ------------------------------------------------
            "context_quality": context_quality,
            "context_reason": context_reason,
            # ------------------------------------------------
            # Citation validation
            # ------------------------------------------------
            "citation_valid": citation_valid,
            # ------------------------------------------------
            # Grounding
            # ------------------------------------------------
            "is_grounded": is_grounded,
            "claims": claims,
            "grounding_scores": grounding_scores,
            "grounding_details": grounding_details,
            "grounding_feedback": grounding_feedback,
            # ------------------------------------------------
            # Answer evaluation
            # ------------------------------------------------
            "answer_supported": answer_supported,
            "answer_useful": answer_useful,
            "support_score": support_score,
            "usefulness_score": usefulness_score,
            "validation_passed": validation_passed,
            # ------------------------------------------------
            # Retry / revision
            # ------------------------------------------------
            "revision_count": revision_count,
            "retrieval_rewrite_count": (retrieval_rewrite_count),
            "web_rewrite_count": (web_rewrite_count),
            "support_retry_count": (support_retry_count),
            "max_revisions": max_revisions,
            # ------------------------------------------------
            # Performance
            # ------------------------------------------------
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
