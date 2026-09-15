# ============================================================
# KnowledgeMesh · Enterprise Agentic RAG API
#
# IMPORTANT:
# logfire MUST be configured before importing application
# modules so spans from all modules are captured.
# ============================================================

import os
import time
from typing import Optional

import logfire
from dotenv import load_dotenv


# ============================================================
# ENVIRONMENT
# ============================================================

# Load .env from the project root.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(dotenv_path=ENV_PATH, override=False)


# ============================================================
# LOGFIRE
# ============================================================

# Do not configure Logfire with None.
LOGFIRE_TOKEN = os.getenv("LOGFIRE_TOKEN")

if LOGFIRE_TOKEN:
    logfire.configure(
        token=LOGFIRE_TOKEN,
    )
else:
    print(
        "⚠️ LOGFIRE_TOKEN is not configured. Continuing without Logfire cloud telemetry."
    )


# ============================================================
# APPLICATION IMPORTS
# ============================================================

from fastapi import FastAPI, Response
from pydantic import BaseModel, Field

from app.agents.graph import rag_agent
from app.guardrails import initialize_rails, guard


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="KnowledgeMesh · Enterprise Agentic RAG API",
    description=(
        "Enterprise Agentic RAG using LangGraph, "
        "Qdrant, FlashRank, Portkey, NeMo Guardrails "
        "and conversational memory."
    ),
    version="2.0.0",
)


# ============================================================
# STARTUP
# ============================================================


@app.on_event("startup")
def startup_event():
    """
    Initialise components that should be ready before
    serving requests.
    """

    logfire.info("🚀 KnowledgeMesh API startup")

    try:
        # ----------------------------------------------------
        # NeMo Guardrails
        # ----------------------------------------------------

        initialize_rails()

        logfire.info("✅ NeMo Guardrails initialised successfully")

    except Exception:
        logfire.exception("❌ Failed to initialise NeMo Guardrails")
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
# ROOT
# ============================================================


@app.get("/")
def home():
    return {
        "message": "KnowledgeMesh Enterprise Agentic RAG API is live.",
        "service": "KnowledgeMesh",
        "status": "ok",
        "version": "2.0.0",
    }


# ============================================================
# HEALTH CHECK
# ============================================================


@app.get("/health")
def health():
    """
    Lightweight liveness check.

    Does not call Qdrant, the LLM, or LangGraph.
    """

    return {
        "status": "ok",
        "service": "KnowledgeMesh",
    }


# ============================================================
# READINESS CHECK
# ============================================================


@app.get("/ready")
def ready():
    """
    Lightweight readiness check.
    """

    checks = {
        "api": "ok",
        "rag_agent": ("ok" if rag_agent is not None else "error"),
        "guardrails": "ok",
    }

    overall_status = (
        "ready" if all(value == "ok" for value in checks.values()) else "not_ready"
    )

    return {
        "status": overall_status,
        "service": "KnowledgeMesh",
        "checks": checks,
    }


# ============================================================
# GRAPH VISUALISATION
# ============================================================


@app.get("/graph")
def get_graph_image():
    """
    Return the current LangGraph as a Mermaid-generated PNG.
    """

    try:
        png_bytes = rag_agent.get_graph().draw_mermaid_png()

        return Response(
            content=png_bytes,
            media_type="image/png",
        )

    except Exception as exc:
        logfire.exception("❌ Could not generate graph image")

        return {"error": (f"Could not generate graph image: {exc}")}


# ============================================================
# RAG QUERY
# ============================================================


@app.post("/query")
def query(request: QueryRequest):
    """
    Execute the KnowledgeMesh Agentic RAG pipeline.

    Current flow:

        User Query
             ↓
        NeMo Guardrails
             ↓
        LangGraph Planner
             ↓
        ┌─────────────────────────┐
        │                         │
        ▼                         ▼
    Conversation            Private Retrieval
        │                         │
        │                         ▼
        │                     Qdrant
        │                         │
        │                     FlashRank
        │                         │
        └─────────────┬───────────┘
                      ▼
                  Responder
                      ↓
                 Final Answer

    Future agentic stages can add:

        Document Grader
             ↓
        Query Rewrite
             ↓
        Private KB Retry
             ↓
        Web Search
             ↓
        Web Evidence Grader
             ↓
        IsSUP
             ↓
        IsUSE
             ↓
        Bounded Revision
    """

    # ========================================================
    # REQUEST PREPARATION
    # ========================================================

    q = request.q.strip()

    thread_id = request.thread_id.strip() if request.thread_id else "default_user"

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    if not q:
        return {
            "question": "",
            "answer": "Please provide a question.",
            "thought_process": [],
            "status": "invalid_request",
            "sources": [],
            "search_query": None,
            "web_search_used": False,
            "context_quality": None,
            "answer_supported": None,
            "answer_useful": None,
        }

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
        "web_documents": [],
        "plan": [
            "Start",
        ],
        "status": "Initializing Graph...",
        "search_query": q,
        "web_search_used": False,
        "retrieval_required": True,
        "web_search_required": False,
        "context_quality": "unknown",
        "answer_supported": False,
        "answer_useful": False,
        "support_score": 0.0,
        "usefulness_score": 0.0,
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
            logfire.info(
                "🛡️ Request blocked by guardrails",
                thread_id=thread_id,
            )

            elapsed_ms = round(
                (time.perf_counter() - start_time) * 1000,
                2,
            )

            return {
                "question": q,
                "answer": rail_response,
                "thought_process": [
                    "Intent: Guardrails Fired",
                    "Retrieval: Skipped",
                ],
                "status": "Blocked by guardrails.",
                "sources": [],
                "search_query": None,
                "web_search_used": False,
                "context_quality": None,
                "answer_supported": None,
                "answer_useful": None,
                "support_score": None,
                "usefulness_score": None,
                "revision_count": 0,
                "latency_ms": elapsed_ms,
            }

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

        if final_output is None:
            final_output = {}

        # ----------------------------------------------------
        # Answer
        # ----------------------------------------------------

        answer = (
            final_output.get("final_answer")
            or final_output.get("answer")
            or "I was unable to generate an answer."
        )

        # ----------------------------------------------------
        # Reasoning / execution trace
        # ----------------------------------------------------

        plan = final_output.get("plan") or []

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        status = final_output.get(
            "status",
            "completed",
        )

        # ----------------------------------------------------
        # Private KB documents
        # ----------------------------------------------------

        documents = final_output.get("documents") or []

        # ----------------------------------------------------
        # Web documents
        # ----------------------------------------------------

        web_documents = final_output.get("web_documents") or []

        # ----------------------------------------------------
        # Query metadata
        # ----------------------------------------------------

        search_query = (
            final_output.get("search_query") or final_output.get("current_query") or q
        )

        # ----------------------------------------------------
        # Agentic evaluation metadata
        # ----------------------------------------------------

        context_quality = final_output.get(
            "context_quality",
            "unknown",
        )

        answer_supported = final_output.get(
            "answer_supported",
            None,
        )

        answer_useful = final_output.get(
            "answer_useful",
            None,
        )

        support_score = final_output.get(
            "support_score",
            None,
        )

        usefulness_score = final_output.get(
            "usefulness_score",
            None,
        )

        revision_count = final_output.get(
            "revision_count",
            0,
        )

        # ----------------------------------------------------
        # Web-search metadata
        # ----------------------------------------------------

        web_search_used = bool(
            final_output.get(
                "web_search_used",
                False,
            )
            or len(web_documents) > 0
        )

        # ----------------------------------------------------
        # Combined sources
        # ----------------------------------------------------

        all_sources = []

        for document in documents:
            if isinstance(document, dict):
                document.setdefault(
                    "source_type",
                    "private_kb",
                )

                all_sources.append(document)

            else:
                all_sources.append(
                    {
                        "content": str(document),
                        "source_type": "private_kb",
                    }
                )

        for document in web_documents:
            if isinstance(document, dict):
                document.setdefault(
                    "source_type",
                    "web",
                )

                all_sources.append(document)

            else:
                all_sources.append(
                    {
                        "content": str(document),
                        "source_type": "web",
                    }
                )

        # ====================================================
        # PERFORMANCE
        # ====================================================

        elapsed_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2,
        )

        # ====================================================
        # COMPLETION LOG
        # ====================================================

        logfire.info(
            "✅ RAG request completed",
            thread_id=thread_id,
            documents_retrieved=len(documents),
            web_documents_retrieved=len(web_documents),
            web_search_used=web_search_used,
            context_quality=context_quality,
            answer_supported=answer_supported,
            answer_useful=answer_useful,
            revision_count=revision_count,
            latency_ms=elapsed_ms,
        )

        # ====================================================
        # API RESPONSE
        # ====================================================

        return {
            "question": q,
            "answer": answer,
            "thought_process": plan,
            "status": status,
            # All sources for backwards compatibility
            "sources": all_sources,
            # Explicit source separation
            "private_sources": documents,
            "web_sources": web_documents,
            # Retrieval metadata
            "search_query": search_query,
            "retrieval_used": (len(documents) > 0),
            "web_search_used": web_search_used,
            # Agentic evaluation
            "context_quality": context_quality,
            "answer_supported": answer_supported,
            "answer_useful": answer_useful,
            "support_score": support_score,
            "usefulness_score": usefulness_score,
            # Revision loop metadata
            "revision_count": revision_count,
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
            "❌ Backend Execution Failed",
            thread_id=thread_id,
            error_type=type(exc).__name__,
            latency_ms=elapsed_ms,
        )

        return {
            "question": q,
            "answer": (
                "I apologize, but I encountered an internal "
                "error while processing your request. "
                "Please try again later."
            ),
            "thought_process": ["Error encountered during execution."],
            "status": "error",
            "sources": [],
            "private_sources": [],
            "web_sources": [],
            "search_query": q,
            "retrieval_used": False,
            "web_search_used": False,
            "context_quality": "error",
            "answer_supported": None,
            "answer_useful": None,
            "support_score": None,
            "usefulness_score": None,
            "revision_count": 0,
            "latency_ms": elapsed_ms,
        }
