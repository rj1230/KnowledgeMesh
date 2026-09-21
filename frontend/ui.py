"""
KnowledgeMesh · Agentic RAG Console

Professional Streamlit frontend for the KnowledgeMesh FastAPI backend.

Backend endpoint:
    POST http://localhost:8000/query

Request:
    {"q": "...", "thread_id": "..."}

The interface visualizes an agentic-RAG execution path:
Guardrails -> Planner -> Private Retrieval -> Reranking -> Grading ->
Web Fallback -> Response Synthesis -> Citation Validation -> Grounding Critic.
"""

from contextlib import nullcontext
from datetime import datetime
import html
import os
import re
import textwrap
import time
import uuid

import logfire
import requests
import streamlit as st
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ============================================================
# ENVIRONMENT
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")

if not os.path.exists(ENV_PATH):
    parent_env = os.path.join(os.path.dirname(PROJECT_ROOT), ".env")
    if os.path.exists(parent_env):
        ENV_PATH = parent_env

load_dotenv(dotenv_path=ENV_PATH, override=False)

DEFAULT_BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

BACKEND_TIMEOUT_SECONDS = int(os.getenv("BACKEND_TIMEOUT_SECONDS", "180"))


# ============================================================
# LOGFIRE
# ============================================================

LOGFIRE_OK = False
LOGFIRE_ERROR = None

try:
    logfire_token = os.getenv("LOGFIRE_TOKEN")

    if logfire_token:
        logfire.configure(token=logfire_token)
        LOGFIRE_OK = True
    else:
        print("Logfire token is not configured for the Streamlit UI.")

except Exception as exc:
    LOGFIRE_ERROR = str(exc)
    print(f"Streamlit Logfire initialization failed: {exc}")


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="KnowledgeMesh",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# GENERIC HELPERS
# ============================================================


def clean_html(markup: str) -> str:
    return textwrap.dedent(markup).strip()


def display_html(markup: str):
    markup = clean_html(markup)

    native_html_renderer = getattr(st, "html", None)

    if callable(native_html_renderer):
        native_html_renderer(markup)
    else:
        st.markdown(markup, unsafe_allow_html=True)


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def safe_url(value) -> str:
    value = str(value or "").strip()

    if value.startswith(("http://", "https://")):
        return value

    return ""


def format_score(value) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.3f}"

    return "—"


def format_ms(value) -> str:
    if value is None:
        return "—"

    try:
        milliseconds = float(value)

        if milliseconds >= 1000:
            return f"{milliseconds / 1000:.2f}s"

        return f"{milliseconds:.0f}ms"

    except (TypeError, ValueError):
        return "—"


def format_seconds(value) -> str:
    if value is None:
        return "—"

    try:
        return f"{float(value):.2f}s"

    except (TypeError, ValueError):
        return "—"


def tri_state_chip(
    label_true: str,
    label_false: str,
    label_unknown: str,
    value,
    unknown_ok: bool = True,
) -> str:
    if value is True:
        return f'<span class="km-chip success">{esc(label_true)}</span>'

    if value is False:
        return f'<span class="km-chip danger">{esc(label_false)}</span>'

    if unknown_ok:
        return f'<span class="km-chip">{esc(label_unknown)}</span>'

    return ""


def _trace_context():
    if LOGFIRE_OK:
        return logfire.span("KnowledgeMesh UI operation")

    return nullcontext()


# ============================================================
# PIPELINE CONFIGURATION
# ============================================================

PIPELINE_STAGES = [
    ("guardrails", "Guardrails", "Safety and policy check"),
    ("planner", "Planner", "Intent and search planning"),
    ("retrieval", "Qdrant", "Private knowledge retrieval"),
    ("reranking", "FlashRank", "Semantic reranking"),
    ("grader", "Grader", "Evidence and context evaluation"),
    ("web_search", "Web Search", "External fallback evidence"),
    ("responder", "LLM", "Grounded response synthesis"),
    ("validator", "Validator", "Citation validation"),
    ("critic", "Critic", "Grounding review"),
]

TRACE_PATTERNS = [
    (re.compile(r"guardrail", re.IGNORECASE), "guardrails", "GRD"),
    (re.compile(r"intent|planner|planning", re.IGNORECASE), "planner", "PLN"),
    (
        re.compile(
            r"retriev|qdrant|context retrieved|knowledge retrieval",
            re.IGNORECASE,
        ),
        "retrieval",
        "RET",
    ),
    (
        re.compile(
            r"rerank|flashrank|semantic reranking",
            re.IGNORECASE,
        ),
        "reranking",
        "RER",
    ),
    (
        re.compile(
            r"grader|document grade|context quality|relevance",
            re.IGNORECASE,
        ),
        "grader",
        "GRD",
    ),
    (
        re.compile(
            r"web search|web fallback|external search|web evidence",
            re.IGNORECASE,
        ),
        "web_search",
        "WEB",
    ),
    (
        re.compile(r"citation|cite", re.IGNORECASE),
        "validator",
        "VAL",
    ),
    (
        re.compile(r"ground|support|critic|revis", re.IGNORECASE),
        "critic",
        "CRT",
    ),
    (
        re.compile(r"respond|response|synthes|answer|llm", re.IGNORECASE),
        "responder",
        "LLM",
    ),
]


# ============================================================
# TRACE / STATUS HELPERS
# ============================================================


def classify_step(step):
    if isinstance(step, dict):
        stage = step.get("stage") or step.get("node") or ""
        detail = step.get("detail") or step.get("message") or str(step)
        searchable_text = f"{stage} {detail}"
    else:
        detail = str(step)
        searchable_text = detail

    for pattern, stage_key, code in TRACE_PATTERNS:
        if pattern.search(searchable_text):
            return stage_key, code, detail

    return "", "STEP", detail


def infer_query_type(thought_process):
    text = " ".join(str(step) for step in (thought_process or [])).lower()

    if "conversational" in text:
        return "conversational"

    if "retrieval: skipped" in text:
        return "conversational"

    if "intent: technical" in text:
        return "technical"

    return "technical"


def extract_search_query(thought_process):
    for step in thought_process or []:
        text = str(step)

        if text.lower().startswith("search term:"):
            return text.split(":", 1)[1].strip()

    return None


def infer_visited_nodes(
    thought_process,
    sources,
    status,
    context_quality=None,
    web_search_used=False,
    citation_valid=None,
    is_grounded=None,
):
    visited = set()
    steps = thought_process or []

    combined_text = " ".join(str(step) for step in steps).lower()

    status_text = str(status or "").lower()

    for step in steps:
        stage_key, _, _ = classify_step(step)

        if stage_key:
            visited.add(stage_key)

    if "guardrail" in combined_text or "guardrail" in status_text:
        visited.add("guardrails")

    if (
        "intent:" in combined_text
        or "search term:" in combined_text
        or "conversational" in combined_text
    ):
        visited.add("planner")

    if sources:
        visited.add("retrieval")
        visited.add("reranking")

    if (
        "document grade" in combined_text
        or "context quality" in combined_text
        or context_quality is not None
    ):
        visited.add("grader")

    if web_search_used:
        visited.add("web_search")

    if status:
        visited.add("responder")

    if citation_valid is not None:
        visited.add("validator")

    if is_grounded is not None:
        visited.add("critic")

    return visited


def is_rate_limit_error(value) -> bool:
    text = str(value or "").lower()

    markers = [
        "ratelimit",
        "rate limit",
        "too many requests",
        "429",
    ]

    return any(marker in text for marker in markers)


def normalize_stage_state(value) -> str:
    value = str(value or "").lower().strip()

    valid_states = {
        "success",
        "fallback",
        "failed",
        "skipped",
        "pending",
    }

    return value if value in valid_states else "pending"


def stage_statuses(trace: dict) -> dict:
    """
    Creates transparent UI stage states from current backend payload fields.

    The preferred long-term backend contract is an explicit `node_statuses`
    dictionary. Until that exists, this avoids falsely showing every stage
    as successful when the backend only gives partial telemetry.
    """
    trace = trace or {}

    steps = trace.get("steps", []) or []
    steps_text = " ".join(str(step) for step in steps).lower()
    status_text = str(trace.get("status") or "").lower()

    query_type = trace.get("query_type", "technical")

    private_sources = (
        trace.get(
            "private_sources",
            trace.get("sources", []),
        )
        or []
    )

    web_sources = trace.get("web_sources", []) or []

    private_retrieval_status = trace.get(
        "private_retrieval_status",
        "Not used",
    )

    context_quality = str(trace.get("context_quality") or "").strip().lower()

    web_search_used = bool(trace.get("web_search_used"))

    citation_valid = trace.get("citation_valid")
    is_grounded = trace.get("is_grounded")

    revision_count = trace.get("revision_count")
    support_retry_count = trace.get("support_retry_count")

    grader_failed = (
        "document grade: failed" in steps_text
        or "grader failed" in steps_text
        or is_rate_limit_error(steps_text)
        or is_rate_limit_error(status_text)
    )

    generation_failed = (
        "generation rate-limited" in status_text
        or "generation failed" in status_text
        or "response generation failed" in status_text
    )

    statuses = {
        "guardrails": {
            "state": "pending",
            "detail": "Not reported",
        },
        "planner": {
            "state": "pending",
            "detail": "Not reported",
        },
        "retrieval": {
            "state": "skipped",
            "detail": "Not required",
        },
        "reranking": {
            "state": "skipped",
            "detail": "Not required",
        },
        "grader": {
            "state": "skipped",
            "detail": "Not run",
        },
        "web_search": {
            "state": "skipped",
            "detail": "Not required",
        },
        "responder": {
            "state": "pending",
            "detail": "No response state reported",
        },
        "validator": {
            "state": "skipped",
            "detail": "Not checked",
        },
        "critic": {
            "state": "skipped",
            "detail": "Not checked",
        },
    }

    if "guardrail" in steps_text or "guardrail" in status_text:
        statuses["guardrails"] = {
            "state": "success",
            "detail": "Completed",
        }

    if (
        "intent:" in steps_text
        or "search term:" in steps_text
        or "planner" in steps_text
        or query_type == "conversational"
    ):
        statuses["planner"] = {
            "state": "success",
            "detail": (
                "Conversation path"
                if query_type == "conversational"
                else query_type.title()
            ),
        }

    if query_type == "conversational":
        statuses["retrieval"] = {
            "state": "skipped",
            "detail": "Conversation path",
        }
        statuses["reranking"] = {
            "state": "skipped",
            "detail": "Conversation path",
        }
        statuses["grader"] = {
            "state": "skipped",
            "detail": "Conversation path",
        }
        statuses["web_search"] = {
            "state": "skipped",
            "detail": "Conversation path",
        }
        statuses["responder"] = {
            "state": "success",
            "detail": "Memory-based answer",
        }

        return statuses

    retrieval_attempted = private_retrieval_status in {
        "Used",
        "Attempted",
    }

    if retrieval_attempted:
        statuses["retrieval"] = {
            "state": "success" if private_sources else "failed",
            "detail": (
                f"{len(private_sources)} private source(s)"
                if private_sources
                else "No documents returned"
            ),
        }

        statuses["reranking"] = {
            "state": "success" if private_sources else "skipped",
            "detail": (
                f"Top {len(private_sources)} context chunk(s)"
                if private_sources
                else "No context to rerank"
            ),
        }

    if grader_failed:
        statuses["grader"] = {
            "state": "failed",
            "detail": "Rate-limited or failed",
        }
    elif trace.get("context_quality") is not None:
        statuses["grader"] = {
            "state": "success",
            "detail": f"Context: {context_quality or 'reported'}",
        }

    if web_search_used:
        statuses["web_search"] = {
            "state": "fallback",
            "detail": f"{len(web_sources)} web source(s)",
        }

    if generation_failed:
        statuses["responder"] = {
            "state": "failed",
            "detail": "Generation failed or rate-limited",
        }
    elif status_text:
        statuses["responder"] = {
            "state": "fallback" if web_search_used else "success",
            "detail": (
                "Generated with web fallback" if web_search_used else "Answer generated"
            ),
        }

    if citation_valid is True:
        statuses["validator"] = {
            "state": "success",
            "detail": "Citations valid",
        }
    elif citation_valid is False:
        statuses["validator"] = {
            "state": "failed",
            "detail": "Citation mismatch",
        }

    if is_grounded is True:
        statuses["critic"] = {
            "state": "success",
            "detail": "Grounded",
        }
    elif is_grounded is False:
        statuses["critic"] = {
            "state": "failed",
            "detail": "Not grounded",
        }
    elif revision_count not in (None, 0) or support_retry_count not in (None, 0):
        statuses["critic"] = {
            "state": "fallback",
            "detail": (
                f"Revisions {revision_count or 0} · Retries {support_retry_count or 0}"
            ),
        }

    return statuses


def run_outcome(trace: dict) -> dict:
    trace = trace or {}

    citation_valid = trace.get("citation_valid")
    is_grounded = trace.get("is_grounded")
    web_search_used = bool(trace.get("web_search_used"))

    status_text = str(trace.get("status") or "").lower()
    steps_text = " ".join(str(step) for step in trace.get("steps", [])).lower()

    generation_failed = (
        "generation rate-limited" in status_text
        or "generation failed" in status_text
        or "response generation failed" in status_text
    )

    if generation_failed:
        return {
            "kind": "failed",
            "title": "Response generation issue",
            "description": (
                "The workflow completed partially, but final response "
                "generation encountered an execution issue."
            ),
        }

    if citation_valid is False or is_grounded is False:
        return {
            "kind": "warning",
            "title": "Completed with validation warning",
            "description": (
                "The system produced a response, but citation or grounding "
                "review reported a quality issue."
            ),
        }

    if web_search_used:
        return {
            "kind": "fallback",
            "title": "Completed with web fallback",
            "description": (
                "Private knowledge was supplemented with external evidence "
                "to improve answer coverage."
            ),
        }

    if (
        "rate limit" in steps_text
        or "ratelimit" in steps_text
        or "failed" in steps_text
    ):
        return {
            "kind": "warning",
            "title": "Completed with recovery",
            "description": (
                "The agentic workflow recovered from a partial execution or "
                "evaluation issue."
            ),
        }

    if trace.get("query_type") == "conversational":
        return {
            "kind": "memory",
            "title": "Conversation response",
            "description": (
                "The response used the conversational memory path without "
                "private knowledge retrieval."
            ),
        }

    return {
        "kind": "success",
        "title": "Completed successfully",
        "description": (
            "The agent completed planning, evidence processing, response "
            "generation, and available validation checks."
        ),
    }


def outcome_badge_class(kind: str) -> str:
    return {
        "success": "success",
        "memory": "ok",
        "fallback": "warning",
        "warning": "warning",
        "failed": "danger",
    }.get(kind, "")


def render_run_banner(trace: dict) -> str:
    trace = trace or {}

    outcome = run_outcome(trace)
    kind = outcome_badge_class(outcome["kind"])

    private_sources = (
        trace.get(
            "private_sources",
            trace.get("sources", []),
        )
        or []
    )

    web_sources = trace.get("web_sources", []) or []
    answer_sources = trace.get("answer_sources", []) or []

    context_quality = trace.get("context_quality")
    context_label = str(context_quality).title() if context_quality else "Not reported"

    web_used = bool(trace.get("web_search_used"))

    return f"""
        <div class="km-run-banner {esc(kind)}">
            <div class="km-run-banner-main">
                <div class="km-run-banner-title">
                    <span class="km-run-indicator"></span>
                    {esc(outcome["title"])}
                </div>
                <div class="km-run-banner-description">
                    {esc(outcome["description"])}
                </div>
            </div>

            <div class="km-run-banner-stats">
                <div class="km-run-stat">
                    <span>Total time</span>
                    <strong>{esc(format_seconds(trace.get("latency")))}</strong>
                </div>
                <div class="km-run-stat">
                    <span>Private evidence</span>
                    <strong>{len(private_sources)}</strong>
                </div>
                <div class="km-run-stat">
                    <span>Web evidence</span>
                    <strong>{len(web_sources) if web_used else "—"}</strong>
                </div>
                <div class="km-run-stat">
                    <span>Cited sources</span>
                    <strong>{len(answer_sources)}</strong>
                </div>
                <div class="km-run-stat">
                    <span>Context</span>
                    <strong>{esc(context_label)}</strong>
                </div>
            </div>
        </div>
    """


def render_pipeline_rail(trace=None):
    trace = trace or {}
    statuses = stage_statuses(trace)

    icon_map = {
        "success": "✓",
        "fallback": "↻",
        "failed": "✕",
        "skipped": "—",
        "pending": "·",
    }

    nodes = []
    total = len(PIPELINE_STAGES)

    for index, (key, title, default_detail) in enumerate(
        PIPELINE_STAGES,
        start=1,
    ):
        stage = statuses.get(key, {})

        state = normalize_stage_state(stage.get("state"))
        detail = stage.get("detail") or default_detail
        icon = icon_map.get(state, "·")

        connector = ""

        if index < total:
            connector = f'<div class="km-step-line {esc(state)}"></div>'

        nodes.append(
            f"""
            <div class="km-step {esc(state)}">
                <div class="km-step-marker">
                    <div class="km-step-dot">{esc(icon)}</div>
                    {connector}
                </div>
                <div class="km-step-body">
                    <strong>{esc(title)}</strong>
                    <span>{esc(detail)}</span>
                </div>
            </div>
            """
        )

    return f'<div class="km-pipeline">{"".join(nodes)}</div>'


def render_recovery_notice(trace: dict) -> str:
    trace = trace or {}

    steps_text = " ".join(str(item) for item in trace.get("steps", []))

    status_text = str(trace.get("status") or "")

    grader_failure = (
        "document grade: failed" in steps_text.lower()
        or "grader failed" in steps_text.lower()
        or is_rate_limit_error(steps_text)
        or is_rate_limit_error(status_text)
    )

    web_search_used = bool(trace.get("web_search_used"))
    revision_count = trace.get("revision_count") or 0
    retry_count = trace.get("support_retry_count") or 0

    citation_valid = trace.get("citation_valid")
    is_grounded = trace.get("is_grounded")

    details = []

    if grader_failure:
        details.append("Document grading did not complete successfully.")

    if web_search_used:
        details.append("External web evidence was used as a fallback.")

    if revision_count:
        details.append(f"Answer revisions: {revision_count}.")

    if retry_count:
        details.append(f"Support retries: {retry_count}.")

    if citation_valid is False:
        details.append("Citation validation reported a mismatch.")

    if is_grounded is False:
        details.append("Grounding review reported an issue.")

    if not details:
        return ""

    return f"""
        <div class="km-recovery-card">
            <div class="km-recovery-title">
                ⚠ Recovery or validation event
            </div>
            <div class="km-recovery-body">
                {esc(" ".join(details))}
            </div>
        </div>
    """


def render_quality_summary(trace: dict) -> str:
    trace = trace or {}

    citation_valid = trace.get("citation_valid")
    is_grounded = trace.get("is_grounded")

    answer_supported = trace.get("answer_supported")
    answer_useful = trace.get("answer_useful")

    support_score = trace.get("support_score")
    usefulness_score = trace.get("usefulness_score")

    grounding_scores = trace.get("grounding_scores") or {}

    chips = [
        tri_state_chip(
            "Citations valid",
            "Citation mismatch",
            "Citations not checked",
            citation_valid,
        ),
        tri_state_chip(
            "Grounded",
            "Not grounded",
            "Grounding not checked",
            is_grounded,
        ),
    ]

    if answer_supported is not None:
        chips.append(
            tri_state_chip(
                "Supported",
                "Not supported",
                "",
                answer_supported,
                unknown_ok=False,
            )
        )

    if answer_useful is not None:
        chips.append(
            tri_state_chip(
                "Useful",
                "Not useful",
                "",
                answer_useful,
                unknown_ok=False,
            )
        )

    if support_score is not None:
        chips.append(
            '<span class="km-chip">'
            f'Support <span class="num">{float(support_score):.2f}</span>'
            "</span>"
        )

    if usefulness_score is not None:
        chips.append(
            '<span class="km-chip">'
            f'Usefulness <span class="num">{float(usefulness_score):.2f}</span>'
            "</span>"
        )

    if isinstance(grounding_scores, dict):
        for key, value in grounding_scores.items():
            if isinstance(value, (int, float)):
                readable_key = str(key).replace("_", " ").title()

                chips.append(
                    '<span class="km-chip">'
                    f"{esc(readable_key)} "
                    f'<span class="num">{float(value):.2f}</span>'
                    "</span>"
                )

    return "".join(chip for chip in chips if chip)


def render_recovery_summary(trace: dict) -> str:
    trace = trace or {}

    recovery_items = [
        ("Query rewrites", trace.get("retrieval_rewrite_count")),
        ("Answer revisions", trace.get("revision_count")),
        ("Support retries", trace.get("support_retry_count")),
        ("Web rewrites", trace.get("web_rewrite_count")),
    ]

    chips = []

    for label, value in recovery_items:
        if value is not None:
            chips.append(
                '<span class="km-chip">'
                f'{esc(label)} <span class="num">{int(value)}</span>'
                "</span>"
            )

    return "".join(chips)


# ============================================================
# SOURCE NORMALIZATION
# ============================================================


def normalize_origin(value) -> str:
    value = str(value or "").strip().lower()

    if value in {"private", "internal", "knowledge_base", "kb"}:
        return "private"

    if value in {"web", "external", "internet"}:
        return "web"

    return "unknown"


def normalize_source(source, index, origin="private"):
    origin = normalize_origin(origin)

    if isinstance(source, dict):
        content = (
            source.get("content") or source.get("text") or source.get("snippet") or ""
        )

        name = (
            source.get("source")
            or source.get("filename")
            or source.get("title")
            or "Unknown document"
        )

        source_type = source.get("source_type") or "unknown"

        document_id = source.get("id") or source.get("document_id")

        chunk_id = source.get("chunk_id")

        citation_id = source.get("citation_id") or source.get("citation") or chunk_id

        vector_score = source.get("score")
        rerank_score = source.get("rerank_score")

        grader_score = source.get("grader_score")
        grader_relevant = source.get("grader_relevant")
        grader_reason = source.get("grader_reason")

        relevance = source.get("relevance") or source.get("rank")
        url = source.get("url")

    else:
        content = str(source)
        name = "Unknown document"
        source_type = "unknown"

        document_id = None
        chunk_id = None
        citation_id = None

        vector_score = None
        rerank_score = None

        grader_score = None
        grader_relevant = None
        grader_reason = None

        relevance = None
        url = None

    return {
        "n": index,
        "text": str(content),
        "name": str(name),
        "source_type": str(source_type),
        "origin": origin,
        "id": document_id,
        "chunk_id": chunk_id,
        "citation_id": citation_id,
        "score": vector_score,
        "rerank_score": rerank_score,
        "grader_score": grader_score,
        "grader_relevant": grader_relevant,
        "grader_reason": grader_reason,
        "relevance": relevance,
        "url": url,
    }


def source_type_label(value):
    normalized = str(value or "").strip().lower()

    if normalized == "true":
        return "Trusted"

    if normalized in {"false", "noisy"}:
        return "Noisy"

    if not normalized:
        return "Unknown"

    return normalized.title()


def render_source_cards(source_list):
    cards = []

    origin_map = {
        "private": ("ok", "PRIVATE"),
        "web": ("warning", "WEB"),
        "unknown": ("", "UNKNOWN"),
    }

    for source in source_list:
        source_name = source.get("name", "Unknown document")
        source_type = source_type_label(source.get("source_type"))

        origin = normalize_origin(source.get("origin"))
        origin_class, origin_label = origin_map[origin]

        vector_score = format_score(source.get("score"))
        rerank_score = format_score(source.get("rerank_score"))
        grader_score = format_score(source.get("grader_score"))

        grader_relevant = source.get("grader_relevant")
        grader_reason = source.get("grader_reason")

        document_id = source.get("id") or "Unknown"
        citation_id = source.get("citation_id")

        source_url = safe_url(source.get("url"))

        if source_url:
            link_html = (
                f'<a class="km-source-link" href="{esc(source_url)}" '
                'target="_blank" rel="noopener noreferrer">'
                "Open source</a>"
            )
        elif origin == "web":
            link_html = '<span class="km-source-score">No URL provided</span>'
        else:
            link_html = ""

        score_html = ""

        if origin == "private":
            score_html = (
                f'<span class="km-source-score">'
                f"vector {esc(vector_score)}</span>"
                f'<span class="km-source-score">'
                f"rerank {esc(rerank_score)}</span>"
            )

        grader_html = ""

        if grader_score != "—":
            if grader_relevant is True:
                grader_label = "relevant"
            elif grader_relevant is False:
                grader_label = "rejected"
            else:
                grader_label = "reported"

            grader_html = (
                f'<span class="km-source-score">'
                f"grader {esc(grader_score)} · {esc(grader_label)}"
                "</span>"
            )

        citation_html = ""

        if citation_id:
            citation_html = (
                f'<span class="km-source-score">cite [{esc(citation_id)}]</span>'
            )

        reason_html = ""

        if grader_reason:
            reason_html = (
                f'<div class="km-source-reason">Grader: {esc(grader_reason)}</div>'
            )

        cards.append(
            f"""
            <div class="km-source">
                <div class="km-source-meta">
                    <span class="km-source-number">[{source["n"]}]</span>
                    <span class="km-chip {origin_class} km-origin-badge">
                        {esc(origin_label)}
                    </span>
                    <span class="km-source-name">{esc(source_name)}</span>
                    <span class="km-source-type">{esc(source_type)}</span>
                    {score_html}
                    {grader_html}
                    {citation_html}
                    {link_html}
                </div>

                <div class="km-source-body">
                    {esc(source.get("text", ""))}
                </div>

                {reason_html}

                <div class="km-source-id">
                    ID: {esc(document_id)}
                </div>
            </div>
            """
        )

    return "".join(cards)


def render_citation_provenance_table(provenance_list):
    if not provenance_list:
        return ""

    rows = []

    for entry in provenance_list:
        if not isinstance(entry, dict):
            continue

        citation = entry.get("citation") or entry.get("citation_id") or "—"

        source = entry.get("source") or entry.get("name") or "—"

        origin = str(entry.get("origin") or "—").upper()

        document = entry.get("document") or entry.get("document_id") or "—"

        chunk = entry.get("chunk") or entry.get("chunk_id") or "—"

        url = safe_url(entry.get("url"))

        url_cell = (
            f'<a class="km-source-link" href="{esc(url)}" '
            'target="_blank" rel="noopener noreferrer">Open</a>'
            if url
            else "—"
        )

        origin_style = ""

        if origin == "PRIVATE":
            origin_style = ' style="color:var(--km-success);"'
        elif origin == "WEB":
            origin_style = ' style="color:var(--km-warning);"'

        rows.append(
            f"""
            <tr>
                <td>[{esc(citation)}]</td>
                <td>{esc(source)}</td>
                <td{origin_style}>{esc(origin)}</td>
                <td>{esc(document)}</td>
                <td>{esc(chunk)}</td>
                <td>{url_cell}</td>
            </tr>
            """
        )

    if not rows:
        return ""

    return f"""
        <table class="km-provenance-table">
            <thead>
                <tr>
                    <th>Citation</th>
                    <th>Source</th>
                    <th>Origin</th>
                    <th>Document</th>
                    <th>Chunk</th>
                    <th>URL</th>
                </tr>
            </thead>
            <tbody>
                {"".join(rows)}
            </tbody>
        </table>
    """


# ============================================================
# TRANSCRIPT EXPORT
# ============================================================


def transcript_markdown(messages):
    lines = [
        "# KnowledgeMesh Transcript",
        "",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
    ]

    for message in messages:
        role = message.get("role", "assistant")

        speaker = "You" if role == "user" else "KnowledgeMesh"

        lines.extend(
            [
                f"## {speaker}",
                "",
                str(message.get("content", "")),
                "",
            ]
        )

        trace = message.get("trace")

        if not trace:
            continue

        lines.extend(
            [
                "### Agentic-RAG Run Summary",
                "",
                f"- Status: {trace.get('status') or 'Not reported'}",
                f"- Total latency: {format_seconds(trace.get('latency'))}",
                f"- Backend latency: {format_ms(trace.get('backend_latency_ms'))}",
                f"- Retrieval latency: {format_ms(trace.get('retrieval_latency_ms'))}",
                f"- Reranking latency: {format_ms(trace.get('rerank_latency_ms'))}",
                f"- Grading latency: {format_ms(trace.get('grader_latency_ms'))}",
                f"- Generation latency: {format_ms(trace.get('generation_latency_ms'))}",
                "",
            ]
        )

        for label, value in (
            ("Context quality", trace.get("context_quality")),
            ("Context reason", trace.get("context_reason")),
            ("Citation valid", trace.get("citation_valid")),
            ("Grounded", trace.get("is_grounded")),
            ("Support score", trace.get("support_score")),
            ("Usefulness score", trace.get("usefulness_score")),
        ):
            if value is not None:
                lines.append(f"- {label}: {value}")

        private_sources = trace.get(
            "private_sources",
            trace.get("sources", []),
        )

        web_sources = trace.get("web_sources", [])
        answer_sources = trace.get("answer_sources", [])

        lines.extend(
            [
                f"- Private sources retrieved: {len(private_sources)}",
                f"- Web sources used: {len(web_sources)}",
                f"- Sources used in answer: {len(answer_sources)}",
                "",
            ]
        )

        for label, value in (
            ("Query rewrites", trace.get("retrieval_rewrite_count")),
            ("Answer revisions", trace.get("revision_count")),
            ("Support retries", trace.get("support_retry_count")),
            ("Web rewrites", trace.get("web_rewrite_count")),
        ):
            if value is not None:
                lines.append(f"- {label}: {value}")

        lines.append("")

        search_query = trace.get("search_query")

        if search_query:
            lines.extend(
                [
                    f"**Planner search query:** `{search_query}`",
                    "",
                ]
            )

        steps = trace.get("steps", [])

        if steps:
            lines.extend(["### Reasoning Trace", ""])

            for step in steps:
                lines.append(f"- {step}")

            lines.append("")

    return "\n".join(lines)


# ============================================================
# BACKEND HEALTH
# ============================================================


@st.cache_data(ttl=10, show_spinner=False)
def check_backend_health(backend_url):
    try:
        response = requests.get(
            f"{backend_url}/health",
            timeout=4,
        )
        return response.ok

    except requests.RequestException:
        return False


@st.cache_data(ttl=10, show_spinner=False)
def check_backend_ready(backend_url):
    try:
        response = requests.get(
            f"{backend_url}/ready",
            timeout=4,
        )

        if not response.ok:
            return False

        payload = response.json()

        return payload.get("status") == "ready"

    except (requests.RequestException, ValueError):
        return False


# ============================================================
# SESSION STATE
# ============================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "latencies" not in st.session_state:
    st.session_state.latencies = []

if "session_started_at" not in st.session_state:
    st.session_state.session_started_at = time.strftime("%H:%M:%S")

if "backend_url" not in st.session_state:
    st.session_state.backend_url = DEFAULT_BACKEND_URL

if "http_session" not in st.session_state:
    http_session = requests.Session()

    retry_strategy = Retry(
        total=2,
        backoff_factor=0.5,
        status_forcelist=[502, 503, 504],
        allowed_methods=["GET", "POST"],
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)

    http_session.mount("http://", adapter)
    http_session.mount("https://", adapter)

    st.session_state.http_session = http_session


# ============================================================
# CSS
# ============================================================

CUSTOM_CSS = """
<style>
:root {
    --km-bg: #0a0c11;
    --km-panel: #11141c;
    --km-panel-alt: #161a24;
    --km-panel-raised: #1b202b;

    --km-border: #2a3040;
    --km-border-soft: #202633;

    --km-text: #f1f3f8;
    --km-muted: #9aa2b4;
    --km-muted-soft: #6f778a;

    --km-accent: #7c86f5;
    --km-accent-strong: #adb5ff;
    --km-accent-soft: rgba(124, 134, 245, .14);
    --km-accent-border: rgba(124, 134, 245, .40);

    --km-success: #58d6a0;
    --km-success-soft: rgba(88, 214, 160, .13);

    --km-warning: #e5ad5d;
    --km-warning-soft: rgba(229, 173, 93, .13);

    --km-danger: #ef737e;
    --km-danger-soft: rgba(239, 115, 126, .13);

    --km-radius-sm: 8px;
    --km-radius-md: 12px;
    --km-radius-lg: 16px;

    --km-shadow:
        0 1px 2px rgba(0, 0, 0, .32),
        0 10px 28px -16px rgba(0, 0, 0, .70);
}

html,
body,
[class*="css"] {
    font-family:
        "Inter",
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
}

.stApp {
    background:
        radial-gradient(
            1050px 520px at 12% -10%,
            rgba(124, 134, 245, .08),
            transparent 62%
        ),
        var(--km-bg);
    color: var(--km-text);
}

header[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu,
footer {
    visibility: hidden;
}

::selection {
    background: var(--km-accent-soft);
}

@keyframes kmFadeIn {
    from {
        opacity: 0;
        transform: translateY(5px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* ============================================================
   SIDEBAR
   ============================================================ */

section[data-testid="stSidebar"] {
    background: var(--km-panel);
    border-right: 1px solid var(--km-border-soft);
}

section[data-testid="stSidebar"] > div {
    padding-top: .65rem;
}

section[data-testid="stSidebar"] * {
    color: var(--km-text);
}

.km-brand {
    display: flex;
    align-items: center;
    gap: 11px;
    padding: 6px 0 18px;
    margin-bottom: 18px;
    border-bottom: 1px solid var(--km-border-soft);
}

.km-brand-mark {
    display: grid;
    place-items: center;
    width: 34px;
    height: 34px;
    border-radius: 9px;
    background: linear-gradient(150deg, var(--km-accent), #5d65ce);
    box-shadow: 0 4px 14px -4px rgba(124, 134, 245, .60);
    color: #0a0c11;
    font-size: 15px;
    font-weight: 850;
}

.km-brand strong {
    display: block;
    color: #ffffff;
    font-size: 14.5px;
    font-weight: 720;
    letter-spacing: -.01em;
}

.km-brand small {
    display: block;
    margin-top: 2px;
    color: var(--km-muted-soft);
    font-size: 10.5px;
}

.km-rail-label {
    margin: 20px 0 9px 1px;
    color: var(--km-muted-soft);
    font-size: 10px;
    font-weight: 750;
    letter-spacing: .09em;
    text-transform: uppercase;
}

.km-status-card {
    padding: 12px 13px;
    border: 1px solid var(--km-border-soft);
    border-radius: var(--km-radius-md);
    background: var(--km-panel-alt);
}

.km-status-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    padding: 5px 0;
    font-size: 11px;
}

.km-status-row + .km-status-row {
    border-top: 1px solid var(--km-border-soft);
}

.km-status-row .label {
    color: var(--km-muted-soft);
}

.km-status-row .value {
    max-width: 150px;
    overflow: hidden;
    color: var(--km-text);
    font-family: "SF Mono", "JetBrains Mono", monospace;
    font-size: 10px;
    text-align: right;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.km-capability-list {
    display: grid;
    gap: 7px;
}

.km-capability-row {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--km-muted);
    font-size: 11px;
}

.km-capability-row .mark {
    color: var(--km-success);
    font-weight: 750;
}

.km-rail-footer {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 22px;
    padding-top: 16px;
    border-top: 1px solid var(--km-border-soft);
}

.km-rail-footer strong {
    display: block;
    color: var(--km-text);
    font-size: 11.5px;
    font-weight: 650;
}

.km-rail-footer small {
    display: block;
    margin-top: 3px;
    color: var(--km-muted-soft);
    font-size: 9.5px;
    line-height: 1.5;
}

.km-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    flex: none;
    border-radius: 50%;
    background: var(--km-muted-soft);
}

.km-dot.success {
    background: var(--km-success);
    box-shadow: 0 0 10px var(--km-success-soft);
}

.km-dot.warning {
    background: var(--km-warning);
    box-shadow: 0 0 10px var(--km-warning-soft);
}

.km-dot.danger {
    background: var(--km-danger);
    box-shadow: 0 0 10px var(--km-danger-soft);
}

section[data-testid="stSidebar"] button {
    border: 1px solid var(--km-border) !important;
    border-radius: 8px !important;
    background: var(--km-panel-raised) !important;
    color: var(--km-text) !important;
    font-size: 11.5px !important;
}

section[data-testid="stSidebar"] button:hover {
    border-color: var(--km-accent-border) !important;
}

section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] div[data-baseweb="input"] {
    border-color: var(--km-border) !important;
    background: var(--km-panel-raised) !important;
    color: var(--km-text) !important;
    font-family: "SF Mono", "JetBrains Mono", monospace !important;
    font-size: 10.5px !important;
}

/* ============================================================
   TOPBAR
   ============================================================ */

.km-topbar {
    padding-top: 3px;
}

.km-kicker {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    margin-bottom: 10px;
    color: var(--km-accent-strong);
    font-family: "SF Mono", "JetBrains Mono", monospace;
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: .08em;
    text-transform: uppercase;
}

.km-kicker::before {
    content: "";
    width: 14px;
    height: 1px;
    background: var(--km-accent);
}

.km-topbar h1 {
    margin: 0 0 9px;
    color: #ffffff;
    font-size: 27px;
    font-weight: 760;
    letter-spacing: -.03em;
}

.km-topbar p {
    max-width: 780px;
    margin: 0;
    color: var(--km-muted);
    font-size: 13px;
    line-height: 1.65;
}

.km-engine-pill {
    display: inline-flex;
    align-items: center;
    gap: 9px;
    margin-top: 14px;
    padding: 7px 13px;
    border: 1px solid var(--km-border);
    border-radius: 999px;
    background: var(--km-panel-alt);
    color: var(--km-muted);
    font-family: "SF Mono", "JetBrains Mono", monospace;
    font-size: 10px;
}

/* ============================================================
   CONSOLE
   ============================================================ */

.km-console-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 2px 12px;
    margin: 24px 0 16px;
    border-bottom: 1px solid var(--km-border-soft);
}

.km-console-live {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--km-text);
    font-size: 12.5px;
    font-weight: 650;
}

.km-session-id {
    color: var(--km-muted-soft);
    font-family: "SF Mono", "JetBrains Mono", monospace;
    font-size: 10px;
}

.km-msg-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 6px;
}

.km-msg-row.user {
    justify-content: flex-end;
}

.km-msg-avatar {
    display: grid;
    place-items: center;
    width: 26px;
    height: 26px;
    border-radius: 7px;
    background: linear-gradient(150deg, var(--km-accent), #5d65ce);
    color: #0a0c11;
    font-family: "SF Mono", "JetBrains Mono", monospace;
    font-size: 11px;
    font-weight: 850;
}

.km-msg-label {
    color: var(--km-muted-soft);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: .05em;
    text-transform: uppercase;
}

.km-msg-label.user {
    text-align: right;
}

[class*="st-key-km_bubble_"] {
    max-width: min(860px, 92%);
    padding: 15px 17px;
    border: 1px solid var(--km-border-soft);
    border-radius: 4px 14px 14px 14px;
    background: var(--km-panel-alt);
    color: var(--km-text);
    font-size: 13.5px;
    line-height: 1.7;
    box-shadow: var(--km-shadow);
    animation: kmFadeIn .24s ease both;
}

[class*="st-key-km_user_bubble_"] {
    margin-left: auto;
    border-radius: 14px 4px 14px 14px;
    background: var(--km-panel-raised);
}

[class*="st-key-km_bubble_intro"] {
    border-left: 2px solid var(--km-accent);
    background: linear-gradient(
        180deg,
        var(--km-accent-soft),
        var(--km-panel-alt) 60%
    );
}

[class*="st-key-km_bubble_"] p {
    margin-bottom: 9px;
}

[class*="st-key-km_bubble_"] p:last-child {
    margin-bottom: 0;
}

[class*="st-key-km_bubble_"] code {
    padding: 2px 5px;
    border: 1px solid var(--km-border);
    border-radius: 4px;
    background: var(--km-panel);
    font-size: 12px;
}

/* ============================================================
   RUN SUMMARY
   ============================================================ */

.km-run-banner {
    display: flex;
    align-items: stretch;
    justify-content: space-between;
    gap: 22px;
    margin: 16px 0;
    padding: 16px 18px;
    border: 1px solid var(--km-border-soft);
    border-radius: var(--km-radius-md);
    background: linear-gradient(
        115deg,
        var(--km-panel-alt),
        var(--km-panel)
    );
    box-shadow: var(--km-shadow);
}

.km-run-banner.success {
    border-left: 3px solid var(--km-success);
}

.km-run-banner.ok {
    border-left: 3px solid var(--km-accent);
}

.km-run-banner.warning {
    border-left: 3px solid var(--km-warning);
}

.km-run-banner.danger {
    border-left: 3px solid var(--km-danger);
}

.km-run-banner-main {
    min-width: 0;
    flex: 1.2;
}

.km-run-banner-title {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--km-text);
    font-size: 13px;
    font-weight: 750;
}

.km-run-indicator {
    display: inline-block;
    width: 8px;
    height: 8px;
    flex: none;
    border-radius: 50%;
    background: var(--km-accent);
}

.km-run-banner.success .km-run-indicator {
    background: var(--km-success);
    box-shadow: 0 0 12px var(--km-success-soft);
}

.km-run-banner.ok .km-run-indicator {
    background: var(--km-accent);
    box-shadow: 0 0 12px var(--km-accent-soft);
}

.km-run-banner.warning .km-run-indicator {
    background: var(--km-warning);
    box-shadow: 0 0 12px var(--km-warning-soft);
}

.km-run-banner.danger .km-run-indicator {
    background: var(--km-danger);
    box-shadow: 0 0 12px var(--km-danger-soft);
}

.km-run-banner-description {
    max-width: 520px;
    margin-top: 5px;
    color: var(--km-muted);
    font-size: 11px;
    line-height: 1.55;
}

.km-run-banner-stats {
    display: grid;
    grid-template-columns: repeat(5, minmax(72px, 1fr));
    align-items: center;
    gap: 14px;
    min-width: 430px;
}

.km-run-stat {
    min-width: 0;
    padding-left: 12px;
    border-left: 1px solid var(--km-border-soft);
}

.km-run-stat span {
    display: block;
    overflow: hidden;
    color: var(--km-muted-soft);
    font-size: 9px;
    font-weight: 700;
    letter-spacing: .04em;
    text-overflow: ellipsis;
    text-transform: uppercase;
    white-space: nowrap;
}

.km-run-stat strong {
    display: block;
    margin-top: 4px;
    overflow: hidden;
    color: var(--km-text);
    font-family: "SF Mono", "JetBrains Mono", monospace;
    font-size: 12px;
    font-weight: 700;
    text-overflow: ellipsis;
    white-space: nowrap;
}

/* ============================================================
   PIPELINE
   ============================================================ */

.km-pipeline {
    display: flex;
    width: 100%;
    margin: 18px 0;
    padding: 16px 18px;
    border: 1px solid var(--km-border-soft);
    border-radius: var(--km-radius-lg);
    background: var(--km-panel);
    box-shadow: var(--km-shadow);
    overflow-x: auto;
}

.km-step {
    display: flex;
    flex: 1;
    flex-direction: column;
    align-items: flex-start;
    gap: 8px;
    min-width: 108px;
    opacity: .45;
}

.km-step.success,
.km-step.fallback,
.km-step.failed,
.km-step.skipped {
    opacity: 1;
}

.km-step-marker {
    display: flex;
    align-items: center;
    width: 100%;
}

.km-step-dot {
    display: grid;
    place-items: center;
    width: 26px;
    height: 26px;
    flex: none;
    border: 1px solid var(--km-border);
    border-radius: 50%;
    background: var(--km-panel-alt);
    color: var(--km-muted-soft);
    font-family: "SF Mono", "JetBrains Mono", monospace;
    font-size: 11px;
    font-weight: 850;
}

.km-step.success .km-step-dot {
    border-color: rgba(88, 214, 160, .6);
    background: var(--km-success);
    color: #0a0c11;
    box-shadow: 0 0 0 4px var(--km-success-soft);
}

.km-step.fallback .km-step-dot {
    border-color: rgba(229, 173, 93, .6);
    background: var(--km-warning);
    color: #0a0c11;
    box-shadow: 0 0 0 4px var(--km-warning-soft);
}

.km-step.failed .km-step-dot {
    border-color: rgba(239, 115, 126, .7);
    background: var(--km-danger);
    color: #0a0c11;
    box-shadow: 0 0 0 4px var(--km-danger-soft);
}

.km-step.skipped {
    opacity: .58;
}

.km-step.skipped .km-step-dot {
    border-style: dashed;
    color: var(--km-muted-soft);
}

.km-step-line {
    flex: 1;
    height: 1px;
    margin: 0 6px;
    background: var(--km-border);
}

.km-step-line.success {
    background: rgba(88, 214, 160, .52);
}

.km-step-line.fallback {
    background: rgba(229, 173, 93, .55);
}

.km-step-line.failed {
    background: rgba(239, 115, 126, .55);
}

.km-step-line.skipped,
.km-step-line.pending {
    background: var(--km-border);
}

.km-step-body strong {
    display: block;
    color: var(--km-text);
    font-size: 11.5px;
    font-weight: 700;
}

.km-step.success .km-step-body strong {
    color: var(--km-success);
}

.km-step.fallback .km-step-body strong {
    color: var(--km-warning);
}

.km-step.failed .km-step-body strong {
    color: var(--km-danger);
}

.km-step-body span {
    display: block;
    margin-top: 2px;
    color: var(--km-muted-soft);
    font-size: 9.5px;
    line-height: 1.4;
}

/* ============================================================
   RECOVERY + TABS
   ============================================================ */

.km-recovery-card {
    margin: 12px 0;
    padding: 11px 13px;
    border: 1px solid rgba(229, 173, 93, .35);
    border-radius: var(--km-radius-sm);
    background: var(--km-warning-soft);
}

.km-recovery-title {
    color: var(--km-warning);
    font-size: 11.5px;
    font-weight: 700;
}

.km-recovery-body {
    margin-top: 4px;
    color: var(--km-muted);
    font-size: 10.5px;
    line-height: 1.55;
}

.km-tab-panel {
    padding: 14px;
    margin: 8px 0 10px;
    border: 1px solid var(--km-border-soft);
    border-radius: var(--km-radius-sm);
    background: var(--km-panel);
}

.km-tab-heading {
    margin-bottom: 10px;
    color: var(--km-muted-soft);
    font-size: 9.5px;
    font-weight: 750;
    letter-spacing: .08em;
    text-transform: uppercase;
}

.km-info-card {
    padding: 12px 13px;
    margin: 10px 0;
    border: 1px solid var(--km-border-soft);
    border-radius: var(--km-radius-sm);
    background: var(--km-panel-alt);
}

.km-info-card-title {
    margin-bottom: 6px;
    color: var(--km-accent-strong);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: .04em;
    text-transform: uppercase;
}

.km-info-card-body {
    color: var(--km-muted);
    font-size: 11.5px;
    line-height: 1.6;
}

.km-info-card-body code {
    padding: 3px 6px;
    border: 1px solid var(--km-border);
    border-radius: 5px;
    background: var(--km-panel-raised);
    color: var(--km-text);
    font-size: 10.5px;
}

.km-status-note {
    margin: 11px 0 0;
    color: var(--km-muted-soft);
    font-size: 10.5px;
    line-height: 1.55;
}

/* ============================================================
   CHIPS + EVIDENCE
   ============================================================ */

.km-signal-row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}

.km-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 10px;
    border: 1px solid var(--km-border);
    border-radius: 999px;
    background: var(--km-panel-raised);
    color: var(--km-muted);
    font-size: 10.5px;
    font-weight: 500;
}

.km-chip::before {
    content: "";
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: var(--km-muted-soft);
}

.km-chip.ok {
    border-color: var(--km-accent-border);
    background: var(--km-accent-soft);
    color: var(--km-accent-strong);
}

.km-chip.ok::before {
    background: var(--km-accent);
}

.km-chip.success {
    border-color: rgba(88, 214, 160, .35);
    background: var(--km-success-soft);
    color: var(--km-success);
}

.km-chip.success::before {
    background: var(--km-success);
}

.km-chip.warning {
    border-color: rgba(229, 173, 93, .35);
    background: var(--km-warning-soft);
    color: var(--km-warning);
}

.km-chip.warning::before {
    background: var(--km-warning);
}

.km-chip.danger {
    border-color: rgba(239, 115, 126, .35);
    background: var(--km-danger-soft);
    color: var(--km-danger);
}

.km-chip.danger::before {
    background: var(--km-danger);
}

.km-chip .num {
    font-family: "SF Mono", "JetBrains Mono", monospace;
    font-weight: 700;
}

.km-sources {
    display: grid;
    gap: 8px;
    margin-top: 10px;
}

.km-source {
    padding: 12px 13px;
    border: 1px solid var(--km-border-soft);
    border-radius: var(--km-radius-sm);
    background: var(--km-panel-raised);
}

.km-source:hover {
    border-color: var(--km-border);
}

.km-source-meta {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
    color: var(--km-muted);
    font-family: "SF Mono", "JetBrains Mono", monospace;
    font-size: 10px;
}

.km-source-number {
    color: var(--km-accent-strong);
    font-weight: 700;
}

.km-source-name {
    color: var(--km-text);
    font-weight: 700;
}

.km-source-type {
    padding: 2px 7px;
    border: 1px solid var(--km-border);
    border-radius: 999px;
    background: var(--km-panel);
    color: var(--km-muted);
}

.km-origin-badge {
    padding: 2px 8px;
}

.km-source-score {
    color: var(--km-muted-soft);
}

.km-source-body {
    max-height: 190px;
    overflow-y: auto;
    color: var(--km-muted);
    font-size: 11.5px;
    line-height: 1.6;
}

.km-source-reason {
    margin-top: 7px;
    color: var(--km-muted-soft);
    font-size: 10px;
}

.km-source-id {
    margin-top: 8px;
    color: var(--km-muted-soft);
    font-family: "SF Mono", "JetBrains Mono", monospace;
    font-size: 9px;
    word-break: break-all;
}

.km-source-link {
    color: var(--km-accent-strong);
    font-weight: 600;
    text-decoration: none;
}

.km-source-link:hover {
    text-decoration: underline;
}

.km-empty-note {
    margin-top: 10px;
    padding: 10px 11px;
    border: 1px dashed var(--km-border);
    border-radius: var(--km-radius-sm);
    color: var(--km-muted-soft);
    font-size: 10.5px;
    line-height: 1.55;
}

.km-provenance-table {
    width: 100%;
    margin-top: 10px;
    border-collapse: collapse;
    font-size: 10.5px;
}

.km-provenance-table th {
    padding: 7px 8px;
    border-bottom: 1px solid var(--km-border);
    color: var(--km-muted-soft);
    font-size: 9px;
    font-weight: 700;
    letter-spacing: .04em;
    text-align: left;
    text-transform: uppercase;
}

.km-provenance-table td {
    padding: 7px 8px;
    border-bottom: 1px solid var(--km-border-soft);
    color: var(--km-muted);
    font-family: "SF Mono", "JetBrains Mono", monospace;
}

/* ============================================================
   TRACE + NATIVE STREAMLIT WIDGETS
   ============================================================ */

.km-trace-list {
    margin: 9px 0 0;
    padding-left: 20px;
    color: var(--km-muted);
    font-size: 11.5px;
    line-height: 1.75;
}

.km-trace-list b {
    color: var(--km-accent-strong);
}

div[data-testid="stMetric"] {
    padding: 12px;
    border: 1px solid var(--km-border-soft);
    border-radius: var(--km-radius-sm);
    background: var(--km-panel);
}

div[data-testid="stMetricLabel"] {
    color: var(--km-muted-soft) !important;
    font-size: 10px !important;
}

div[data-testid="stMetricValue"] {
    color: var(--km-text) !important;
    font-family: "SF Mono", "JetBrains Mono", monospace;
    font-size: 18px !important;
}

button[data-baseweb="tab"] {
    height: 38px;
    padding: 0 13px;
    color: var(--km-muted) !important;
    font-size: 11.5px !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--km-accent-strong) !important;
}

div[data-baseweb="tab-highlight"] {
    background-color: var(--km-accent) !important;
}

.stButton > button {
    border: 1px solid var(--km-border) !important;
    border-radius: 8px !important;
    background: var(--km-panel-raised) !important;
    color: var(--km-text) !important;
    font-size: 12px !important;
    font-weight: 500 !important;
}

.stButton > button:hover {
    border-color: var(--km-accent-border) !important;
}

div[data-testid="stChatInput"] {
    border-top: 1px solid var(--km-border-soft);
    background: var(--km-panel);
}

div[data-testid="stChatInput"] textarea {
    border: 1px solid var(--km-border) !important;
    border-radius: 10px !important;
    background: var(--km-panel-raised) !important;
    color: var(--km-text) !important;
}

div[data-testid="stChatInput"] textarea:focus {
    border-color: var(--km-accent) !important;
    box-shadow: 0 0 0 1px var(--km-accent-soft) !important;
}

div[data-testid="stExpander"] {
    border: 1px solid var(--km-border-soft);
    border-radius: var(--km-radius-sm);
    background: var(--km-panel);
}

/* ============================================================
   EMPTY STATE
   ============================================================ */

.km-starter-card {
    height: 100%;
    padding: 14px 15px;
    border: 1px solid var(--km-border-soft);
    border-radius: var(--km-radius-md);
    background: var(--km-panel-alt);
}

.km-starter-card .icon {
    margin-bottom: 6px;
    font-size: 17px;
}

.km-starter-card .title {
    margin-bottom: 3px;
    color: var(--km-text);
    font-size: 12px;
    font-weight: 650;
}

.km-starter-card .desc {
    color: var(--km-muted-soft);
    font-size: 10.5px;
    line-height: 1.5;
}

/* ============================================================
   RESPONSIVE
   ============================================================ */

@media (max-width: 1100px) {
    .km-run-banner {
        flex-direction: column;
        gap: 14px;
    }

    .km-run-banner-stats {
        width: 100%;
        min-width: 0;
        grid-template-columns: repeat(5, minmax(0, 1fr));
    }
}

@media (max-width: 900px) {
    .km-topbar h1 {
        font-size: 22px;
    }

    .km-pipeline {
        padding: 14px;
        overflow-x: auto;
    }

    .km-step {
        min-width: 108px;
    }

    [class*="st-key-km_bubble_"] {
        max-width: 100%;
    }
}

@media (max-width: 700px) {
    .km-run-banner-stats {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .km-run-stat:nth-child(odd) {
        padding-left: 0;
        border-left: 0;
    }
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# SESSION ACTIONS
# ============================================================


def start_new_session():
    old_session_id = st.session_state.session_id

    if LOGFIRE_OK:
        logfire.info(
            "KnowledgeMesh session reset",
            old_session_id=old_session_id,
        )

    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.latencies = []
    st.session_state.session_started_at = time.strftime("%H:%M:%S")

    check_backend_health.clear()
    check_backend_ready.clear()


def derive_completion_state(
    status_text,
    citation_valid,
    is_grounded,
    web_search_used,
    thought_process,
):
    status_text = str(status_text or "").lower()

    trace_text = " ".join(str(step) for step in (thought_process or [])).lower()

    hard_failure = (
        "generation rate-limited" in status_text
        or "generation failed" in status_text
        or "response generation failed" in status_text
    )

    if hard_failure:
        return "error", "Completed with generation issue"

    if citation_valid is False or is_grounded is False:
        return "complete", "Completed with validation warning"

    if web_search_used:
        return "complete", "Completed with web fallback"

    if (
        "rate limit" in trace_text
        or "ratelimit" in trace_text
        or "failed" in trace_text
    ):
        return "complete", "Completed with recovery"

    return "complete", "Completed successfully"


def ask(question: str):
    question = (question or "").strip()

    if not question:
        return

    backend_url = st.session_state.backend_url

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    response = None

    try:
        with st.status(
            (
                "Running Guardrails → Planner → Qdrant → FlashRank → "
                "Grader → LLM → Validator → Critic…"
            ),
            expanded=True,
        ) as status:
            start_time = time.perf_counter()

            st.write("Connecting to KnowledgeMesh backend…")

            payload = {
                "q": question,
                "thread_id": st.session_state.session_id,
            }

            with _trace_context():
                response = st.session_state.http_session.post(
                    f"{backend_url}/query",
                    json=payload,
                    timeout=BACKEND_TIMEOUT_SECONDS,
                )

            elapsed = time.perf_counter() - start_time

            if response.status_code == 422:
                status.update(
                    label="Request rejected by backend validation (422)",
                    state="error",
                    expanded=False,
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            "**The request format was rejected by the backend.**"
                        ),
                        "error_detail": {
                            "status_code": response.status_code,
                            "body": response.text[:4000],
                        },
                    }
                )

                return

            if response.status_code != 200:
                status.update(
                    label=f"Backend error ({response.status_code})",
                    state="error",
                    expanded=False,
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            "**KnowledgeMesh backend returned an unexpected "
                            f"response (HTTP {response.status_code}).**"
                        ),
                        "error_detail": {
                            "status_code": response.status_code,
                            "body": response.text[:4000],
                        },
                    }
                )

                return

            data = response.json()

            backend_latency_ms = data.get("latency_ms")
            retrieval_latency_ms = data.get("retrieval_latency_ms")
            rerank_latency_ms = data.get("rerank_latency_ms")
            grader_latency_ms = data.get("grader_latency_ms")
            generation_latency_ms = data.get("generation_latency_ms")

            context_quality = data.get("context_quality")
            context_reason = data.get("context_reason")

            should_search_web = data.get("should_search_web")
            web_search_required = data.get("web_search_required")

            web_search_used = bool(data.get("web_search_used", False))

            retrieval_used = bool(data.get("retrieval_used", False))

            citation_valid = data.get("citation_valid")
            is_grounded = data.get("is_grounded")

            answer_supported = data.get("answer_supported")
            answer_useful = data.get("answer_useful")

            support_score = data.get("support_score")
            usefulness_score = data.get("usefulness_score")

            grounding_scores = data.get("grounding_scores")
            grounding_feedback = data.get("grounding_feedback")

            revision_count = data.get("revision_count")
            support_retry_count = data.get("support_retry_count")

            retrieval_rewrite_count = data.get("retrieval_rewrite_count")

            web_rewrite_count = data.get("web_rewrite_count")

            citation_provenance = data.get("citation_provenance") or []

            thought_process = data.get("thought_process", []) or []

            raw_private_sources = (
                data.get("private_sources") or data.get("sources") or []
            )

            raw_answer_sources = data.get("answer_sources") or []
            raw_web_sources = data.get("web_sources") or []

            private_sources = [
                normalize_source(source, index + 1, origin="private")
                for index, source in enumerate(raw_private_sources)
            ]

            web_sources = [
                normalize_source(source, index + 1, origin="web")
                for index, source in enumerate(raw_web_sources)
            ]

            private_by_id = {
                source["id"]: source
                for source in private_sources
                if source.get("id") is not None
            }

            web_by_id = {
                source["id"]: source
                for source in web_sources
                if source.get("id") is not None
            }

            answer_sources = []

            for raw_source in raw_answer_sources:
                source_id = (
                    raw_source.get("id") if isinstance(raw_source, dict) else None
                )

                if source_id is not None and source_id in private_by_id:
                    answer_sources.append(private_by_id[source_id])
                    continue

                if source_id is not None and source_id in web_by_id:
                    answer_sources.append(web_by_id[source_id])
                    continue

                raw_origin = (
                    raw_source.get("origin")
                    if isinstance(raw_source, dict)
                    else "unknown"
                )

                answer_sources.append(
                    normalize_source(
                        raw_source,
                        len(answer_sources) + 1,
                        origin=raw_origin,
                    )
                )

            status_text = data.get(
                "status",
                "Response generated.",
            )

            query_type = infer_query_type(thought_process)

            search_query = data.get("search_query") or extract_search_query(
                thought_process
            )

            visited_nodes = infer_visited_nodes(
                thought_process,
                private_sources,
                status_text,
                context_quality=context_quality,
                web_search_used=web_search_used,
                citation_valid=citation_valid,
                is_grounded=is_grounded,
            )

            if query_type == "conversational" or not retrieval_used:
                private_retrieval_status = "Not used"
            elif web_search_used:
                private_retrieval_status = "Attempted"
            else:
                private_retrieval_status = "Used"

            completion_state, completion_label = derive_completion_state(
                status_text,
                citation_valid,
                is_grounded,
                web_search_used,
                thought_process,
            )

            status.update(
                label=f"{completion_label} in {elapsed:.2f}s",
                state=completion_state,
                expanded=False,
            )

        trace = {
            "steps": thought_process,
            "visited": list(visited_nodes),
            "query_type": query_type,
            "search_query": search_query,
            "sources": private_sources,
            "private_sources": private_sources,
            "web_sources": web_sources,
            "answer_sources": answer_sources,
            "citation_provenance": citation_provenance,
            "context_quality": context_quality,
            "context_reason": context_reason,
            "should_search_web": should_search_web,
            "web_search_required": web_search_required,
            "web_search_used": web_search_used,
            "private_retrieval_status": private_retrieval_status,
            "citation_valid": citation_valid,
            "is_grounded": is_grounded,
            "answer_supported": answer_supported,
            "answer_useful": answer_useful,
            "support_score": support_score,
            "usefulness_score": usefulness_score,
            "grounding_scores": grounding_scores,
            "grounding_feedback": grounding_feedback,
            "revision_count": revision_count,
            "support_retry_count": support_retry_count,
            "retrieval_rewrite_count": retrieval_rewrite_count,
            "web_rewrite_count": web_rewrite_count,
            "latency": elapsed,
            "backend_latency_ms": backend_latency_ms,
            "retrieval_latency_ms": retrieval_latency_ms,
            "rerank_latency_ms": rerank_latency_ms,
            "grader_latency_ms": grader_latency_ms,
            "generation_latency_ms": generation_latency_ms,
            "status": status_text,
            "http_status": response.status_code,
        }

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": data.get(
                    "answer",
                    "No response was returned.",
                ),
                "trace": trace,
            }
        )

        st.session_state.latencies.append(elapsed)

        if LOGFIRE_OK:
            logfire.info(
                "KnowledgeMesh response rendered",
                query_type=query_type,
                private_source_count=len(private_sources),
                web_source_count=len(web_sources),
                answer_source_count=len(answer_sources),
                citation_valid=citation_valid,
                is_grounded=is_grounded,
                latency_seconds=elapsed,
                retrieval_latency_ms=retrieval_latency_ms,
                rerank_latency_ms=rerank_latency_ms,
                grader_latency_ms=grader_latency_ms,
                generation_latency_ms=generation_latency_ms,
            )

    except requests.exceptions.ConnectionError:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    "**Unable to reach KnowledgeMesh backend.**\n\n"
                    f"I could not connect to `{backend_url}`.\n\n"
                    "Check that FastAPI is running:\n\n"
                    "```powershell\n"
                    "uvicorn app.main:app --reload --host 0.0.0.0 --port 8000\n"
                    "```"
                ),
            }
        )

    except requests.exceptions.Timeout:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    "**KnowledgeMesh took too long to respond.**\n\n"
                    "The backend may be loading a model or processing a "
                    "long-running agentic-RAG request.\n\n"
                    f"Current UI timeout: `{BACKEND_TIMEOUT_SECONDS}` seconds."
                ),
            }
        )

    except requests.exceptions.RequestException as exc:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "**Network request failed.**",
                "error_detail": {
                    "body": str(exc),
                },
            }
        )

    except ValueError as exc:
        body_preview = response.text[:4000] if response is not None else ""

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    "**Invalid backend response.**\n\n"
                    "The backend did not return valid JSON."
                ),
                "error_detail": {
                    "status_code": (
                        response.status_code if response is not None else None
                    ),
                    "body": body_preview or str(exc),
                },
            }
        )

    except Exception as exc:
        if LOGFIRE_OK:
            logfire.exception(
                "KnowledgeMesh UI request failed",
                error_type=type(exc).__name__,
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "**Request failed.**",
                "error_detail": {
                    "body": str(exc),
                },
            }
        )


# ============================================================
# BACKEND STATUS
# ============================================================

backend_online = check_backend_health(st.session_state.backend_url)

backend_ready = check_backend_ready(st.session_state.backend_url)

if backend_ready:
    system_dot_class = "success"
    system_text = "KnowledgeMesh ready"

elif backend_online:
    system_dot_class = "warning"
    system_text = "Backend online · not ready"

else:
    system_dot_class = "danger"
    system_text = "Backend unreachable"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    display_html(
        """
        <div class="km-brand">
            <div class="km-brand-mark">◈</div>
            <div>
                <strong>KnowledgeMesh</strong>
                <small>Agentic RAG console</small>
            </div>
        </div>

        <div class="km-rail-label">Session</div>
        """
    )

    user_turns = len(
        [
            message
            for message in st.session_state.messages
            if message.get("role") == "user"
        ]
    )

    display_html(
        f"""
        <div class="km-status-card">
            <div class="km-status-row">
                <span class="label">Backend</span>
                <span class="value">
                    <span class="km-dot {esc(system_dot_class)}"></span>
                    {esc(system_text)}
                </span>
            </div>

            <div class="km-status-row">
                <span class="label">Session</span>
                <span class="value">
                    {esc(st.session_state.session_id[:8].upper())}
                </span>
            </div>

            <div class="km-status-row">
                <span class="label">Started</span>
                <span class="value">
                    {esc(st.session_state.session_started_at)}
                </span>
            </div>

            <div class="km-status-row">
                <span class="label">Questions</span>
                <span class="value">{user_turns}</span>
            </div>
        </div>
        """
    )

    display_html('<div class="km-rail-label">Backend connection</div>')

    backend_url_input = st.text_input(
        "Backend URL",
        value=st.session_state.backend_url,
        label_visibility="collapsed",
        key="km_backend_url_input",
        placeholder="http://127.0.0.1:8000",
    ).rstrip("/")

    if backend_url_input != st.session_state.backend_url:
        st.session_state.backend_url = backend_url_input

        check_backend_health.clear()
        check_backend_ready.clear()

        st.rerun()

    sidebar_left, sidebar_right = st.columns(2)

    with sidebar_left:
        if st.button(
            "New chat",
            width="stretch",
            key="km_new_session_sidebar",
        ):
            start_new_session()
            st.rerun()

    with sidebar_right:
        transcript = transcript_markdown(st.session_state.messages)

        st.download_button(
            "Export",
            data=transcript,
            file_name=(f"knowledgemesh-{st.session_state.session_id[:8]}.md"),
            mime="text/markdown",
            width="stretch",
            disabled=not bool(st.session_state.messages),
            key="km_export_transcript",
        )

    display_html(
        """
        <div class="km-rail-label">Agentic workflow</div>

        <div class="km-capability-list">
            <div class="km-capability-row">
                <span class="mark">✓</span>
                <span>Intent-aware planning</span>
            </div>

            <div class="km-capability-row">
                <span class="mark">✓</span>
                <span>Private knowledge retrieval</span>
            </div>

            <div class="km-capability-row">
                <span class="mark">✓</span>
                <span>Semantic reranking</span>
            </div>

            <div class="km-capability-row">
                <span class="mark">✓</span>
                <span>Evidence and context grading</span>
            </div>

            <div class="km-capability-row">
                <span class="mark">✓</span>
                <span>External web fallback</span>
            </div>

            <div class="km-capability-row">
                <span class="mark">✓</span>
                <span>Citation and grounding review</span>
            </div>
        </div>

        <div class="km-rail-footer">
            <span class="km-dot success"></span>
            <div>
                <strong>Traceable responses</strong>
                <small>
                    Inspect evidence, fallback behavior, validation,
                    latency, and execution state for every answer.
                </small>
            </div>
        </div>
        """
    )


# ============================================================
# TOPBAR
# ============================================================

display_html(
    """
    <div class="km-topbar">
        <div class="km-kicker">KnowledgeMesh</div>

        <h1>Agentic RAG Console</h1>

        <p>
            Ask questions over private knowledge. KnowledgeMesh plans the
            request, retrieves and reranks evidence, evaluates context,
            activates web fallback only when needed, and reviews citations
            and grounding before presenting the final answer.
        </p>

        <div class="km-engine-pill">
            <span class="km-dot success"></span>
            Private retrieval · Evidence grading · Web fallback · Validation
        </div>
    </div>
    """
)


# ============================================================
# CONSOLE HEADER
# ============================================================

console_left, console_right = st.columns([5, 1])

with console_left:
    display_html(
        f"""
        <div class="km-console-head">
            <div class="km-console-live">
                <span class="km-dot {esc(system_dot_class)}"></span>
                Console
            </div>

            <span class="km-session-id">
                {esc(st.session_state.session_id[:8].upper())}
            </span>
        </div>
        """
    )

with console_right:
    if st.button(
        "New conversation",
        width="stretch",
        key="km_new_session_top",
    ):
        start_new_session()
        st.rerun()


# ============================================================
# EMPTY STATE
# ============================================================

if not st.session_state.messages:
    display_html(
        """
        <div class="km-msg-row">
            <div class="km-msg-avatar">K</div>
            <div class="km-msg-label">KnowledgeMesh</div>
        </div>
        """
    )

    with st.container(key="km_bubble_intro"):
        st.markdown(
            """
Welcome to **KnowledgeMesh**.

Ask a question about your private knowledge base. The system can:

**Plan → Retrieve → Rerank → Grade → Use web fallback when needed → Answer → Validate citations → Review grounding**

Each response includes a concise run summary, an agentic pipeline timeline, evidence provenance, quality signals, performance metrics, and a detailed execution trace.
            """
        )

    starter_prompts = [
        (
            "🔎",
            "Explain a concept",
            "What is loop engineering?",
        ),
        (
            "📚",
            "Summarize documentation",
            "Summarize the key points from our documentation.",
        ),
        (
            "🛠️",
            "Troubleshoot",
            "What does the documentation say about rate limiting?",
        ),
    ]

    starter_columns = st.columns(len(starter_prompts))

    for column, (icon, label, question) in zip(
        starter_columns,
        starter_prompts,
    ):
        with column:
            display_html(
                f"""
                <div class="km-starter-card">
                    <div class="icon">{esc(icon)}</div>
                    <div class="title">{esc(label)}</div>
                    <div class="desc">{esc(question)}</div>
                </div>
                """
            )

            st.write("")

            if st.button(
                "Ask",
                key=f"km_starter_{label}",
                width="stretch",
            ):
                ask(question)
                st.rerun()


# ============================================================
# CHAT HISTORY
# ============================================================

else:
    for index, message in enumerate(st.session_state.messages):
        role = message.get("role", "assistant")

        if role == "user":
            display_html(
                """
                <div class="km-msg-row user">
                    <div class="km-msg-label user">You</div>
                </div>
                """
            )

            with st.container(key=f"km_user_bubble_{index}"):
                st.markdown(message.get("content", ""))

            continue

        display_html(
            """
            <div class="km-msg-row">
                <div class="km-msg-avatar">K</div>
                <div class="km-msg-label">KnowledgeMesh</div>
            </div>
            """
        )

        with st.container(key=f"km_bubble_{index}"):
            st.markdown(message.get("content", "No response."))

        error_detail = message.get("error_detail")

        if error_detail:
            with st.expander("Technical error details", expanded=False):
                if error_detail.get("status_code") is not None:
                    st.write(f"HTTP status: `{error_detail['status_code']}`")

                st.code(
                    error_detail.get("body", ""),
                    language="text",
                )

            continue

        trace = message.get("trace")

        if not trace:
            continue

        private_source_list = trace.get(
            "private_sources",
            trace.get("sources", []),
        )

        web_source_list = trace.get("web_sources", [])
        answer_source_list = trace.get("answer_sources", [])

        private_count = len(private_source_list)
        web_count = len(web_source_list)
        answer_count = len(answer_source_list)

        query_type = trace.get("query_type", "technical")
        search_query = trace.get("search_query")

        context_reason = trace.get("context_reason")
        grounding_feedback = trace.get("grounding_feedback")

        citation_provenance = trace.get(
            "citation_provenance",
            [],
        )

        status_text = trace.get("status")

        # Professional summary immediately after the assistant answer.
        display_html(render_run_banner(trace))

        # Clear visual representation of execution stages.
        display_html(render_pipeline_rail(trace))

        # Render only when fallback, recovery, or validation event occurred.
        recovery_notice = render_recovery_notice(trace)

        if recovery_notice:
            display_html(recovery_notice)

        overview_tab, evidence_tab, quality_tab, performance_tab, trace_tab = st.tabs(
            [
                "Overview",
                f"Evidence ({private_count + web_count})",
                "Quality",
                "Performance",
                "Trace",
            ]
        )

        with overview_tab:
            overview_chips = [
                (
                    '<span class="km-chip">'
                    f'Mode <span class="num">'
                    f"{esc(query_type.title())}</span></span>"
                ),
                (
                    '<span class="km-chip">'
                    "Private retrieval "
                    f'<span class="num">'
                    f"{esc(trace.get('private_retrieval_status', 'Not reported'))}"
                    "</span></span>"
                ),
                (
                    '<span class="km-chip">'
                    "Web fallback "
                    f'<span class="num">'
                    f"{'Used' if trace.get('web_search_used') else 'Not used'}"
                    "</span></span>"
                ),
                (
                    '<span class="km-chip">'
                    f'Private sources <span class="num">'
                    f"{private_count}</span></span>"
                ),
                (
                    '<span class="km-chip">'
                    f'Answer sources <span class="num">'
                    f"{answer_count}</span></span>"
                ),
            ]

            display_html(
                """
                <div class="km-tab-panel">
                    <div class="km-tab-heading">Run overview</div>
                    <div class="km-signal-row">
                """
                + "".join(overview_chips)
                + """
                    </div>
                </div>
                """
            )

            if context_reason:
                display_html(
                    f"""
                    <div class="km-info-card">
                        <div class="km-info-card-title">
                            Context evaluation
                        </div>
                        <div class="km-info-card-body">
                            {esc(context_reason)}
                        </div>
                    </div>
                    """
                )

            if search_query:
                display_html(
                    f"""
                    <div class="km-info-card">
                        <div class="km-info-card-title">
                            Planner search query
                        </div>
                        <div class="km-info-card-body">
                            <code>{esc(search_query)}</code>
                        </div>
                    </div>
                    """
                )

            if status_text:
                display_html(
                    f"""
                    <div class="km-status-note">
                        Status: {esc(status_text)}
                    </div>
                    """
                )

        with evidence_tab:
            if private_source_list:
                st.markdown("#### Private knowledge sources")

                display_html(
                    '<div class="km-sources">'
                    f"{render_source_cards(private_source_list)}"
                    "</div>"
                )

            if web_source_list:
                st.markdown("#### Web fallback sources")

                display_html(
                    '<div class="km-sources">'
                    f"{render_source_cards(web_source_list)}"
                    "</div>"
                )

            if answer_source_list:
                st.markdown("#### Sources used in the answer")

                display_html(
                    '<div class="km-sources">'
                    f"{render_source_cards(answer_source_list)}"
                    "</div>"
                )

            elif private_source_list or web_source_list:
                display_html(
                    """
                    <div class="km-empty-note">
                        Retrieved evidence is available above, but the backend
                        did not record any source as directly used in the final
                        answer.
                    </div>
                    """
                )

            provenance_table_html = render_citation_provenance_table(
                citation_provenance
            )

            if provenance_table_html:
                st.markdown("#### Citation provenance")
                display_html(provenance_table_html)

            if not (
                private_source_list
                or web_source_list
                or answer_source_list
                or citation_provenance
            ):
                st.info("No source evidence was reported for this response.")

        with quality_tab:
            quality_html = render_quality_summary(trace)

            if quality_html:
                display_html(
                    """
                    <div class="km-tab-panel">
                        <div class="km-tab-heading">
                            Validation signals
                        </div>
                        <div class="km-signal-row">
                    """
                    + quality_html
                    + """
                        </div>
                    </div>
                    """
                )

            else:
                st.info("No validation results were reported for this response.")

            recovery_html = render_recovery_summary(trace)

            if recovery_html:
                display_html(
                    """
                    <div class="km-tab-panel">
                        <div class="km-tab-heading">
                            Self-RAG and recovery
                        </div>
                        <div class="km-signal-row">
                    """
                    + recovery_html
                    + """
                        </div>
                    </div>
                    """
                )

            if grounding_feedback:
                display_html(
                    f"""
                    <div class="km-info-card">
                        <div class="km-info-card-title">
                            Grounding feedback
                        </div>
                        <div class="km-info-card-body">
                            {esc(grounding_feedback)}
                        </div>
                    </div>
                    """
                )

        with performance_tab:
            metric_1, metric_2, metric_3, metric_4, metric_5 = st.columns(5)

            metric_1.metric(
                "Total",
                format_seconds(trace.get("latency")),
                help="End-to-end Streamlit request duration.",
            )

            metric_2.metric(
                "Retrieval",
                format_ms(trace.get("retrieval_latency_ms")),
                help="Private knowledge retrieval latency.",
            )

            metric_3.metric(
                "Reranking",
                format_ms(trace.get("rerank_latency_ms")),
                help="Semantic reranking latency.",
            )

            metric_4.metric(
                "Grading",
                format_ms(trace.get("grader_latency_ms")),
                help="Document and context evaluation latency.",
            )

            metric_5.metric(
                "Generation",
                format_ms(trace.get("generation_latency_ms")),
                help="LLM response generation latency.",
            )

            backend_latency_ms = trace.get("backend_latency_ms")

            if backend_latency_ms is not None:
                display_html(
                    f"""
                    <div class="km-status-note">
                        Backend-reported total latency:
                        <strong>{esc(format_ms(backend_latency_ms))}</strong>
                    </div>
                    """
                )

        with trace_tab:
            steps = trace.get("steps", [])

            if steps:
                trace_items = []

                for step in steps:
                    _, code, detail = classify_step(step)

                    trace_items.append(f"<li><b>{esc(code)}</b> — {esc(detail)}</li>")

                display_html(
                    f"""
                    <div class="km-tab-panel">
                        <div class="km-tab-heading">
                            Execution trace
                        </div>
                        <ol class="km-trace-list">
                            {"".join(trace_items)}
                        </ol>
                    </div>
                    """
                )

            else:
                st.info("No reasoning trace was reported for this response.")


# ============================================================
# CHAT INPUT
# ============================================================

prompt = st.chat_input("Ask about your documentation...")

if prompt:
    ask(prompt)
    st.rerun()


# ============================================================
# FOOTER
# ============================================================

display_html(
    """
    <div style="
        display:flex;
        flex-wrap:wrap;
        gap:8px;
        margin-top:10px;
        color:var(--km-muted-soft);
        font-size:10px;
    ">
        <span>KnowledgeMesh Agentic RAG</span>
        <span>·</span>
        <span>Private retrieval</span>
        <span>·</span>
        <span>Evidence grading</span>
        <span>·</span>
        <span>Web fallback</span>
        <span>·</span>
        <span>Answer validation</span>
    </div>
    """
)

