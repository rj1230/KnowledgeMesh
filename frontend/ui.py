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
    "http://localhost:8000",
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
# ICONS
# ============================================================
# A small inline-SVG icon set used in place of emoji/unicode glyphs.
# Every path inherits color via currentColor, so icons track the design
# tokens defined in CUSTOM_CSS automatically.

_ICON_PATHS = {
    "search": '<circle cx="11" cy="11" r="6.5"/><path d="M20 20l-4.6-4.6"/>',
    "layers": (
        '<path d="M12 3.5 3.5 8 12 12.5 20.5 8z"/>'
        '<path d="M3.5 13 12 17.5 20.5 13"/>'
        '<path d="M3.5 18 12 22.5 20.5 18"/>'
    ),
    "settings": (
        '<circle cx="12" cy="12" r="3"/>'
        '<path d="M12 3v2.4M12 18.6V21M4.9 4.9l1.7 1.7'
        "M17.4 17.4l1.7 1.7M3 12h2.4M18.6 12H21"
        'M4.9 19.1l1.7-1.7M17.4 6.6l1.7-1.7"/>'
    ),
    "check": '<path d="M5 12.5 9.5 17 19 7"/>',
    "refresh": (
        '<path d="M4 12a8 8 0 0 1 14-5.3L20 8"/>'
        '<path d="M20 4v4h-4"/>'
        '<path d="M20 12a8 8 0 0 1-14 5.3L4 16"/>'
        '<path d="M4 20v-4h4"/>'
    ),
    "cross": '<path d="M6 6l12 12"/><path d="M18 6 6 18"/>',
    "dash": '<path d="M6 12h12"/>',
    "dot": '<circle cx="12" cy="12" r="2.2" fill="currentColor" stroke="none"/>',
    "warning": (
        '<path d="M12 3.5 22 20.5H2z"/>'
        '<path d="M12 9.5v5"/>'
        '<circle cx="12" cy="17.4" r=".9" fill="currentColor" stroke="none"/>'
    ),
}


def icon(name: str, size: int = 14) -> str:
    """Renders a small trusted inline SVG icon from the KM icon set."""

    paths = _ICON_PATHS.get(name, "")

    return (
        f'<svg class="km-icon" width="{size}" height="{size}" '
        'viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
        f'aria-hidden="true">{paths}</svg>'
    )


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
        "success": icon("check", 12),
        "fallback": icon("refresh", 12),
        "failed": icon("cross", 12),
        "skipped": icon("dash", 12),
        "pending": icon("dot", 12),
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
        icon_markup = icon_map.get(state, icon("dot", 12))

        connector = ""

        if index < total:
            connector = f'<div class="km-step-line {esc(state)}"></div>'

        nodes.append(
            f"""
            <div class="km-step {esc(state)}">
                <div class="km-step-marker">
                    <div class="km-step-dot">{icon_markup}</div>
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
                {icon("warning", 13)} Recovery or validation event
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

    grounding_details = trace.get("grounding_details") or {}

    claim_count = grounding_details.get("claim_count")
    atomic_claim_count = grounding_details.get("atomic_claim_count")
    unsupported_atomic_count = grounding_details.get("unsupported_atomic_count")
    entailment_threshold = grounding_details.get("entailment_threshold")

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

    if claim_count:
        chips.append(
            '<span class="km-chip">'
            f'Claims checked <span class="num">{int(claim_count)}</span>'
            "</span>"
        )

    if atomic_claim_count:
        unsupported_class = " danger" if unsupported_atomic_count else ""

        chips.append(
            f'<span class="km-chip{unsupported_class}">'
            f'Atomic claims <span class="num">{int(atomic_claim_count)}</span>'
            "</span>"
        )

    if unsupported_atomic_count:
        chips.append(
            '<span class="km-chip danger">'
            f'Unsupported <span class="num">{int(unsupported_atomic_count)}</span>'
            "</span>"
        )

    if entailment_threshold is not None:
        chips.append(
            '<span class="km-chip">'
            f'Entailment threshold <span class="num">'
            f"{float(entailment_threshold):.2f}</span></span>"
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

    if value in {"private", "internal", "knowledge_base", "kb", "private_kb"}:
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

        origin_raw = entry.get("origin") or entry.get("source_type")
        origin_key = normalize_origin(origin_raw)

        if origin_key == "private":
            origin = "PRIVATE"
        elif origin_key == "web":
            origin = "WEB"
        else:
            origin = str(origin_raw or "—").upper()

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


def render_latency_waterfall(trace: dict) -> str:
    """
    Renders a proportional stage-latency bar from the same
    retrieval/rerank/grader/generation timings already shown as
    raw numbers in the Performance tab.
    """

    trace = trace or {}

    segments = [
        ("Retrieval", trace.get("retrieval_latency_ms"), "var(--km-accent)"),
        ("Reranking", trace.get("rerank_latency_ms"), "var(--km-success)"),
        ("Grading", trace.get("grader_latency_ms"), "var(--km-warning)"),
        ("Generation", trace.get("generation_latency_ms"), "var(--km-danger)"),
    ]

    numeric_segments = [
        (label, float(value), color)
        for label, value, color in segments
        if isinstance(value, (int, float)) and value > 0
    ]

    if not numeric_segments:
        return ""

    total = sum(value for _, value, _ in numeric_segments)

    bars = []
    legend = []

    for label, value, color in numeric_segments:
        share = (value / total) * 100 if total else 0

        bars.append(
            f'<div class="km-waterfall-seg" style="width:{share:.2f}%;'
            f'background:{color};" title="{esc(label)}: {esc(format_ms(value))}">'
            "</div>"
        )

        legend.append(
            f"""
            <div class="km-waterfall-legend-item">
                <span class="km-waterfall-swatch" style="background:{color};"></span>
                <span>{esc(label)}</span>
                <span class="km-waterfall-legend-value">{esc(format_ms(value))}</span>
            </div>
            """
        )

    return f"""
        <div class="km-waterfall">
            <div class="km-waterfall-bar">{"".join(bars)}</div>
            <div class="km-waterfall-legend">{"".join(legend)}</div>
        </div>
    """


def render_claims(claims, grounding_details=None) -> str:
    """
    Renders the Grounding Critic's per-claim entailment review.

    The backend decomposes the answer into atomic claims and checks
    each one for citation and entailment support (see
    _normalize_grounding_scores / _extract_grounding_details on the
    API side). This was already computed but not previously shown.
    """

    if not claims:
        return ""

    grounding_details = grounding_details or {}

    uncited = set(grounding_details.get("uncited_claims") or [])
    invalid_citations = set(grounding_details.get("invalid_citations") or [])

    cards = []

    for claim in claims:
        if not isinstance(claim, dict):
            continue

        text = str(claim.get("claim") or "")

        if not text:
            continue

        supported = claim.get("supported")
        score = claim.get("score")
        citations = claim.get("citations") or []

        state = (
            "supported" if supported else "unsupported" if supported is False else ""
        )

        badge = tri_state_chip(
            "Supported",
            "Unsupported",
            "Unchecked",
            supported,
        )

        chips = [badge]

        if isinstance(score, (int, float)):
            chips.append(
                '<span class="km-chip">'
                f'Entailment <span class="num">{float(score):.2f}</span>'
                "</span>"
            )

        if text in uncited or not citations:
            chips.append('<span class="km-chip danger">Uncited</span>')

        if text in invalid_citations:
            chips.append('<span class="km-chip danger">Invalid citation</span>')

        citations_line = ""

        if citations:
            citation_labels = " · ".join(f"[{esc(c)}]" for c in citations)

            citations_line = (
                f'<div class="km-claim-citations">Cited: {citation_labels}</div>'
            )

        cards.append(
            f"""
            <div class="km-claim {esc(state)}">
                <div class="km-claim-meta">{"".join(chips)}</div>
                <div class="km-claim-text">{esc(text)}</div>
                {citations_line}
            </div>
            """
        )

    if not cards:
        return ""

    return f'<div class="km-claims">{"".join(cards)}</div>'


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
    --km-bg: #070a0f;
    --km-bg-2: #0b1018;
    --km-panel: #0e141d;
    --km-panel-alt: #111923;
    --km-panel-raised: #151f2c;

    --km-border: #243143;
    --km-border-soft: #1a2634;
    --km-border-strong: #314158;

    --km-text: #f4f7fb;
    --km-text-soft: #c6cfdb;
    --km-muted: #8996a8;
    --km-muted-soft: #5f6d80;

    --km-accent: #6d8cff;
    --km-accent-strong: #9db2ff;
    --km-accent-soft: rgba(109, 140, 255, .11);
    --km-accent-border: rgba(109, 140, 255, .34);

    --km-success: #48c997;
    --km-success-soft: rgba(72, 201, 151, .10);
    --km-warning: #e5aa5a;
    --km-warning-soft: rgba(229, 170, 90, .10);
    --km-danger: #ee7180;
    --km-danger-soft: rgba(238, 113, 128, .10);

    --km-radius-xs: 6px;
    --km-radius-sm: 9px;
    --km-radius-md: 13px;
    --km-radius-lg: 17px;

    --km-shadow-sm: 0 2px 8px rgba(0, 0, 0, .20);
    --km-shadow-md: 0 10px 30px rgba(0, 0, 0, .20);
}

html, body, [class*="css"] {
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 78% 0%, rgba(109, 140, 255, .055), transparent 27rem),
        radial-gradient(circle at 15% 20%, rgba(72, 201, 151, .025), transparent 22rem),
        var(--km-bg);
    color: var(--km-text);
}

header[data-testid="stHeader"] {
    background: rgba(7, 10, 15, .72);
}

#MainMenu, footer {
    visibility: hidden;
}

::selection {
    background: rgba(109, 140, 255, .24);
}

@keyframes kmFadeIn {
    from { opacity: 0; transform: translateY(5px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes kmPulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(72, 201, 151, .0); }
    50% { box-shadow: 0 0 0 5px rgba(72, 201, 151, .08); }
}

.km-icon {
    flex: none;
    vertical-align: -2px;
}

/* ============================================================
   SIDEBAR — compact product navigation
   ============================================================ */

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c121a 0%, #0a0f16 100%);
    border-right: 1px solid var(--km-border-soft);
}

section[data-testid="stSidebar"] > div {
    padding-top: .8rem;
}

section[data-testid="stSidebar"] * {
    color: var(--km-text);
}

.km-brand {
    display: flex;
    align-items: center;
    gap: 11px;
    padding: 4px 2px 18px;
    margin-bottom: 17px;
    border-bottom: 1px solid var(--km-border-soft);
}

.km-brand-mark {
    display: grid;
    place-items: center;
    width: 34px;
    height: 34px;
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 10px;
    background: linear-gradient(145deg, #718dff, #526fdf);
    color: #fff;
    font-size: 15px;
    font-weight: 800;
    box-shadow: 0 7px 20px rgba(82, 111, 223, .22);
}

.km-brand strong {
    display: block;
    color: #fff;
    font-size: 14px;
    font-weight: 760;
    letter-spacing: -.01em;
}

.km-brand small {
    display: block;
    margin-top: 2px;
    color: var(--km-muted-soft);
    font-size: 10px;
}

.km-rail-label {
    margin: 19px 0 8px 1px;
    color: #647287;
    font-size: 9.5px;
    font-weight: 750;
    letter-spacing: .10em;
    text-transform: uppercase;
}

.km-status-card {
    padding: 10px 12px;
    border: 1px solid var(--km-border-soft);
    border-radius: var(--km-radius-md);
    background: rgba(17, 25, 35, .78);
    box-shadow: var(--km-shadow-sm);
}

.km-status-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    padding: 6px 0;
    font-size: 10.5px;
}

.km-status-row + .km-status-row {
    border-top: 1px solid rgba(36, 49, 67, .58);
}

.km-status-row .label {
    color: var(--km-muted);
}

.km-status-row .value {
    max-width: 155px;
    overflow: hidden;
    color: var(--km-text-soft);
    font-family: "SF Mono", "JetBrains Mono", monospace;
    font-size: 9.5px;
    text-align: right;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.km-capability-list {
    display: grid;
    gap: 2px;
}

.km-capability-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 5px 6px;
    border-radius: 7px;
    color: var(--km-muted);
    font-size: 10.5px;
    transition: background .15s ease, color .15s ease;
}

.km-capability-row:hover {
    background: rgba(255,255,255,.025);
    color: var(--km-text-soft);
}

.km-capability-row .mark {
    color: var(--km-success);
    font-weight: 750;
}

.km-rail-footer {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin-top: 20px;
    padding: 13px 0 0;
    border-top: 1px solid var(--km-border-soft);
}

.km-rail-footer strong {
    display: block;
    color: var(--km-text-soft);
    font-size: 10.5px;
    font-weight: 700;
}

.km-rail-footer small {
    display: block;
    margin-top: 3px;
    color: var(--km-muted-soft);
    font-size: 9px;
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
    animation: kmPulse 2.4s infinite;
}

.km-dot.warning { background: var(--km-warning); }
.km-dot.danger { background: var(--km-danger); }

section[data-testid="stSidebar"] button,
section[data-testid="stSidebar"] .stDownloadButton button {
    min-height: 34px;
    border: 1px solid var(--km-border) !important;
    border-radius: 8px !important;
    background: #121b27 !important;
    color: var(--km-text-soft) !important;
    font-size: 10.5px !important;
    font-weight: 600 !important;
    transition: border-color .15s ease, background .15s ease;
}

section[data-testid="stSidebar"] button:hover,
section[data-testid="stSidebar"] .stDownloadButton button:hover {
    border-color: var(--km-accent-border) !important;
    background: #172235 !important;
}

section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] div[data-baseweb="input"] {
    border-color: var(--km-border) !important;
    background: #101823 !important;
    color: var(--km-text) !important;
    font-family: "SF Mono", "JetBrains Mono", monospace !important;
    font-size: 9.5px !important;
}

/* ============================================================
   TOP HEADER — product identity
   ============================================================ */

.km-topbar {
    position: relative;
    padding: 7px 0 0;
}

.km-topbar::before {
    content: "";
    display: block;
    width: 42px;
    height: 3px;
    margin-bottom: 15px;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--km-accent), #8ca4ff);
}

.km-topbar h1 {
    margin: 0 0 7px;
    color: #fff;
    font-size: 29px;
    font-weight: 790;
    letter-spacing: -.035em;
}

.km-topbar p {
    max-width: 820px;
    margin: 0;
    color: var(--km-muted);
    font-size: 12.5px;
    line-height: 1.7;
}

/* ============================================================
   CONSOLE HEADER
   ============================================================ */

.km-console-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 12px;
    margin: 24px 0 13px;
    border: 1px solid var(--km-border-soft);
    border-radius: var(--km-radius-md);
    background: rgba(14, 20, 29, .72);
    box-shadow: var(--km-shadow-sm);
}

.km-console-live {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--km-text-soft);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: .02em;
}

.km-session-id {
    padding: 4px 7px;
    border: 1px solid var(--km-border);
    border-radius: 6px;
    background: #0b1119;
    color: var(--km-muted-soft);
    font-family: "SF Mono", "JetBrains Mono", monospace;
    font-size: 8.5px;
    letter-spacing: .06em;
}

.km-msg-row {
    display: flex;
    align-items: center;
    gap: 9px;
    margin: 18px 0 6px;
}

.km-msg-row.user {
    justify-content: flex-end;
    margin-top: 20px;
}

.km-msg-avatar {
    display: grid;
    place-items: center;
    width: 25px;
    height: 25px;
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 8px;
    background: linear-gradient(145deg, #718dff, #526fdf);
    color: #fff;
    font-family: "SF Mono", "JetBrains Mono", monospace;
    font-size: 9px;
    font-weight: 800;
    box-shadow: 0 5px 16px rgba(82,111,223,.18);
}

.km-msg-label {
    color: #748297;
    font-size: 10px;
    font-weight: 700;
}

.km-msg-label.user {
    color: #8e9db1;
    text-align: right;
}

[class*="st-key-km_bubble_"] {
    max-width: min(900px, 94%);
    padding: 17px 19px;
    border: 1px solid var(--km-border-soft);
    border-radius: 5px 15px 15px 15px;
    background:
        linear-gradient(180deg, rgba(17,25,35,.96), rgba(14,20,29,.96));
    color: var(--km-text);
    font-size: 13.2px;
    line-height: 1.75;
    box-shadow: var(--km-shadow-md);
    animation: kmFadeIn .22s ease both;
}

[class*="st-key-km_user_bubble_"] {
    margin-left: auto;
    border-radius: 15px 5px 15px 15px;
    border-color: #2c3a4e;
    background: #131d2a;
    box-shadow: var(--km-shadow-sm);
}

[class*="st-key-km_bubble_intro"] {
    max-width: 900px;
    border-left: 2px solid var(--km-accent);
    background: linear-gradient(135deg, rgba(17,25,35,.98), rgba(13,19,28,.98));
}

[class*="st-key-km_bubble_"] p {
    margin-bottom: 9px;
}

[class*="st-key-km_bubble_"] p:last-child {
    margin-bottom: 0;
}

[class*="st-key-km_bubble_"] code {
    padding: 2px 6px;
    border: 1px solid var(--km-border);
    border-radius: 5px;
    background: #0a1018;
    color: var(--km-accent-strong);
    font-size: 11.5px;
}

/* ============================================================
   RUN SUMMARY — executive status card
   ============================================================ */

.km-run-banner {
    display: flex;
    align-items: stretch;
    justify-content: space-between;
    gap: 20px;
    margin: 15px 0 12px;
    padding: 15px 17px;
    border: 1px solid var(--km-border-soft);
    border-radius: var(--km-radius-md);
    background: linear-gradient(135deg, #101822, #0d141d);
    box-shadow: var(--km-shadow-sm);
    animation: kmFadeIn .2s ease both;
}

.km-run-banner.success { border-left: 3px solid var(--km-success); }
.km-run-banner.ok { border-left: 3px solid var(--km-accent); }
.km-run-banner.warning { border-left: 3px solid var(--km-warning); }
.km-run-banner.danger { border-left: 3px solid var(--km-danger); }

.km-run-banner-main {
    min-width: 0;
    flex: 1.2;
}

.km-run-banner-title {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--km-text);
    font-size: 12.5px;
    font-weight: 750;
}

.km-run-indicator {
    width: 7px;
    height: 7px;
    flex: none;
    border-radius: 50%;
    background: var(--km-accent);
}

.km-run-banner.success .km-run-indicator { background: var(--km-success); }
.km-run-banner.ok .km-run-indicator { background: var(--km-accent); }
.km-run-banner.warning .km-run-indicator { background: var(--km-warning); }
.km-run-banner.danger .km-run-indicator { background: var(--km-danger); }

.km-run-banner-description {
    max-width: 540px;
    margin-top: 5px;
    color: var(--km-muted);
    font-size: 10px;
    line-height: 1.55;
}

.km-run-banner-stats {
    display: grid;
    grid-template-columns: repeat(5, minmax(66px, 1fr));
    align-items: center;
    gap: 12px;
    min-width: 440px;
}

.km-run-stat {
    min-width: 0;
    padding-left: 11px;
    border-left: 1px solid var(--km-border-soft);
}

.km-run-stat span {
    display: block;
    overflow: hidden;
    color: #68778b;
    font-size: 8px;
    font-weight: 750;
    letter-spacing: .08em;
    text-overflow: ellipsis;
    text-transform: uppercase;
    white-space: nowrap;
}

.km-run-stat strong {
    display: block;
    margin-top: 4px;
    overflow: hidden;
    color: var(--km-text-soft);
    font-family: "SF Mono", "JetBrains Mono", monospace;
    font-size: 11px;
    font-weight: 700;
    text-overflow: ellipsis;
    white-space: nowrap;
}

/* ============================================================
   PIPELINE — visual agent execution timeline
   ============================================================ */

.km-pipeline {
    display: flex;
    width: 100%;
    margin: 13px 0;
    padding: 15px 16px 14px;
    border: 1px solid var(--km-border-soft);
    border-radius: var(--km-radius-lg);
    background: linear-gradient(180deg, #0f161f, #0c121a);
    box-shadow: var(--km-shadow-md);
    overflow-x: auto;
    animation: kmFadeIn .2s ease both;
}

.km-step {
    display: flex;
    flex: 1;
    flex-direction: column;
    align-items: flex-start;
    gap: 7px;
    min-width: 105px;
    opacity: .38;
}

.km-step.success,
.km-step.fallback,
.km-step.failed,
.km-step.skipped { opacity: 1; }

.km-step-marker {
    display: flex;
    align-items: center;
    width: 100%;
}

.km-step-dot {
    display: grid;
    place-items: center;
    width: 27px;
    height: 27px;
    flex: none;
    border: 1px solid #2b394c;
    border-radius: 50%;
    background: #111923;
    color: #657387;
    font-family: "SF Mono", "JetBrains Mono", monospace;
    font-size: 10px;
    font-weight: 850;
}

.km-step.success .km-step-dot {
    border-color: rgba(72,201,151,.8);
    background: var(--km-success);
    color: #06110c;
    box-shadow: 0 0 0 3px var(--km-success-soft);
}

.km-step.fallback .km-step-dot {
    border-color: rgba(229,170,90,.8);
    background: var(--km-warning);
    color: #171006;
    box-shadow: 0 0 0 3px var(--km-warning-soft);
}

.km-step.failed .km-step-dot {
    border-color: rgba(238,113,128,.8);
    background: var(--km-danger);
    color: #17080b;
    box-shadow: 0 0 0 3px var(--km-danger-soft);
}

.km-step.skipped { opacity: .52; }

.km-step.skipped .km-step-dot {
    border-style: dashed;
    color: var(--km-muted-soft);
}

.km-step-line {
    flex: 1;
    height: 1px;
    margin: 0 6px;
    background: #263448;
}

.km-step-line.success { background: rgba(72,201,151,.48); }
.km-step-line.fallback { background: rgba(229,170,90,.48); }
.km-step-line.failed { background: rgba(238,113,128,.48); }
.km-step-line.skipped,
.km-step-line.pending { background: #243143; }

.km-step-body strong {
    display: block;
    color: var(--km-text-soft);
    font-size: 10.5px;
    font-weight: 700;
}

.km-step.success .km-step-body strong { color: var(--km-success); }
.km-step.fallback .km-step-body strong { color: var(--km-warning); }
.km-step.failed .km-step-body strong { color: var(--km-danger); }

.km-step-body span {
    display: block;
    margin-top: 2px;
    color: #637187;
    font-size: 8.5px;
    line-height: 1.4;
}

/* ============================================================
   RECOVERY / TABS / INFORMATION
   ============================================================ */

.km-recovery-card {
    margin: 11px 0;
    padding: 10px 12px;
    border: 1px solid rgba(229,170,90,.27);
    border-radius: var(--km-radius-sm);
    background: var(--km-warning-soft);
}

.km-recovery-title {
    color: var(--km-warning);
    font-size: 10.5px;
    font-weight: 700;
}

.km-recovery-body {
    margin-top: 4px;
    color: var(--km-muted);
    font-size: 10px;
    line-height: 1.55;
}

.km-tab-panel {
    padding: 13px;
    margin: 7px 0 9px;
    border: 1px solid var(--km-border-soft);
    border-radius: var(--km-radius-sm);
    background: #0d141d;
}

.km-tab-heading {
    margin-bottom: 9px;
    color: #738197;
    font-size: 9px;
    font-weight: 750;
    letter-spacing: .09em;
    text-transform: uppercase;
}

.km-info-card {
    padding: 11px 12px;
    margin: 9px 0;
    border: 1px solid var(--km-border-soft);
    border-radius: var(--km-radius-sm);
    background: #111923;
}

.km-info-card-title {
    margin-bottom: 5px;
    color: var(--km-accent-strong);
    font-size: 10px;
    font-weight: 700;
}

.km-info-card-body {
    color: var(--km-muted);
    font-size: 10.5px;
    line-height: 1.6;
}

.km-info-card-body code {
    padding: 3px 6px;
    border: 1px solid var(--km-border);
    border-radius: 5px;
    background: #0a1018;
    color: var(--km-text-soft);
    font-size: 10px;
}

.km-status-note {
    margin: 10px 0 0;
    color: var(--km-muted-soft);
    font-size: 9.5px;
    line-height: 1.55;
}

/* ============================================================
   CHIPS / EVIDENCE
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
    padding: 4px 8px;
    border: 1px solid var(--km-border);
    border-radius: 999px;
    background: #121b26;
    color: #8997aa;
    font-size: 9.5px;
    font-weight: 550;
}

.km-chip::before {
    content: "";
    width: 4px;
    height: 4px;
    border-radius: 50%;
    background: #647287;
}

.km-chip.ok {
    border-color: var(--km-accent-border);
    background: var(--km-accent-soft);
    color: var(--km-accent-strong);
}

.km-chip.ok::before { background: var(--km-accent); }

.km-chip.success {
    border-color: rgba(72,201,151,.30);
    background: var(--km-success-soft);
    color: var(--km-success);
}

.km-chip.success::before { background: var(--km-success); }

.km-chip.warning {
    border-color: rgba(229,170,90,.30);
    background: var(--km-warning-soft);
    color: var(--km-warning);
}

.km-chip.warning::before { background: var(--km-warning); }

.km-chip.danger {
    border-color: rgba(238,113,128,.30);
    background: var(--km-danger-soft);
    color: var(--km-danger);
}

.km-chip.danger::before { background: var(--km-danger); }

.km-chip .num {
    color: inherit;
    font-family: "SF Mono", "JetBrains Mono", monospace;
    font-weight: 750;
}

.km-sources {
    display: grid;
    gap: 7px;
    margin-top: 9px;
}

.km-source {
    padding: 11px 12px;
    border: 1px solid var(--km-border-soft);
    border-radius: var(--km-radius-sm);
    background: #111923;
    animation: kmFadeIn .2s ease both;
    transition: border-color .15s ease, transform .15s ease;
}

.km-source:hover {
    border-color: #2c3b50;
    transform: translateY(-1px);
}

.km-source-meta {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 7px;
    margin-bottom: 7px;
    color: var(--km-muted);
    font-family: "SF Mono", "JetBrains Mono", monospace;
    font-size: 9px;
}

.km-source-number {
    color: var(--km-accent-strong);
    font-weight: 700;
}

.km-source-name {
    color: var(--km-text-soft);
    font-weight: 700;
}

.km-source-type {
    padding: 2px 6px;
    border: 1px solid var(--km-border);
    border-radius: 999px;
    background: #0d141d;
    color: var(--km-muted);
}

.km-origin-badge { padding: 2px 7px; }
.km-source-score { color: #68778b; }

.km-source-body {
    max-height: 190px;
    overflow-y: auto;
    color: #9aa7b8;
    font-size: 10.5px;
    line-height: 1.6;
}

.km-source-reason {
    margin-top: 7px;
    color: var(--km-muted-soft);
    font-size: 9px;
}

.km-source-id {
    margin-top: 7px;
    color: #566477;
    font-family: "SF Mono", "JetBrains Mono", monospace;
    font-size: 8.5px;
    word-break: break-all;
}

.km-source-link {
    color: var(--km-accent-strong);
    font-weight: 650;
    text-decoration: none;
}

.km-source-link:hover { text-decoration: underline; }

.km-empty-note {
    margin-top: 9px;
    padding: 9px 10px;
    border: 1px dashed #2b394b;
    border-radius: var(--km-radius-sm);
    color: var(--km-muted-soft);
    font-size: 9.5px;
    line-height: 1.55;
}

.km-provenance-table {
    width: 100%;
    margin-top: 9px;
    border-collapse: collapse;
    font-size: 9.5px;
}

.km-provenance-table th {
    padding: 7px 8px;
    border-bottom: 1px solid var(--km-border);
    color: #66758a;
    font-size: 8px;
    font-weight: 750;
    letter-spacing: .08em;
    text-align: left;
    text-transform: uppercase;
}

.km-provenance-table td {
    padding: 7px 8px;
    border-bottom: 1px solid var(--km-border-soft);
    color: #8997a9;
    font-family: "SF Mono", "JetBrains Mono", monospace;
}

/* ============================================================
   PERFORMANCE / CLAIMS / TRACE
   ============================================================ */

.km-waterfall {
    margin: 2px 0 18px;
}

.km-waterfall-bar {
    display: flex;
    height: 9px;
    overflow: hidden;
    border: 1px solid var(--km-border-soft);
    border-radius: 999px;
    background: #121b26;
}

.km-waterfall-seg { height: 100%; }
.km-waterfall-seg + .km-waterfall-seg {
    border-left: 1px solid var(--km-bg);
}

.km-waterfall-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
    margin-top: 9px;
}

.km-waterfall-legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
    color: var(--km-muted);
    font-size: 9.5px;
}

.km-waterfall-swatch {
    width: 7px;
    height: 7px;
    flex: none;
    border-radius: 2px;
}

.km-waterfall-legend-value {
    color: var(--km-text-soft);
    font-family: "SF Mono", "JetBrains Mono", monospace;
    font-weight: 700;
}

.km-claims {
    display: grid;
    gap: 7px;
    margin-top: 9px;
}

.km-claim {
    padding: 10px 12px;
    border: 1px solid var(--km-border-soft);
    border-left: 2px solid #2a394d;
    border-radius: var(--km-radius-sm);
    background: #111923;
}

.km-claim.supported { border-left-color: var(--km-success); }
.km-claim.unsupported { border-left-color: var(--km-danger); }

.km-claim-meta {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 7px;
    margin-bottom: 5px;
}

.km-claim-text {
    color: var(--km-text-soft);
    font-size: 10.5px;
    line-height: 1.55;
}

.km-claim-citations {
    margin-top: 6px;
    color: #637187;
    font-family: "SF Mono", "JetBrains Mono", monospace;
    font-size: 8.5px;
}

.km-trace-list {
    margin: 8px 0 0;
    padding-left: 19px;
    color: #8997a9;
    font-size: 10.5px;
    line-height: 1.75;
}

.km-trace-list b { color: var(--km-accent-strong); }

/* ============================================================
   STREAMLIT NATIVE WIDGETS
   ============================================================ */

div[data-testid="stMetric"] {
    min-height: 75px;
    padding: 11px 12px;
    border: 1px solid var(--km-border-soft);
    border-radius: var(--km-radius-sm);
    background: linear-gradient(180deg, #101822, #0d141d);
    box-shadow: var(--km-shadow-sm);
}

div[data-testid="stMetricLabel"] {
    color: #6f7e92 !important;
    font-size: 9px !important;
    font-weight: 650 !important;
    letter-spacing: .04em;
    text-transform: uppercase;
}

div[data-testid="stMetricValue"] {
    color: var(--km-text-soft) !important;
    font-family: "SF Mono", "JetBrains Mono", monospace;
    font-size: 17px !important;
}

button[data-baseweb="tab"] {
    height: 37px;
    padding: 0 12px;
    border-radius: 7px 7px 0 0;
    color: #6f7e92 !important;
    font-size: 10.5px !important;
    font-weight: 650 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--km-accent-strong) !important;
}

div[data-baseweb="tab-highlight"] {
    background-color: var(--km-accent) !important;
}

.stButton > button {
    min-height: 35px;
    border: 1px solid var(--km-border) !important;
    border-radius: 8px !important;
    background: #121b27 !important;
    color: var(--km-text-soft) !important;
    font-size: 10.5px !important;
    font-weight: 650 !important;
    transition: all .15s ease;
}

.stButton > button:hover {
    border-color: var(--km-accent-border) !important;
    background: #172235 !important;
    color: #fff !important;
    transform: translateY(-1px);
}

div[data-testid="stChatInput"] {
    border-top: 1px solid var(--km-border-soft);
    background: linear-gradient(180deg, rgba(7,10,15,.86), rgba(7,10,15,.98));
    padding-top: 10px;
}

div[data-testid="stChatInput"] textarea {
    min-height: 48px !important;
    border: 1px solid #2a394c !important;
    border-radius: 12px !important;
    background: #101923 !important;
    color: var(--km-text) !important;
    box-shadow: 0 8px 28px rgba(0,0,0,.18);
}

div[data-testid="stChatInput"] textarea::placeholder {
    color: #66758a !important;
}

div[data-testid="stChatInput"] textarea:focus {
    border-color: var(--km-accent) !important;
    box-shadow: 0 0 0 3px rgba(109,140,255,.10) !important;
}

div[data-testid="stExpander"] {
    border: 1px solid var(--km-border-soft);
    border-radius: var(--km-radius-md);
    background: #0d141d;
    box-shadow: var(--km-shadow-sm);
}

div[data-testid="stExpander"] summary {
    color: var(--km-text-soft);
    font-size: 10.5px;
    font-weight: 700;
}

/* ============================================================
   EMPTY STATE
   ============================================================ */

.km-starter-card {
    height: 100%;
    min-height: 125px;
    padding: 15px;
    border: 1px solid var(--km-border-soft);
    border-radius: var(--km-radius-md);
    background: linear-gradient(145deg, #111923, #0d141d);
    box-shadow: var(--km-shadow-sm);
    transition: transform .15s ease, border-color .15s ease;
}

.km-starter-card:hover {
    border-color: #2c3b50;
    transform: translateY(-2px);
}

.km-starter-card .icon {
    display: grid;
    place-items: center;
    width: 31px;
    height: 31px;
    margin-bottom: 10px;
    border: 1px solid var(--km-accent-border);
    border-radius: 9px;
    background: var(--km-accent-soft);
    color: var(--km-accent-strong);
}

.km-starter-card .title {
    margin-bottom: 4px;
    color: var(--km-text-soft);
    font-size: 11px;
    font-weight: 700;
}

.km-starter-card .desc {
    color: #718095;
    font-size: 9.5px;
    line-height: 1.55;
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
    .km-topbar h1 { font-size: 24px; }

    .km-pipeline {
        padding: 13px;
        overflow-x: auto;
    }

    .km-step { min-width: 105px; }

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

    .km-topbar h1 { font-size: 22px; }
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
            grounding_details = data.get("grounding_details") or {}
            claims = data.get("claims") or []

            revision_count = data.get("revision_count")
            support_retry_count = data.get("support_retry_count")

            retrieval_rewrite_count = data.get("retrieval_rewrite_count")

            web_rewrite_count = data.get("web_rewrite_count")

            # The backend returns citation_provenance as a dict keyed by
            # citation id (see _normalize_citation_provenance in the API),
            # not a list — normalize it here so the provenance table and
            # claim citations below can rely on a consistent list shape.
            raw_citation_provenance = data.get("citation_provenance") or {}

            if isinstance(raw_citation_provenance, dict):
                citation_provenance = list(raw_citation_provenance.values())
            elif isinstance(raw_citation_provenance, list):
                citation_provenance = raw_citation_provenance
            else:
                citation_provenance = []

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
            "grounding_details": grounding_details,
            "claims": claims,
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
        placeholder="http://localhost:8000",
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
        f"""
        <div class="km-rail-label">Agentic workflow</div>

        <div class="km-capability-list">
            <div class="km-capability-row">
                <span class="mark">{icon("check", 12)}</span>
                <span>Intent-aware planning</span>
            </div>

            <div class="km-capability-row">
                <span class="mark">{icon("check", 12)}</span>
                <span>Private knowledge retrieval</span>
            </div>

            <div class="km-capability-row">
                <span class="mark">{icon("check", 12)}</span>
                <span>Semantic reranking</span>
            </div>

            <div class="km-capability-row">
                <span class="mark">{icon("check", 12)}</span>
                <span>Evidence and context grading</span>
            </div>

            <div class="km-capability-row">
                <span class="mark">{icon("check", 12)}</span>
                <span>External web fallback</span>
            </div>

            <div class="km-capability-row">
                <span class="mark">{icon("check", 12)}</span>
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
        <h1>Agentic RAG Console</h1>

        <p>
            Ask questions over your private knowledge base. Each answer is
            planned, retrieved, reranked, and graded — then, only when local
            evidence falls short, supplemented with a web search — before
            citations and grounding are checked against the final answer.
        </p>
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
            "search",
            "Explain a concept",
            "What is loop engineering?",
        ),
        (
            "layers",
            "Summarize documentation",
            "Summarize the key points from our documentation.",
        ),
        (
            "settings",
            "Troubleshoot",
            "What does the documentation say about rate limiting?",
        ),
    ]

    starter_columns = st.columns(len(starter_prompts))

    for column, (icon_name, label, question) in zip(
        starter_columns,
        starter_prompts,
    ):
        with column:
            display_html(
                f"""
                <div class="km-starter-card">
                    <div class="icon">{icon(icon_name, 18)}</div>
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
    last_message_index = len(st.session_state.messages) - 1

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

        with st.expander(
            "Execution details",
            expanded=(index == last_message_index),
        ):
            # Clear visual representation of execution stages.
            display_html(render_pipeline_rail(trace))

            # Render only when fallback, recovery, or validation event occurred.
            recovery_notice = render_recovery_notice(trace)

            if recovery_notice:
                display_html(recovery_notice)

            overview_tab, evidence_tab, quality_tab, performance_tab, trace_tab = (
                st.tabs(
                    [
                        "Overview",
                        f"Evidence ({private_count + web_count})",
                        "Quality",
                        "Performance",
                        "Trace",
                    ]
                )
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

                claims = trace.get("claims") or []
                grounding_details = trace.get("grounding_details") or {}

                claims_html = render_claims(claims, grounding_details)

                if claims_html:
                    st.markdown("#### Claim-level grounding review")
                    display_html(claims_html)

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
                    feedback_items = (
                        grounding_feedback
                        if isinstance(grounding_feedback, list)
                        else [grounding_feedback]
                    )

                    feedback_html = "".join(
                        f"<li>{esc(item)}</li>"
                        for item in feedback_items
                        if str(item).strip()
                    )

                    if feedback_html:
                        display_html(
                            f"""
                            <div class="km-info-card">
                                <div class="km-info-card-title">
                                    Grounding feedback
                                </div>
                                <div class="km-info-card-body">
                                    <ul class="km-trace-list">{feedback_html}</ul>
                                </div>
                            </div>
                            """
                        )

            with performance_tab:
                waterfall_html = render_latency_waterfall(trace)

                if waterfall_html:
                    display_html(waterfall_html)

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

                        trace_items.append(
                            f"<li><b>{esc(code)}</b> — {esc(detail)}</li>"
                        )

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
        margin-top:10px;
        color:var(--km-muted-soft);
        font-size:10.5px;
        line-height:1.5;
    ">
        KnowledgeMesh — self-correcting agentic RAG, running against your
        local FastAPI backend.
    </div>
    """
)
