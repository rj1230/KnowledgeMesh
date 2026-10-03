"""
KnowledgeMesh · Agentic RAG Console

Streamlit frontend for the KnowledgeMesh FastAPI backend.
    POST {BACKEND_URL}/query   {"q": "...", "thread_id": "..."}
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

DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
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
    return value if value.startswith(("http://", "https://")) else ""


def format_score(value) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else "—"


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


def tri_state_chip(label_true, label_false, label_unknown, value, unknown_ok=True):
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
# ICONS (inline SVG, colour follows currentColor)
# ============================================================

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
        '<path d="M4 12a8 8 0 0 1 14-5.3L20 8"/><path d="M20 4v4h-4"/>'
        '<path d="M20 12a8 8 0 0 1-14 5.3L4 16"/><path d="M4 20v-4h4"/>'
    ),
    "cross": '<path d="M6 6l12 12"/><path d="M18 6 6 18"/>',
    "dash": '<path d="M6 12h12"/>',
    "dot": '<circle cx="12" cy="12" r="2.2" fill="currentColor" stroke="none"/>',
    "warning": (
        '<path d="M12 3.5 22 20.5H2z"/><path d="M12 9.5v5"/>'
        '<circle cx="12" cy="17.4" r=".9" fill="currentColor" stroke="none"/>'
    ),
}


def icon(name: str, size: int = 14) -> str:
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

STAGE_TITLES = {key: title for key, title, _ in PIPELINE_STAGES}

STAGE_LATENCY_FIELDS = {
    "retrieval": "retrieval_latency_ms",
    "reranking": "rerank_latency_ms",
    "grader": "grader_latency_ms",
    "responder": "generation_latency_ms",
}

TRACE_PATTERNS = [
    (re.compile(r"guardrail", re.I), "guardrails", "GRD"),
    (re.compile(r"intent|planner|planning", re.I), "planner", "PLN"),
    (
        re.compile(r"retriev|qdrant|context retrieved|knowledge retrieval", re.I),
        "retrieval",
        "RET",
    ),
    (re.compile(r"rerank|flashrank|semantic reranking", re.I), "reranking", "RER"),
    (
        re.compile(r"grader|document grade|context quality|relevance", re.I),
        "grader",
        "GRD",
    ),
    (
        re.compile(r"web search|web fallback|external search|web evidence", re.I),
        "web_search",
        "WEB",
    ),
    (re.compile(r"citation|cite", re.I), "validator", "VAL"),
    (re.compile(r"ground|support|critic|revis", re.I), "critic", "CRT"),
    (re.compile(r"respond|response|synthes|answer|llm", re.I), "responder", "LLM"),
]


# ============================================================
# LATENCY HELPERS
# ============================================================


def stage_latency_ms(trace, key):
    field = STAGE_LATENCY_FIELDS.get(key)
    value = (trace or {}).get(field) if field else None

    return float(value) if isinstance(value, (int, float)) else None


def backend_total_ms(trace):
    trace = trace or {}
    value = trace.get("backend_latency_ms")

    if isinstance(value, (int, float)) and value > 0:
        return float(value)

    latency = trace.get("latency")

    return float(latency) * 1000 if isinstance(latency, (int, float)) else None


def bottleneck_key(trace):
    total = backend_total_ms(trace)
    timed = {k: stage_latency_ms(trace, k) for k in STAGE_LATENCY_FIELDS}
    timed = {k: v for k, v in timed.items() if v}

    if not total or not timed:
        return None

    key = max(timed, key=timed.get)

    return key if timed[key] / total >= 0.4 else None


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

    if "conversational" in text or "retrieval: skipped" in text:
        return "conversational"

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
        visited.update({"retrieval", "reranking"})

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

    return any(
        m in text for m in ("ratelimit", "rate limit", "too many requests", "429")
    )


def normalize_stage_state(value) -> str:
    value = str(value or "").lower().strip()

    return (
        value
        if value in {"success", "fallback", "failed", "skipped", "pending"}
        else "pending"
    )


def _generation_failed(status_text: str) -> bool:
    return (
        "generation rate-limited" in status_text
        or "generation failed" in status_text
        or "response generation failed" in status_text
    )


def stage_statuses(trace: dict) -> dict:
    """UI stage states from the payload fields the backend currently sends."""
    trace = trace or {}

    steps_text = " ".join(str(s) for s in trace.get("steps", []) or []).lower()
    status_text = str(trace.get("status") or "").lower()
    query_type = trace.get("query_type", "technical")

    private_sources = trace.get("private_sources", trace.get("sources", [])) or []
    web_sources = trace.get("web_sources", []) or []
    retrieval_status = trace.get("private_retrieval_status", "Not used")
    context_quality = str(trace.get("context_quality") or "").strip().lower()
    web_used = bool(trace.get("web_search_used"))
    citation_valid = trace.get("citation_valid")
    is_grounded = trace.get("is_grounded")
    revisions = trace.get("revision_count")
    retries = trace.get("support_retry_count")

    grader_failed = (
        "document grade: failed" in steps_text
        or "grader failed" in steps_text
        or is_rate_limit_error(steps_text)
        or is_rate_limit_error(status_text)
    )

    def s(state, detail):
        return {"state": state, "detail": detail}

    out = {
        "guardrails": s("pending", "Not reported"),
        "planner": s("pending", "Not reported"),
        "retrieval": s("skipped", "Not required"),
        "reranking": s("skipped", "Not required"),
        "grader": s("skipped", "Not run"),
        "web_search": s("skipped", "Not required"),
        "responder": s("pending", "No response state reported"),
        "validator": s("skipped", "Not checked"),
        "critic": s("skipped", "Not checked"),
    }

    if "guardrail" in steps_text or "guardrail" in status_text:
        out["guardrails"] = s("success", "Completed")

    if (
        "intent:" in steps_text
        or "search term:" in steps_text
        or "planner" in steps_text
        or query_type == "conversational"
    ):
        out["planner"] = s(
            "success",
            "Conversation path"
            if query_type == "conversational"
            else query_type.title(),
        )

    if query_type == "conversational":
        for key in ("retrieval", "reranking", "grader", "web_search"):
            out[key] = s("skipped", "Conversation path")
        out["responder"] = s("success", "Memory-based answer")

        return out

    if retrieval_status in {"Used", "Attempted"}:
        out["retrieval"] = s(
            "success" if private_sources else "failed",
            f"{len(private_sources)} private source(s)"
            if private_sources
            else "No documents returned",
        )
        out["reranking"] = s(
            "success" if private_sources else "skipped",
            f"Top {len(private_sources)} context chunk(s)"
            if private_sources
            else "No context to rerank",
        )

    if grader_failed:
        out["grader"] = s("failed", "Rate-limited or failed")
    elif trace.get("context_quality") is not None:
        out["grader"] = s("success", f"Context: {context_quality or 'reported'}")

    if web_used:
        out["web_search"] = s("fallback", f"{len(web_sources)} web source(s)")

    if _generation_failed(status_text):
        out["responder"] = s("failed", "Generation failed or rate-limited")
    elif status_text:
        out["responder"] = s(
            "fallback" if web_used else "success",
            "Generated with web fallback" if web_used else "Answer generated",
        )

    if citation_valid is True:
        out["validator"] = s("success", "Citations valid")
    elif citation_valid is False:
        out["validator"] = s("failed", "Citation mismatch")

    if is_grounded is True:
        out["critic"] = s("success", "Grounded")
    elif is_grounded is False:
        out["critic"] = s("failed", "Not grounded")
    elif revisions not in (None, 0) or retries not in (None, 0):
        out["critic"] = s(
            "fallback", f"Revisions {revisions or 0} · Retries {retries or 0}"
        )

    return out


def run_outcome(trace: dict) -> dict:
    trace = trace or {}

    status_text = str(trace.get("status") or "").lower()
    steps_text = " ".join(str(step) for step in trace.get("steps", [])).lower()

    if _generation_failed(status_text):
        return {
            "kind": "failed",
            "title": "Response generation issue",
            "description": (
                "The workflow completed partially, but final response "
                "generation encountered an execution issue."
            ),
        }

    if trace.get("citation_valid") is False or trace.get("is_grounded") is False:
        return {
            "kind": "warning",
            "title": "Completed with validation warning",
            "description": (
                "The system produced a response, but citation or grounding "
                "review reported a quality issue."
            ),
        }

    if trace.get("web_search_used"):
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


# ---- Part 2 continues with: render_run_banner, render_pipeline_rail, ... ----
# ============================================================
# PART 2 · RENDER FUNCTIONS
# (continues directly after ui_part1.py)
# ============================================================


def render_run_banner(trace: dict) -> str:
    trace = trace or {}
    outcome = run_outcome(trace)
    kind = outcome_badge_class(outcome["kind"])

    private_sources = trace.get("private_sources", trace.get("sources", [])) or []
    web_sources = trace.get("web_sources", []) or []
    answer_sources = trace.get("answer_sources", []) or []

    quality = trace.get("context_quality")
    context_label = str(quality).title() if quality else "Not reported"
    web_used = bool(trace.get("web_search_used"))

    stats = [
        ("Total time", esc(format_seconds(trace.get("latency")))),
        ("Private evidence", len(private_sources)),
        ("Web evidence", len(web_sources) if web_used else "—"),
        ("Cited sources", len(answer_sources)),
        ("Context", esc(context_label)),
    ]
    stats_html = "".join(
        f'<div class="km-run-stat"><span>{label}</span><strong>{value}</strong></div>'
        for label, value in stats
    )

    return f"""
        <div class="km-run-banner {esc(kind)}">
            <div class="km-run-banner-main">
                <div class="km-run-banner-title">
                    <span class="km-run-indicator"></span>{esc(outcome["title"])}
                </div>
                <div class="km-run-banner-description">{esc(outcome["description"])}</div>
            </div>
            <div class="km-run-banner-stats">{stats_html}</div>
        </div>
    """


def render_pipeline_rail(trace=None):
    trace = trace or {}
    statuses = stage_statuses(trace)
    slow_key = bottleneck_key(trace)

    icon_map = {
        "success": icon("check", 12),
        "fallback": icon("refresh", 12),
        "failed": icon("cross", 12),
        "skipped": icon("dash", 12),
        "pending": icon("dot", 12),
    }

    nodes = []
    total = len(PIPELINE_STAGES)

    for index, (key, title, default_detail) in enumerate(PIPELINE_STAGES, start=1):
        stage = statuses.get(key, {})
        state = normalize_stage_state(stage.get("state"))
        detail = stage.get("detail") or default_detail
        icon_markup = icon_map.get(state, icon("dot", 12))

        latency_ms = stage_latency_ms(trace, key)
        slow_cls = "slow" if key == slow_key else ""

        if latency_ms is not None:
            latency_html = f'<em class="km-step-time">{esc(format_ms(latency_ms))}</em>'
        elif key in STAGE_LATENCY_FIELDS and state != "skipped":
            latency_html = '<em class="km-step-time muted">no timing</em>'
        else:
            latency_html = ""

        connector = (
            f'<div class="km-step-line {esc(state)}"></div>' if index < total else ""
        )

        nodes.append(
            f"""
            <div class="km-step {esc(state)} {slow_cls}">
                <div class="km-step-marker">
                    <div class="km-step-dot">{icon_markup}</div>
                    {connector}
                </div>
                <div class="km-step-body">
                    <strong>{esc(title)}</strong>
                    <span>{esc(detail)}</span>
                    {latency_html}
                </div>
            </div>
            """
        )

    return f'<div class="km-pipeline">{"".join(nodes)}</div>'


def render_latency_panel(trace: dict) -> str:
    trace = trace or {}
    total = backend_total_ms(trace)

    if not total:
        return ""

    slow = bottleneck_key(trace)
    segments, legend, known = [], [], 0.0

    for key in ("retrieval", "reranking", "grader", "responder"):
        value = stage_latency_ms(trace, key)

        if not value:
            continue

        known += value
        label = STAGE_TITLES[key]
        css = "warn" if key == slow else ""
        color = "var(--km-warning)" if key == slow else "var(--km-success)"

        segments.append(
            f'<i class="km-lat-seg {css}" style="flex:{value:.2f}" '
            f'title="{esc(label)}: {esc(format_ms(value))}"></i>'
        )
        legend.append(
            f'<span><span style="color:{color}">&#9632;</span> {esc(label)} '
            f"<em>{esc(format_ms(value))}</em></span>"
        )

    rest = max(total - known, 0)

    if rest > 0:
        segments.append(
            f'<i class="km-lat-seg unknown" style="flex:{rest:.2f}" '
            'title="Not itemised"></i>'
        )
        legend.append(
            f"<span>&#9640; Not itemised <em>{esc(format_ms(rest))}</em></span>"
        )

    latency = trace.get("latency")
    ui_overhead = (
        max(latency * 1000 - total, 0) if isinstance(latency, (int, float)) else None
    )

    note = ""

    if slow:
        share = stage_latency_ms(trace, slow) / total * 100
        note = (
            f'<div class="km-lat-note">{esc(STAGE_TITLES[slow])} took '
            f"{share:.0f}% of the run.</div>"
        )

    return f"""
        <div class="km-lat">
            <div class="km-lat-head">
                <span>Where the time went</span>
                <span>Backend <b>{esc(format_ms(total))}</b> &middot;
                UI and network <b>{esc(format_ms(ui_overhead))}</b></span>
            </div>
            <div class="km-lat-bar">{"".join(segments)}</div>
            <div class="km-lat-legend">{"".join(legend)}</div>
            {note}
        </div>
    """


def render_stage_detail(trace: dict, key: str) -> str:
    trace = trace or {}
    stage = stage_statuses(trace).get(key, {})
    ms = stage_latency_ms(trace, key)
    total = backend_total_ms(trace)
    share = f"{ms / total * 100:.1f}%" if ms and total else "—"

    tip_html = ""

    if key == bottleneck_key(trace):
        idea = (
            "Lower top-k, keep the vector client warm between requests, and "
            "check whether the embedding call is the slow part."
            if key == "retrieval"
            else "This stage dominates the run. Profile it first."
        )
        tip_html = (
            '<div class="km-stage-box tip"><h5>Speed-up idea</h5>'
            f"<p>{esc(idea)}</p></div>"
        )
    elif ms is None and key in STAGE_LATENCY_FIELDS:
        tip_html = (
            '<div class="km-stage-box tip"><h5>No timing reported</h5>'
            "<p>Return this stage's latency from the API so it shows here.</p></div>"
        )

    return f"""
        <div class="km-stage-detail">
            <div class="km-stage-box">
                <h5>{esc(STAGE_TITLES.get(key, key))}</h5>
                <p>{esc(stage.get("detail", "Not reported"))}</p>
                <div class="nums">
                    <div><span>Time</span><strong>{esc(format_ms(ms))}</strong></div>
                    <div><span>Share of backend</span><strong>{esc(share)}</strong></div>
                </div>
            </div>
            {tip_html}
        </div>
    """


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

    revisions = trace.get("revision_count") or 0
    retries = trace.get("support_retry_count") or 0
    details = []

    if grader_failure:
        details.append("Document grading did not complete successfully.")
    if trace.get("web_search_used"):
        details.append("External web evidence was used as a fallback.")
    if revisions:
        details.append(f"Answer revisions: {revisions}.")
    if retries:
        details.append(f"Support retries: {retries}.")
    if trace.get("citation_valid") is False:
        details.append("Citation validation reported a mismatch.")
    if trace.get("is_grounded") is False:
        details.append("Grounding review reported an issue.")

    if not details:
        return ""

    return f"""
        <div class="km-recovery-card">
            <div class="km-recovery-title">{icon("warning", 13)} Recovery or validation event</div>
            <div class="km-recovery-body">{esc(" ".join(details))}</div>
        </div>
    """


def _score_chip(label, value, danger=False):
    css = " danger" if danger else ""

    return (
        f'<span class="km-chip{css}">{esc(label)} '
        f'<span class="num">{value}</span></span>'
    )


def render_quality_summary(trace: dict) -> str:
    trace = trace or {}
    details = trace.get("grounding_details") or {}

    chips = [
        tri_state_chip(
            "Citations valid",
            "Citation mismatch",
            "Citations not checked",
            trace.get("citation_valid"),
        ),
        tri_state_chip(
            "Grounded",
            "Not grounded",
            "Grounding not checked",
            trace.get("is_grounded"),
        ),
    ]

    if trace.get("answer_supported") is not None:
        chips.append(
            tri_state_chip(
                "Supported", "Not supported", "", trace["answer_supported"], False
            )
        )

    if trace.get("answer_useful") is not None:
        chips.append(
            tri_state_chip("Useful", "Not useful", "", trace["answer_useful"], False)
        )

    if trace.get("support_score") is not None:
        chips.append(_score_chip("Support", f"{float(trace['support_score']):.2f}"))

    if trace.get("usefulness_score") is not None:
        chips.append(
            _score_chip("Usefulness", f"{float(trace['usefulness_score']):.2f}")
        )

    if details.get("claim_count"):
        chips.append(_score_chip("Claims checked", int(details["claim_count"])))

    unsupported = details.get("unsupported_atomic_count")

    if details.get("atomic_claim_count"):
        chips.append(
            _score_chip(
                "Atomic claims", int(details["atomic_claim_count"]), bool(unsupported)
            )
        )

    if unsupported:
        chips.append(_score_chip("Unsupported", int(unsupported), True))

    if details.get("entailment_threshold") is not None:
        chips.append(
            _score_chip(
                "Entailment threshold", f"{float(details['entailment_threshold']):.2f}"
            )
        )

    return "".join(chip for chip in chips if chip)


def render_recovery_summary(trace: dict) -> str:
    trace = trace or {}
    items = [
        ("Query rewrites", trace.get("retrieval_rewrite_count")),
        ("Answer revisions", trace.get("revision_count")),
        ("Support retries", trace.get("support_retry_count")),
        ("Web rewrites", trace.get("web_rewrite_count")),
    ]

    return "".join(_score_chip(label, int(v)) for label, v in items if v is not None)


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

    if not isinstance(source, dict):
        source = {"content": str(source)}

    chunk_id = source.get("chunk_id")

    return {
        "n": index,
        "text": str(
            source.get("content") or source.get("text") or source.get("snippet") or ""
        ),
        "name": str(
            source.get("source")
            or source.get("filename")
            or source.get("title")
            or "Unknown document"
        ),
        "source_type": str(source.get("source_type") or "unknown"),
        "origin": origin,
        "id": source.get("id") or source.get("document_id"),
        "chunk_id": chunk_id,
        "citation_id": source.get("citation_id") or source.get("citation") or chunk_id,
        "score": source.get("score"),
        "rerank_score": source.get("rerank_score"),
        "grader_score": source.get("grader_score"),
        "grader_relevant": source.get("grader_relevant"),
        "grader_reason": source.get("grader_reason"),
        "relevance": source.get("relevance") or source.get("rank"),
        "url": source.get("url"),
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
    origin_map = {
        "private": ("ok", "PRIVATE"),
        "web": ("warning", "WEB"),
        "unknown": ("", "UNKNOWN"),
    }
    cards = []

    for source in source_list:
        origin = normalize_origin(source.get("origin"))
        origin_class, origin_label = origin_map[origin]
        url = safe_url(source.get("url"))

        if url:
            link_html = (
                f'<a class="km-source-link" href="{esc(url)}" target="_blank" '
                'rel="noopener noreferrer">Open source</a>'
            )
        elif origin == "web":
            link_html = '<span class="km-source-score">No URL provided</span>'
        else:
            link_html = ""

        score_html = ""

        if origin == "private":
            score_html = (
                f'<span class="km-source-score">vector {esc(format_score(source.get("score")))}</span>'
                f'<span class="km-source-score">rerank {esc(format_score(source.get("rerank_score")))}</span>'
            )

        grader_html = ""
        grader_score = format_score(source.get("grader_score"))

        if grader_score != "—":
            relevant = source.get("grader_relevant")
            label = (
                "relevant"
                if relevant is True
                else "rejected"
                if relevant is False
                else "reported"
            )
            grader_html = f'<span class="km-source-score">grader {esc(grader_score)} · {label}</span>'

        citation_html = (
            f'<span class="km-source-score">cite [{esc(source["citation_id"])}]</span>'
            if source.get("citation_id")
            else ""
        )
        reason_html = (
            f'<div class="km-source-reason">Grader: {esc(source["grader_reason"])}</div>'
            if source.get("grader_reason")
            else ""
        )

        cards.append(
            f"""
            <div class="km-source">
                <div class="km-source-meta">
                    <span class="km-source-number">[{source["n"]}]</span>
                    <span class="km-chip {origin_class} km-origin-badge">{esc(origin_label)}</span>
                    <span class="km-source-name">{esc(source.get("name", "Unknown document"))}</span>
                    <span class="km-source-type">{esc(source_type_label(source.get("source_type")))}</span>
                    {score_html}{grader_html}{citation_html}{link_html}
                </div>
                <div class="km-source-body">{esc(source.get("text", ""))}</div>
                {reason_html}
                <div class="km-source-id">ID: {esc(source.get("id") or "Unknown")}</div>
            </div>
            """
        )

    return "".join(cards)


def render_citation_provenance_table(provenance_list):
    rows = []

    for entry in provenance_list or []:
        if not isinstance(entry, dict):
            continue

        raw_origin = entry.get("origin") or entry.get("source_type")
        key = normalize_origin(raw_origin)
        origin = {"private": "PRIVATE", "web": "WEB"}.get(
            key, str(raw_origin or "—").upper()
        )
        color = {"PRIVATE": "var(--km-success)", "WEB": "var(--km-warning)"}.get(origin)
        style = f' style="color:{color};"' if color else ""

        url = safe_url(entry.get("url"))
        url_cell = (
            f'<a class="km-source-link" href="{esc(url)}" target="_blank" '
            'rel="noopener noreferrer">Open</a>'
            if url
            else "—"
        )

        rows.append(
            f"""
            <tr>
                <td>[{esc(entry.get("citation") or entry.get("citation_id") or "—")}]</td>
                <td>{esc(entry.get("source") or entry.get("name") or "—")}</td>
                <td{style}>{esc(origin)}</td>
                <td>{esc(entry.get("document") or entry.get("document_id") or "—")}</td>
                <td>{esc(entry.get("chunk") or entry.get("chunk_id") or "—")}</td>
                <td>{url_cell}</td>
            </tr>
            """
        )

    if not rows:
        return ""

    return f"""
        <table class="km-provenance-table">
            <thead><tr>
                <th>Citation</th><th>Source</th><th>Origin</th>
                <th>Document</th><th>Chunk</th><th>URL</th>
            </tr></thead>
            <tbody>{"".join(rows)}</tbody>
        </table>
    """


def render_claims(claims, grounding_details=None) -> str:
    if not claims:
        return ""

    details = grounding_details or {}
    uncited = set(details.get("uncited_claims") or [])
    invalid = set(details.get("invalid_citations") or [])
    cards = []

    for claim in claims:
        text = str(claim.get("claim") or "") if isinstance(claim, dict) else ""

        if not text:
            continue

        supported = claim.get("supported")
        score = claim.get("score")
        citations = claim.get("citations") or []
        state = (
            "supported" if supported else "unsupported" if supported is False else ""
        )

        chips = [tri_state_chip("Supported", "Unsupported", "Unchecked", supported)]

        if isinstance(score, (int, float)):
            chips.append(_score_chip("Entailment", f"{float(score):.2f}"))
        if text in uncited or not citations:
            chips.append('<span class="km-chip danger">Uncited</span>')
        if text in invalid:
            chips.append('<span class="km-chip danger">Invalid citation</span>')

        cited = (
            '<div class="km-claim-citations">Cited: '
            + " · ".join(f"[{esc(c)}]" for c in citations)
            + "</div>"
            if citations
            else ""
        )

        cards.append(
            f"""
            <div class="km-claim {esc(state)}">
                <div class="km-claim-meta">{"".join(chips)}</div>
                <div class="km-claim-text">{esc(text)}</div>
                {cited}
            </div>
            """
        )

    return f'<div class="km-claims">{"".join(cards)}</div>' if cards else ""


# ---- Part 3 continues with: transcript export, backend health, session
# ---- state and history helpers, and the CSS.
# ============================================================
# PART 3 · TRANSCRIPT, HEALTH, SESSION, CSS
# (continues directly after ui_part2.py)
# ============================================================


def transcript_markdown(messages):
    lines = [
        "# KnowledgeMesh Transcript",
        "",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
    ]

    for message in messages:
        speaker = "You" if message.get("role") == "user" else "KnowledgeMesh"
        lines += [f"## {speaker}", "", str(message.get("content", "")), ""]

        trace = message.get("trace")

        if not trace:
            continue

        private = trace.get("private_sources", trace.get("sources", []))

        lines += [
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

        for label, key in (
            ("Context quality", "context_quality"),
            ("Context reason", "context_reason"),
            ("Citation valid", "citation_valid"),
            ("Grounded", "is_grounded"),
            ("Support score", "support_score"),
            ("Usefulness score", "usefulness_score"),
            ("Query rewrites", "retrieval_rewrite_count"),
            ("Answer revisions", "revision_count"),
            ("Support retries", "support_retry_count"),
            ("Web rewrites", "web_rewrite_count"),
        ):
            if trace.get(key) is not None:
                lines.append(f"- {label}: {trace[key]}")

        lines += [
            f"- Private sources retrieved: {len(private)}",
            f"- Web sources used: {len(trace.get('web_sources', []))}",
            f"- Sources used in answer: {len(trace.get('answer_sources', []))}",
            "",
        ]

        if trace.get("search_query"):
            lines += [f"**Planner search query:** `{trace['search_query']}`", ""]

        if trace.get("steps"):
            lines += ["### Reasoning Trace", ""]
            lines += [f"- {step}" for step in trace["steps"]]
            lines.append("")

    return "\n".join(lines)


# ============================================================
# BACKEND HEALTH
# ============================================================


@st.cache_data(ttl=10, show_spinner=False)
def check_backend_health(backend_url):
    try:
        return requests.get(f"{backend_url}/health", timeout=4).ok
    except requests.RequestException:
        return False


@st.cache_data(ttl=10, show_spinner=False)
def check_backend_ready(backend_url):
    try:
        response = requests.get(f"{backend_url}/ready", timeout=4)

        return response.ok and response.json().get("status") == "ready"

    except (requests.RequestException, ValueError):
        return False


@st.cache_data(ttl=60, show_spinner=False)
def fetch_kb_stats(backend_url):
    """Optional GET /stats -> {collection, documents, chunks, last_indexed}."""
    try:
        response = requests.get(f"{backend_url}/stats", timeout=4)

        if response.ok:
            payload = response.json()

            return payload if isinstance(payload, dict) else None

    except (requests.RequestException, ValueError):
        pass

    return None


# ============================================================
# SESSION STATE
# ============================================================

for _key, _default in (
    ("session_id", lambda: str(uuid.uuid4())),
    ("messages", list),
    ("latencies", list),
    ("history", list),
    ("session_started_at", lambda: time.strftime("%H:%M:%S")),
    ("backend_url", lambda: DEFAULT_BACKEND_URL),
):
    if _key not in st.session_state:
        st.session_state[_key] = _default()

if "http_session" not in st.session_state:
    _http = requests.Session()
    _adapter = HTTPAdapter(
        max_retries=Retry(
            total=2,
            backoff_factor=0.5,
            status_forcelist=[502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
    )
    _http.mount("http://", _adapter)
    _http.mount("https://", _adapter)
    st.session_state.http_session = _http


def session_title(messages):
    for message in messages:
        if message.get("role") == "user":
            text = str(message.get("content", "")).strip().replace("\n", " ")

            return text if len(text) <= 38 else text[:35] + "…"

    return "New conversation"


def archive_session():
    messages = st.session_state.messages

    if not messages:
        return

    st.session_state.history.insert(
        0,
        {
            "id": st.session_state.session_id,
            "title": session_title(messages),
            "started": st.session_state.session_started_at,
            "questions": len([m for m in messages if m.get("role") == "user"]),
            "messages": list(messages),
            "latencies": list(st.session_state.latencies),
        },
    )
    st.session_state.history = st.session_state.history[:20]


def restore_session(index):
    entry = st.session_state.history.pop(index)

    archive_session()

    st.session_state.session_id = entry["id"]
    st.session_state.messages = entry["messages"]
    st.session_state.latencies = entry["latencies"]
    st.session_state.session_started_at = entry["started"]


def start_new_session():
    old_session_id = st.session_state.session_id

    archive_session()

    if LOGFIRE_OK:
        logfire.info("KnowledgeMesh session reset", old_session_id=old_session_id)

    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.latencies = []
    st.session_state.session_started_at = time.strftime("%H:%M:%S")

    check_backend_health.clear()
    check_backend_ready.clear()


def derive_completion_state(
    status_text, citation_valid, is_grounded, web_search_used, thought_process
):
    status_text = str(status_text or "").lower()
    trace_text = " ".join(str(step) for step in (thought_process or [])).lower()

    if _generation_failed(status_text):
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


# ============================================================
# CSS
# ============================================================

CUSTOM_CSS = """
<style>
:root{--km-bg:#080d18;--km-panel:#0f1626;--km-panel-alt:#131b2e;--km-panel-raised:#19233a;
--km-border:#2a3555;--km-border-soft:#1c2743;--km-text:#eef2fb;--km-text-soft:#c9d2e4;
--km-muted:#8b98b5;--km-muted-soft:#5d6b8a;--km-accent:#7c93ff;--km-accent-strong:#a5b6ff;
--km-accent-soft:rgba(124,147,255,.11);--km-accent-border:rgba(124,147,255,.34);
--km-success:#3fcf9f;--km-success-soft:rgba(63,207,159,.10);--km-warning:#f2b45c;
--km-warning-soft:rgba(242,180,92,.10);--km-danger:#f2708a;--km-danger-soft:rgba(242,112,138,.10);
--km-radius-sm:9px;--km-radius-md:13px;--km-radius-lg:17px;
--km-shadow-sm:0 2px 8px rgba(0,0,0,.20);--km-shadow-md:0 10px 30px rgba(0,0,0,.20);
--km-mono:"SF Mono","JetBrains Mono",monospace}
html,body,[class*="css"]{font-family:"Inter",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.stApp{background:radial-gradient(circle at 78% 0%,rgba(124,147,255,.07),transparent 27rem),var(--km-bg);color:var(--km-text)}
header[data-testid="stHeader"]{background:rgba(8,13,24,.72)}
#MainMenu,footer{visibility:hidden}
::selection{background:rgba(124,147,255,.24)}
@keyframes kmFadeIn{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:none}}
@keyframes kmPulse{0%,100%{box-shadow:0 0 0 0 rgba(63,207,159,0)}50%{box-shadow:0 0 0 5px rgba(63,207,159,.10)}}
.km-icon{flex:none;vertical-align:-2px}

/* sidebar */
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d1424,#0a101c);border-right:1px solid var(--km-border-soft)}
section[data-testid="stSidebar"]>div{padding-top:.8rem}
section[data-testid="stSidebar"] *{color:var(--km-text)}
.km-brand{display:flex;align-items:center;gap:11px;padding:4px 2px 16px;margin-bottom:12px;border-bottom:1px solid var(--km-border-soft)}
.km-brand-mark{display:grid;place-items:center;width:34px;height:34px;border-radius:10px;background:linear-gradient(145deg,var(--km-accent-strong),var(--km-accent));color:#0a1024;font-size:15px;font-weight:800;box-shadow:0 7px 20px rgba(124,147,255,.22)}
.km-brand strong{display:block;color:#fff;font-size:14px;font-weight:760;letter-spacing:-.01em}
.km-brand small{display:block;margin-top:2px;color:var(--km-muted-soft);font-size:10px}
.km-rail-label{margin:19px 0 8px 1px;color:var(--km-muted-soft);font-size:11px;font-weight:650}
.km-status-card,.km-side-card{border:1px solid var(--km-border-soft);border-radius:12px;background:var(--km-panel-alt);padding:3px 12px;box-shadow:var(--km-shadow-sm)}
.km-side-card.kb{padding:10px 12px 3px}
.km-status-row{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:7px 0;font-size:11px}
.km-status-row+.km-status-row{border-top:1px solid var(--km-border-soft)}
.km-status-row .label{color:var(--km-muted)}
.km-status-row .value{max-width:160px;overflow:hidden;color:var(--km-text-soft);font-family:var(--km-mono);font-size:10px;text-align:right;text-overflow:ellipsis;white-space:nowrap}
.km-kb-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;font-size:12px;color:var(--km-text-soft);font-weight:700}
.km-kb-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:8px}
.km-kb-grid b{display:block;font:700 16px var(--km-mono);color:var(--km-text)}
.km-kb-grid span{font-size:10.5px;color:var(--km-muted-soft)}
.km-hist-active{padding:8px 10px;margin-bottom:4px;border-left:2px solid var(--km-accent);border-radius:9px;background:var(--km-panel-raised)}
.km-hist-active b{display:block;color:var(--km-text-soft);font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.km-hist-active small{color:var(--km-muted-soft);font-size:10.5px}
.km-capability-list{display:grid;gap:2px}
.km-capability-row{display:flex;align-items:center;gap:8px;padding:5px 6px;border-radius:7px;color:var(--km-muted);font-size:10.5px}
.km-capability-row .mark{color:var(--km-success)}
.km-dot{display:inline-block;width:7px;height:7px;flex:none;border-radius:50%;background:var(--km-muted-soft)}
.km-dot.success{background:var(--km-success);animation:kmPulse 2.4s infinite}
.km-dot.warning{background:var(--km-warning)}.km-dot.danger{background:var(--km-danger)}
section[data-testid="stSidebar"] button,section[data-testid="stSidebar"] .stDownloadButton button{min-height:34px;border:1px solid var(--km-border)!important;border-radius:8px!important;background:var(--km-panel-raised)!important;color:var(--km-text-soft)!important;font-size:11px!important;font-weight:600!important;transition:border-color .15s}
section[data-testid="stSidebar"] button:hover{border-color:var(--km-accent-border)!important}
section[data-testid="stSidebar"] button[kind="primary"]{border:0!important;background:linear-gradient(145deg,var(--km-accent-strong),var(--km-accent))!important;box-shadow:0 8px 20px -10px var(--km-accent)}
section[data-testid="stSidebar"] button[kind="primary"] p{color:#0a1024!important;font-weight:700}
section[data-testid="stSidebar"] input{border-color:var(--km-border)!important;background:var(--km-panel-alt)!important;color:var(--km-text)!important;font-size:11px!important}

/* header + chat */
.km-topbar{padding:7px 0 0}
.km-topbar::before{content:"";display:block;width:42px;height:3px;margin-bottom:15px;border-radius:999px;background:linear-gradient(90deg,var(--km-accent),var(--km-success))}
.km-topbar h1{margin:0 0 7px;color:#fff;font-size:29px;font-weight:790;letter-spacing:-.035em}
.km-topbar p{max-width:820px;margin:0;color:var(--km-muted);font-size:12.5px;line-height:1.7}
.km-console-head{display:flex;align-items:center;justify-content:space-between;padding:10px 12px;margin:24px 0 13px;border:1px solid var(--km-border-soft);border-radius:var(--km-radius-md);background:rgba(15,22,38,.72);box-shadow:var(--km-shadow-sm)}
.km-console-live{display:flex;align-items:center;gap:8px;color:var(--km-text-soft);font-size:11px;font-weight:700}
.km-session-id{padding:4px 7px;border:1px solid var(--km-border);border-radius:6px;background:#0a101c;color:var(--km-muted-soft);font:8.5px var(--km-mono);letter-spacing:.06em}
.km-msg-row{display:flex;align-items:center;gap:9px;margin:18px 0 6px}
.km-msg-row.user{justify-content:flex-end;margin-top:20px}
.km-msg-avatar{display:grid;place-items:center;width:25px;height:25px;border-radius:8px;background:linear-gradient(145deg,var(--km-accent-strong),var(--km-accent));color:#0a1024;font:800 9px var(--km-mono)}
.km-msg-label{color:var(--km-muted);font-size:10px;font-weight:700}
[class*="st-key-km_bubble_"]{max-width:min(900px,94%);padding:17px 19px;border:1px solid var(--km-border-soft);border-radius:5px 15px 15px 15px;background:linear-gradient(180deg,rgba(19,27,46,.96),rgba(15,22,38,.96));color:var(--km-text);font-size:13.2px;line-height:1.75;box-shadow:var(--km-shadow-md);animation:kmFadeIn .22s ease both}
[class*="st-key-km_user_bubble_"]{margin-left:auto;border-radius:15px 5px 15px 15px;border-color:var(--km-border);background:var(--km-panel-raised)}
[class*="st-key-km_bubble_intro"]{max-width:900px;border-left:2px solid var(--km-accent)}
[class*="st-key-km_bubble_"] p{margin-bottom:9px}[class*="st-key-km_bubble_"] p:last-child{margin-bottom:0}
[class*="st-key-km_bubble_"] code{padding:2px 6px;border:1px solid var(--km-border);border-radius:5px;background:#0a101c;color:var(--km-accent-strong);font-size:11.5px}

/* run banner */
.km-run-banner{display:flex;align-items:stretch;justify-content:space-between;gap:20px;margin:15px 0 12px;padding:15px 17px;border:1px solid var(--km-border-soft);border-left:3px solid var(--km-muted-soft);border-radius:var(--km-radius-md);background:linear-gradient(135deg,var(--km-panel-alt),var(--km-panel));box-shadow:var(--km-shadow-sm);animation:kmFadeIn .2s ease both}
.km-run-banner.success{border-left-color:var(--km-success)}.km-run-banner.ok{border-left-color:var(--km-accent)}
.km-run-banner.warning{border-left-color:var(--km-warning)}.km-run-banner.danger{border-left-color:var(--km-danger)}
.km-run-banner-main{min-width:0;flex:1.2}
.km-run-banner-title{display:flex;align-items:center;gap:8px;color:var(--km-text);font-size:12.5px;font-weight:750}
.km-run-indicator{width:7px;height:7px;flex:none;border-radius:50%;background:var(--km-accent)}
.km-run-banner.success .km-run-indicator{background:var(--km-success)}.km-run-banner.warning .km-run-indicator{background:var(--km-warning)}.km-run-banner.danger .km-run-indicator{background:var(--km-danger)}
.km-run-banner-description{max-width:540px;margin-top:5px;color:var(--km-muted);font-size:10px;line-height:1.55}
.km-run-banner-stats{display:grid;grid-template-columns:repeat(5,minmax(66px,1fr));align-items:center;gap:12px;min-width:440px}
.km-run-stat{min-width:0;padding-left:11px;border-left:1px solid var(--km-border-soft)}
.km-run-stat span{display:block;overflow:hidden;color:var(--km-muted-soft);font-size:9px;font-weight:700;text-overflow:ellipsis;white-space:nowrap}
.km-run-stat strong{display:block;margin-top:4px;overflow:hidden;color:var(--km-text-soft);font:700 11px var(--km-mono);text-overflow:ellipsis;white-space:nowrap}

/* latency + pipeline */
.km-lat{margin:2px 0 14px}
.km-lat-head{display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;margin-bottom:7px;color:var(--km-muted);font-size:11px}
.km-lat-head b{color:var(--km-text-soft)}
.km-lat-bar{display:flex;gap:2px;height:10px}
.km-lat-seg{display:block;min-width:4px;height:100%;border-radius:3px;background:var(--km-success)}
.km-lat-seg.warn{background:var(--km-warning)}
.km-lat-seg.unknown{background:repeating-linear-gradient(45deg,var(--km-muted-soft) 0 3px,transparent 3px 6px)}
.km-lat-legend{display:flex;flex-wrap:wrap;gap:16px;margin-top:9px;color:var(--km-muted);font-size:10.5px}
.km-lat-legend em{color:var(--km-text-soft);font:700 10.5px var(--km-mono)}
.km-lat-note{margin-top:8px;color:var(--km-warning);font-size:11px}
.km-pipeline{display:flex;width:100%;margin:13px 0;padding:15px 16px 14px;border:1px solid var(--km-border-soft);border-radius:var(--km-radius-lg);background:linear-gradient(180deg,var(--km-panel-alt),var(--km-panel));box-shadow:var(--km-shadow-md);overflow-x:auto}
.km-step{display:flex;flex:1;flex-direction:column;align-items:flex-start;gap:7px;min-width:105px;opacity:.38}
.km-step.success,.km-step.fallback,.km-step.failed,.km-step.skipped{opacity:1}
.km-step.skipped{opacity:.52}
.km-step-marker{display:flex;align-items:center;width:100%}
.km-step-dot{display:grid;place-items:center;width:27px;height:27px;flex:none;border:1px solid var(--km-border);border-radius:50%;background:var(--km-panel-alt);color:var(--km-muted-soft)}
.km-step.success .km-step-dot{border-color:var(--km-success);background:var(--km-success);color:#06110c;box-shadow:0 0 0 3px var(--km-success-soft)}
.km-step.fallback .km-step-dot{border-color:var(--km-warning);background:var(--km-warning);color:#171006;box-shadow:0 0 0 3px var(--km-warning-soft)}
.km-step.failed .km-step-dot{border-color:var(--km-danger);background:var(--km-danger);color:#17080b;box-shadow:0 0 0 3px var(--km-danger-soft)}
.km-step.skipped .km-step-dot{border-style:dashed}
.km-step.slow .km-step-dot{border-color:var(--km-warning);background:var(--km-warning);color:#1a1204;box-shadow:0 0 0 4px var(--km-warning-soft)}
.km-step-line{flex:1;height:1px;margin:0 6px;background:var(--km-border)}
.km-step-line.success{background:rgba(63,207,159,.48)}.km-step-line.fallback{background:rgba(242,180,92,.48)}.km-step-line.failed{background:rgba(242,112,138,.48)}
.km-step-body strong{display:block;color:var(--km-text-soft);font-size:10.5px;font-weight:700}
.km-step.success .km-step-body strong{color:var(--km-success)}.km-step.fallback .km-step-body strong,.km-step.slow .km-step-body strong{color:var(--km-warning)}.km-step.failed .km-step-body strong{color:var(--km-danger)}
.km-step-body span{display:block;margin-top:2px;color:var(--km-muted-soft);font-size:8.5px;line-height:1.4}
.km-step-time{display:block;margin-top:2px;color:var(--km-muted);font:600 10px var(--km-mono);font-style:normal}
.km-step-time.muted{color:var(--km-muted-soft);font-weight:500}
.km-stage-detail{display:grid;grid-template-columns:1.2fr 1fr;gap:12px;margin:10px 0 12px}
.km-stage-box{border:1px solid var(--km-border-soft);border-radius:10px;background:var(--km-panel-alt);padding:12px 14px}
.km-stage-box h5{margin:0 0 4px;color:var(--km-accent-strong);font-size:12px}
.km-stage-box.tip h5{color:var(--km-warning)}
.km-stage-box p{margin:0;color:var(--km-muted);font-size:11.5px;line-height:1.6}
.km-stage-box .nums{display:flex;gap:26px;margin-top:10px}
.km-stage-box .nums span{display:block;color:var(--km-muted-soft);font-size:10px;font-weight:600}
.km-stage-box .nums strong{color:var(--km-text-soft);font:700 13px var(--km-mono)}

/* panels, chips, evidence */
.km-recovery-card{margin:11px 0;padding:10px 12px;border:1px solid rgba(242,180,92,.27);border-radius:var(--km-radius-sm);background:var(--km-warning-soft)}
.km-recovery-title{color:var(--km-warning);font-size:10.5px;font-weight:700}
.km-recovery-body{margin-top:4px;color:var(--km-muted);font-size:10px;line-height:1.55}
.km-tab-panel{padding:13px;margin:7px 0 9px;border:1px solid var(--km-border-soft);border-radius:var(--km-radius-sm);background:var(--km-panel)}
.km-tab-heading{margin-bottom:9px;color:var(--km-muted);font-size:10px;font-weight:700}
.km-info-card{padding:11px 12px;margin:9px 0;border:1px solid var(--km-border-soft);border-radius:var(--km-radius-sm);background:var(--km-panel-alt)}
.km-info-card-title{margin-bottom:5px;color:var(--km-accent-strong);font-size:10px;font-weight:700}
.km-info-card-body{color:var(--km-muted);font-size:10.5px;line-height:1.6}
.km-info-card-body code{padding:3px 6px;border:1px solid var(--km-border);border-radius:5px;background:#0a101c;color:var(--km-text-soft);font-size:10px}
.km-status-note{margin:10px 0 0;color:var(--km-muted-soft);font-size:9.5px;line-height:1.55}
.km-signal-row{display:flex;flex-wrap:wrap;gap:6px}
.km-chip{display:inline-flex;align-items:center;gap:6px;padding:4px 8px;border:1px solid var(--km-border);border-radius:999px;background:var(--km-panel-raised);color:var(--km-muted);font-size:9.5px;font-weight:550}
.km-chip::before{content:"";width:4px;height:4px;border-radius:50%;background:var(--km-muted-soft)}
.km-chip.ok{border-color:var(--km-accent-border);background:var(--km-accent-soft);color:var(--km-accent-strong)}.km-chip.ok::before{background:var(--km-accent)}
.km-chip.success{border-color:rgba(63,207,159,.30);background:var(--km-success-soft);color:var(--km-success)}.km-chip.success::before{background:var(--km-success)}
.km-chip.warning{border-color:rgba(242,180,92,.30);background:var(--km-warning-soft);color:var(--km-warning)}.km-chip.warning::before{background:var(--km-warning)}
.km-chip.danger{border-color:rgba(242,112,138,.30);background:var(--km-danger-soft);color:var(--km-danger)}.km-chip.danger::before{background:var(--km-danger)}
.km-chip .num{color:inherit;font-family:var(--km-mono);font-weight:750}
.km-sources{display:grid;gap:7px;margin-top:9px}
.km-source{padding:11px 12px;border:1px solid var(--km-border-soft);border-radius:var(--km-radius-sm);background:var(--km-panel-alt);animation:kmFadeIn .2s ease both;transition:border-color .15s}
.km-source:hover{border-color:var(--km-border)}
.km-source-meta{display:flex;flex-wrap:wrap;align-items:center;gap:7px;margin-bottom:7px;color:var(--km-muted);font:9px var(--km-mono)}
.km-source-number{color:var(--km-accent-strong);font-weight:700}.km-source-name{color:var(--km-text-soft);font-weight:700}
.km-source-type{padding:2px 6px;border:1px solid var(--km-border);border-radius:999px;background:var(--km-panel);color:var(--km-muted)}
.km-origin-badge{padding:2px 7px}.km-source-score{color:var(--km-muted-soft)}
.km-source-body{max-height:190px;overflow-y:auto;color:var(--km-muted);font-size:10.5px;line-height:1.6}
.km-source-reason{margin-top:7px;color:var(--km-muted-soft);font-size:9px}
.km-source-id{margin-top:7px;color:var(--km-muted-soft);font:8.5px var(--km-mono);word-break:break-all}
.km-source-link{color:var(--km-accent-strong);font-weight:650;text-decoration:none}.km-source-link:hover{text-decoration:underline}
.km-empty-note{margin-top:9px;padding:9px 10px;border:1px dashed var(--km-border);border-radius:var(--km-radius-sm);color:var(--km-muted-soft);font-size:9.5px;line-height:1.55}
.km-provenance-table{width:100%;margin-top:9px;border-collapse:collapse;font-size:9.5px}
.km-provenance-table th{padding:7px 8px;border-bottom:1px solid var(--km-border);color:var(--km-muted-soft);font-size:9px;font-weight:700;text-align:left}
.km-provenance-table td{padding:7px 8px;border-bottom:1px solid var(--km-border-soft);color:var(--km-muted);font-family:var(--km-mono)}
.km-claims{display:grid;gap:7px;margin-top:9px}
.km-claim{padding:10px 12px;border:1px solid var(--km-border-soft);border-left:2px solid var(--km-border);border-radius:var(--km-radius-sm);background:var(--km-panel-alt)}
.km-claim.supported{border-left-color:var(--km-success)}.km-claim.unsupported{border-left-color:var(--km-danger)}
.km-claim-meta{display:flex;flex-wrap:wrap;align-items:center;gap:7px;margin-bottom:5px}
.km-claim-text{color:var(--km-text-soft);font-size:10.5px;line-height:1.55}
.km-claim-citations{margin-top:6px;color:var(--km-muted-soft);font:8.5px var(--km-mono)}
.km-trace-list{margin:8px 0 0;padding-left:19px;color:var(--km-muted);font-size:10.5px;line-height:1.75}.km-trace-list b{color:var(--km-accent-strong)}

/* streamlit widgets */
button[data-baseweb="tab"]{height:37px;padding:0 12px;color:var(--km-muted)!important;font-size:10.5px!important;font-weight:650!important}
button[data-baseweb="tab"][aria-selected="true"]{color:var(--km-accent-strong)!important}
div[data-baseweb="tab-highlight"]{background-color:var(--km-accent)!important}
.stButton>button{min-height:35px;border:1px solid var(--km-border)!important;border-radius:8px!important;background:var(--km-panel-raised)!important;color:var(--km-text-soft)!important;font-size:10.5px!important;font-weight:650!important;transition:all .15s}
.stButton>button:hover{border-color:var(--km-accent-border)!important;color:#fff!important}
div[data-testid="stChatInput"]{border-top:1px solid var(--km-border-soft);background:linear-gradient(180deg,rgba(8,13,24,.86),rgba(8,13,24,.98));padding-top:10px}
div[data-testid="stChatInput"] textarea{min-height:48px!important;border:1px solid var(--km-border)!important;border-radius:12px!important;background:var(--km-panel-alt)!important;color:var(--km-text)!important}
div[data-testid="stChatInput"] textarea:focus{border-color:var(--km-accent)!important;box-shadow:0 0 0 3px rgba(124,147,255,.10)!important}
div[data-testid="stExpander"]{border:1px solid var(--km-border-soft);border-radius:var(--km-radius-md);background:var(--km-panel);box-shadow:var(--km-shadow-sm)}
div[data-testid="stExpander"] summary{color:var(--km-text-soft);font-size:10.5px;font-weight:700}

/* empty state */
.km-starter-card{height:100%;min-height:125px;padding:15px;border:1px solid var(--km-border-soft);border-radius:var(--km-radius-md);background:linear-gradient(145deg,var(--km-panel-alt),var(--km-panel));box-shadow:var(--km-shadow-sm);transition:transform .15s,border-color .15s}
.km-starter-card:hover{border-color:var(--km-border);transform:translateY(-2px)}
.km-starter-card .icon{display:grid;place-items:center;width:31px;height:31px;margin-bottom:10px;border:1px solid var(--km-accent-border);border-radius:9px;background:var(--km-accent-soft);color:var(--km-accent-strong)}
.km-starter-card .title{margin-bottom:4px;color:var(--km-text-soft);font-size:11px;font-weight:700}
.km-starter-card .desc{color:var(--km-muted);font-size:9.5px;line-height:1.55}

@media(max-width:1100px){.km-run-banner{flex-direction:column;gap:14px}.km-run-banner-stats{width:100%;min-width:0}}
@media(max-width:900px){.km-topbar h1{font-size:24px}.km-stage-detail{grid-template-columns:1fr}[class*="st-key-km_bubble_"]{max-width:100%}}
@media(max-width:700px){.km-run-banner-stats{grid-template-columns:repeat(2,minmax(0,1fr))}.km-run-stat:nth-child(odd){padding-left:0;border-left:0}}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---- Part 4 continues with: ask(), backend status, sidebar, topbar,
# ---- empty state, chat history with tabs, chat input, and footer.
