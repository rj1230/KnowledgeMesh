"""
KnowledgeMesh · Agentic RAG Console

Frontend for the KnowledgeMesh FastAPI backend.

Backend:
    POST http://localhost:8000/query

Request:
    {"q": "...", "thread_id": "..."}

Response: see original docstring / API — unchanged from the base app.
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
LOGFIRE_PROJECT_URL = os.getenv("LOGFIRE_PROJECT_URL")


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
# RENDERING HELPERS
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
    value = str(value or "")
    if value.startswith("http://") or value.startswith("https://"):
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
        value = float(value)
        if value >= 1000:
            return f"{value / 1000:.2f}s"
        return f"{value:.0f}ms"
    except (TypeError, ValueError):
        return "—"


def _trace_context():
    if LOGFIRE_OK:
        return logfire.span("KnowledgeMesh UI operation")
    return nullcontext()


# ============================================================
# PIPELINE
# ============================================================

PIPELINE_STAGES = [
    ("guardrails", "Guardrails", "Safety and policy check"),
    ("planner", "Planner", "Intent and search planning"),
    ("retrieval", "Qdrant", "Vector knowledge retrieval"),
    ("reranking", "FlashRank", "Semantic reranking"),
    ("grader", "Grader", "Evidence relevance evaluation"),
    ("responder", "LLM", "Grounded response synthesis"),
]

TRACE_PATTERNS = [
    (re.compile(r"guardrail", re.IGNORECASE), "guardrails", "GRD"),
    (re.compile(r"intent|planner|planning", re.IGNORECASE), "planner", "PLN"),
    (
        re.compile(
            r"retriev|qdrant|context retrieved|knowledge retrieval", re.IGNORECASE
        ),
        "retrieval",
        "RET",
    ),
    (
        re.compile(r"rerank|flashrank|semantic reranking", re.IGNORECASE),
        "reranking",
        "RER",
    ),
    (
        re.compile(r"grader|document grade|context quality|relevance", re.IGNORECASE),
        "grader",
        "GRD",
    ),
    (
        re.compile(r"respond|response|synthes|answer|llm", re.IGNORECASE),
        "responder",
        "LLM",
    ),
]


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
            return (stage_key, code, detail)

    return ("", "STEP", detail)


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
    thought_process, sources, status, context_quality=None, web_search_used=False
):
    visited = set()
    steps = thought_process or []
    status_text = str(status or "").lower()
    combined_text = " ".join(str(step) for step in steps).lower()

    for step in steps:
        stage_key, _, _ = classify_step(step)
        if stage_key:
            visited.add(stage_key)

    if "guardrail" in combined_text or "guardrail" in status_text or steps:
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
        or context_quality
    ):
        visited.add("grader")

    if status:
        visited.add("responder")

    return visited


# ============================================================
# SOURCE NORMALIZATION
# ============================================================


def normalize_source(source, index):
    if isinstance(source, dict):
        content = source.get("content") or source.get("text") or ""
        name = (
            source.get("source")
            or source.get("filename")
            or source.get("title")
            or "Unknown document"
        )
        source_type = source.get("source_type") or "unknown"
        document_id = source.get("id")
        vector_score = source.get("score")
        rerank_score = source.get("rerank_score")
        grader_score = source.get("grader_score")
        grader_relevant = source.get("grader_relevant")
        grader_reason = source.get("grader_reason")
        url = source.get("url")
    else:
        content = str(source)
        name = "Unknown document"
        source_type = "unknown"
        document_id = None
        vector_score = None
        rerank_score = None
        grader_score = None
        grader_relevant = None
        grader_reason = None
        url = None

    return {
        "n": index,
        "text": str(content),
        "name": str(name),
        "source_type": str(source_type),
        "id": document_id,
        "score": vector_score,
        "rerank_score": rerank_score,
        "grader_score": grader_score,
        "grader_relevant": grader_relevant,
        "grader_reason": grader_reason,
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


# ============================================================
# PIPELINE RAIL
# ============================================================


def render_pipeline_rail(trace=None):
    trace = trace or {}
    visited = set(trace.get("visited", []))
    query_type = trace.get("query_type", "technical")
    sources = trace.get("sources", [])
    source_count = len(sources)
    context_quality = trace.get("context_quality")

    nodes = []
    total = len(PIPELINE_STAGES)

    for index, (key, title, subtitle) in enumerate(PIPELINE_STAGES, start=1):
        active = key in visited
        current_subtitle = subtitle

        if trace:
            if key == "planner" and query_type == "conversational":
                current_subtitle = "Conversation memory path"
            elif key == "retrieval" and source_count:
                current_subtitle = f"{source_count} document{'s' if source_count != 1 else ''} retrieved"
            elif key == "reranking" and source_count:
                current_subtitle = f"Top {source_count} context chunks"
            elif key == "grader":
                current_subtitle = (
                    f"Context: {str(context_quality).title()}"
                    if context_quality
                    else "Evidence relevance evaluation"
                )
            elif key == "responder" and query_type == "conversational":
                current_subtitle = "Memory-based response"
            elif key == "responder" and source_count:
                current_subtitle = "Grounded synthesis"

        state_class = " active" if active else ""
        line_class = " active" if (active and index < total) else ""

        nodes.append(
            f"""
            <div class="km-step{state_class}">
                <div class="km-step-marker">
                    <div class="km-step-dot">{index}</div>
                    {"" if index == total else f'<div class="km-step-line{line_class}"></div>'}
                </div>
                <div class="km-step-body">
                    <strong>{esc(title)}</strong>
                    <span>{esc(current_subtitle)}</span>
                </div>
            </div>
            """
        )

    return f'<div class="km-pipeline">{"".join(nodes)}</div>'


# ============================================================
# TRANSCRIPT
# ============================================================


def transcript_markdown(messages):
    lines = [
        "# KnowledgeMesh transcript",
        "",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
    ]

    for message in messages:
        role = message.get("role", "assistant")
        speaker = "You" if role == "user" else "KnowledgeMesh"
        lines.extend([f"## {speaker}", "", str(message.get("content", "")), ""])

        trace = message.get("trace")
        if not trace:
            continue

        lines.extend(
            [
                "### Pipeline metrics",
                "",
                f"- Total latency: {format_ms(trace.get('backend_latency_ms'))}",
                f"- Retrieval: {format_ms(trace.get('retrieval_latency_ms'))}",
                f"- Reranking: {format_ms(trace.get('rerank_latency_ms'))}",
                f"- Grader: {format_ms(trace.get('grader_latency_ms'))}",
                f"- Generation: {format_ms(trace.get('generation_latency_ms'))}",
                "",
            ]
        )

        context_quality = trace.get("context_quality")
        if context_quality:
            lines.append(f"- Context quality: {context_quality}")

        support_score = trace.get("support_score")
        if support_score is not None:
            lines.append(f"- Support score: {support_score}")

        usefulness_score = trace.get("usefulness_score")
        if usefulness_score is not None:
            lines.append(f"- Usefulness score: {usefulness_score}")

        lines.append("")

        search_query = trace.get("search_query")
        if search_query:
            lines.extend([f"**Planner search query:** `{search_query}`", ""])

        sources = trace.get("sources", [])
        if sources:
            lines.extend(["### Retrieved sources", ""])
            for source in sources:
                lines.extend(
                    [
                        f"**[{source['n']}] {source['name']}**",
                        "",
                        f"- Source type: {source_type_label(source['source_type'])}",
                        f"- Vector score: {format_score(source['score'])}",
                        f"- Rerank score: {format_score(source['rerank_score'])}",
                        f"- Grader score: {format_score(source.get('grader_score'))}",
                        "",
                        source["text"],
                        "",
                    ]
                )

        steps = trace.get("steps", [])
        if steps:
            lines.extend(["### Reasoning trace", ""])
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
        response = requests.get(f"{backend_url}/health", timeout=4)
        return response.ok
    except requests.RequestException:
        return False


@st.cache_data(ttl=10, show_spinner=False)
def check_backend_ready(backend_url):
    try:
        response = requests.get(f"{backend_url}/ready", timeout=4)
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
    --km-bg: #0b0d12;
    --km-panel: #12141b;
    --km-panel-alt: #171a23;
    --km-panel-raised: #1c1f2a;
    --km-border: #262a37;
    --km-border-soft: #1d2029;

    --km-text: #edeef3;
    --km-muted: #9297a8;
    --km-muted-soft: #666c7d;

    --km-accent: #7c86f5;
    --km-accent-strong: #a8b1ff;
    --km-accent-soft: rgba(124, 134, 245, .13);
    --km-accent-border: rgba(124, 134, 245, .35);

    --km-success: #5fd9a4;
    --km-success-soft: rgba(95, 217, 164, .13);

    --km-warning: #e0a95c;
    --km-warning-soft: rgba(224, 169, 92, .13);

    --km-danger: #f0707a;
    --km-danger-soft: rgba(240, 112, 122, .13);

    --km-radius-sm: 8px;
    --km-radius-md: 12px;
    --km-radius-lg: 16px;

    --km-shadow: 0 1px 2px rgba(0, 0, 0, .35), 0 8px 24px -12px rgba(0, 0, 0, .5);
}

html, body, [class*="css"] {
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.stApp {
    background:
        radial-gradient(1100px 480px at 12% -8%, rgba(124,134,245,.06), transparent 60%),
        var(--km-bg);
    color: var(--km-text);
}

header[data-testid="stHeader"] { background: transparent; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

@keyframes kmFadeIn {
    from { opacity: 0; transform: translateY(5px); }
    to { opacity: 1; transform: translateY(0); }
}

::selection { background: var(--km-accent-soft); }

/* ============================================================
   SIDEBAR
   ============================================================ */

section[data-testid="stSidebar"] {
    background: var(--km-panel);
    border-right: 1px solid var(--km-border-soft);
}

section[data-testid="stSidebar"] > div { padding-top: .5rem; }
section[data-testid="stSidebar"] * { color: var(--km-text); }

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
    flex: none;
    border-radius: 9px;
    background: linear-gradient(155deg, var(--km-accent), #5b63c7);
    box-shadow: 0 4px 14px -4px rgba(124,134,245,.55);
    color: #0b0d12;
    font-weight: 800;
    font-size: 15px;
}

.km-brand strong {
    display: block;
    color: #ffffff;
    font-size: 14.5px;
    font-weight: 700;
    letter-spacing: -.01em;
}

.km-brand small {
    display: block;
    margin-top: 2px;
    color: var(--km-muted-soft);
    font-size: 10.5px;
    letter-spacing: .01em;
}

.km-rail-label {
    display: flex;
    align-items: center;
    gap: 7px;
    margin: 20px 0 9px 1px;
    color: var(--km-muted-soft);
    font-size: 10.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .08em;
}

.km-status-card {
    padding: 13px 14px;
    border: 1px solid var(--km-border-soft);
    border-radius: var(--km-radius-md);
    background: var(--km-panel-alt);
}

.km-status-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 4px 0;
    font-size: 11.5px;
}

.km-status-row + .km-status-row { border-top: 1px solid var(--km-border-soft); }

.km-status-row .label {
    color: var(--km-muted-soft);
    font-weight: 500;
}

.km-status-row .value {
    color: var(--km-text);
    font-family: "SF Mono", "JetBrains Mono", monospace;
    font-size: 11px;
    font-weight: 500;
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
    font-weight: 600;
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

.km-dot.success { background: var(--km-success); box-shadow: 0 0 10px var(--km-success-soft); }
.km-dot.warning { background: var(--km-warning); box-shadow: 0 0 10px var(--km-warning-soft); }
.km-dot.danger  { background: var(--km-danger);  box-shadow: 0 0 10px var(--km-danger-soft); }

section[data-testid="stSidebar"] button {
    background: var(--km-panel-raised) !important;
    border: 1px solid var(--km-border) !important;
    color: var(--km-text) !important;
    font-size: 12px !important;
    border-radius: 8px !important;
    transition: border-color .15s ease, transform .1s ease;
}

section[data-testid="stSidebar"] button:hover {
    border-color: var(--km-accent-border) !important;
}

section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] div[data-baseweb="input"] {
    background: var(--km-panel-raised) !important;
    border-color: var(--km-border) !important;
    color: var(--km-text) !important;
    font-family: "SF Mono", "JetBrains Mono", monospace !important;
    font-size: 11.5px !important;
}

/* ============================================================
   TOPBAR
   ============================================================ */

.km-topbar { padding-top: 2px; }

.km-kicker {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    margin-bottom: 10px;
    color: var(--km-accent-strong);
    font-family: "SF Mono", "JetBrains Mono", monospace;
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: .07em;
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
    font-size: 26px;
    font-weight: 750;
    letter-spacing: -.025em;
}

.km-topbar p {
    max-width: 760px;
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
    font-size: 10.5px;
}

/* ============================================================
   PIPELINE STEPPER
   ============================================================ */

.km-pipeline {
    display: flex;
    width: 100%;
    margin: 22px 0 24px;
    padding: 16px 18px;
    border: 1px solid var(--km-border-soft);
    border-radius: var(--km-radius-lg);
    background: var(--km-panel);
    box-shadow: var(--km-shadow);
}

.km-step {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 8px;
    flex: 1;
    min-width: 0;
    opacity: .4;
    transition: opacity .2s ease;
}

.km-step.active { opacity: 1; }

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
    color: var(--km-accent-strong);
    font-family: "SF Mono", "JetBrains Mono", monospace;
    font-size: 10.5px;
    font-weight: 700;
    transition: all .2s ease;
}

.km-step.active .km-step-dot {
    border-color: var(--km-accent);
    background: var(--km-accent);
    color: #0b0d12;
    box-shadow: 0 0 0 4px var(--km-accent-soft);
}

.km-step-line {
    flex: 1;
    height: 1px;
    margin: 0 6px;
    background: var(--km-border);
}

.km-step-line.active { background: var(--km-accent-border); }

.km-step-body strong {
    display: block;
    color: var(--km-text);
    font-size: 11.5px;
    font-weight: 700;
}

.km-step-body span {
    display: block;
    margin-top: 2px;
    color: var(--km-muted-soft);
    font-size: 9.5px;
    line-height: 1.4;
}

/* ============================================================
   CONSOLE
   ============================================================ */

.km-console-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 2px 12px;
    margin-bottom: 16px;
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

.km-msg-row.user { justify-content: flex-end; }

.km-msg-avatar {
    display: grid;
    place-items: center;
    width: 26px;
    height: 26px;
    border-radius: 7px;
    background: linear-gradient(155deg, var(--km-accent), #5b63c7);
    color: #0b0d12;
    font-family: "SF Mono", "JetBrains Mono", monospace;
    font-size: 11px;
    font-weight: 800;
}

.km-msg-label {
    color: var(--km-muted-soft);
    font-size: 10.5px;
    font-weight: 650;
    text-transform: uppercase;
    letter-spacing: .04em;
}

.km-msg-label.user { text-align: right; }

[class*="st-key-km_bubble_"] {
    max-width: min(850px, 92%);
    padding: 14px 17px;
    border: 1px solid var(--km-border-soft);
    border-radius: 4px 14px 14px 14px;
    background: var(--km-panel-alt);
    color: var(--km-text);
    font-size: 13.5px;
    line-height: 1.7;
    box-shadow: var(--km-shadow);
    animation: kmFadeIn .25s ease both;
}

[class*="st-key-km_user_bubble_"] {
    margin-left: auto;
    border-radius: 14px 4px 14px 14px;
    background: var(--km-panel-raised);
}

[class*="st-key-km_bubble_intro"] {
    border-left: 2px solid var(--km-accent);
    background: linear-gradient(180deg, var(--km-accent-soft), var(--km-panel-alt) 60%);
}

[class*="st-key-km_bubble_"] p { margin-bottom: 9px; }
[class*="st-key-km_bubble_"] p:last-child { margin-bottom: 0; }

[class*="st-key-km_bubble_"] code {
    padding: 2px 5px;
    border: 1px solid var(--km-border);
    border-radius: 4px;
    background: var(--km-panel);
    font-size: 12px;
}

[class*="st-key-km_bubble_"] pre {
    border: 1px solid var(--km-border) !important;
    border-radius: 8px !important;
}

/* ============================================================
   SIGNAL READOUT
   ============================================================ */

.km-signal-readout {
    max-width: min(850px, 92%);
    padding: 12px 14px;
    margin: -6px 0 20px;
    border: 1px solid var(--km-border-soft);
    border-top: 0;
    border-radius: 0 0 14px 14px;
    background: var(--km-panel);
}

.km-signal-row { display: flex; flex-wrap: wrap; gap: 6px; }

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

.km-chip.ok      { border-color: var(--km-accent-border); background: var(--km-accent-soft); color: var(--km-accent-strong); }
.km-chip.ok::before { background: var(--km-accent); }

.km-chip.success { border-color: rgba(95,217,164,.35); background: var(--km-success-soft); color: var(--km-success); }
.km-chip.success::before { background: var(--km-success); }

.km-chip.danger  { border-color: rgba(240,112,122,.35); background: var(--km-danger-soft); color: var(--km-danger); }
.km-chip.danger::before { background: var(--km-danger); }

.km-chip .num { font-family: "SF Mono", "JetBrains Mono", monospace; font-weight: 700; }

/* ============================================================
   SOURCES
   ============================================================ */

.km-details { margin-top: 10px; }

.km-details summary {
    cursor: pointer;
    color: var(--km-muted);
    font-size: 11.5px;
    font-weight: 500;
    padding: 2px 0;
}

.km-details summary:hover { color: var(--km-accent-strong); }

.km-sources { display: grid; gap: 8px; margin-top: 10px; }

.km-source {
    padding: 12px 13px;
    border: 1px solid var(--km-border-soft);
    border-radius: var(--km-radius-sm);
    background: var(--km-panel-raised);
    transition: border-color .15s ease;
}

.km-source:hover { border-color: var(--km-border); }

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

.km-source-number { color: var(--km-accent-strong); font-weight: 700; }
.km-source-name { color: var(--km-text); font-weight: 700; font-family: inherit; }

.km-source-type {
    padding: 2px 7px;
    border: 1px solid var(--km-border);
    border-radius: 999px;
    background: var(--km-panel);
    color: var(--km-muted);
}

.km-source-score { color: var(--km-muted-soft); }

.km-source-body {
    max-height: 190px;
    overflow-y: auto;
    color: var(--km-muted);
    font-size: 11.5px;
    line-height: 1.6;
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
    text-decoration: none;
    font-weight: 600;
}

.km-source-link:hover { text-decoration: underline; }

/* ============================================================
   TRACE
   ============================================================ */

.km-trace-list {
    margin: 9px 0 0;
    padding-left: 20px;
    color: var(--km-muted);
    font-size: 11.5px;
    line-height: 1.75;
}

.km-trace-list b { color: var(--km-accent-strong); }

/* ============================================================
   EMPTY STATE
   ============================================================ */

.km-starter-card {
    padding: 14px 15px;
    border: 1px solid var(--km-border-soft);
    border-radius: var(--km-radius-md);
    background: var(--km-panel-alt);
    height: 100%;
}

.km-starter-card .icon { font-size: 17px; margin-bottom: 6px; }

.km-starter-card .title {
    color: var(--km-text);
    font-size: 12px;
    font-weight: 650;
    margin-bottom: 3px;
}

.km-starter-card .desc {
    color: var(--km-muted-soft);
    font-size: 10.5px;
    line-height: 1.5;
}

/* ============================================================
   INPUTS
   ============================================================ */

.stButton > button {
    border: 1px solid var(--km-border) !important;
    border-radius: 8px !important;
    background: var(--km-panel-raised) !important;
    color: var(--km-text) !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    transition: border-color .15s ease !important;
}

.stButton > button:hover { border-color: var(--km-accent-border) !important; }

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
   MOBILE
   ============================================================ */

@media (max-width: 900px) {
    .km-topbar h1 { font-size: 21px; }

    .km-pipeline { overflow-x: auto; padding: 14px; }
    .km-step { min-width: 145px; }

    [class*="st-key-km_bubble_"],
    .km-signal-readout { max-width: 100%; }
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
        logfire.info("KnowledgeMesh session reset", old_session_id=old_session_id)

    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.latencies = []
    st.session_state.session_started_at = time.strftime("%H:%M:%S")

    check_backend_health.clear()
    check_backend_ready.clear()


def ask(question: str):
    question = (question or "").strip()
    if not question:
        return

    backend_url = st.session_state.backend_url

    st.session_state.messages.append({"role": "user", "content": question})

    try:
        with st.status(
            "Running Guardrails → Planner → Qdrant → FlashRank → Grader → LLM…",
            expanded=True,
        ) as status:
            start_time = time.perf_counter()
            st.write("Connecting to KnowledgeMesh backend…")

            payload = {"q": question, "thread_id": st.session_state.session_id}

            with _trace_context():
                response = st.session_state.http_session.post(
                    f"{backend_url}/query",
                    json=payload,
                    timeout=BACKEND_TIMEOUT_SECONDS,
                )

            elapsed = time.perf_counter() - start_time

            if response.status_code != 200:
                raise RuntimeError(
                    f"Backend returned HTTP {response.status_code}: {response.text[:500]}"
                )

            data = response.json()

            backend_latency_ms = data.get("latency_ms")
            retrieval_latency_ms = data.get("retrieval_latency_ms")
            rerank_latency_ms = data.get("rerank_latency_ms")
            grader_latency_ms = data.get("grader_latency_ms")
            generation_latency_ms = data.get("generation_latency_ms")
            context_quality = data.get("context_quality")
            answer_supported = data.get("answer_supported")
            answer_useful = data.get("answer_useful")
            support_score = data.get("support_score")
            usefulness_score = data.get("usefulness_score")
            revision_count = data.get("revision_count", 0)
            web_search_used = bool(data.get("web_search_used", False))
            retrieval_used = bool(data.get("retrieval_used", False))

            thought_process = data.get("thought_process", []) or []

            raw_sources = data.get("sources", []) or []
            if not raw_sources:
                raw_sources = data.get("private_sources", []) or []

            sources = [
                normalize_source(source, index + 1)
                for index, source in enumerate(raw_sources)
            ]

            status_text = data.get("status", "Response generated.")
            query_type = infer_query_type(thought_process)
            search_query = data.get("search_query") or extract_search_query(
                thought_process
            )

            visited_nodes = infer_visited_nodes(
                thought_process,
                sources,
                status_text,
                context_quality=context_quality,
                web_search_used=web_search_used,
            )

            st.write(f"Intent: **{query_type.title()}**")

            if search_query:
                st.write(f"Planner query: `{search_query}`")

            if retrieval_used:
                st.write(f"Retrieved and reranked **{len(sources)}** documents.")
            else:
                st.write("Retrieval was not required for this turn.")

            if context_quality:
                st.write(f"Context quality: **{context_quality}**")

            if web_search_used:
                st.write("External web search was used.")

            if grader_latency_ms is not None:
                st.write(f"Document grading: **{format_ms(grader_latency_ms)}**")

            if generation_latency_ms is not None:
                st.write(f"LLM generation: **{format_ms(generation_latency_ms)}**")

            status.update(
                label=f"Completed in {elapsed:.2f}s", state="complete", expanded=False
            )

        if query_type == "conversational":
            verdict = {"kind": "memory", "label": "Conversation memory"}
        elif answer_supported is True and answer_useful is True:
            verdict = {"kind": "grounded", "label": "Supported & useful"}
        elif answer_supported is True:
            verdict = {"kind": "grounded", "label": "Answer supported"}
        elif context_quality == "strong":
            verdict = {"kind": "grounded", "label": "Strong retrieved context"}
        elif sources:
            verdict = {"kind": "grounded", "label": "Retrieved context"}
        else:
            verdict = {"kind": "no_context", "label": "No retrieved context"}

        trace = {
            "steps": thought_process,
            "visited": list(visited_nodes),
            "query_type": query_type,
            "search_query": search_query,
            "sources": sources,
            "context_quality": context_quality,
            "answer_supported": answer_supported,
            "answer_useful": answer_useful,
            "support_score": support_score,
            "usefulness_score": usefulness_score,
            "revision_count": revision_count,
            "web_search_used": web_search_used,
            "latency": elapsed,
            "backend_latency_ms": backend_latency_ms,
            "retrieval_latency_ms": retrieval_latency_ms,
            "rerank_latency_ms": rerank_latency_ms,
            "grader_latency_ms": grader_latency_ms,
            "generation_latency_ms": generation_latency_ms,
            "status": status_text,
            "verdict": verdict,
        }

        assistant_message = {
            "role": "assistant",
            "content": data.get("answer", "No response was returned."),
            "trace": trace,
        }

        st.session_state.messages.append(assistant_message)
        st.session_state.latencies.append(elapsed)

        if LOGFIRE_OK:
            logfire.info(
                "KnowledgeMesh response rendered",
                query_type=query_type,
                source_count=len(sources),
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
                    "**Backend unavailable**\n\n"
                    f"I could not connect to `{backend_url}`.\n\n"
                    "Start FastAPI with:\n\n"
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
                    "**Request timed out.**\n\n"
                    "The backend may still be loading the embedding model or "
                    "processing the request.\n\n"
                    f"Current UI timeout: `{BACKEND_TIMEOUT_SECONDS}` seconds."
                ),
            }
        )

    except requests.exceptions.RequestException as exc:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"**Network request failed**\n\n`{str(exc)}`",
            }
        )

    except ValueError:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "**Invalid backend response**\n\nThe backend did not return valid JSON.",
            }
        )

    except Exception as exc:
        if LOGFIRE_OK:
            logfire.exception(
                "KnowledgeMesh UI request failed", error_type=type(exc).__name__
            )

        st.session_state.messages.append(
            {"role": "assistant", "content": f"**Request failed**\n\n`{str(exc)}`"}
        )


# ============================================================
# BACKEND STATE
# ============================================================

backend_online = check_backend_health(st.session_state.backend_url)
backend_ready = check_backend_ready(st.session_state.backend_url)

if backend_ready:
    system_dot_class = "success"
    system_text = "KnowledgeMesh ready"
elif backend_online:
    system_dot_class = "warning"
    system_text = "Backend online"
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

    if st.session_state.latencies:
        average_latency = sum(st.session_state.latencies) / len(
            st.session_state.latencies
        )
        average_latency_text = f"{average_latency:.2f}s"
    else:
        average_latency_text = "—"

    tracing_text = "Connected" if LOGFIRE_OK else "Standby"

    display_html(
        f"""
        <div class="km-status-card">
            <div class="km-status-row">
                <span class="label">Session</span>
                <span class="value">{esc(st.session_state.session_id[:8])}</span>
            </div>
            <div class="km-status-row">
                <span class="label">Started</span>
                <span class="value">{esc(st.session_state.session_started_at)}</span>
            </div>
            <div class="km-status-row">
                <span class="label">Turns</span>
                <span class="value">{len(st.session_state.messages) // 2}</span>
            </div>
            <div class="km-status-row">
                <span class="label">Avg latency</span>
                <span class="value">{average_latency_text}</span>
            </div>
            <div class="km-status-row">
                <span class="label">Backend</span>
                <span class="value">{esc(system_text)}</span>
            </div>
            <div class="km-status-row">
                <span class="label">Tracing</span>
                <span class="value">{esc(tracing_text)}</span>
            </div>
        </div>
        """
    )

    display_html('<div class="km-rail-label">Actions</div>')

    if st.session_state.messages:
        st.download_button(
            "Export transcript",
            data=transcript_markdown(st.session_state.messages),
            file_name=f"knowledgemesh_{st.session_state.session_id[:8]}.md",
            mime="text/markdown",
            width="stretch",
        )

    if st.button("New session", width="stretch"):
        start_new_session()
        st.rerun()

    if LOGFIRE_PROJECT_URL:
        st.link_button("Open Logfire dashboard", LOGFIRE_PROJECT_URL, width="stretch")

    display_html('<div class="km-rail-label">Settings</div>')

    with st.expander("Backend connection", expanded=False):
        new_backend_url = st.text_input(
            "Backend URL",
            value=st.session_state.backend_url,
            label_visibility="collapsed",
        ).rstrip("/")

        if new_backend_url and new_backend_url != st.session_state.backend_url:
            st.session_state.backend_url = new_backend_url
            check_backend_health.clear()
            check_backend_ready.clear()
            st.rerun()

        st.caption(f"Request timeout: {BACKEND_TIMEOUT_SECONDS}s")

    display_html(
        f"""
        <div class="km-rail-footer">
            <span class="km-dot {system_dot_class}"></span>
            <div>
                <strong>{esc(system_text)}</strong>
                <small>FastAPI · Qdrant · FlashRank · Grader · Portkey</small>
            </div>
        </div>
        """
    )


# ============================================================
# TOPBAR
# ============================================================

display_html(
    f"""
    <div class="km-topbar">
        <div class="km-kicker">Enterprise knowledge system</div>
        <h1>KnowledgeMesh</h1>
        <p>
            Agentic RAG over your internal documentation. The planner decides
            whether retrieval is needed, Qdrant retrieves candidate context,
            FlashRank reranks the strongest chunks, the document grader
            evaluates evidence quality, and the LLM synthesizes a grounded
            response.
        </p>
        <div class="km-engine-pill">
            <span class="km-dot {system_dot_class}"></span>
            {esc(system_text)}
        </div>
    </div>
    """
)


# ============================================================
# LAST PIPELINE TRACE
# ============================================================

last_trace = None
for message in reversed(st.session_state.messages):
    if message.get("role") == "assistant" and message.get("trace"):
        last_trace = message["trace"]
        break

display_html(render_pipeline_rail(last_trace))


# ============================================================
# CONSOLE HEADER
# ============================================================

console_left, console_right = st.columns([5, 1])

with console_left:
    display_html(
        f"""
        <div class="km-console-head">
            <div class="km-console-live">
                <span class="km-dot {system_dot_class}"></span>
                Console
            </div>
            <span class="km-session-id">{esc(st.session_state.session_id[:8].upper())}</span>
        </div>
        """
    )

with console_right:
    if st.button("New session", width="stretch", key="km_new_session_top"):
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

            Ask a question about your enterprise documentation. The system
            will **Plan → Retrieve → Rerank → Grade → Answer**.

            You can inspect retrieved documents, retrieval scores, reranking
            scores, document relevance grades, latency, and the planner
            reasoning trace for every response.
            """
        )

    starter_prompts = [
        ("🔎", "Explain a concept", "What is loop engineering?"),
        (
            "📚",
            "Summarize documentation",
            "Summarize the key points from our documentation.",
        ),
        ("🛠️", "Troubleshoot", "What does the documentation say about rate limiting?"),
    ]

    starter_columns = st.columns(len(starter_prompts))

    for column, (icon, label, question) in zip(starter_columns, starter_prompts):
        with column:
            display_html(
                f"""
                <div class="km-starter-card">
                    <div class="icon">{icon}</div>
                    <div class="title">{esc(label)}</div>
                    <div class="desc">{esc(question)}</div>
                </div>
                """
            )
            st.write("")
            if st.button("Ask", key=f"km_starter_{label}", width="stretch"):
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

        trace = message.get("trace")
        if not trace:
            continue

        verdict = trace.get("verdict", {})
        verdict_kind = verdict.get("kind")
        verdict_label = verdict.get("label", "")
        query_type = trace.get("query_type", "technical")
        source_list = trace.get("sources", [])
        source_count = len(source_list)
        latency = trace.get("latency")
        search_query = trace.get("search_query")
        status_text = trace.get("status")
        context_quality = trace.get("context_quality")
        answer_supported = trace.get("answer_supported")
        answer_useful = trace.get("answer_useful")
        support_score = trace.get("support_score")
        usefulness_score = trace.get("usefulness_score")
        revision_count = trace.get("revision_count", 0)
        web_search_used = trace.get("web_search_used", False)

        verdict_class = {
            "grounded": "ok",
            "memory": "success",
            "no_context": "danger",
        }.get(verdict_kind, "")

        readout_parts = []

        if verdict_label:
            readout_parts.append(
                f'<span class="km-chip {verdict_class}">{esc(verdict_label)}</span>'
            )

        readout_parts.append(
            f'<span class="km-chip">Mode <span class="num">{esc(query_type.title())}</span></span>'
        )
        readout_parts.append(
            f'<span class="km-chip">Sources <span class="num">{source_count}</span></span>'
        )

        if context_quality:
            readout_parts.append(
                f'<span class="km-chip">Context <span class="num">{esc(str(context_quality).title())}</span></span>'
            )

        if web_search_used:
            readout_parts.append('<span class="km-chip ok">Web fallback</span>')

        if latency is not None:
            readout_parts.append(
                f'<span class="km-chip">Total <span class="num">{float(latency):.2f}s</span></span>'
            )

        latency_items = [
            ("Retrieval", trace.get("retrieval_latency_ms")),
            ("Rerank", trace.get("rerank_latency_ms")),
            ("Grader", trace.get("grader_latency_ms")),
            ("LLM", trace.get("generation_latency_ms")),
        ]

        for label, value in latency_items:
            if value is None:
                continue
            readout_parts.append(
                f'<span class="km-chip">{esc(label)} <span class="num">{esc(format_ms(value))}</span></span>'
            )

        readout_html = (
            '<div class="km-signal-readout">'
            '<div class="km-signal-row">' + "".join(readout_parts) + "</div>"
        )

        if (
            context_quality is not None
            or answer_supported is not None
            or answer_useful is not None
        ):
            evaluation_parts = []

            if answer_supported is not None:
                supported_class = "success" if answer_supported else "danger"
                supported_label = "Supported" if answer_supported else "Not supported"
                evaluation_parts.append(
                    f'<span class="km-chip {supported_class}">{supported_label}</span>'
                )

            if answer_useful is not None:
                useful_class = "success" if answer_useful else "danger"
                useful_label = "Useful" if answer_useful else "Not useful"
                evaluation_parts.append(
                    f'<span class="km-chip {useful_class}">{useful_label}</span>'
                )

            if support_score is not None:
                evaluation_parts.append(
                    f'<span class="km-chip">Support <span class="num">{float(support_score):.2f}</span></span>'
                )

            if usefulness_score is not None:
                evaluation_parts.append(
                    f'<span class="km-chip">Usefulness <span class="num">{float(usefulness_score):.2f}</span></span>'
                )

            evaluation_parts.append(
                f'<span class="km-chip">Revisions <span class="num">{int(revision_count or 0)}</span></span>'
            )

            readout_html += (
                '<div style="margin-top:8px;display:flex;flex-wrap:wrap;gap:6px;">'
                + "".join(evaluation_parts)
                + "</div>"
            )

        if search_query:
            readout_html += f"""
                <details class="km-details">
                    <summary>Planner search query</summary>
                    <div style="padding-top:9px;color:var(--km-muted);font-size:11.5px;">
                        <code>{esc(search_query)}</code>
                    </div>
                </details>
            """

        if source_list:
            source_cards = []

            for source in source_list:
                source_name = source.get("name", "Unknown document")
                source_type = source_type_label(source.get("source_type"))
                vector_score = format_score(source.get("score"))
                rerank_score = format_score(source.get("rerank_score"))
                grader_score = format_score(source.get("grader_score"))
                grader_relevant = source.get("grader_relevant")
                grader_reason = source.get("grader_reason")
                document_id = source.get("id") or "Unknown"
                source_url = safe_url(source.get("url"))

                link_html = ""
                if source_url:
                    link_html = (
                        f'<a class="km-source-link" href="{esc(source_url)}" '
                        f'target="_blank" rel="noopener noreferrer">Open source</a>'
                    )

                grader_html = ""
                if grader_score != "—":
                    grader_label = "relevant" if grader_relevant else "rejected"
                    grader_html = f'<span class="km-source-score">grader {esc(grader_score)} · {esc(grader_label)}</span>'

                reason_html = ""
                if grader_reason:
                    reason_html = (
                        f'<div style="margin-top:7px;color:var(--km-muted-soft);font-size:10px;">'
                        f"Grader: {esc(grader_reason)}</div>"
                    )

                source_cards.append(
                    f"""
                    <div class="km-source">
                        <div class="km-source-meta">
                            <span class="km-source-number">[{source["n"]}]</span>
                            <span class="km-source-name">{esc(source_name)}</span>
                            <span class="km-source-type">{esc(source_type)}</span>
                            <span class="km-source-score">vector {esc(vector_score)}</span>
                            <span class="km-source-score">rerank {esc(rerank_score)}</span>
                            {grader_html}
                            {link_html}
                        </div>
                        <div class="km-source-body">{esc(source.get("text", ""))}</div>
                        {reason_html}
                        <div class="km-source-id">ID: {esc(document_id)}</div>
                    </div>
                    """
                )

            readout_html += f"""
                <details class="km-details">
                    <summary>View retrieved context · {source_count}</summary>
                    <div class="km-sources">{"".join(source_cards)}</div>
                </details>
            """

        steps = trace.get("steps", [])
        if steps:
            trace_items = []
            for step in steps:
                _, code, detail = classify_step(step)
                trace_items.append(f"<li><b>{esc(code)}</b> — {esc(detail)}</li>")

            readout_html += f"""
                <details class="km-details">
                    <summary>Inspect reasoning trace</summary>
                    <ol class="km-trace-list">{"".join(trace_items)}</ol>
                </details>
            """

        if status_text:
            readout_html += (
                f'<div style="margin-top:10px;color:var(--km-muted-soft);font-size:10.5px;">'
                f"Status: {esc(status_text)}</div>"
            )

        readout_html += "</div>"

        display_html(readout_html)


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
        display:flex;flex-wrap:wrap;gap:8px;margin-top:8px;
        color:var(--km-muted-soft);font-size:10px;
    ">
        <span>KnowledgeMesh Agentic RAG</span>
        <span>·</span><span>Qdrant retrieval</span>
        <span>·</span><span>FlashRank reranking</span>
        <span>·</span><span>Document grading</span>
        <span>·</span><span>Portkey synthesis</span>
        <span>·</span><span>Logfire observability</span>
    </div>
    """
)
