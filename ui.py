"""
KnowledgeMesh · Agentic RAG Console

Frontend for the current KnowledgeMesh FastAPI backend.

Backend request:
    POST http://localhost:8000/query

Payload:
    {
        "q": "What is loop engineering?",
        "thread_id": "..."
    }

Expected response:
    {
        "question": "...",
        "answer": "...",
        "thought_process": [...],
        "status": "...",
        "sources": [
            {
                "id": "...",
                "content": "...",
                "source": "...",
                "source_type": "true",
                "score": 0.74,
                "rerank_score": 0.98
            }
        ]
    }
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

ENV_PATH = os.path.join(
    PROJECT_ROOT,
    ".env",
)

# Fallback in case ui.py is inside a subdirectory.
if not os.path.exists(ENV_PATH):
    parent_env = os.path.join(
        os.path.dirname(PROJECT_ROOT),
        ".env",
    )

    if os.path.exists(parent_env):
        ENV_PATH = parent_env

load_dotenv(
    dotenv_path=ENV_PATH,
    override=False,
)

BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://localhost:8000",
).rstrip("/")

BACKEND_TIMEOUT_SECONDS = int(
    os.getenv(
        "BACKEND_TIMEOUT_SECONDS",
        "180",
    )
)

LOGFIRE_PROJECT_URL = os.getenv(
    "LOGFIRE_PROJECT_URL",
)


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
        print("⚠️ LOGFIRE_TOKEN is not configured for the Streamlit UI.")

except Exception as exc:
    LOGFIRE_ERROR = str(exc)

    print(f"⚠️ Streamlit Logfire initialization failed: {exc}")


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="KnowledgeMesh",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# RENDERING HELPERS
# ============================================================


def clean_html(markup: str) -> str:
    """
    Normalize HTML indentation.

    Important:
    Do not escape this output because it is intended to be
    rendered as HTML.
    """

    return textwrap.dedent(markup).strip()


def display_html(markup: str):
    """
    Centralized HTML renderer.

    Streamlit versions that support st.html() use it directly.
    The fallback uses st.markdown(..., unsafe_allow_html=True).
    """

    markup = clean_html(markup)

    native_html_renderer = getattr(
        st,
        "html",
        None,
    )

    if callable(native_html_renderer):
        native_html_renderer(markup)

    else:
        st.markdown(
            markup,
            unsafe_allow_html=True,
        )


def esc(value) -> str:
    """
    Escape dynamic values before inserting them into HTML.
    """

    return html.escape(
        str(value),
        quote=True,
    )


def safe_url(value) -> str:
    """
    Allow only HTTP and HTTPS URLs.
    """

    value = str(value or "")

    if value.startswith("http://"):
        return value

    if value.startswith("https://"):
        return value

    return ""


def format_score(value) -> str:
    """
    Format numeric scores safely.
    """

    if isinstance(value, (int, float)):
        return f"{value:.3f}"

    return "—"


def _trace_context():
    """
    Return a Logfire context when available.
    """

    if LOGFIRE_OK:
        return logfire.span("KnowledgeMesh UI operation")

    return nullcontext()


# ============================================================
# PIPELINE CONFIGURATION
# ============================================================

PIPELINE_STAGES = [
    (
        "guardrails",
        "Guardrails",
        "Safety and policy check",
    ),
    (
        "planner",
        "Planner",
        "Intent and search planning",
    ),
    (
        "retrieval",
        "Qdrant",
        "Vector knowledge retrieval",
    ),
    (
        "reranking",
        "FlashRank",
        "Context relevance ranking",
    ),
    (
        "responder",
        "LLM",
        "Grounded response synthesis",
    ),
]


TRACE_PATTERNS = [
    (
        re.compile(
            r"guardrail",
            re.IGNORECASE,
        ),
        "guardrails",
        "GRD",
    ),
    (
        re.compile(
            r"intent|planner|planning",
            re.IGNORECASE,
        ),
        "planner",
        "PLN",
    ),
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
            r"respond|response|synthes|answer|llm",
            re.IGNORECASE,
        ),
        "responder",
        "LLM",
    ),
]


def classify_step(step):
    """
    Convert a backend thought_process entry into a UI stage.
    """

    if isinstance(step, dict):
        stage = step.get("stage") or step.get("node") or ""

        detail = step.get("detail") or step.get("message") or str(step)

        searchable_text = f"{stage} {detail}"

    else:
        detail = str(step)

        searchable_text = detail

    for pattern, stage_key, code in TRACE_PATTERNS:
        if pattern.search(searchable_text):
            return (
                stage_key,
                code,
                detail,
            )

    return (
        "",
        "STEP",
        detail,
    )


def infer_query_type(thought_process):
    """
    Infer whether the backend handled the turn as conversational
    or technical.
    """

    text = " ".join(str(step) for step in (thought_process or [])).lower()

    if "conversational" in text:
        return "conversational"

    if "retrieval: skipped" in text:
        return "conversational"

    if "intent: technical" in text:
        return "technical"

    return "technical"


def extract_search_query(thought_process):
    """
    Extract the planner-generated search term.
    """

    for step in thought_process or []:
        text = str(step)

        if text.lower().startswith("search term:"):
            return text.split(
                ":",
                1,
            )[1].strip()

    return None


def infer_visited_nodes(
    thought_process,
    sources,
    status,
):
    """
    Derive active pipeline stages from the actual backend response.

    The current backend returns thought_process rather than explicit
    node telemetry, so this function derives the visual state.
    """

    visited = set()

    steps = thought_process or []

    status_text = str(status or "").lower()

    combined_text = " ".join(str(step) for step in steps).lower()

    for step in steps:
        stage_key, _, _ = classify_step(step)

        if stage_key:
            visited.add(stage_key)

    # A successful request always passed through the guardrail check.
    if "guardrail" in combined_text or "guardrail" in status_text or steps:
        visited.add("guardrails")

    # Planner is present when an intent/search decision exists.
    if (
        "intent:" in combined_text
        or "search term:" in combined_text
        or "conversational" in combined_text
    ):
        visited.add("planner")

    # Sources indicate retrieval and reranking occurred.
    if sources:
        visited.add("retrieval")
        visited.add("reranking")

    # A response status indicates the responder completed.
    if status:
        visited.add("responder")

    return visited


# ============================================================
# SOURCE NORMALIZATION
# ============================================================


def normalize_source(
    source,
    index,
):
    """
    Normalize the current structured source response.

    Supports:
    - Current dictionary sources
    - Legacy string sources
    """

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

        url = source.get("url")

    else:
        content = str(source)

        name = "Unknown document"

        source_type = "unknown"

        document_id = None

        vector_score = None

        rerank_score = None

        url = None

    return {
        "n": index,
        "text": str(content),
        "name": str(name),
        "source_type": str(source_type),
        "id": document_id,
        "score": vector_score,
        "rerank_score": rerank_score,
        "url": url,
    }


def source_type_label(value):
    """
    Human-friendly source type.
    """

    normalized = str(value or "").strip().lower()

    if normalized == "true":
        return "Trusted"

    if normalized in {
        "false",
        "noisy",
    }:
        return "Noisy"

    if not normalized:
        return "Unknown"

    return normalized.title()


# ============================================================
# PIPELINE HTML
# ============================================================


def render_pipeline_rail(trace=None):
    """
    Build the pipeline HTML.
    """

    trace = trace or {}

    visited = set(
        trace.get(
            "visited",
            [],
        )
    )

    query_type = trace.get(
        "query_type",
        "technical",
    )

    sources = trace.get(
        "sources",
        [],
    )

    source_count = len(sources)

    nodes = []

    for index, (
        key,
        title,
        subtitle,
    ) in enumerate(
        PIPELINE_STAGES,
        start=1,
    ):
        active = key in visited

        current_subtitle = subtitle

        if trace:
            if key == "planner" and query_type == "conversational":
                current_subtitle = "Conversation memory path"

            elif key == "retrieval" and source_count:
                current_subtitle = (
                    f"{source_count} document"
                    f"{'s' if source_count != 1 else ''} retrieved"
                )

            elif key == "reranking" and source_count:
                current_subtitle = f"Top {source_count} context chunks"

            elif key == "responder" and query_type == "conversational":
                current_subtitle = "Memory-based response"

            elif key == "responder" and source_count:
                current_subtitle = "Grounded synthesis"

        active_class = " active" if active else ""

        nodes.append(
            f"""
            <div class="km-pipeline-node{active_class}">

                <div class="km-node-index">
                    {index}
                </div>

                <div class="km-node-content">

                    <strong>
                        {esc(title)}
                    </strong>

                    <span>
                        {esc(current_subtitle)}
                    </span>

                </div>

            </div>
            """
        )

    return f"""
    <div class="km-pipeline">
        {"".join(nodes)}
    </div>
    """


# ============================================================
# TRANSCRIPT EXPORT
# ============================================================


def transcript_markdown(messages):
    """
    Export conversation and retrieved source metadata.
    """

    lines = [
        "# KnowledgeMesh transcript",
        "",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
    ]

    for message in messages:
        role = message.get(
            "role",
            "assistant",
        )

        speaker = "You" if role == "user" else "KnowledgeMesh"

        lines.extend(
            [
                f"## {speaker}",
                "",
                str(
                    message.get(
                        "content",
                        "",
                    )
                ),
                "",
            ]
        )

        trace = message.get("trace")

        if not trace:
            continue

        search_query = trace.get("search_query")

        if search_query:
            lines.extend(
                [
                    f"**Planner search query:** `{search_query}`",
                    "",
                ]
            )

        sources = trace.get(
            "sources",
            [],
        )

        if sources:
            lines.extend(
                [
                    "### Retrieved sources",
                    "",
                ]
            )

            for source in sources:
                lines.extend(
                    [
                        (f"**[{source['n']}] {source['name']}**"),
                        "",
                        (f"- Source type: {source_type_label(source['source_type'])}"),
                        (f"- Vector score: {format_score(source['score'])}"),
                        (f"- Rerank score: {format_score(source['rerank_score'])}"),
                        "",
                        source["text"],
                        "",
                    ]
                )

        steps = trace.get(
            "steps",
            [],
        )

        if steps:
            lines.extend(
                [
                    "### Reasoning trace",
                    "",
                ]
            )

            for step in steps:
                lines.append(f"- {step}")

            lines.append("")

    return "\n".join(lines)


# ============================================================
# BACKEND HEALTH
# ============================================================


@st.cache_data(
    ttl=10,
    show_spinner=False,
)
def check_backend_health():

    try:
        response = requests.get(
            f"{BACKEND_URL}/health",
            timeout=4,
        )

        return response.ok

    except requests.RequestException:
        return False


@st.cache_data(
    ttl=10,
    show_spinner=False,
)
def check_backend_ready():

    try:
        response = requests.get(
            f"{BACKEND_URL}/ready",
            timeout=4,
        )

        if not response.ok:
            return False

        payload = response.json()

        return payload.get("status") == "ready"

    except (
        requests.RequestException,
        ValueError,
    ):
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


if "http_session" not in st.session_state:
    http_session = requests.Session()

    retry_strategy = Retry(
        total=2,
        backoff_factor=0.5,
        status_forcelist=[
            502,
            503,
            504,
        ],
        allowed_methods=[
            "GET",
            "POST",
        ],
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)

    http_session.mount(
        "http://",
        adapter,
    )

    http_session.mount(
        "https://",
        adapter,
    )

    st.session_state.http_session = http_session


# ============================================================
# CSS
# ============================================================

CUSTOM_CSS = """
<style>

:root {
    --km-bg: #0d0f14;
    --km-panel: #14171f;
    --km-panel-alt: #191d27;
    --km-border: #282d39;
    --km-border-soft: #20242e;

    --km-text: #e8eaf0;
    --km-muted: #8a90a3;
    --km-muted-soft: #656c7e;

    --km-accent: #818cf8;
    --km-accent-strong: #a5b4fc;
    --km-accent-soft: rgba(129, 140, 248, .14);

    --km-success: #6ee7b7;
    --km-success-soft: rgba(110, 231, 183, .14);

    --km-warning: #e0a95c;
    --km-warning-soft: rgba(224, 169, 92, .14);

    --km-danger: #f0707a;
    --km-danger-soft: rgba(240, 112, 122, .14);
}

html,
body,
[class*="css"] {
    font-family: Inter, sans-serif;
}

.stApp {
    background: var(--km-bg);
    color: var(--km-text);
}

header[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

@keyframes kmFadeIn {
    from {
        opacity: 0;
        transform: translateY(4px);
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
    border-right: 1px solid var(--km-border);
}

section[data-testid="stSidebar"] > div {
    padding-top: .6rem;
}

section[data-testid="stSidebar"] * {
    color: var(--km-text);
}

.km-brand {
    display: flex;
    align-items: center;
    gap: 10px;

    padding: 4px 0 17px;
    margin-bottom: 16px;

    border-bottom: 1px solid var(--km-border);
}

.km-brand strong {
    display: block;

    color: #ffffff;

    font-family: monospace;
    font-size: 15px;
    font-weight: 700;
}

.km-brand small {
    display: block;

    margin-top: 3px;

    color: var(--km-muted);

    font-size: 10.5px;
}

.km-rail-label {
    display: flex;
    align-items: center;
    gap: 7px;

    margin: 18px 0 8px 2px;

    color: var(--km-muted);

    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .06em;
}

.km-rail-label::before {
    content: "";

    width: 3px;
    height: 12px;

    border-radius: 3px;

    background: var(--km-accent);
}

.km-status-card {
    padding: 13px 14px;

    border: 1px solid var(--km-border);
    border-radius: 10px;

    background: var(--km-panel-alt);
}

.km-status-card .kv {
    color: var(--km-muted);

    font-family: monospace;
    font-size: 11px;
    line-height: 2;
}

.km-status-card .kv b {
    color: var(--km-text);
    font-weight: 500;
}

.km-rail-footer {
    display: flex;
    align-items: center;
    gap: 10px;

    margin-top: 22px;
    padding-top: 16px;

    border-top: 1px solid var(--km-border);
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

    font-size: 10px;
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
    box-shadow: 0 0 12px var(--km-success-soft);
}

.km-dot.warning {
    background: var(--km-warning);
    box-shadow: 0 0 12px var(--km-warning-soft);
}

.km-dot.danger {
    background: var(--km-danger);
    box-shadow: 0 0 12px var(--km-danger-soft);
}

section[data-testid="stSidebar"] button {
    background: var(--km-panel-alt) !important;

    border: 1px solid var(--km-border) !important;

    color: var(--km-text) !important;

    font-size: 12px !important;
}

section[data-testid="stSidebar"] button:hover {
    border-color: var(--km-accent) !important;
}


/* ============================================================
   TOPBAR
   ============================================================ */

.km-topbar {
    padding-top: 4px;
}

.km-kicker {
    margin-bottom: 7px;

    color: var(--km-accent-strong);

    font-family: monospace;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: .04em;
    text-transform: uppercase;
}

.km-topbar h1 {
    margin: 0 0 8px;

    color: var(--km-text);

    font-size: 28px;
    font-weight: 700;
    letter-spacing: -.02em;
}

.km-topbar p {
    max-width: 800px;

    margin: 0;

    color: var(--km-muted);

    font-size: 13px;
    line-height: 1.6;
}

.km-engine-pill {
    display: inline-flex;
    align-items: center;
    gap: 9px;

    margin-top: 13px;
    padding: 8px 13px;

    border: 1px solid var(--km-border);
    border-radius: 999px;

    background: var(--km-panel);

    color: var(--km-muted);

    font-family: monospace;
    font-size: 11px;
}

.km-signal-divider {
    margin: 15px 0 7px;

    color: var(--km-border);
}

.km-signal-divider svg {
    display: block;

    width: 100%;
    height: 18px;
}


/* ============================================================
   PIPELINE
   ============================================================ */

.km-pipeline {
    display: flex;
    gap: 0;

    width: 100%;

    margin: 10px 0 22px;
}

.km-pipeline-node {
    position: relative;

    display: flex;
    align-items: flex-start;
    gap: 9px;

    flex: 1;

    min-width: 0;
    padding-right: 15px;

    opacity: .35;
}

.km-pipeline-node.active {
    opacity: 1;
}

.km-pipeline-node:not(:last-child)::after {
    content: "";

    position: absolute;
    top: 14px;
    left: calc(100% - 7px);

    width: calc(100% - 20px);
    height: 1px;

    background: var(--km-border);
}

.km-node-index {
    display: grid;
    place-items: center;

    width: 28px;
    height: 28px;

    flex: none;

    border: 1px solid var(--km-border);
    border-radius: 50%;

    background: var(--km-panel);

    color: var(--km-accent-strong);

    font-family: monospace;
    font-size: 11px;
    font-weight: 700;
}

.km-pipeline-node.active .km-node-index {
    border-color: var(--km-accent);

    background: var(--km-accent);

    color: #111321;

    box-shadow: 0 0 0 3px var(--km-accent-soft);
}

.km-node-content {
    min-width: 0;
}

.km-node-content strong {
    display: block;

    color: var(--km-text);

    font-size: 12px;
    font-weight: 700;
}

.km-node-content span {
    display: block;

    margin-top: 3px;

    color: var(--km-muted-soft);

    font-size: 10px;
    line-height: 1.35;
}


/* ============================================================
   CONSOLE
   ============================================================ */

.km-console-head {
    display: flex;
    align-items: center;
    justify-content: space-between;

    padding: 0 2px 11px;
    margin-bottom: 15px;

    border-bottom: 1px solid var(--km-border-soft);
}

.km-console-live {
    display: flex;
    align-items: center;
    gap: 8px;

    color: var(--km-text);

    font-size: 12.5px;
    font-weight: 600;
}

.km-session-id {
    color: var(--km-muted-soft);

    font-family: monospace;
    font-size: 10px;
}

.km-msg-row {
    display: flex;
    align-items: center;
    gap: 10px;

    margin-bottom: 5px;
}

.km-msg-row.user {
    justify-content: flex-end;
}

.km-msg-avatar {
    display: grid;
    place-items: center;

    width: 28px;
    height: 28px;

    border: 1px solid var(--km-border);
    border-radius: 8px;

    background: var(--km-panel-alt);

    color: var(--km-accent-strong);

    font-family: monospace;
    font-size: 12px;
    font-weight: 700;
}

.km-msg-label {
    color: var(--km-muted-soft);

    font-size: 11px;
    font-weight: 600;
}

.km-msg-label.user {
    text-align: right;
}

[class*="st-key-km_bubble_"] {
    max-width: min(850px, 90%);

    padding: 14px 16px;

    border: 1px solid var(--km-border-soft);
    border-radius: 4px 12px 12px 12px;

    background: var(--km-panel-alt);

    color: var(--km-text);

    font-size: 13.5px;
    line-height: 1.65;

    animation: kmFadeIn .25s ease both;
}

[class*="st-key-km_user_bubble_"] {
    margin-left: auto;

    border-radius: 12px 4px 12px 12px;

    background: var(--km-panel);
}

[class*="st-key-km_bubble_intro"] {
    border-left: 2px solid var(--km-accent);
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

[class*="st-key-km_bubble_"] pre {
    border: 1px solid var(--km-border) !important;
    border-radius: 8px !important;
}


/* ============================================================
   SIGNAL READOUT
   ============================================================ */

.km-signal-readout {
    max-width: min(850px, 90%);

    padding: 10px 13px;
    margin: -5px 0 18px;

    border: 1px solid var(--km-border-soft);
    border-top: 0;
    border-radius: 0 0 12px 12px;

    background: var(--km-panel);
}

.km-signal-row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}

.km-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;

    padding: 5px 9px;

    border: 1px solid var(--km-border);
    border-radius: 999px;

    background: var(--km-panel-alt);

    color: var(--km-muted);

    font-size: 10.5px;
}

.km-chip::before {
    content: "";

    width: 6px;
    height: 6px;

    border-radius: 50%;

    background: var(--km-muted-soft);
}

.km-chip.ok {
    border-color: var(--km-accent-soft);

    background: var(--km-accent-soft);

    color: var(--km-accent-strong);
}

.km-chip.ok::before {
    background: var(--km-accent);
}

.km-chip.success {
    border-color: var(--km-success-soft);

    background: var(--km-success-soft);

    color: var(--km-success);
}

.km-chip.success::before {
    background: var(--km-success);
}

.km-chip.danger {
    border-color: var(--km-danger-soft);

    background: var(--km-danger-soft);

    color: var(--km-danger);
}

.km-chip.danger::before {
    background: var(--km-danger);
}

.km-chip .num {
    font-family: monospace;
    font-weight: 700;
}


/* ============================================================
   SOURCES
   ============================================================ */

.km-details {
    margin-top: 10px;
}

.km-details summary {
    cursor: pointer;

    color: var(--km-muted);

    font-size: 11.5px;
}

.km-sources {
    display: grid;
    gap: 8px;

    margin-top: 10px;
}

.km-source {
    padding: 11px 12px;

    border: 1px solid var(--km-border-soft);
    border-radius: 8px;

    background: var(--km-panel-alt);
}

.km-source-meta {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;

    margin-bottom: 8px;

    color: var(--km-muted);

    font-family: monospace;
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

.km-source-id {
    margin-top: 8px;

    color: var(--km-muted-soft);

    font-family: monospace;
    font-size: 9px;

    word-break: break-all;
}

.km-source-link {
    color: var(--km-accent-strong);
    text-decoration: none;
}


/* ============================================================
   REASONING TRACE
   ============================================================ */

.km-trace-list {
    margin: 9px 0 0;
    padding-left: 20px;

    color: var(--km-muted);

    font-size: 11.5px;
    line-height: 1.7;
}

.km-trace-list b {
    color: var(--km-accent-strong);
}


/* ============================================================
   INPUTS AND BUTTONS
   ============================================================ */

.stButton > button {
    border: 1px solid var(--km-border) !important;
    border-radius: 8px !important;

    background: var(--km-panel-alt) !important;

    color: var(--km-text) !important;

    font-size: 12px !important;
}

.stButton > button:hover {
    border-color: var(--km-accent) !important;
}

div[data-testid="stChatInput"] {
    border-top: 1px solid var(--km-border-soft);
    background: var(--km-panel);
}

div[data-testid="stChatInput"] textarea {
    border: 1px solid var(--km-border) !important;
    border-radius: 10px !important;

    background: var(--km-panel) !important;
    color: var(--km-text) !important;
}

div[data-testid="stChatInput"] textarea:focus {
    border-color: var(--km-accent) !important;
    box-shadow: 0 0 0 1px var(--km-accent-soft) !important;
}

div[data-testid="stExpander"] {
    border: 1px solid var(--km-border);
    border-radius: 9px;

    background: var(--km-panel);
}


/* ============================================================
   MOBILE
   ============================================================ */

@media (max-width: 900px) {

    .km-topbar h1 {
        font-size: 23px;
    }

    .km-pipeline {
        overflow-x: auto;
        padding-bottom: 8px;
    }

    .km-pipeline-node {
        min-width: 155px;
    }

    [class*="st-key-km_bubble_"],
    .km-signal-readout {
        max-width: 100%;
    }

}

</style>
"""

# CSS is the only place where st.markdown is intentionally used
# with unsafe_allow_html=True.
st.markdown(
    CUSTOM_CSS,
    unsafe_allow_html=True,
)


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


def ask(question: str):
    """
    Send a question to the KnowledgeMesh backend.
    """

    question = (question or "").strip()

    if not question:
        return

    # Add the user message immediately.
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    try:
        with st.status(
            "Running Guardrails → Planner → Qdrant → FlashRank → LLM…",
            expanded=True,
        ) as status:
            start_time = time.perf_counter()

            st.write("Connecting to KnowledgeMesh backend…")

            payload = {
                "q": question,
                "thread_id": (st.session_state.session_id),
            }

            with _trace_context():
                response = st.session_state.http_session.post(
                    f"{BACKEND_URL}/query",
                    json=payload,
                    timeout=BACKEND_TIMEOUT_SECONDS,
                )

            elapsed = time.perf_counter() - start_time

            if response.status_code != 200:
                raise RuntimeError(
                    f"Backend returned HTTP "
                    f"{response.status_code}: "
                    f"{response.text[:500]}"
                )

            data = response.json()

            thought_process = (
                data.get(
                    "thought_process",
                    [],
                )
                or []
            )

            raw_sources = (
                data.get(
                    "sources",
                    [],
                )
                or []
            )

            sources = [
                normalize_source(
                    source,
                    index + 1,
                )
                for index, source in enumerate(raw_sources)
            ]

            status_text = data.get(
                "status",
                "Response generated.",
            )

            query_type = infer_query_type(thought_process)

            search_query = extract_search_query(thought_process)

            visited_nodes = infer_visited_nodes(
                thought_process,
                sources,
                status_text,
            )

            st.write(f"Intent: **{query_type.title()}**")

            if search_query:
                st.write(f"Planner query: `{search_query}`")

            if sources:
                st.write(f"Retrieved and reranked **{len(sources)}** documents.")

            else:
                st.write("Retrieval was not required for this turn.")

            status.update(
                label=f"Completed in {elapsed:.2f}s",
                state="complete",
                expanded=False,
            )

        # ----------------------------------------------------
        # Verdict
        # ----------------------------------------------------

        if query_type == "conversational":
            verdict = {
                "kind": "memory",
                "label": "💬 Conversation memory",
            }

        elif sources:
            verdict = {
                "kind": "grounded",
                "label": "✓ Grounded in knowledge base",
            }

        else:
            verdict = {
                "kind": "no_context",
                "label": "No retrieved context",
            }

        # ----------------------------------------------------
        # Trace
        # ----------------------------------------------------

        trace = {
            "steps": thought_process,
            "sources": sources,
            "verdict": verdict,
            "latency": elapsed,
            "visited": list(visited_nodes),
            "query_type": query_type,
            "search_query": search_query,
            "status": status_text,
        }

        # ----------------------------------------------------
        # Assistant response
        # ----------------------------------------------------

        assistant_message = {
            "role": "assistant",
            "content": data.get(
                "answer",
                "No response was returned.",
            ),
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
            )

    except requests.exceptions.ConnectionError:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    "⚠️ **Backend unavailable**\n\n"
                    f"I could not connect to `{BACKEND_URL}`.\n\n"
                    "Start FastAPI with:\n\n"
                    "```powershell\n"
                    "uvicorn app.main:app --reload "
                    "--host 0.0.0.0 --port 8000\n"
                    "```"
                ),
            }
        )

    except requests.exceptions.Timeout:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    "⏳ **Request timed out.**\n\n"
                    "The backend may still be loading the "
                    "embedding model or processing the request.\n\n"
                    f"Current UI timeout: "
                    f"`{BACKEND_TIMEOUT_SECONDS}` seconds."
                ),
            }
        )

    except requests.exceptions.RequestException as exc:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (f"⚠️ **Network request failed**\n\n`{str(exc)}`"),
            }
        )

    except ValueError:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    "⚠️ **Invalid backend response**\n\n"
                    "The backend did not return valid JSON."
                ),
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
                "content": (f"⚠️ **Request failed**\n\n`{str(exc)}`"),
            }
        )


# ============================================================
# BACKEND STATE
# ============================================================

backend_online = check_backend_health()
backend_ready = check_backend_ready()

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

            <svg
                width="30"
                height="30"
                viewBox="0 0 34 34"
                fill="none"
            >

                <circle
                    cx="8"
                    cy="9"
                    r="2.4"
                    fill="#818cf8"
                />

                <circle
                    cx="26"
                    cy="7"
                    r="2.4"
                    fill="#818cf8"
                />

                <circle
                    cx="17"
                    cy="18"
                    r="2.6"
                    fill="#818cf8"
                />

                <circle
                    cx="7"
                    cy="27"
                    r="2.4"
                    fill="#818cf8"
                />

                <circle
                    cx="27"
                    cy="26"
                    r="2.4"
                    fill="#818cf8"
                />

                <path
                    d="
                    M8 9L17 18
                    M26 7L17 18
                    M17 18L7 27
                    M17 18L27 26
                    M8 9L26 7
                    "
                    stroke="#818cf8"
                    stroke-width="1.3"
                    stroke-opacity="0.55"
                />

            </svg>

            <div>

                <strong>
                    KnowledgeMesh
                </strong>

                <small>
                    Agentic RAG console
                </small>

            </div>

        </div>

        <div class="km-rail-label">
            Session
        </div>
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

            <div class="kv">

                <b>Session</b>
                &nbsp;
                {esc(st.session_state.session_id[:8])}

                <br/>

                <b>Started</b>
                &nbsp;
                {esc(st.session_state.session_started_at)}

                <br/>

                <b>Turns</b>
                &nbsp;
                {len(st.session_state.messages) // 2}

                <br/>

                <b>Avg latency</b>
                &nbsp;
                {average_latency_text}

                <br/>

                <b>Backend</b>
                &nbsp;
                {esc(system_text)}

                <br/>

                <b>Tracing</b>
                &nbsp;
                {esc(tracing_text)}

            </div>

        </div>
        """
    )

    display_html(
        """
        <div class="km-rail-label">
            Actions
        </div>
        """
    )

    if st.session_state.messages:
        st.download_button(
            "⬇️ Export transcript",
            data=transcript_markdown(st.session_state.messages),
            file_name=(f"knowledgemesh_{st.session_state.session_id[:8]}.md"),
            mime="text/markdown",
            width="stretch",
        )

    if st.button(
        "🗑️ New session",
        width="stretch",
    ):
        start_new_session()

        st.rerun()

    if LOGFIRE_PROJECT_URL:
        st.link_button(
            "↗ Open Logfire dashboard",
            LOGFIRE_PROJECT_URL,
            width="stretch",
        )

    display_html(
        f"""
        <div class="km-rail-footer">

            <span class="km-dot {system_dot_class}"></span>

            <div>

                <strong>
                    {esc(system_text)}
                </strong>

                <small>
                    FastAPI · Qdrant · FlashRank · Portkey
                </small>

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

        <div class="km-kicker">
            Enterprise knowledge system
        </div>

        <h1>
            ✦ KnowledgeMesh
        </h1>

        <p>
            Agentic RAG over your internal documentation.
            The planner decides whether retrieval is needed,
            Qdrant finds relevant context, FlashRank reranks
            the strongest chunks, and the LLM synthesizes
            a grounded response.
        </p>

        <div class="km-engine-pill">

            <span class="km-dot {system_dot_class}"></span>

            {esc(system_text)}

        </div>

    </div>

    <div class="km-signal-divider">

        <svg
            viewBox="0 0 1200 20"
            preserveAspectRatio="none"
        >

            <path
                d="
                M0,10
                L360,10
                L380,2
                L400,18
                L420,10
                L1200,10
                "
                fill="none"
                stroke="currentColor"
                stroke-width="1.5"
            />

        </svg>

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

            <span class="km-session-id">

                {esc(st.session_state.session_id[:8].upper())}

            </span>

        </div>
        """
    )

with console_right:
    if st.button(
        "New session",
        width="stretch",
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

            <div class="km-msg-avatar">
                K
            </div>

            <div class="km-msg-label">
                KnowledgeMesh
            </div>

        </div>
        """
    )

    with st.container(key="km_bubble_intro"):
        st.markdown(
            """
            Welcome to **KnowledgeMesh**.

            Ask a question about your enterprise
            documentation. The system will:

            **Plan → Retrieve → Rerank → Answer**

            You can inspect retrieved documents, vector
            similarity scores, reranking scores, and the
            planner reasoning trace.
            """
        )

    starter_prompts = [
        (
            "🔎 Explain a concept",
            "What is loop engineering?",
        ),
        (
            "📚 Summarize documentation",
            "Summarize the key points from our documentation.",
        ),
        (
            "🛠️ Troubleshoot",
            "What does the documentation say about rate limiting?",
        ),
    ]

    starter_columns = st.columns(len(starter_prompts))

    for column, (
        label,
        question,
    ) in zip(
        starter_columns,
        starter_prompts,
    ):
        with column:
            if st.button(
                label,
                width="stretch",
            ):
                ask(question)

                st.rerun()


# ============================================================
# CHAT HISTORY
# ============================================================

else:
    for index, message in enumerate(st.session_state.messages):
        role = message.get(
            "role",
            "assistant",
        )

        # ----------------------------------------------------
        # USER MESSAGE
        # ----------------------------------------------------

        if role == "user":
            display_html(
                """
                <div class="km-msg-row user">

                    <div class="km-msg-label user">
                        You
                    </div>

                </div>
                """
            )

            with st.container(key=f"km_user_bubble_{index}"):
                st.markdown(
                    message.get(
                        "content",
                        "",
                    )
                )

            continue

        # ----------------------------------------------------
        # ASSISTANT MESSAGE
        # ----------------------------------------------------

        display_html(
            """
            <div class="km-msg-row">

                <div class="km-msg-avatar">
                    K
                </div>

                <div class="km-msg-label">
                    KnowledgeMesh
                </div>

            </div>
            """
        )

        with st.container(key=f"km_bubble_{index}"):
            st.markdown(
                message.get(
                    "content",
                    "No response.",
                )
            )

        trace = message.get("trace")

        if not trace:
            continue

        # ----------------------------------------------------
        # TRACE DATA
        # ----------------------------------------------------

        verdict = trace.get(
            "verdict",
            {},
        )

        verdict_kind = verdict.get("kind")

        verdict_label = verdict.get(
            "label",
            "",
        )

        query_type = trace.get(
            "query_type",
            "technical",
        )

        source_list = trace.get(
            "sources",
            [],
        )

        source_count = len(source_list)

        latency = trace.get("latency")

        search_query = trace.get("search_query")

        status_text = trace.get("status")

        # ----------------------------------------------------
        # SIGNAL READOUT
        # ----------------------------------------------------

        verdict_class = {
            "grounded": "ok",
            "memory": "success",
            "no_context": "danger",
        }.get(
            verdict_kind,
            "",
        )

        readout_parts = []

        if verdict_label:
            readout_parts.append(
                f"""
                <span class="km-chip {verdict_class}">
                    {esc(verdict_label)}
                </span>
                """
            )

        readout_parts.append(
            f"""
            <span class="km-chip">

                Mode
                <span class="num">
                    {esc(query_type.title())}
                </span>

            </span>
            """
        )

        readout_parts.append(
            f"""
            <span class="km-chip">

                Sources
                <span class="num">
                    {source_count}
                </span>

            </span>
            """
        )

        if latency is not None:
            readout_parts.append(
                f"""
                <span class="km-chip">
                    {float(latency):.2f}s round trip
                </span>
                """
            )

        readout_html = (
            '<div class="km-signal-readout">'
            '<div class="km-signal-row">' + "".join(readout_parts) + "</div>"
        )

        # ----------------------------------------------------
        # SEARCH QUERY
        # ----------------------------------------------------

        if search_query:
            readout_html += (
                """
                <details class="km-details">

                    <summary>
                        Planner search query
                    </summary>

                    <div
                        style="
                            padding-top: 9px;
                            color: var(--km-muted);
                            font-size: 11.5px;
                        "
                    >
                """
                f"""
                        <code>
                            {esc(search_query)}
                        </code>
                """
                """
                    </div>

                </details>
                """
            )

        # ----------------------------------------------------
        # SOURCES
        # ----------------------------------------------------

        if source_list:
            source_cards = []

            for source in source_list:
                source_name = source.get(
                    "name",
                    "Unknown document",
                )

                source_type = source_type_label(source.get("source_type"))

                vector_score = format_score(source.get("score"))

                rerank_score = format_score(source.get("rerank_score"))

                document_id = source.get("id") or "Unknown"

                source_url = safe_url(source.get("url"))

                link_html = ""

                if source_url:
                    link_html = f"""
                        <a
                            class="km-source-link"
                            href="{esc(source_url)}"
                            target="_blank"
                            rel="noopener noreferrer"
                        >
                            Open source
                        </a>
                        """

                source_cards.append(
                    f"""
                    <div class="km-source">

                        <div class="km-source-meta">

                            <span class="km-source-number">
                                [{source["n"]}]
                            </span>

                            <span class="km-source-name">
                                {esc(source_name)}
                            </span>

                            <span class="km-source-type">
                                {esc(source_type)}
                            </span>

                            <span class="km-source-score">
                                vector {esc(vector_score)}
                            </span>

                            <span class="km-source-score">
                                rerank {esc(rerank_score)}
                            </span>

                            {link_html}

                        </div>

                        <div class="km-source-body">
                            {esc(source.get("text", ""))}
                        </div>

                        <div class="km-source-id">
                            ID: {esc(document_id)}
                        </div>

                    </div>
                    """
                )

            readout_html += f"""
                <details class="km-details">

                    <summary>
                        View retrieved context · {source_count}
                    </summary>

                    <div class="km-sources">
                        {"".join(source_cards)}
                    </div>

                </details>
                """

        # ----------------------------------------------------
        # REASONING TRACE
        # ----------------------------------------------------

        steps = trace.get(
            "steps",
            [],
        )

        if steps:
            trace_items = []

            for step in steps:
                _, code, detail = classify_step(step)

                trace_items.append(
                    f"""
                    <li>
                        <b>{esc(code)}</b>
                        — {esc(detail)}
                    </li>
                    """
                )

            readout_html += f"""
                <details class="km-details">

                    <summary>
                        Inspect reasoning trace
                    </summary>

                    <ol class="km-trace-list">
                        {"".join(trace_items)}
                    </ol>

                </details>
                """

        # ----------------------------------------------------
        # BACKEND STATUS
        # ----------------------------------------------------

        if status_text:
            readout_html += f"""
                <div
                    style="
                        margin-top: 10px;
                        color: var(--km-muted-soft);
                        font-size: 10.5px;
                    "
                >
                    Status: {esc(status_text)}
                </div>
                """

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
    <div
        style="
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 8px;
            color: var(--km-muted-soft);
            font-size: 10.5px;
        "
    >

        <span>
            KnowledgeMesh Agentic RAG
        </span>

        <span>·</span>

        <span>
            Qdrant retrieval
        </span>

        <span>·</span>

        <span>
            FlashRank reranking
        </span>

        <span>·</span>

        <span>
            Portkey synthesis
        </span>

        <span>·</span>

        <span>
            Logfire observability
        </span>

    </div>
    """
)
