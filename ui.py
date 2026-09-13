import os
import re
import textwrap
import time
import uuid
from datetime import datetime

import logfire
import requests
import streamlit as st
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# Load environment variables explicitly from the root directory
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(dotenv_path=env_path)


# Initialize Logfire
LOGFIRE_ERROR_DETAIL = None
try:
    token = os.getenv("LOGFIRE_TOKEN")
    if not token:
        print("ERROR: LOGFIRE_TOKEN is empty or None!")
    logfire.configure(token=token)
    # logfire.instrument_requests() # Disabled due to OpenTelemetry bug on Windows: MeterProvider.get_meter() got multiple values for argument 'version'
    LOGFIRE_STATUS = "Connected & Tracing"
    LOGFIRE_OK = True
except Exception as e:
    print(f"Logfire Init Error in UI: {e}")
    LOGFIRE_ERROR_DETAIL = str(e)
    LOGFIRE_STATUS = "Standby"
    LOGFIRE_OK = False

LOGFIRE_PROJECT_URL = os.getenv(
    "LOGFIRE_PROJECT_URL"
)  # optional, for a "view trace" link


# --- PAGE CONFIG ---
st.set_page_config(
    page_title="KnowledgeMesh",
    page_icon="🕸️",
    layout="wide",
)

AI_AVATAR = "🤖"
USER_AVATAR = "👤"
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


def render_html(markup: str) -> str:
    """Collapse per-line leading whitespace before handing markup to
    st.markdown(unsafe_allow_html=True).

    Any function that builds HTML/SVG with an f-string carries its
    Python source indentation straight into the returned string. Once
    that indented, multi-line text is interpolated into an outer
    st.markdown() call, Streamlit's markdown parser reads the
    indented lines as an *indented code block* rather than HTML —
    which is exactly what showed up as a raw-text panel with a copy
    button instead of the rendered pipeline graph. Stripping leading
    whitespace from every line (after dedenting the block as a whole)
    avoids that regardless of which function nested the string, or
    how deeply.
    """
    dedented = textwrap.dedent(markup).strip("\n")
    return "\n".join(line.strip() for line in dedented.splitlines())


# ============================================================
# PIPELINE MODEL
# ============================================================
# Fixed node layout for the pipeline graph. Positions are hand-tuned
# for the 660x190 viewBox below, not computed — this is a small,
# known topology (Guardrails -> Planner -> Retriever -> Grader ->
# {Responder | Web Fallback -> Responder}) so a layout algorithm
# would be overkill.
PIPELINE_NODES = {
    "guardrails": {"label": "Guardrails", "x": 50, "y": 90},
    "planner": {"label": "Planner", "x": 190, "y": 90},
    "retriever": {"label": "Retriever", "x": 330, "y": 90},
    "grader": {"label": "Grader", "x": 470, "y": 90},
    "fallback": {"label": "Web Fallback", "x": 470, "y": 160},
    "responder": {"label": "Responder", "x": 610, "y": 90},
}
PIPELINE_EDGES = [
    ("guardrails", "planner"),
    ("planner", "retriever"),
    ("retriever", "grader"),
    ("grader", "responder"),
    ("grader", "fallback"),
    ("fallback", "responder"),
]

# Reasoning-step text gets matched against these to (a) classify which
# pipeline node it belongs to, for the graph, and (b) render a short
# stage code instead of emoji in the trace log.
STAGE_PATTERNS = [
    (re.compile(r"guardrail", re.I), "guardrails", "GRD-IN"),
    (re.compile(r"planner|planning", re.I), "planner", "PLN"),
    (re.compile(r"retriev", re.I), "retriever", "RET"),
    (re.compile(r"grad(e|er|ing)|judge", re.I), "grader", "GRD"),
    (re.compile(r"web|tavily|fallback", re.I), "fallback", "WEB"),
    (re.compile(r"respond|synthes|answer", re.I), "responder", "RES"),
]


def classify_step(step) -> tuple[str, str, str]:
    """Return (node_key, stage_code, detail_text) for one reasoning step."""
    if isinstance(step, dict):
        stage = step.get("stage") or step.get("node") or ""
        detail = step.get("detail") or step.get("message") or str(step)
        haystack = f"{stage} {detail}"
    else:
        detail = str(step)
        haystack = detail

    for pattern, node_key, code in STAGE_PATTERNS:
        if pattern.search(haystack):
            return node_key, code, detail

    return "", "STEP", detail


def render_pipeline_svg(visited: set) -> str:
    """A small node-graph diagram of the agent pipeline. Visited nodes
    (from the latest turn's trace) light up in accent teal; the rest
    stay as dim outlines so the shape of the whole system is always
    visible, not just the part that just ran."""

    def edge_path(a, b):
        ax, ay = PIPELINE_NODES[a]["x"], PIPELINE_NODES[a]["y"]
        bx, by = PIPELINE_NODES[b]["x"], PIPELINE_NODES[b]["y"]
        active = a in visited and b in visited
        color = "var(--km-accent)" if active else "var(--km-border)"
        width = 1.6 if active else 1.2
        if ay == by:
            return f'<line x1="{ax + 16}" y1="{ay}" x2="{bx - 16}" y2="{by}" stroke="{color}" stroke-width="{width}" />'
        midx = (ax + bx) / 2
        return (
            f'<path d="M {ax + 11} {ay + 11} C {midx} {ay + 40}, {midx} {by - 40}, {bx + 11} {by - 11}" '
            f'fill="none" stroke="{color}" stroke-width="{width}" />'
        )

    edges_svg = "".join(edge_path(a, b) for a, b in PIPELINE_EDGES)

    nodes_svg = []
    for key, node in PIPELINE_NODES.items():
        on = key in visited
        fill = "var(--km-accent)" if on else "var(--km-panel)"
        stroke = "var(--km-accent)" if on else "var(--km-border)"
        text_color = "#12131f" if on else "var(--km-muted)"
        label_color = "var(--km-text)" if on else "var(--km-muted)"
        glow = (
            f'<circle cx="{node["x"]}" cy="{node["y"]}" r="15" fill="var(--km-accent)" opacity="0.1" />'
            if on
            else ""
        )
        nodes_svg.append(
            f"""
            {glow}
            <circle cx="{node["x"]}" cy="{node["y"]}" r="11" fill="{fill}" stroke="{stroke}" stroke-width="1.6" />
            <text x="{node["x"]}" y="{node["y"] + 4}" text-anchor="middle" font-size="10" font-weight="700"
                  font-family="JetBrains Mono, monospace" fill="{text_color}">{node["label"][0]}</text>
            <text x="{node["x"]}" y="{node["y"] + 30}" text-anchor="middle" font-size="10.5"
                  font-family="Inter, sans-serif" fill="{label_color}">{node["label"]}</text>
            """
        )

    return f"""
    <svg viewBox="0 0 660 185" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg">
        {edges_svg}
        {"".join(nodes_svg)}
    </svg>
    """


def render_mesh_backdrop(seed_id: str) -> str:
    """Faint decorative node-and-edge mesh used behind the hero panel —
    a literal visual for 'KnowledgeMesh' instead of a stock gradient."""
    pts = [
        (30, 20),
        (110, 55),
        (60, 95),
        (150, 15),
        (210, 70),
        (170, 120),
        (250, 40),
        (300, 90),
        (260, 140),
        (340, 60),
        (380, 110),
        (40, 140),
        (330, 20),
        (400, 30),
    ]
    edges = [
        (0, 1),
        (1, 2),
        (0, 2),
        (1, 3),
        (3, 4),
        (4, 5),
        (2, 5),
        (4, 6),
        (6, 7),
        (7, 8),
        (5, 8),
        (7, 9),
        (9, 10),
        (8, 10),
        (2, 11),
        (11, 5),
        (3, 12),
        (12, 13),
        (9, 13),
    ]
    lines = "".join(
        f'<line x1="{pts[a][0]}" y1="{pts[a][1]}" x2="{pts[b][0]}" y2="{pts[b][1]}" '
        f'stroke="#818cf8" stroke-opacity="0.14" stroke-width="1" />'
        for a, b in edges
    )
    dots = "".join(
        f'<circle cx="{x}" cy="{y}" r="2.2" fill="#818cf8" fill-opacity="0.3" />'
        for x, y in pts
    )
    return (
        f'<svg class="km-mesh-bg" viewBox="0 0 420 160" preserveAspectRatio="xMidYMid slice" '
        f'xmlns="http://www.w3.org/2000/svg" aria-hidden="true">{lines}{dots}</svg>'
    )


def normalize_source(source, index: int) -> dict:
    if isinstance(source, dict):
        text = source.get("content") or source.get("text") or ""
        name = source.get("source") or source.get("filename") or source.get("title")
        score = source.get("score")
    else:
        text = str(source)
        name, score = None, None
    return {"n": index, "text": text, "name": name, "score": score}


def to_transcript_markdown(messages: list) -> str:
    lines = [f"# KnowledgeMesh transcript — {datetime.now():%Y-%m-%d %H:%M}\n"]
    for m in messages:
        speaker = "You" if m["role"] == "user" else "KnowledgeMesh"
        lines.append(f"**{speaker}:** {m['content']}\n")
    return "\n".join(lines)


# --- SESSION STATE ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    logfire.info(f"✨ New User Session Created: {st.session_state.session_id}")
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_started_at" not in st.session_state:
    st.session_state.session_started_at = time.strftime("%H:%M:%S")
if "last_trace" not in st.session_state:
    st.session_state.last_trace = None
if "latencies" not in st.session_state:
    st.session_state.latencies = []
if "http_session" not in st.session_state:
    _s = requests.Session()
    _retry = Retry(total=2, backoff_factor=0.5, status_forcelist=[502, 503, 504])
    _adapter = HTTPAdapter(max_retries=_retry)
    _s.mount("http://", _adapter)
    _s.mount("https://", _adapter)
    st.session_state.http_session = _s


# ============================================================
# STYLING
# ============================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --km-bg: #0d0f14;
        --km-panel: #14171f;
        --km-panel-alt: #191d27;
        --km-border: #262b36;
        --km-border-soft: #1e222c;
        --km-accent: #818cf8;
        --km-accent-strong: #a5b4fc;
        --km-accent-dim: #818cf833;
        --km-accent-solid: #6366f1;
        --km-text: #e8eaf0;
        --km-muted: #8a90a3;
        --km-muted-soft: #666c7f;
        --km-danger: #f0707a;
        --km-warn: #e0a95c;
        --km-info: #6fb7e0;
        --km-radius: 10px;
        --km-shadow-sm: 0 1px 2px rgba(0,0,0,0.24);
        --km-shadow-md: 0 4px 16px rgba(0,0,0,0.28);
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: var(--km-bg); color: var(--km-text); }

    @keyframes kmFadeIn { from { opacity: 0; transform: translateY(3px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes kmPulse { 0%, 100% { opacity: 1; } 50% { opacity: .5; } }

    /* ===================== SIDEBAR ===================== */
    section[data-testid="stSidebar"] { background: var(--km-panel); border-right: 1px solid var(--km-border); }
    section[data-testid="stSidebar"] * { color: var(--km-text) !important; }
    section[data-testid="stSidebar"] > div { padding-top: 0.5rem; }

    .km-sidebar-brand {
        display: flex; align-items: center; gap: 10px;
        padding: 4px 0 16px 0; border-bottom: 1px solid var(--km-border); margin-bottom: 16px;
    }
    .km-sidebar-brand .mark {
        width: 34px; height: 34px; border-radius: 9px;
        background: var(--km-panel-alt); border: 1px solid var(--km-border);
        display: flex; align-items: center; justify-content: center;
        font-size: 1rem; flex-shrink: 0;
    }
    .km-sidebar-brand .name { font-weight: 600; font-size: 0.98rem; line-height: 1.15; color: #fff !important; letter-spacing: -0.01em; }
    .km-sidebar-brand .tag { font-size: 0.72rem; color: var(--km-muted) !important; margin-top: 1px; }

    .km-sidebar-label {
        font-size: 0.76rem; font-weight: 600; color: var(--km-muted) !important;
        margin: 18px 0 8px 2px; display: flex; align-items: center; gap: 6px;
    }
    .km-sidebar-label::before { content: ""; width: 3px; height: 12px; background: var(--km-accent); border-radius: 2px; display: inline-block; }

    .km-system-card {
        background: var(--km-panel-alt); border: 1px solid var(--km-border);
        border-radius: var(--km-radius); padding: 12px 14px;
    }
    .km-system-card .row { display: flex; align-items: center; gap: 8px; font-size: 0.85rem; margin-bottom: 8px; }
    .km-system-card .row:last-child { margin-bottom: 0; }
    .km-system-card { box-shadow: var(--km-shadow-sm); }
    .km-system-card .dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
    .km-system-card .dot.ok { background: var(--km-accent); }
    .km-system-card .dot.warn { background: var(--km-warn); animation: kmPulse 2.4s ease-in-out infinite; }
    .km-system-card .kv { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: var(--km-muted) !important; line-height: 2; }
    .km-system-card .kv b { color: var(--km-text) !important; font-weight: 500; }
    .km-system-card hr { border: none; border-top: 1px solid var(--km-border-soft); margin: 9px 0; }

    .km-sidebar-footer {
        margin-top: 26px; padding-top: 14px; border-top: 1px solid var(--km-border);
        font-size: 0.7rem; color: var(--km-muted-soft) !important; text-align: center; letter-spacing: 0.01em;
    }

    section[data-testid="stSidebar"] button {
        background: var(--km-panel-alt) !important; border: 1px solid var(--km-border) !important;
        color: var(--km-text) !important; font-weight: 500 !important; font-size: 0.86rem !important;
        transition: border-color 0.15s ease, color 0.15s ease;
    }
    section[data-testid="stSidebar"] button:hover { border-color: var(--km-danger) !important; color: var(--km-danger) !important; }

    /* ===================== HEADER ===================== */
    .km-title {
        font-size: 1.7rem; font-weight: 650; color: var(--km-text); letter-spacing: -0.015em;
        display: flex; align-items: center; gap: 9px; margin: 0;
    }
    .km-title .spark { color: var(--km-accent); font-size: 1.3rem; }
    .km-subtitle { color: var(--km-muted); font-size: 0.88rem; margin-top: 4px; }

    .km-badge {
        background: var(--km-panel); border: 1px solid var(--km-border);
        color: var(--km-muted); font-family: 'JetBrains Mono', monospace;
        font-size: 0.74rem; padding: 5px 12px; border-radius: 999px;
        display: inline-flex; align-items: center; gap: 7px; white-space: nowrap;
    }
    .km-badge .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--km-accent); }

    [data-testid="stMetric"] {
        background: var(--km-panel); border: 1px solid var(--km-border);
        border-radius: 10px; padding: 10px 16px; transition: border-color 0.15s ease;
        box-shadow: var(--km-shadow-sm);
    }
    [data-testid="stMetric"]:hover { border-color: var(--km-accent-dim); }
    [data-testid="stMetricLabel"] { color: var(--km-muted) !important; font-size: 0.75rem !important; font-weight: 500 !important; }
    [data-testid="stMetricValue"] { color: var(--km-text) !important; font-family: 'JetBrains Mono', monospace; font-size: 1.3rem !important; }

    /* ===================== PANELS ===================== */
    .km-panel-heading {
        font-size: 0.8rem; font-weight: 600; color: var(--km-text); text-transform: uppercase; letter-spacing: 0.04em;
        display: flex; align-items: center; gap: 7px; margin: 2px 0 12px 0;
    }
    .km-panel-heading::before { content: ""; width: 3px; height: 12px; background: var(--km-accent); border-radius: 2px; }
    .km-panel-heading .count { color: var(--km-muted); font-weight: 400; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; text-transform: none; letter-spacing: 0; }

    .km-hero-panel {
        position: relative; overflow: hidden;
        background: var(--km-panel); border: 1px solid var(--km-border);
        border-radius: 14px; padding: 36px 40px; margin-bottom: 22px;
        box-shadow: var(--km-shadow-md);
    }
    .km-mesh-bg { position: absolute; inset: 0; width: 100%; height: 100%; z-index: 0; opacity: 0.7; }
    .km-hero-panel .hero-content { position: relative; z-index: 1; max-width: 560px; }
    .km-hero-panel .eyebrow {
        color: var(--km-accent-strong); font-family: 'JetBrains Mono', monospace;
        font-size: 0.74rem; margin-bottom: 10px; letter-spacing: 0.02em;
    }
    .km-hero-panel .title-line { color: #fff; font-weight: 600; font-size: 1.3rem; line-height: 1.4; margin-bottom: 12px; letter-spacing: -0.01em; }
    .km-hero-panel .body-text { color: var(--km-muted); font-size: 0.9rem; line-height: 1.7; }

    .km-graph-card {
        background: var(--km-panel); border: 1px solid var(--km-border);
        border-radius: var(--km-radius); padding: 16px 6px 6px 6px; margin-bottom: 16px;
        box-shadow: var(--km-shadow-sm);
    }

    .km-trace-step {
        display: flex; gap: 10px; align-items: flex-start;
        background: var(--km-panel); border: 1px solid var(--km-border);
        border-radius: 8px; padding: 9px 12px; margin-bottom: 6px; font-size: 0.82rem;
        animation: kmFadeIn 0.25s ease both;
    }
    .km-trace-step .code {
        flex-shrink: 0; font-family: 'JetBrains Mono', monospace; font-size: 0.66rem;
        font-weight: 700; color: var(--km-accent-strong); background: var(--km-panel-alt);
        border: 1px solid var(--km-border); border-radius: 5px; padding: 2px 6px;
        min-width: 46px; text-align: center; letter-spacing: 0.01em;
    }
    .km-trace-step .detail { color: var(--km-text); line-height: 1.5; }

    .km-verdict {
        display: inline-flex; align-items: center; gap: 6px;
        font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
        padding: 5px 11px; border-radius: 6px; margin-bottom: 12px; border: 1px solid;
        animation: kmFadeIn 0.25s ease both;
    }
    .km-verdict.grounded { color: var(--km-accent-strong); border-color: var(--km-accent-dim); background: #818cf80d; }
    .km-verdict.fallback { color: var(--km-info); border-color: #6fb7e04d; background: #6fb7e00d; }
    .km-verdict.blocked { color: var(--km-danger); border-color: #f0707a4d; background: #f0707a0d; }

    .km-source-card {
        background: var(--km-panel); border: 1px solid var(--km-border);
        border-radius: 8px; padding: 10px 12px; margin-bottom: 8px; font-size: 0.81rem;
        transition: border-color 0.15s ease;
    }
    .km-source-card:hover { border-color: var(--km-accent-dim); }
    .km-source-card .meta {
        display: flex; justify-content: space-between; align-items: center;
        font-family: 'JetBrains Mono', monospace; font-size: 0.67rem; color: var(--km-muted); margin-bottom: 5px;
    }
    .km-source-card .meta .n { color: var(--km-accent-strong); font-weight: 700; }
    .km-source-card .body { color: var(--km-text); line-height: 1.55; }

    /* Chat */
    div[data-testid="stChatMessage"] {
        background: var(--km-panel); border: 1px solid var(--km-border);
        border-radius: 12px; animation: kmFadeIn 0.25s ease both;
        box-shadow: var(--km-shadow-sm);
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
        background: var(--km-panel-alt);
    }
    div[data-testid="stChatInput"] textarea {
        background: var(--km-panel) !important; color: var(--km-text) !important;
        border: 1px solid var(--km-border) !important; border-radius: 10px !important;
    }
    div[data-testid="stChatInput"] textarea:focus { border-color: var(--km-accent) !important; box-shadow: 0 0 0 1px var(--km-accent-dim) !important; }
    div[data-testid="stExpander"] { background: var(--km-panel); border: 1px solid var(--km-border); border-radius: 9px; }

    hr { border-color: var(--km-border-soft) !important; }

    @media (max-width: 900px) {
        .km-hero-panel { padding: 26px 22px; }
        .km-hero-panel .title-line { font-size: 1.15rem; }
        .km-title { font-size: 1.4rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown(
        render_html(
            """
            <div class="km-sidebar-brand">
                <div class="mark">🕸️</div>
                <div><div class="name">KnowledgeMesh</div><div class="tag">Agentic RAG Console</div></div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    avg_latency = (
        sum(st.session_state.latencies) / len(st.session_state.latencies)
        if st.session_state.latencies
        else None
    )
    latency_line = f"{avg_latency:.1f}s" if avg_latency is not None else "—"
    status_dot = "ok" if LOGFIRE_OK else "warn"
    status_text = "Tracing connected" if LOGFIRE_OK else "Tracing in standby"

    st.markdown('<div class="km-sidebar-label">System</div>', unsafe_allow_html=True)
    st.markdown(
        render_html(
            f"""
            <div class="km-system-card">
                <div class="row"><span class="dot {status_dot}"></span><span>{status_text}</span></div>
                <hr/>
                <div class="kv">
                    <b>Session</b> &nbsp;{st.session_state.session_id[:8]}<br/>
                    <b>Started</b> &nbsp;{st.session_state.session_started_at}<br/>
                    <b>Turns</b> &nbsp;{len(st.session_state.messages) // 2}<br/>
                    <b>Avg latency</b> &nbsp;{latency_line}
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )
    if not LOGFIRE_OK and LOGFIRE_ERROR_DETAIL:
        with st.expander("Tracing error detail"):
            st.code(LOGFIRE_ERROR_DETAIL)

    st.markdown('<div class="km-sidebar-label">Actions</div>', unsafe_allow_html=True)

    if st.session_state.messages:
        st.download_button(
            "⬇️  Export transcript",
            data=to_transcript_markdown(st.session_state.messages),
            file_name=f"knowledgemesh_{st.session_state.session_id[:8]}.md",
            mime="text/markdown",
            width="stretch",
        )

    if st.button("🗑️  Clear history & memory", width="stretch"):
        logfire.warn(
            f"🗑️ Memory Wipe Triggered for session: {st.session_state.session_id}"
        )
        st.session_state.messages = []
        st.session_state.last_trace = None
        st.session_state.latencies = []
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.session_started_at = time.strftime("%H:%M:%S")
        st.rerun()

    if LOGFIRE_PROJECT_URL:
        st.link_button("↗ Open Logfire dashboard", LOGFIRE_PROJECT_URL, width="stretch")

    st.markdown(
        render_html(
            '<div class="km-sidebar-footer">KnowledgeMesh &middot; Self-Correcting Agentic RAG</div>'
        ),
        unsafe_allow_html=True,
    )

# --- HEADER ---
header_col, metric_col1, metric_col2, metric_col3 = st.columns([3, 1, 1, 1])
with header_col:
    st.markdown(
        render_html(
            """
            <p class="km-title"><span class="spark">✦</span> KnowledgeMesh</p>
            <p class="km-subtitle">Self-correcting agentic RAG over your internal documentation</p>
            """
        ),
        unsafe_allow_html=True,
    )
with metric_col1:
    st.metric("Turns", len(st.session_state.messages) // 2)
with metric_col2:
    st.metric("Avg latency", f"{avg_latency:.1f}s" if avg_latency is not None else "—")
with metric_col3:
    st.metric("Trace", "Live" if LOGFIRE_OK else "Standby")

st.markdown(
    render_html(
        f"""
        <div style="display:flex; justify-content:flex-end; margin:6px 0 16px 0;">
            <div class="km-badge"><span class="dot"></span> Pipeline armed · {time.strftime("%H:%M")}</div>
        </div>
        <hr style="border-color:var(--km-border); margin:0 0 22px 0;"/>
        """
    ),
    unsafe_allow_html=True,
)

# --- LAYOUT ---
chat_col, trace_col = st.columns([2, 1], gap="large")

with trace_col:
    visited = (
        set(st.session_state.last_trace["visited"])
        if st.session_state.last_trace
        else set()
    )

    st.markdown(
        render_html(
            f'<div class="km-panel-heading">Pipeline'
            + (
                f' <span class="count">· {len(visited)}/6 nodes traversed</span>'
                if visited
                else ""
            )
            + "</div>"
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        render_html(f'<div class="km-graph-card">{render_pipeline_svg(visited)}</div>'),
        unsafe_allow_html=True,
    )

    if st.session_state.last_trace is None:
        st.caption("The graph and trace for your next question will light up here.")
    else:
        trace = st.session_state.last_trace

        if trace.get("verdict"):
            v = trace["verdict"]
            v_class = {
                "grounded": "grounded",
                "fallback": "fallback",
                "blocked": "blocked",
            }.get(v.get("kind"), "grounded")
            st.markdown(
                render_html(
                    f'<div class="km-verdict {v_class}">{v.get("label", "")}</div>'
                ),
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div class="km-panel-heading">Reasoning trace</div>',
            unsafe_allow_html=True,
        )
        for step in trace.get("steps", []):
            _, code, detail = classify_step(step)
            st.markdown(
                render_html(
                    f'<div class="km-trace-step"><div class="code">{code}</div><div class="detail">{detail}</div></div>'
                ),
                unsafe_allow_html=True,
            )

        st.markdown(
            render_html(
                f'<div class="km-panel-heading" style="margin-top:18px;">Sources '
                f'<span class="count">· {len(trace.get("sources", []))}</span></div>'
            ),
            unsafe_allow_html=True,
        )

        if trace.get("sources"):
            with st.expander("View retrieved context", expanded=False):
                for source in trace["sources"]:
                    meta_bits = []
                    if source["name"]:
                        meta_bits.append(source["name"])
                    if source["score"] is not None:
                        meta_bits.append(
                            f"score {source['score']:.2f}"
                            if isinstance(source["score"], (int, float))
                            else str(source["score"])
                        )
                    meta_line = " · ".join(meta_bits) if meta_bits else "chunk"
                    st.markdown(
                        render_html(
                            f"""
                            <div class="km-source-card">
                                <div class="meta"><span class="n">[{source["n"]}]</span><span>{meta_line}</span></div>
                                <div class="body">{source["text"]}</div>
                            </div>
                            """
                        ),
                        unsafe_allow_html=True,
                    )
        else:
            st.caption("No sources retrieved for this turn.")

        if trace.get("latency") is not None:
            st.caption(f"Round trip: {trace['latency']:.2f}s")

with chat_col:
    if not st.session_state.messages:
        st.markdown(
            render_html(
                f"""
                <div class="km-hero-panel">
                    {render_mesh_backdrop(st.session_state.session_id)}
                    <div class="hero-content">
                        <div class="eyebrow">Agentic RAG · Reasoning Steps · Retrieved Sources</div>
                        <div class="title-line">Ask a question, watch it think, check its work.</div>
                        <div class="body-text">
                            KnowledgeMesh routes your question through a Planner → Retriever → Grader
                            pipeline, falls back to live web search when local retrieval is
                            insufficient, and cites every chunk it used — with each session traced
                            end-to-end in Logfire.
                        </div>
                    </div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    for message in st.session_state.messages:
        avatar = AI_AVATAR if message["role"] == "assistant" else USER_AVATAR
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about your documentation..."):
        with logfire.span(
            "💬 User Chat Interaction",
            user_query=prompt,
            session_id=st.session_state.session_id,
        ):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar=USER_AVATAR):
                st.markdown(prompt)

            with st.chat_message("assistant", avatar=AI_AVATAR):
                with st.status(
                    "Running Planner → Retriever → Grader…", expanded=True
                ) as status:
                    data = None
                    try:
                        start = time.perf_counter()
                        with logfire.span("📡 Calling RAG Backend"):
                            url = f"{BACKEND_URL}/query"
                            payload = {
                                "q": prompt,
                                "thread_id": st.session_state.session_id,
                            }
                            response = st.session_state.http_session.post(
                                url, json=payload, timeout=60
                            )
                        elapsed = time.perf_counter() - start

                        if response.status_code != 200:
                            raise RuntimeError(
                                f"Backend returned HTTP {response.status_code}: {response.text[:300]}"
                            )

                        data = response.json()
                        st.session_state.latencies.append(elapsed)

                        raw_steps = data.get("thought_process", [])
                        visited_nodes = set()
                        for step in raw_steps:
                            node_key, code, detail = classify_step(step)
                            if node_key:
                                visited_nodes.add(node_key)
                            st.write(f"`{code}` {detail}")

                        status.update(
                            label="✅ Answer synthesized",
                            state="complete",
                            expanded=False,
                        )

                        raw_sources = data.get("sources", [])
                        sources = [
                            normalize_source(s, i + 1)
                            for i, s in enumerate(raw_sources)
                        ]
                        if sources:
                            visited_nodes.add("responder")

                        verdict = None
                        if data.get("web_fallback_used"):
                            verdict = {
                                "kind": "fallback",
                                "label": "🌐 Web fallback used",
                            }
                            visited_nodes.add("fallback")
                        elif data.get("guardrail_blocked"):
                            verdict = {
                                "kind": "blocked",
                                "label": "🛡️ Blocked by guardrails",
                            }
                            visited_nodes.add("guardrails")
                        elif sources:
                            verdict = {
                                "kind": "grounded",
                                "label": "✅ Grounded in retrieved sources",
                            }

                        st.session_state.last_trace = {
                            "steps": raw_steps,
                            "sources": sources,
                            "verdict": verdict,
                            "latency": elapsed,
                            "visited": list(visited_nodes),
                        }

                    except requests.exceptions.ConnectionError:
                        logfire.error(
                            "❌ UI-Backend Connection Failed: connection refused"
                        )
                        status.update(label="❌ Backend unreachable", state="error")
                        st.error(
                            f"Can't reach the backend at {BACKEND_URL}. Is it running?"
                        )
                        st.stop()
                    except requests.exceptions.Timeout:
                        logfire.error("❌ UI-Backend Connection Failed: timeout")
                        status.update(label="❌ Request timed out", state="error")
                        st.error("The backend didn't respond in time.")
                        st.stop()
                    except Exception as e:
                        logfire.error(f"❌ UI-Backend Connection Failed: {e}")
                        status.update(label="❌ Request failed", state="error")
                        st.error(str(e))
                        st.stop()

                full_answer = data.get("answer", "No response.")
                st.markdown(full_answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_answer}
                )
                logfire.info("✅ Chat cycle completed successfully.")

        st.rerun()
