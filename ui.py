"""
KnowledgeMesh · Agentic RAG Console

Single-file Streamlit frontend for the KnowledgeMesh FastAPI backend.
    POST {BACKEND_URL}/query   {"q": "...", "thread_id": "..."}

The stylesheet is embedded below (APP_CSS), so this file runs on its own.
Optional files next to it:
    .streamlit/config.toml    (theme + minimal toolbar)
    assets/favicon.png, assets/favicon-warning.png, assets/favicon-error.png

Environment:
    BACKEND_URL                http://localhost:8000
    BACKEND_TIMEOUT_SECONDS    180
    LOGFIRE_TOKEN              optional
    KM_DEBUG=1                 shows developer hints, raw error bodies and the
                               editable backend URL (leave off for shared use)

Requires Streamlit >= 1.37 for the evidence dialog (older versions fall back
to an inline expander).
"""

import base64
import html
import inspect
import os
import re
import textwrap
import time
import uuid
from contextlib import nullcontext
from datetime import datetime
from functools import lru_cache

import logfire
import requests
import streamlit as st
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from fastapi.testclient import TestClient


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

def _secret_or_env(name, default=None):
    """Read a Streamlit secret first, then fall back to the environment."""
    try:
        value = st.secrets.get(name)
        if value not in (None, ""):
            return value
    except Exception:
        pass

    return os.getenv(name, default)


STREAMLIT_CLOUD_MODE = str(
    _secret_or_env("STREAMLIT_CLOUD_MODE", "false")
).strip().lower() in {"1", "true", "yes", "on"}

print(f"KnowledgeMesh runtime: STREAMLIT_CLOUD_MODE={STREAMLIT_CLOUD_MODE}")

if STREAMLIT_CLOUD_MODE:
    DEFAULT_BACKEND_URL = "in-process://knowledgemesh"
else:
    DEFAULT_BACKEND_URL = os.getenv(
        "BACKEND_URL",
        "http://localhost:8000",
    ).rstrip("/")

BACKEND_TIMEOUT_SECONDS = int(os.getenv("BACKEND_TIMEOUT_SECONDS", "180"))
DEBUG_UI = os.getenv("KM_DEBUG", "0").strip().lower() in {"1", "true", "yes"}

# Streamlit renamed use_container_width -> width="stretch"; support both.
_STRETCH = (
    {"width": "stretch"}
    if "width" in inspect.signature(st.button).parameters
    else {"use_container_width": True}
)


@st.cache_resource(show_spinner=False)
def get_inprocess_backend():
    """Create a FastAPI TestClient for Streamlit Cloud mode."""
    secret_names = (
        "QDRANT_CLUSTER_ENDPOINT",
        "QDRANT_API_KEY",
        "GROQ_API_KEY",
        "TAVILY_API_KEY",
        "PORTKEY_API_KEY",
        "PORTKEY_CONFIG_SLUG",
        "GROQ_FALLBACK_API_KEY",
        "LANGSMITH_API_KEY",
        "LANGSMITH_PROJECT",
        "LANGSMITH_ENDPOINT",
        "LOGFIRE_TOKEN",
    )

    for name in secret_names:
        value = _secret_or_env(name)

        if value not in (None, ""):
            os.environ[name] = str(value)

    from app.main import app

    return TestClient(app)


# ============================================================
# LOGFIRE (configured once per process, not on every rerun)
# ============================================================


@st.cache_resource(show_spinner=False)
def init_logfire():
    token = os.getenv("LOGFIRE_TOKEN")

    if not token:
        return False, None

    try:
        logfire.configure(token=token)

        try:
            # Propagates trace context to the FastAPI backend.
            logfire.instrument_requests()
        except Exception as exc:  # optional instrumentation package missing
            print(f"Logfire requests instrumentation skipped: {exc}")

        return True, None

    except Exception as exc:
        print(f"Streamlit Logfire initialization failed: {exc}")
        return False, str(exc)


LOGFIRE_OK, LOGFIRE_ERROR = init_logfire()


def _trace_context():
    if LOGFIRE_OK:
        return logfire.span("KnowledgeMesh UI operation")

    return nullcontext()


# ============================================================
# SMALL HELPERS
# ============================================================


def safe_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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


def format_2dp(value) -> str:
    number = safe_float(value)
    return f"{number:.2f}" if number is not None else "—"


def format_ms(value) -> str:
    milliseconds = safe_float(value)

    if milliseconds is None:
        return "—"

    if milliseconds >= 1000:
        return f"{milliseconds / 1000:.2f}s"

    return f"{milliseconds:.0f}ms"


def format_seconds(value) -> str:
    seconds = safe_float(value)
    return f"{seconds:.2f}s" if seconds is not None else "—"


def tri_state_chip(label_true, label_false, label_unknown, value, unknown_ok=True):
    if value is True:
        return f'<span class="km-chip success">{esc(label_true)}</span>'

    if value is False:
        return f'<span class="km-chip danger">{esc(label_false)}</span>'

    if unknown_ok:
        return f'<span class="km-chip">{esc(label_unknown)}</span>'

    return ""


# ============================================================
# FAILURE / RECOVERY DETECTION
# ============================================================

_RATE_LIMIT_RE = re.compile(
    r"ratelimit|rate[\s_-]?limit|too many requests|(?:error|status|http)\D{0,12}\b429\b",
    re.I,
)
_FAILURE_RE = re.compile(
    r"document grade: failed|grader failed|generation failed|"
    r"response generation failed|retrieval failed|web search failed",
    re.I,
)


def is_rate_limit_error(value) -> bool:
    return bool(_RATE_LIMIT_RE.search(str(value or "")))


def has_recovery_signal(text) -> bool:
    return is_rate_limit_error(text) or bool(_FAILURE_RE.search(str(text or "")))


def _generation_failed(status_text: str) -> bool:
    status_text = str(status_text or "").lower()

    return (
        "generation rate-limited" in status_text
        or "generation failed" in status_text
        or "response generation failed" in status_text
    )


# ============================================================
# PAGE CONFIG + FAVICON
# ============================================================


def _favicon_variant() -> str:
    for message in reversed(st.session_state.get("messages", [])):
        if message.get("role") != "assistant":
            continue

        trace = message.get("trace") or {}

        if not trace or _generation_failed(trace.get("status")):
            return "favicon-error"

        if (
            trace.get("citation_valid") is False
            or trace.get("is_grounded") is False
            or trace.get("web_search_used")
        ):
            return "favicon-warning"

        break

    return "favicon"


def _page_icon():
    """assets/<variant>.png when present, otherwise a text glyph."""
    path = os.path.join(PROJECT_ROOT, "assets", f"{_favicon_variant()}.png")

    try:
        if os.path.exists(path):
            from PIL import Image

            return Image.open(path)
    except Exception:
        pass

    return "◈"


st.set_page_config(
    page_title="KnowledgeMesh",
    page_icon=_page_icon(),
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLES (embedded) — dark enterprise console: graphite surfaces, hairline
# borders, one indigo→cyan accent, Inter for UI and answers, mono for ids
# and numbers.
# ============================================================

APP_CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root{
  color-scheme:dark;
  --km-bg:#080b11; --km-surface:#0f131c; --km-surface-2:#141926; --km-surface-3:#1b2231;
  --km-line:rgba(148,163,184,.14); --km-line-strong:rgba(148,163,184,.28);
  --km-ink:#f1f4fa; --km-ink-2:#c5ccda; --km-muted:#8d97ab; --km-faint:#6b7589;
  --km-accent:#7b8cff; --km-accent-strong:#b0baff; --km-accent-soft:rgba(123,140,255,.13); --km-accent-border:rgba(123,140,255,.42);
  --km-grad:linear-gradient(135deg,#7b8cff 0%,#4fd1e8 100%);
  --km-success:#34d399; --km-success-soft:rgba(52,211,153,.12); --km-success-border:rgba(52,211,153,.36);
  --km-warning:#fbbf24; --km-warning-soft:rgba(251,191,36,.11); --km-warning-border:rgba(251,191,36,.36);
  --km-danger:#f87171;  --km-danger-soft:rgba(248,113,113,.11);  --km-danger-border:rgba(248,113,113,.36);
  --km-sans:"Inter",system-ui,-apple-system,"Segoe UI",sans-serif;
  --km-mono:"JetBrains Mono",ui-monospace,Consolas,monospace;
  --km-r-sm:8px; --km-r-md:12px; --km-r-lg:16px;
  --km-shadow:0 1px 0 rgba(255,255,255,.035) inset,0 10px 28px -14px rgba(0,0,0,.7);
}

html,body,[class*="css"],.stApp{font-family:var(--km-sans);-webkit-font-smoothing:antialiased}
.stApp{color:var(--km-ink);background:
  radial-gradient(1100px 520px at 78% -8%,rgba(123,140,255,.13),transparent 60%),
  radial-gradient(800px 460px at -8% 0%,rgba(79,209,232,.06),transparent 55%),
  var(--km-bg);background-attachment:fixed}
header[data-testid="stHeader"]{background:transparent}
[data-testid="stAppDeployButton"],#MainMenu,footer{display:none!important}
::selection{background:var(--km-accent-soft)}
:focus-visible{outline:2px solid var(--km-accent)!important;outline-offset:2px}
.km-icon{flex:none;display:inline-block;vertical-align:-2px}
[data-testid="stMainBlockContainer"],.block-container{max-width:960px;margin:0 auto;padding:2.2rem 2rem 9rem}

/* ---------- top bar ---------- */
.km-topbar{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:14px;margin:0 0 22px;padding-bottom:18px;border-bottom:1px solid var(--km-line)}
.km-topbar-title{color:var(--km-ink);font-size:18px;font-weight:600;letter-spacing:-.015em}
.km-topbar-sub{margin-top:2px;color:var(--km-muted);font-size:12.5px}
.km-topbar-pills{display:flex;flex-wrap:wrap;gap:8px}
.km-pill{display:inline-flex;align-items:center;gap:7px;padding:5px 12px;border:1px solid var(--km-line);border-radius:999px;background:var(--km-surface);color:var(--km-ink-2);font-size:12px;font-weight:500}
.km-pill.mono{font:500 11.5px var(--km-mono)}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d111a 0%,#090c13 100%);border-right:1px solid var(--km-line)}
section[data-testid="stSidebar"]>div{padding-top:1rem}
.km-brand{display:flex;align-items:center;gap:11px;padding:2px 2px 18px;margin-bottom:14px;border-bottom:1px solid var(--km-line)}
.km-brand-mark{display:grid;place-items:center;width:34px;height:34px;border-radius:10px;background:var(--km-grad);color:#06101a;font-size:16px;font-weight:700;box-shadow:0 8px 20px -8px rgba(123,140,255,.7)}
.km-brand strong{display:block;color:var(--km-ink);font-size:15px;font-weight:600;letter-spacing:-.01em}
.km-brand small{display:block;margin-top:1px;color:var(--km-muted);font-size:12px}
.km-rail-label{margin:24px 0 9px 2px;color:var(--km-faint);font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase}
.km-status-card,.km-side-card{padding:2px 14px;border:1px solid var(--km-line);border-radius:var(--km-r-md);background:var(--km-surface)}
.km-side-card.kb{padding:13px 14px 2px}
.km-status-row{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:9px 0;font-size:12.5px}
.km-status-row+.km-status-row{border-top:1px solid var(--km-line)}
.km-status-row .label{color:var(--km-muted)}
.km-status-row .value{max-width:150px;overflow:hidden;color:var(--km-ink-2);font:500 12px var(--km-mono);text-align:right;text-overflow:ellipsis;white-space:nowrap}
.km-kb-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;color:var(--km-ink);font-size:13px;font-weight:600}
.km-kb-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px}
.km-kb-grid>div{padding:9px 11px;border:1px solid var(--km-line);border-radius:10px;background:var(--km-surface-2)}
.km-kb-grid b{display:block;color:var(--km-ink);font:500 19px var(--km-mono)}
.km-kb-grid span{color:var(--km-muted);font-size:11.5px}
.km-hist-active{margin-bottom:6px;padding:9px 11px;border:1px solid var(--km-accent-border);border-radius:10px;background:var(--km-accent-soft)}
.km-hist-active b{display:block;overflow:hidden;color:var(--km-accent-strong);font-size:13px;font-weight:600;text-overflow:ellipsis;white-space:nowrap}
.km-hist-active small{color:var(--km-muted);font-size:12px}
.km-capability-list{display:grid;gap:2px}
.km-capability-row{display:flex;align-items:center;gap:8px;padding:5px 4px;color:var(--km-ink-2);font-size:12.5px}
.km-dot{display:inline-block;width:8px;height:8px;flex:none;border-radius:50%;background:var(--km-faint)}
.km-dot.success{background:var(--km-success);box-shadow:0 0 0 3px var(--km-success-soft)}
.km-dot.warning{background:var(--km-warning);box-shadow:0 0 0 3px var(--km-warning-soft)}
.km-dot.danger{background:var(--km-danger);box-shadow:0 0 0 3px var(--km-danger-soft)}

section[data-testid="stSidebar"] .stButton>button,section[data-testid="stSidebar"] .stDownloadButton>button{
  min-height:38px;border:1px solid var(--km-line);border-radius:10px;background:var(--km-surface);
  color:var(--km-ink-2);font-size:13px;font-weight:500;justify-content:flex-start;text-align:left;transition:border-color .15s,background .15s,color .15s}
section[data-testid="stSidebar"] .stButton>button:hover,section[data-testid="stSidebar"] .stDownloadButton>button:hover{border-color:var(--km-accent-border);background:var(--km-surface-2);color:var(--km-ink)}
section[data-testid="stSidebar"] .stButton>button[kind="primary"],section[data-testid="stSidebar"] .stButton>button[data-testid="stBaseButton-primary"]{
  justify-content:center;border:0;background:var(--km-grad);color:#06101a;font-weight:600;box-shadow:0 8px 20px -10px rgba(123,140,255,.8)}
section[data-testid="stSidebar"] .stButton>button[kind="primary"]:hover,section[data-testid="stSidebar"] .stButton>button[data-testid="stBaseButton-primary"]:hover{filter:brightness(1.08);color:#06101a}
section[data-testid="stSidebar"] .stDownloadButton>button{justify-content:center}

/* ---------- empty state ---------- */
.km-hero{position:relative;margin:.4rem 0 16px;padding:34px 34px 30px;border:1px solid var(--km-line);border-radius:20px;overflow:hidden;
  background:linear-gradient(180deg,rgba(255,255,255,.04),rgba(255,255,255,.008)),var(--km-surface);box-shadow:var(--km-shadow)}
.km-hero::before{content:"";position:absolute;top:-45%;right:-8%;width:520px;height:400px;background:radial-gradient(closest-side,rgba(123,140,255,.24),transparent);pointer-events:none}
.km-eyebrow{position:relative;display:inline-flex;align-items:center;padding:4px 11px;border:1px solid var(--km-accent-border);border-radius:999px;background:var(--km-accent-soft);color:var(--km-accent-strong);font-size:11px;font-weight:600;letter-spacing:.07em;text-transform:uppercase}
.km-hero h1{position:relative;margin:16px 0 10px;padding:0;color:var(--km-ink);font:700 38px/1.1 var(--km-sans);letter-spacing:-.03em}
.km-hero h1 em{font-style:normal;background:var(--km-grad);-webkit-background-clip:text;background-clip:text;color:transparent}
.km-hero p{position:relative;max-width:60ch;margin:0;color:var(--km-ink-2);font-size:15px;line-height:1.7}
.km-features{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:0 0 26px}
.km-feature{padding:16px 16px 15px;border:1px solid var(--km-line);border-radius:14px;background:var(--km-surface)}
.km-feature-icon{display:grid;place-items:center;width:30px;height:30px;margin-bottom:11px;border-radius:9px;background:var(--km-accent-soft);color:var(--km-accent-strong);font-size:14px;font-weight:600}
.km-feature b{display:block;color:var(--km-ink);font-size:13.5px;font-weight:600}
.km-feature span{display:block;margin-top:4px;color:var(--km-muted);font-size:12.5px;line-height:1.55}
.km-section-label{margin:4px 0 10px;color:var(--km-faint);font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase}
.st-key-km_starters .stButton>button{
  justify-content:space-between;gap:12px;height:auto;min-height:52px;padding:12px 18px;text-align:left;
  border:1px solid var(--km-line);border-radius:12px;background:var(--km-surface);
  color:var(--km-ink);font-size:14.5px;font-weight:500;transition:border-color .15s,background .15s}
.st-key-km_starters .stButton>button::after{content:"→";color:var(--km-faint);transition:transform .15s,color .15s}
.st-key-km_starters .stButton>button:hover{border-color:var(--km-accent-border);background:var(--km-surface-2);color:var(--km-ink)}
.st-key-km_starters .stButton>button:hover::after{color:var(--km-accent);transform:translateX(3px)}
.st-key-km_starters [data-testid="stVerticalBlock"]{gap:.5rem}

/* ---------- conversation ---------- */
.km-msg-row{display:flex;align-items:center;gap:10px;margin:34px 0 10px}
.km-msg-row.user{margin-top:40px}
.km-msg-avatar{display:grid;place-items:center;width:26px;height:26px;border-radius:8px;background:var(--km-grad);color:#06101a;font-size:13px;font-weight:700}
.km-msg-row.user .km-msg-avatar{background:var(--km-surface-3);border:1px solid var(--km-line-strong)}
.km-msg-label{color:var(--km-ink);font-size:13.5px;font-weight:600}
.km-msg-meta{padding:2px 9px;border:1px solid var(--km-line);border-radius:999px;color:var(--km-muted);font:500 11px var(--km-mono)}

[class*="st-key-km_user_bubble_"]{
  max-width:100%;padding:14px 18px;overflow:visible;border:1px solid var(--km-line);border-radius:14px;
  background:var(--km-surface-2);color:var(--km-ink);font:500 15.5px/1.6 var(--km-sans)}
[class*="st-key-km_bubble_"]{
  max-width:100%;padding:22px 26px;border:1px solid var(--km-line);border-radius:var(--km-r-lg);box-shadow:var(--km-shadow);
  background:linear-gradient(180deg,rgba(255,255,255,.028),transparent 45%),var(--km-surface);color:#dfe4ee;font:400 15.5px/1.75 var(--km-sans)}
[class*="st-key-km_bubble_"] p,[class*="st-key-km_bubble_"] li,
[class*="st-key-km_user_bubble_"] p{font-family:inherit;font-size:inherit;line-height:inherit;margin-bottom:.85em}
[class*="st-key-km_bubble_"] p,[class*="st-key-km_bubble_"] li{max-width:72ch}
[class*="st-key-km_bubble_"] p:last-child,[class*="st-key-km_user_bubble_"] p:last-child{margin-bottom:0}
[class*="st-key-km_bubble_"] strong{color:var(--km-ink);font-weight:600}
[class*="st-key-km_bubble_"] code{padding:2px 6px;border:1px solid var(--km-line);border-radius:6px;background:var(--km-surface-2);color:var(--km-accent-strong);font:13px var(--km-mono)}
[class*="st-key-km_bubble_"] pre code{border:0;background:none;color:var(--km-ink-2)}
[data-testid="stMarkdownContainer"] blockquote{margin:6px 0;padding:12px 16px;border:0;border-left:3px solid var(--km-accent);border-radius:0 10px 10px 0;background:var(--km-surface-2);color:var(--km-ink-2)}

/* ---------- run summary ---------- */
.km-run-banner{display:flex;align-items:center;justify-content:space-between;gap:24px;margin:14px 0 10px;padding:16px 18px;border:1px solid var(--km-line);border-left:3px solid var(--km-accent);border-radius:var(--km-r-md);background:var(--km-surface)}
.km-run-banner.success{border-left-color:var(--km-success)}.km-run-banner.warning{border-left-color:var(--km-warning)}.km-run-banner.danger{border-left-color:var(--km-danger)}
.km-run-banner-main{min-width:0;flex:1.2}
.km-run-banner-title{display:flex;align-items:center;gap:9px;color:var(--km-ink);font-size:14px;font-weight:600}
.km-run-indicator{width:8px;height:8px;flex:none;border-radius:50%;background:var(--km-accent);box-shadow:0 0 0 4px var(--km-accent-soft)}
.km-run-banner.success .km-run-indicator{background:var(--km-success);box-shadow:0 0 0 4px var(--km-success-soft)}
.km-run-banner.warning .km-run-indicator{background:var(--km-warning);box-shadow:0 0 0 4px var(--km-warning-soft)}
.km-run-banner.danger .km-run-indicator{background:var(--km-danger);box-shadow:0 0 0 4px var(--km-danger-soft)}
.km-run-banner-description{max-width:52ch;margin-top:5px;color:var(--km-muted);font-size:12.5px;line-height:1.55}
.km-run-banner-stats{display:grid;grid-template-columns:repeat(5,minmax(72px,1fr));gap:4px;min-width:430px}
.km-run-stat{min-width:0;padding-left:12px;border-left:1px solid var(--km-line)}
.km-run-stat span{display:block;overflow:hidden;color:var(--km-muted);font-size:11.5px;text-overflow:ellipsis;white-space:nowrap}
.km-run-stat strong{display:block;margin-top:4px;overflow:hidden;color:var(--km-ink);font:500 13.5px var(--km-mono);text-overflow:ellipsis;white-space:nowrap}

/* ---------- answer health + citation chips ---------- */
.km-health{margin:18px 0 10px}
.km-health-title{margin-bottom:9px;color:var(--km-faint);font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase}
.km-health-grid{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:8px}
.km-health-tile{padding:12px 13px 11px;border:1px solid var(--km-line);border-radius:var(--km-r-md);background:var(--km-surface)}
.km-health-tile span{display:block;overflow:hidden;color:var(--km-muted);font-size:11.5px;text-overflow:ellipsis;white-space:nowrap}
.km-health-tile strong{display:block;margin-top:5px;color:var(--km-ink);font:500 17px var(--km-mono)}
.km-health-tile.success{border-color:var(--km-success-border)}.km-health-tile.success strong{color:var(--km-success)}
.km-health-tile.warning{border-color:var(--km-warning-border)}.km-health-tile.warning strong{color:var(--km-warning)}
.km-health-tile.danger{border-color:var(--km-danger-border)}.km-health-tile.danger strong{color:var(--km-danger)}
.km-meter{display:block;height:4px;margin-top:10px;border-radius:99px;background:rgba(148,163,184,.16);overflow:hidden}
.km-meter b{display:block;height:100%;border-radius:99px;background:var(--km-accent)}
.km-health-tile.success .km-meter b{background:var(--km-success)}
.km-health-tile.warning .km-meter b{background:var(--km-warning)}
.km-health-tile.danger .km-meter b{background:var(--km-danger)}
.km-fail{margin:10px 0 14px;padding:14px 16px;border:1px solid var(--km-danger-border);border-left:3px solid var(--km-danger);border-radius:var(--km-r-md);background:var(--km-danger-soft)}
.km-fail-title{color:var(--km-danger);font-size:14px;font-weight:600}
.km-fail-facts{margin:4px 0 10px;color:var(--km-ink-2);font-size:12.5px}
.km-fail-claim{padding:10px 0;border-top:1px solid var(--km-danger-border);color:var(--km-ink);font-size:13.5px;line-height:1.6}
.km-fail-tags{margin-bottom:3px;color:var(--km-danger);font-size:11.5px;font-weight:600}
.km-fail-cites{margin-top:4px;color:var(--km-muted);font:12px var(--km-mono)}
.km-fail-more{padding-top:8px;color:var(--km-muted);font-size:12.5px}
[class*="st-key-km_cites_"] .stButton>button{min-height:30px;padding:2px 12px;border:1px solid var(--km-accent-border);border-radius:999px;background:var(--km-accent-soft);color:var(--km-accent-strong);font:500 12px var(--km-mono);transition:background .15s,border-color .15s}
[class*="st-key-km_cites_"] .stButton>button:hover{border-color:var(--km-accent);background:var(--km-surface-3);color:var(--km-ink)}
[class*="st-key-km_cites_"] [data-testid="stHorizontalBlock"]{gap:.4rem}

/* ---------- latency + pipeline ---------- */
.km-lat{margin:2px 0 16px}
.km-lat-head{display:flex;flex-wrap:wrap;justify-content:space-between;gap:10px;margin-bottom:9px;color:var(--km-muted);font-size:12.5px}
.km-lat-head b{color:var(--km-ink);font-family:var(--km-mono);font-weight:500}
.km-lat-bar{display:flex;gap:3px;height:8px}
.km-lat-seg{display:block;min-width:4px;height:100%;border-radius:99px;background:var(--km-accent)}
.km-lat-seg.warn{background:var(--km-warning)}
.km-lat-seg.unknown{background:repeating-linear-gradient(45deg,var(--km-line-strong) 0 3px,transparent 3px 6px)}
.km-lat-legend{display:flex;flex-wrap:wrap;gap:6px 18px;margin-top:11px;color:var(--km-muted);font-size:12.5px}
.km-lat-legend em{color:var(--km-ink);font:500 12.5px var(--km-mono);font-style:normal}
.km-lat-note{margin-top:9px;color:var(--km-warning);font-size:12.5px;font-weight:500}

.km-pipeline{display:flex;width:100%;margin:12px 0;padding:20px 18px 16px;border:1px solid var(--km-line);border-radius:var(--km-r-md);background:var(--km-surface);overflow-x:auto}
.km-step{display:flex;flex:1;flex-direction:column;align-items:flex-start;gap:10px;min-width:116px}
.km-step.pending{opacity:.5}
.km-step-marker{display:flex;align-items:center;width:100%}
.km-step-dot{display:grid;place-items:center;width:28px;height:28px;flex:none;border:1px solid var(--km-line-strong);border-radius:50%;background:var(--km-surface-2);color:var(--km-faint)}
.km-step.success .km-step-dot{border-color:var(--km-success);background:var(--km-success);box-shadow:0 0 0 4px var(--km-success-soft)}
.km-step.fallback .km-step-dot{border-color:var(--km-warning);background:var(--km-warning);box-shadow:0 0 0 4px var(--km-warning-soft)}
.km-step.failed .km-step-dot{border-color:var(--km-danger);background:var(--km-danger);box-shadow:0 0 0 4px var(--km-danger-soft)}
.km-step.skipped .km-step-dot{border-style:dashed}
.km-step.slow .km-step-dot{outline:2px solid var(--km-warning);outline-offset:3px}
.km-step-line{flex:1;height:2px;margin:0 8px;border-radius:1px;background:var(--km-line)}
.km-step-line.success{background:var(--km-success)}.km-step-line.fallback{background:var(--km-warning)}.km-step-line.failed{background:var(--km-danger)}
.km-step-body strong{display:block;color:var(--km-ink);font-size:13px;font-weight:600}
.km-step.skipped .km-step-body strong{color:var(--km-muted);font-weight:500}
.km-step-body span{display:block;margin-top:2px;color:var(--km-muted);font-size:12px;line-height:1.45}
.km-step-time{display:block;margin-top:4px;color:var(--km-ink-2);font:500 12px var(--km-mono);font-style:normal}
.km-step-time.muted{color:var(--km-faint);font-weight:400}
.km-stage-detail{display:grid;grid-template-columns:1.2fr 1fr;gap:12px;margin:10px 0 14px}
.km-stage-box{padding:14px 16px;border:1px solid var(--km-line);border-radius:var(--km-r-md);background:var(--km-surface)}
.km-stage-box h5{margin:0 0 5px;color:var(--km-ink);font-size:13.5px;font-weight:600}
.km-stage-box.tip{background:var(--km-warning-soft);border-color:var(--km-warning-border)}
.km-stage-box.tip h5{color:var(--km-warning)}
.km-stage-box p{margin:0;color:var(--km-ink-2);font-size:13px;line-height:1.6}
.km-stage-box .nums{display:flex;gap:28px;margin-top:13px}
.km-stage-box .nums span{display:block;color:var(--km-muted);font-size:11.5px}
.km-stage-box .nums strong{color:var(--km-ink);font:500 15px var(--km-mono)}

/* ---------- panels, chips, evidence ---------- */
.km-recovery-card{margin:12px 0;padding:12px 15px;border:1px solid var(--km-warning-border);border-radius:var(--km-r-md);background:var(--km-warning-soft)}
.km-recovery-title{display:flex;align-items:center;gap:7px;color:var(--km-warning);font-size:13px;font-weight:600}
.km-recovery-body{margin-top:5px;color:var(--km-ink-2);font-size:12.5px;line-height:1.55}
.km-tab-panel{margin:8px 0 10px;padding:15px 16px;border:1px solid var(--km-line);border-radius:var(--km-r-md);background:var(--km-surface)}
.km-tab-heading{margin-bottom:11px;color:var(--km-ink);font-size:13px;font-weight:600}
.km-info-card{margin:10px 0;padding:13px 15px;border:1px solid var(--km-line);border-radius:var(--km-r-md);background:var(--km-surface)}
.km-info-card-title{margin-bottom:5px;color:var(--km-ink);font-size:13px;font-weight:600}
.km-info-card-body{color:var(--km-ink-2);font-size:13px;line-height:1.6}
.km-info-card-body code{padding:2px 6px;border:1px solid var(--km-line);border-radius:6px;background:var(--km-surface-2);color:var(--km-accent-strong);font:12.5px var(--km-mono)}
.km-status-note{margin:10px 0 0;color:var(--km-muted);font-size:12.5px;line-height:1.55}
.km-signal-row{display:flex;flex-wrap:wrap;gap:8px}
.km-chip{display:inline-flex;align-items:center;gap:6px;padding:3px 10px;border:1px solid var(--km-line);border-radius:999px;background:var(--km-surface-2);color:var(--km-ink-2);font-size:12px;font-weight:500}
.km-chip.ok{border-color:var(--km-accent-border);background:var(--km-accent-soft);color:var(--km-accent-strong)}
.km-chip.success{border-color:var(--km-success-border);background:var(--km-success-soft);color:var(--km-success)}
.km-chip.warning{border-color:var(--km-warning-border);background:var(--km-warning-soft);color:var(--km-warning)}
.km-chip.danger{border-color:var(--km-danger-border);background:var(--km-danger-soft);color:var(--km-danger)}
.km-chip .num{color:inherit;font-family:var(--km-mono);font-weight:500}

.km-sources{display:grid;gap:10px;margin-top:10px}
.km-source{padding:14px 16px;border:1px solid var(--km-line);border-radius:var(--km-r-md);background:var(--km-surface)}
.km-source-meta{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin-bottom:9px;color:var(--km-muted);font-size:12px}
.km-source-number{color:var(--km-accent-strong);font:500 12.5px var(--km-mono)}
.km-source-name{color:var(--km-ink);font-size:13px;font-weight:600}
.km-source-type{padding:1px 8px;border:1px solid var(--km-line);border-radius:999px;background:var(--km-surface-2);color:var(--km-ink-2)}
.km-origin-badge{padding:1px 8px}
.km-source-score{color:var(--km-muted);font-family:var(--km-mono)}
.km-source-body{max-height:200px;overflow-y:auto;color:var(--km-ink-2);font-size:13px;line-height:1.7}
.km-source-reason{margin-top:9px;color:var(--km-muted);font-size:12px}
.km-source-id{margin-top:9px;color:var(--km-faint);font:11.5px var(--km-mono);word-break:break-all}
.km-source-link{color:var(--km-accent);font-weight:500;text-decoration:none}.km-source-link:hover{text-decoration:underline}
.km-empty-note{margin-top:10px;padding:11px 13px;border:1px dashed var(--km-line-strong);border-radius:var(--km-r-md);color:var(--km-muted);font-size:12.5px;line-height:1.55}
.km-provenance-table{width:100%;margin-top:10px;border-collapse:collapse;font-size:12.5px}
.km-provenance-table th{padding:9px 8px;border-bottom:1px solid var(--km-line-strong);color:var(--km-muted);font-size:11.5px;font-weight:600;text-align:left}
.km-provenance-table td{padding:9px 8px;border-bottom:1px solid var(--km-line);color:var(--km-ink-2);font-family:var(--km-mono)}
.km-claims{display:grid;gap:8px;margin-top:10px}
.km-claim{padding:12px 15px;border:1px solid var(--km-line);border-radius:var(--km-r-md);background:var(--km-surface)}
.km-claim.supported{border-color:var(--km-success-border)}.km-claim.unsupported{border-color:var(--km-danger-border)}
.km-claim-meta{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin-bottom:7px}
.km-claim-text{color:var(--km-ink);font-size:13.5px;line-height:1.6}
.km-claim-citations{margin-top:7px;color:var(--km-muted);font:12px var(--km-mono)}
.km-trace-list{margin:8px 0 0;padding-left:20px;color:var(--km-ink-2);font-size:13px;line-height:1.8}
.km-trace-list b{color:var(--km-accent-strong);font-family:var(--km-mono);font-weight:500}

/* ---------- Streamlit widgets ---------- */
div[data-baseweb="tab-list"]{gap:4px}
div[data-baseweb="tab-border"]{background-color:var(--km-line)!important}
button[data-baseweb="tab"]{height:42px;padding:0 14px;color:var(--km-muted)!important;font-size:13.5px!important;font-weight:500!important}
button[data-baseweb="tab"]:hover{color:var(--km-ink)!important}
button[data-baseweb="tab"][aria-selected="true"]{color:var(--km-ink)!important}
div[data-baseweb="tab-highlight"]{height:2px!important;background:var(--km-grad)!important}
div[data-testid="stExpander"]{border:1px solid var(--km-line);border-radius:var(--km-r-md);background:var(--km-surface);overflow:hidden}
div[data-testid="stExpander"] summary{padding:.7rem 1rem;color:var(--km-ink);font-size:13.5px;font-weight:600}
div[data-testid="stExpander"] summary:hover{color:var(--km-accent-strong)}
.stButton>button{border-radius:var(--km-r-sm);font-weight:500}
div[role="radiogroup"]{flex-wrap:wrap;gap:6px}
div[role="radiogroup"] label{margin:0;padding:4px 13px;border:1px solid var(--km-line);border-radius:999px;background:var(--km-surface-2);color:var(--km-ink-2);cursor:pointer;transition:border-color .15s,background .15s}
div[role="radiogroup"] label>div:first-child{display:none}
div[role="radiogroup"] label:hover{border-color:var(--km-line-strong)}
div[role="radiogroup"] label:has(input:checked){border-color:var(--km-accent-border);background:var(--km-accent-soft);color:var(--km-accent-strong)}
div[role="radiogroup"] label p{font-size:12.5px;font-weight:500}

/* dialog + metrics (evidence drawer) */
div[role="dialog"]{border:1px solid var(--km-line-strong);border-radius:18px!important;background:var(--km-surface)!important;box-shadow:0 30px 80px -20px rgba(0,0,0,.8)}
[data-testid="stMetric"]{padding:11px 13px;border:1px solid var(--km-line);border-radius:var(--km-r-md);background:var(--km-surface-2)}
[data-testid="stMetricLabel"] p{color:var(--km-muted);font-size:11.5px}
[data-testid="stMetricValue"]{color:var(--km-ink);font:500 20px var(--km-mono)}

/* ---------- chat input ---------- */
[data-testid="stBottom"],[data-testid="stBottom"]>div{background:var(--km-bg)!important}
[data-testid="stBottomBlockContainer"]{max-width:960px;padding:12px 2rem 26px}
div[data-testid="stChatInput"]{border:1px solid var(--km-line-strong);border-radius:var(--km-r-lg);background:var(--km-surface);box-shadow:0 10px 30px -12px rgba(0,0,0,.7);transition:border-color .15s,box-shadow .15s}
div[data-testid="stChatInput"]:focus-within{border-color:var(--km-accent);box-shadow:0 0 0 4px var(--km-accent-soft),0 10px 30px -12px rgba(0,0,0,.7)}
div[data-testid="stChatInput"]>div{border:0!important;background:transparent!important}
div[data-testid="stChatInput"] textarea{min-height:auto!important;border:0!important;background:transparent!important;box-shadow:none!important;color:var(--km-ink)!important;font-size:15px!important}
div[data-testid="stChatInput"] textarea::placeholder{color:var(--km-faint)}
[data-testid="stChatInputSubmitButton"]{color:var(--km-accent)!important}
[data-testid="stChatInputSubmitButton"]:disabled{color:var(--km-faint)!important}

/* ---------- inputs, code, status, scrollbars ---------- */
div[data-baseweb="input"],div[data-baseweb="base-input"]{background:var(--km-surface)!important;border-color:var(--km-line)!important;border-radius:10px!important}
div[data-baseweb="input"]:focus-within{border-color:var(--km-accent)!important}
section[data-testid="stSidebar"] input{color:var(--km-ink)!important;-webkit-text-fill-color:var(--km-ink);font-size:13px!important}
section[data-testid="stSidebar"] input::placeholder{color:var(--km-faint)!important;-webkit-text-fill-color:var(--km-faint);opacity:1}
[data-testid="stCode"],[data-testid="stCode"] pre{background:var(--km-surface-2)!important;border:1px solid var(--km-line);border-radius:var(--km-r-md)}
[data-testid="stCode"] code,[data-testid="stCode"] code span{color:var(--km-ink-2)!important;font-family:var(--km-mono)!important;font-size:13px!important}
[data-testid="stStatusWidget"],div[data-testid="stStatus"]{border:1px solid var(--km-line)!important;border-radius:var(--km-r-md)!important;background:var(--km-surface)!important}
/* Streamlit gives .stMarkdown a negative bottom margin that the last <p>
   margin normally cancels; zero it so bubble text isn't clipped. */
[class*="st-key-km_bubble_"] [data-testid="stMarkdown"],[class*="st-key-km_user_bubble_"] [data-testid="stMarkdown"],
[class*="st-key-km_bubble_"] .stMarkdown,[class*="st-key-km_user_bubble_"] .stMarkdown{margin-bottom:0!important}
hr{border-color:var(--km-line)}
::-webkit-scrollbar{width:10px;height:10px}
::-webkit-scrollbar-thumb{background:var(--km-surface-3);border:2px solid transparent;border-radius:99px;background-clip:padding-box}
::-webkit-scrollbar-track{background:transparent}

/* ---------- responsive + motion ---------- */
@media(max-width:1100px){.km-run-banner{flex-direction:column;align-items:stretch;gap:14px}.km-run-banner-stats{min-width:0}}
@media(max-width:900px){.km-hero{padding:26px 22px}.km-hero h1{font-size:30px}.km-features{grid-template-columns:1fr}.km-stage-detail{grid-template-columns:1fr}.km-health-grid{grid-template-columns:repeat(3,minmax(0,1fr))}[data-testid="stMainBlockContainer"],.block-container{padding:1.6rem 1rem 8rem}[class*="st-key-km_bubble_"]{padding:18px 18px}}
@media(max-width:700px){.km-run-banner-stats{grid-template-columns:repeat(2,minmax(0,1fr))}.km-run-stat:nth-child(odd){padding-left:0;border-left:0}}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
""" + "".join(f".km-w{i}{{width:{i * 5}%}}" for i in range(21))


def load_styles():
    st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)


load_styles()


# ============================================================
# ICONS (inline SVG, wrapped in <img> so sanitisers keep them)
# ============================================================

_ICON_PATHS = {
    "check": '<path d="M5 12.5 9.5 17 19 7"/>',
    "refresh": (
        '<path d="M4 12a8 8 0 0 1 14-5.3L20 8"/><path d="M20 4v4h-4"/>'
        '<path d="M20 12a8 8 0 0 1-14 5.3L4 16"/><path d="M4 20v-4h4"/>'
    ),
    "cross": '<path d="M6 6l12 12"/><path d="M18 6 6 18"/>',
    "dash": '<path d="M6 12h12"/>',
    "user": '<circle cx="12" cy="8" r="3.6"/><path d="M5 20c.8-3.8 3.7-6 7-6s6.2 2.2 7 6"/>',
    "dot": '<circle cx="12" cy="12" r="2.2" fill="currentColor" stroke="none"/>',
    "warning": (
        '<path d="M12 3.5 22 20.5H2z"/><path d="M12 9.5v5"/>'
        '<circle cx="12" cy="17.4" r=".9" fill="currentColor" stroke="none"/>'
    ),
}


@lru_cache(maxsize=128)
def icon(name: str, size: int = 14, color: str = "#c3c9d4") -> str:
    paths = _ICON_PATHS.get(name, "").replace("currentColor", color)

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round">{paths}</svg>'
    )
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")

    return (
        f'<img class="km-icon" src="data:image/svg+xml;base64,{encoded}" '
        f'width="{size}" height="{size}" alt="">'
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


def normalize_stage_state(value) -> str:
    value = str(value or "").lower().strip()

    return (
        value
        if value in {"success", "fallback", "failed", "skipped", "pending"}
        else "pending"
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

    status_text = str(trace.get("status") or "")
    steps_text = " ".join(str(step) for step in trace.get("steps", []))

    if _generation_failed(status_text):
        return {
            "kind": "failed",
            "title": "Response generation issue",
            "description": (
                "The workflow completed partially, but the final response "
                "could not be generated."
            ),
        }

    if trace.get("citation_valid") is False or trace.get("is_grounded") is False:
        return {
            "kind": "warning",
            "title": "Completed with a validation warning",
            "description": (
                "An answer was produced, but citation or grounding review "
                "reported a quality issue."
            ),
        }

    if trace.get("web_search_used"):
        return {
            "kind": "fallback",
            "title": "Completed with web fallback",
            "description": (
                "Private knowledge was supplemented with external evidence. "
                "Web sources are labelled in the Evidence tab."
            ),
        }

    if has_recovery_signal(steps_text):
        return {
            "kind": "warning",
            "title": "Completed after recovery",
            "description": (
                "A stage failed or was rate-limited and the workflow recovered."
            ),
        }

    if trace.get("query_type") == "conversational":
        return {
            "kind": "memory",
            "title": "Conversation response",
            "description": (
                "Answered from conversation memory without searching the "
                "knowledge base."
            ),
        }

    return {
        "kind": "success",
        "title": "Completed successfully",
        "description": (
            "Planning, evidence processing, generation and the available "
            "validation checks all completed."
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


def derive_completion_state(
    status_text, citation_valid, is_grounded, web_search_used, thought_process
):
    trace_text = " ".join(str(step) for step in (thought_process or []))

    if _generation_failed(status_text):
        return "error", "Completed with generation issue"
    if citation_valid is False or is_grounded is False:
        return "complete", "Completed with validation warning"
    if web_search_used:
        return "complete", "Completed with web fallback"
    if has_recovery_signal(trace_text):
        return "complete", "Completed after recovery"

    return "complete", "Completed successfully"


# ============================================================
# RENDER FUNCTIONS
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
        "success": icon("check", 13, "#07130d"),
        "fallback": icon("refresh", 13, "#1a1204"),
        "failed": icon("cross", 13, "#1a0709"),
        "skipped": icon("dash", 13, "#9099a8"),
        "pending": icon("dot", 13, "#5e6878"),
    }

    nodes = []
    total = len(PIPELINE_STAGES)

    for index, (key, title, default_detail) in enumerate(PIPELINE_STAGES, start=1):
        stage = statuses.get(key, {})
        state = normalize_stage_state(stage.get("state"))
        detail = stage.get("detail") or default_detail
        icon_markup = icon_map.get(state, icon_map["pending"])

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
        color = "var(--km-warning)" if key == slow else "var(--km-accent)"

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
        idea = "This stage took the largest share of the run."

        if DEBUG_UI and key == "retrieval":
            idea += (
                " Try a lower top-k, keep the vector client warm between "
                "requests, and check whether the embedding call is the slow part."
            )

        tip_html = (
            '<div class="km-stage-box tip"><h5>Slowest stage</h5>'
            f"<p>{esc(idea)}</p></div>"
        )
    elif DEBUG_UI and ms is None and key in STAGE_LATENCY_FIELDS:
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

    revisions = safe_int(trace.get("revision_count"))
    retries = safe_int(trace.get("support_retry_count"))
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
            <div class="km-recovery-title">{icon("warning", 14, "#e8b25c")} Recovery or validation event</div>
            <div class="km-recovery-body">{esc(" ".join(details))}</div>
        </div>
    """


def _score_chip(label, value, danger=False):
    css = " danger" if danger else ""

    return (
        f'<span class="km-chip{css}">{esc(label)} '
        f'<span class="num">{esc(value)}</span></span>'
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
        chips.append(_score_chip("Support", format_2dp(trace["support_score"])))

    if trace.get("usefulness_score") is not None:
        chips.append(_score_chip("Usefulness", format_2dp(trace["usefulness_score"])))

    if details.get("claim_count"):
        chips.append(_score_chip("Claims checked", safe_int(details["claim_count"])))

    unsupported = safe_int(details.get("unsupported_atomic_count"))

    if details.get("atomic_claim_count"):
        chips.append(
            _score_chip(
                "Atomic claims",
                safe_int(details["atomic_claim_count"]),
                bool(unsupported),
            )
        )

    if unsupported:
        chips.append(_score_chip("Unsupported", unsupported, True))

    if details.get("entailment_threshold") is not None:
        chips.append(
            _score_chip(
                "Entailment threshold", format_2dp(details["entailment_threshold"])
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

    return "".join(
        _score_chip(label, safe_int(value))
        for label, value in items
        if value is not None
    )


# ============================================================
# SOURCE NORMALIZATION + CARDS
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
        "private": ("ok", "Private"),
        "web": ("warning", "Web"),
        "unknown": ("", "Unknown"),
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
            grader_html = f'<span class="km-source-score">grader {esc(grader_score)} ({label})</span>'

        # The answer text cites by citation id, so show that as the label.
        label_value = source.get("citation_id") or source.get("n")
        reason_html = (
            f'<div class="km-source-reason">Grader: {esc(source["grader_reason"])}</div>'
            if source.get("grader_reason")
            else ""
        )

        cards.append(
            f"""
            <div class="km-source">
                <div class="km-source-meta">
                    <span class="km-source-number">[{esc(label_value)}]</span>
                    <span class="km-chip {origin_class} km-origin-badge">{esc(origin_label)}</span>
                    <span class="km-source-name">{esc(source.get("name", "Unknown document"))}</span>
                    <span class="km-source-type">{esc(source_type_label(source.get("source_type")))}</span>
                    {score_html}{grader_html}{link_html}
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
        origin = {"private": "Private", "web": "Web"}.get(key, str(raw_origin or "—"))
        color = {"Private": "var(--km-success)", "Web": "var(--km-warning)"}.get(origin)
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


def _claim_keys(items):
    """Backend lists may hold strings or dicts; return the claim texts as a set."""
    keys = set()

    for item in items or []:
        if isinstance(item, dict):
            text = item.get("claim") or item.get("text") or item.get("statement")
            keys.add(str(text) if text else str(sorted(item.items())))
        else:
            keys.add(str(item))

    return keys


def render_claims(claims, grounding_details=None) -> str:
    if not claims:
        return ""

    details = grounding_details or {}
    uncited = _claim_keys(details.get("uncited_claims"))
    invalid = _claim_keys(details.get("invalid_citations"))
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
            chips.append(_score_chip("Entailment", format_2dp(score)))
        if text in uncited or not citations:
            chips.append('<span class="km-chip danger">Uncited</span>')
        if text in invalid:
            chips.append('<span class="km-chip danger">Invalid citation</span>')

        cited = (
            '<div class="km-claim-citations">Cited: '
            + " ".join(f"[{esc(c)}]" for c in citations)
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


# ============================================================
# ANSWER HEALTH (spec section 4)
# Score tiles, the validation verdict, and — on failure — the exact claims
# behind it rather than a bare red status.
# ============================================================

REVISION_BUDGET = 2  # keep in sync with the backend's max answer revisions

_CONTEXT_TONE = {
    "STRONG": "success",
    "AMBIGUOUS": "warning",
    "WEAK": "danger",
    "INSUFFICIENT": "danger",
}


def _pct(value) -> str:
    number = safe_float(value)
    return f"{number * 100:.0f}%" if number is not None else "—"


def _tone(value, good=0.8, ok=0.6) -> str:
    number = safe_float(value)

    if number is None:
        return ""

    return "success" if number >= good else "warning" if number >= ok else "danger"


def _meter(value) -> str:
    number = safe_float(value)

    if number is None:
        return ""

    step = max(0, min(20, round(number * 20)))

    return f'<i class="km-meter"><b class="km-w{step}"></b></i>'


def _claim_text(claim) -> str:
    return str(claim.get("claim") or "") if isinstance(claim, dict) else ""


def citation_consistency(trace, claims):
    """Share of claims whose citations match; None when it can't be derived."""
    if trace.get("citation_valid") is True:
        return 1.0

    if not claims:
        return None

    bad = _claim_keys((trace.get("grounding_details") or {}).get("invalid_citations"))

    if trace.get("citation_valid") is False and not bad:
        return None

    return max(0.0, 1 - len(bad) / len(claims))


def validation_state(trace):
    if trace.get("citation_valid") is False or trace.get("is_grounded") is False:
        return "FAIL"

    if trace.get("citation_valid") is True or trace.get("is_grounded") is True:
        return "PASS"

    return None


def render_failure_panel(trace, claims, details, revisions) -> str:
    uncited_keys = _claim_keys(details.get("uncited_claims"))

    def is_uncited(claim):
        return not claim.get("citations") or _claim_text(claim) in uncited_keys

    flagged = [c for c in claims if c.get("supported") is False or is_uncited(c)]
    n_unsupported = safe_int(details.get("unsupported_atomic_count")) or sum(
        1 for c in claims if c.get("supported") is False
    )
    n_uncited = sum(1 for c in claims if is_uncited(c))

    facts = []

    if n_unsupported:
        facts.append(
            f"{n_unsupported} unsupported atomic claim{'s' if n_unsupported != 1 else ''}"
        )
    if n_uncited:
        facts.append(f"{n_uncited} uncited claim{'s' if n_uncited != 1 else ''}")
    if safe_int(revisions) >= REVISION_BUDGET:
        facts.append("Revision budget exhausted")

    title = (
        "Grounding validation failed"
        if trace.get("is_grounded") is False
        else "Citation validation failed"
    )

    rows = []

    for claim in flagged[:8]:
        tags = []

        if claim.get("supported") is False:
            tags.append("Unsupported")
        if is_uncited(claim):
            tags.append("Uncited")

        score = safe_float(claim.get("score"))

        if score is not None:
            tags.append(f"entailment {score:.2f}")

        cites = " ".join(f"[{esc(c)}]" for c in claim.get("citations") or [])
        cited_html = f'<div class="km-fail-cites">Cited: {cites}</div>' if cites else ""

        rows.append(
            '<div class="km-fail-claim">'
            f'<div class="km-fail-tags">{esc(" · ".join(tags))}</div>'
            f"<div>{esc(_claim_text(claim))}</div>{cited_html}</div>"
        )

    extra = len(flagged) - 8

    if extra > 0:
        rows.append(
            f'<div class="km-fail-more">+ {extra} more in the Quality tab</div>'
        )

    return (
        '<div class="km-fail">'
        f'<div class="km-fail-title">{esc(title)}</div>'
        f'<div class="km-fail-facts">{esc(" · ".join(facts))}</div>'
        + "".join(rows)
        + "</div>"
    )


def render_answer_health(trace: dict) -> str:
    trace = trace or {}
    signals = (
        "support_score",
        "usefulness_score",
        "citation_valid",
        "is_grounded",
        "context_quality",
    )

    if not any(trace.get(key) is not None for key in signals):
        return ""

    claims = [c for c in trace.get("claims") or [] if isinstance(c, dict)]
    details = trace.get("grounding_details") or {}
    state = validation_state(trace)
    revisions = trace.get("revision_count")
    consistency = citation_consistency(trace, claims)
    quality = str(trace.get("context_quality") or "").upper() or "—"
    revisions_label = (
        f"{safe_int(revisions)} / {REVISION_BUDGET}" if revisions is not None else "—"
    )

    tiles = [
        (
            "Grounding",
            _pct(trace.get("support_score")),
            _tone(trace.get("support_score")),
            _meter(trace.get("support_score")),
        ),
        (
            "Citation consistency",
            _pct(consistency),
            _tone(consistency),
            _meter(consistency),
        ),
        (
            "Answer relevance",
            _pct(trace.get("usefulness_score")),
            _tone(trace.get("usefulness_score")),
            _meter(trace.get("usefulness_score")),
        ),
        ("Context quality", quality, _CONTEXT_TONE.get(quality, ""), ""),
        (
            "Revisions",
            revisions_label,
            "warning" if safe_int(revisions) >= REVISION_BUDGET else "",
            "",
        ),
        (
            "Validation",
            state or "—",
            {"PASS": "success", "FAIL": "danger"}.get(state, ""),
            "",
        ),
    ]
    tiles_html = "".join(
        f'<div class="km-health-tile {tone}"><span>{esc(label)}</span>'
        f"<strong>{esc(value)}</strong>{meter}</div>"
        for label, value, tone, meter in tiles
    )

    markup = (
        '<div class="km-health"><div class="km-health-title">Answer health</div>'
        f'<div class="km-health-grid">{tiles_html}</div></div>'
    )

    if state == "FAIL":
        markup += render_failure_panel(trace, claims, details, revisions)

    return markup


# ============================================================
# INTERACTIVE CITATIONS + EVIDENCE DRAWER (spec sections 3, 5, 11)
# Streamlit markdown can't hold clickable inline markers, so each citation the
# answer uses becomes a button under it; clicking opens that source's exact
# evidence in a dialog (inline expander on older Streamlit).
# ============================================================

_dialog = getattr(st, "dialog", None) or getattr(st, "experimental_dialog", None)
_CITE_RE = re.compile(r"\[([^\[\]]+)\]")
CITES_PER_ROW = 6


def _sources_by_citation(trace):
    items = trace.get("answer_sources") or (
        (trace.get("private_sources") or []) + (trace.get("web_sources") or [])
    )

    return {
        str(s["citation_id"]): s
        for s in items
        if isinstance(s, dict) and s.get("citation_id") is not None
    }


def cited_ids(answer, by_id):
    """Citation ids in the order the answer first uses them."""
    seen = []

    for group in _CITE_RE.findall(answer or ""):
        for part in group.split(","):
            cid = part.strip()

            if cid in by_id and cid not in seen:
                seen.append(cid)

    return seen or list(by_id)


def _grounding_score(trace, cid):
    scores = []

    for claim in trace.get("claims") or []:
        if not isinstance(claim, dict):
            continue

        if cid in {str(c) for c in claim.get("citations") or []}:
            score = safe_float(claim.get("score"))

            if score is not None:
                scores.append(score)

    return max(scores) if scores else None


def _citation_match(trace, score):
    if score is None:
        return "—"

    threshold = (
        safe_float((trace.get("grounding_details") or {}).get("entailment_threshold"))
        or 0.8
    )

    return "Strong" if score >= threshold else "Weak"


def render_evidence_body(source, trace):
    cid = source.get("citation_id") or source.get("n")
    name = source.get("name") or "Unknown document"
    retrieval = (
        source.get("rerank_score")
        if source.get("rerank_score") is not None
        else source.get("score")
    )
    grounding = _grounding_score(trace, str(cid))

    st.markdown(f"**{name}**")

    meta = [
        ("Chunk", source.get("chunk_id")),
        ("Document ID", source.get("id")),
        ("Origin", str(source.get("origin") or "").title()),
    ]
    st.caption(" · ".join(f"{k}: {v}" for k, v in meta if v not in (None, "")))

    text = str(source.get("text") or "").strip() or "No passage text was returned."
    st.markdown("> " + text.replace("\n", "\n> "))

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Retrieval score", format_2dp(retrieval))
    col_b.metric("Grounding score", format_2dp(grounding))
    col_c.metric("Citation match", _citation_match(trace, grounding))

    if source.get("grader_reason"):
        st.caption(f"Grader: {source['grader_reason']}")

    url = safe_url(source.get("url"))

    if url and hasattr(st, "link_button"):
        st.link_button("Open document", url)

    st.caption("Copy citation")
    st.code(f"[{cid}] {name}", language="text")


def open_evidence(source, trace):
    cid = source.get("citation_id") or source.get("n")

    if _dialog is None:
        with st.expander(f"Evidence · [{cid}]", expanded=True):
            render_evidence_body(source, trace)

        return

    @_dialog(f"Evidence · [{cid}]")
    def _show():
        render_evidence_body(source, trace)

    _show()


def render_citation_row(index, answer, trace):
    trace = trace or {}
    by_id = _sources_by_citation(trace)
    ids = cited_ids(answer, by_id)

    if not ids:
        return

    st.caption("Cited sources — select one to see the exact evidence")

    with st.container(key=f"km_cites_{index}"):
        for start in range(0, len(ids), CITES_PER_ROW):
            row = ids[start : start + CITES_PER_ROW]

            for col, cid in zip(st.columns(CITES_PER_ROW), row):
                if col.button(f"[{cid}]", key=f"km_cite_{index}_{cid}"):
                    open_evidence(by_id[cid], trace)


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
        speaker = "You" if message.get("role") == "user" else "KnowledgeMesh"
        lines += [f"## {speaker}", "", str(message.get("content", "")), ""]

        trace = message.get("trace")

        if not trace:
            continue

        private = trace.get("private_sources", trace.get("sources", []))

        lines += [
            "### Run summary",
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

        answer_sources = trace.get("answer_sources", [])

        if answer_sources:
            lines += ["### Cited sources", ""]

            for source in answer_sources:
                label = source.get("citation_id") or source.get("n")
                line = f"- [{label}] {source.get('name', 'Unknown document')} ({source.get('origin', 'unknown')})"

                if safe_url(source.get("url")):
                    line += f" {source['url']}"

                lines.append(line)

            lines.append("")

        if trace.get("search_query"):
            lines += [f"**Planner search query:** `{trace['search_query']}`", ""]

        if trace.get("steps"):
            lines += ["### Reasoning trace", ""]
            lines += [f"- {step}" for step in trace["steps"]]
            lines.append("")

    return "\n".join(lines)


# ============================================================
# BACKEND HEALTH
# ============================================================


def check_backend_health(backend_url):
    try:
        if STREAMLIT_CLOUD_MODE:
            response = get_inprocess_backend().get("/health")
            print(
                f"KnowledgeMesh health response: {response.status_code}",
                flush=True,
            )
            return response.status_code == 200

        response = requests.get(f"{backend_url}/health", timeout=4)
        return response.ok

    except Exception as exc:
        print(
            f"KnowledgeMesh health error: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return False


def check_backend_ready(backend_url):
    try:
        if STREAMLIT_CLOUD_MODE:
            response = get_inprocess_backend().get("/ready")
            return (
                response.status_code == 200 and response.json().get("status") == "ready"
            )

        response = requests.get(f"{backend_url}/ready", timeout=4)
        return response.ok and response.json().get("status") == "ready"

    except (requests.RequestException, ValueError):
        return False
    except Exception as exc:
        print(
            f"KnowledgeMesh ready error: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return False


@st.cache_data(ttl=60, show_spinner=False)
def fetch_kb_stats(backend_url):
    """GET /stats -> backend status and knowledge-base statistics."""
    try:
        if STREAMLIT_CLOUD_MODE:
            response = get_inprocess_backend().get("/stats")
            if response.status_code == 200:
                payload = response.json()
                return payload if isinstance(payload, dict) else None
        else:
            response = requests.get(f"{backend_url}/stats", timeout=4)
            if response.ok:
                payload = response.json()
                return payload if isinstance(payload, dict) else None

    except (requests.RequestException, ValueError):
        pass
    except Exception as exc:
        print(
            f"KnowledgeMesh stats error: {type(exc).__name__}: {exc}",
            flush=True,
        )

    return None


def clear_backend_caches():
    check_backend_health.clear()
    check_backend_ready.clear()
    fetch_kb_stats.clear()


# ============================================================
# SESSION STATE
# ============================================================

for _key, _default in (
    ("session_id", lambda: str(uuid.uuid4())),
    ("messages", list),
    ("latencies", list),
    ("history", list),
    ("pending", lambda: False),
    ("session_started_at", lambda: time.strftime("%H:%M:%S")),
    ("backend_url", lambda: DEFAULT_BACKEND_URL),
):
    if _key not in st.session_state:
        st.session_state[_key] = _default()

if "http_session" not in st.session_state:
    _http = requests.Session()
    # Retry connection errors only. Retrying read timeouts or 5xx on /query
    # would repeat a slow LLM call and multiply the wait.
    _adapter = HTTPAdapter(
        max_retries=Retry(
            connect=2,
            read=0,
            status=0,
            backoff_factor=0.5,
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
    st.session_state.pending = False


def start_new_session():
    old_session_id = st.session_state.session_id

    archive_session()

    if LOGFIRE_OK:
        logfire.info("KnowledgeMesh session reset", old_session_id=old_session_id)

    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.latencies = []
    st.session_state.pending = False
    st.session_state.session_started_at = time.strftime("%H:%M:%S")

    check_backend_health.clear()
    check_backend_ready.clear()


# ============================================================
# REQUEST HANDLING
# ============================================================


def submit_question(question: str):
    """Show the user's message immediately; the request runs on the next rerun."""
    question = (question or "").strip()

    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.pending = True


def _assistant_error(content, status_code=None, body=None):
    message = {"role": "assistant", "content": content}

    # Raw backend bodies are only kept (and shown) in debug mode.
    if DEBUG_UI and (status_code is not None or body):
        message["error_detail"] = {"status_code": status_code, "body": body or ""}

    st.session_state.messages.append(message)


def parse_response(data: dict, elapsed: float, http_status: int):
    """Turn the backend JSON into (answer_text, trace, completion_state, label)."""
    web_search_used = bool(data.get("web_search_used", False))
    retrieval_used = bool(data.get("retrieval_used", False))
    citation_valid = data.get("citation_valid")
    is_grounded = data.get("is_grounded")
    context_quality = data.get("context_quality")

    # citation_provenance arrives as a dict keyed by citation id
    raw_provenance = data.get("citation_provenance") or {}
    citation_provenance = (
        list(raw_provenance.values())
        if isinstance(raw_provenance, dict)
        else raw_provenance
        if isinstance(raw_provenance, list)
        else []
    )

    thought_process = data.get("thought_process", []) or []
    raw_private = data.get("private_sources") or data.get("sources") or []
    raw_web = data.get("web_sources") or []

    private_sources = [
        normalize_source(s, i + 1, origin="private") for i, s in enumerate(raw_private)
    ]
    web_sources = [
        normalize_source(s, i + 1, origin="web") for i, s in enumerate(raw_web)
    ]
    private_by_id = {s["id"]: s for s in private_sources if s.get("id") is not None}
    web_by_id = {s["id"]: s for s in web_sources if s.get("id") is not None}

    answer_sources = []

    for raw in data.get("answer_sources") or []:
        source_id = raw.get("id") if isinstance(raw, dict) else None

        if source_id is not None and source_id in private_by_id:
            answer_sources.append(private_by_id[source_id])
        elif source_id is not None and source_id in web_by_id:
            answer_sources.append(web_by_id[source_id])
        else:
            origin = raw.get("origin") if isinstance(raw, dict) else "unknown"
            answer_sources.append(
                normalize_source(raw, len(answer_sources) + 1, origin=origin)
            )

    status_text = data.get("status", "Response generated.")
    query_type = infer_query_type(thought_process)
    search_query = data.get("search_query") or extract_search_query(thought_process)

    if query_type == "conversational" or not retrieval_used:
        private_retrieval_status = "Not used"
    elif web_search_used:
        private_retrieval_status = "Attempted"
    else:
        private_retrieval_status = "Used"

    completion_state, completion_label = derive_completion_state(
        status_text, citation_valid, is_grounded, web_search_used, thought_process
    )

    passthrough = (
        "context_reason",
        "should_search_web",
        "answer_supported",
        "answer_useful",
        "support_score",
        "usefulness_score",
        "grounding_scores",
        "grounding_feedback",
        "revision_count",
        "support_retry_count",
        "retrieval_rewrite_count",
        "web_rewrite_count",
        "retrieval_latency_ms",
        "rerank_latency_ms",
        "grader_latency_ms",
        "generation_latency_ms",
    )

    trace = {key: data.get(key) for key in passthrough}
    trace.update(
        {
            "steps": thought_process,
            "query_type": query_type,
            "search_query": search_query,
            "sources": private_sources,
            "private_sources": private_sources,
            "web_sources": web_sources,
            "answer_sources": answer_sources,
            "citation_provenance": citation_provenance,
            "context_quality": context_quality,
            "web_search_used": web_search_used,
            "private_retrieval_status": private_retrieval_status,
            "citation_valid": citation_valid,
            "is_grounded": is_grounded,
            "grounding_details": data.get("grounding_details") or {},
            "claims": data.get("claims") or [],
            "latency": elapsed,
            "backend_latency_ms": data.get("latency_ms"),
            "status": status_text,
            "http_status": http_status,
        }
    )

    answer = data.get("answer", "No response was returned.")

    return answer, trace, completion_state, completion_label


def run_pending_query():
    """Runs the request for the last user message and appends the reply."""
    st.session_state.pending = False

    messages = st.session_state.messages

    if not messages or messages[-1].get("role") != "user":
        return

    question = messages[-1].get("content", "")
    backend_url = st.session_state.backend_url
    response = None

    try:
        with st.status(
            "Planning, retrieving, grading and checking the answer…",
            expanded=True,
        ) as status:
            start_time = time.perf_counter()
            st.write("Connecting to the KnowledgeMesh backend…")

            payload = {"q": question, "thread_id": st.session_state.session_id}

            with _trace_context():
                if STREAMLIT_CLOUD_MODE:
                    response = get_inprocess_backend().post(
                        "/query",
                        json=payload,
                    )
                else:
                    response = st.session_state.http_session.post(
                        f"{backend_url}/query",
                        json=payload,
                        timeout=BACKEND_TIMEOUT_SECONDS,
                    )

            elapsed = time.perf_counter() - start_time

            if response.status_code != 200:
                if response.status_code == 422:
                    label = "Request rejected by backend validation (422)"
                    text = "**The backend rejected the request format.**"
                else:
                    label = f"Backend error ({response.status_code})"
                    text = (
                        "**The backend returned an unexpected response "
                        f"(HTTP {response.status_code}).**"
                    )

                status.update(label=label, state="error", expanded=False)
                _assistant_error(text, response.status_code, response.text[:4000])

                return

            try:
                data = response.json()
            except ValueError:
                status.update(
                    label="Invalid backend response", state="error", expanded=False
                )
                _assistant_error(
                    "**Invalid backend response.**\n\nThe backend did not return valid JSON.",
                    response.status_code,
                    response.text[:4000],
                )

                return

            if not isinstance(data, dict):
                status.update(
                    label="Invalid backend response", state="error", expanded=False
                )
                _assistant_error(
                    "**Invalid backend response.**\n\nExpected a JSON object.",
                    response.status_code,
                    response.text[:4000],
                )

                return

            answer, trace, completion_state, completion_label = parse_response(
                data, elapsed, response.status_code
            )
            status.update(
                label=f"{completion_label} in {elapsed:.2f}s",
                state=completion_state,
                expanded=False,
            )

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "trace": trace}
        )
        st.session_state.latencies.append(elapsed)

        if LOGFIRE_OK:
            logfire.info(
                "KnowledgeMesh response rendered",
                query_type=trace["query_type"],
                private_source_count=len(trace["private_sources"]),
                web_source_count=len(trace["web_sources"]),
                answer_source_count=len(trace["answer_sources"]),
                citation_valid=trace["citation_valid"],
                is_grounded=trace["is_grounded"],
                latency_seconds=elapsed,
                retrieval_latency_ms=trace["retrieval_latency_ms"],
                rerank_latency_ms=trace["rerank_latency_ms"],
                grader_latency_ms=trace["grader_latency_ms"],
                generation_latency_ms=trace["generation_latency_ms"],
            )

    except requests.exceptions.ConnectionError:
        if STREAMLIT_CLOUD_MODE:
            _assistant_error(
                "**KnowledgeMesh backend initialization failed.**\n\n"
                "The in-process FastAPI backend could not be reached. "
                "Check the Streamlit Cloud secrets and application logs."
            )
        else:
            _assistant_error(
                "**Unable to reach the KnowledgeMesh backend.**\n\n"
                f"Could not connect to `{backend_url}`.\n\n"
                "Check that FastAPI is running:\n\n"
                "```bash\n"
                "uvicorn app.main:app --reload --host 0.0.0.0 --port 8000\n"
                "```"
            )

    except requests.exceptions.Timeout:
        _assistant_error(
            "**The backend took too long to respond.**\n\n"
            "It may be loading a model or running a long agentic request.\n\n"
            f"Current UI timeout: `{BACKEND_TIMEOUT_SECONDS}` seconds."
        )

    except requests.exceptions.RequestException as exc:
        _assistant_error("**Network request failed.**", body=str(exc))

    except Exception as exc:
        if LOGFIRE_OK:
            logfire.exception(
                "KnowledgeMesh UI request failed", error_type=type(exc).__name__
            )

        _assistant_error("**Request failed.**", body=str(exc))


# ============================================================
# BACKEND STATUS
# ============================================================

backend_online = check_backend_health(st.session_state.backend_url)
backend_ready = (
    check_backend_ready(st.session_state.backend_url) if backend_online else False
)

if backend_ready:
    system_dot_class, system_text = "success", "Ready"
elif backend_online:
    system_dot_class, system_text = "warning", "Online, not ready"
else:
    system_dot_class, system_text = "danger", "Unreachable"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    display_html(
        """
        <div class="km-brand">
            <div class="km-brand-mark">◈</div>
            <div><strong>KnowledgeMesh</strong><small>Agentic RAG console</small></div>
        </div>
        """
    )

    if st.button("New chat", type="primary", key="km_new_session_sidebar", **_STRETCH):
        start_new_session()
        st.rerun()

    search_term = (
        st.text_input(
            "Search conversations",
            key="km_history_search",
            placeholder="Search conversations",
            label_visibility="collapsed",
        )
        .strip()
        .lower()
    )

    display_html('<div class="km-rail-label">Recent</div>')

    user_turns = len([m for m in st.session_state.messages if m.get("role") == "user"])
    current_title = session_title(st.session_state.messages)

    if st.session_state.messages and (
        not search_term or search_term in current_title.lower()
    ):
        display_html(
            f"""
            <div class="km-hist-active">
                <b>{esc(current_title)}</b>
                <small>Current · {user_turns} question(s)</small>
            </div>
            """
        )

    matches = [
        (i, entry)
        for i, entry in enumerate(st.session_state.history)
        if not search_term or search_term in entry["title"].lower()
    ]

    for i, entry in matches:
        if st.button(
            entry["title"],
            key=f"km_hist_{entry['id']}",
            help=f"{entry['started']} · {entry['questions']} question(s)",
            **_STRETCH,
        ):
            restore_session(i)
            st.rerun()

    if not st.session_state.messages and not matches:
        display_html('<div class="km-empty-note">No conversations yet.</div>')

    kb = fetch_kb_stats(st.session_state.backend_url) or {}

    display_html(
        f"""
        <div class="km-rail-label">Knowledge base</div>
        <div class="km-side-card kb">
            <div class="km-kb-head">
                <span>{esc(kb.get("collection", "Knowledge base"))}</span>
                <span class="km-chip ok">Qdrant</span>
            </div>
            <div class="km-kb-grid">
                <div><b>{esc(kb.get("documents", "—"))}</b><span>Documents</span></div>
                <div><b>{esc(kb.get("chunks", "—"))}</b><span>Chunks</span></div>
            </div>
            <div class="km-status-row">
                <span class="label">Last indexed</span>
                <span class="value">{esc(kb.get("last_indexed", "—"))}</span>
            </div>
        </div>

        <div class="km-rail-label">System</div>
        <div class="km-status-card">
            <div class="km-status-row">
                <span class="label">Backend</span>
                <span class="value"><span class="km-dot {esc(system_dot_class)}"></span> {esc(system_text)}</span>
            </div>
            <div class="km-status-row">
                <span class="label">Session</span>
                <span class="value">{esc(st.session_state.session_id[:8].upper())}</span>
            </div>
            <div class="km-status-row">
                <span class="label">Started</span>
                <span class="value">{esc(st.session_state.session_started_at)}</span>
            </div>
        </div>
        """
    )

    with st.expander("Connection"):
        if DEBUG_UI:
            backend_url_input = st.text_input(
                "Backend URL",
                value=st.session_state.backend_url,
                label_visibility="collapsed",
                key="km_backend_url_input",
                placeholder="http://localhost:8000",
            ).rstrip("/")

            if backend_url_input and backend_url_input != st.session_state.backend_url:
                st.session_state.backend_url = backend_url_input
                clear_backend_caches()
                st.rerun()
        else:
            st.caption(f"Backend: {st.session_state.backend_url}")

        if st.button("Refresh status", key="km_test_conn", **_STRETCH):
            clear_backend_caches()
            st.rerun()

    with st.expander(f"Pipeline · {len(PIPELINE_STAGES)} stages"):
        display_html(
            '<div class="km-capability-list">'
            + "".join(
                f'<div class="km-capability-row"><span>{icon("check", 13, "#4cc99a")}</span>'
                f"<span>{esc(title)}</span></div>"
                for _, title, _ in PIPELINE_STAGES
            )
            + "</div>"
        )

    st.download_button(
        "Export transcript",
        data=transcript_markdown(st.session_state.messages)
        if st.session_state.messages
        else "",
        file_name=f"knowledgemesh-{st.session_state.session_id[:8]}.md",
        mime="text/markdown",
        disabled=not bool(st.session_state.messages),
        key="km_export_transcript",
        **_STRETCH,
    )


# ============================================================
# EMPTY STATE
# Edit KB_TOPICS and STARTERS so they match what your knowledge base contains.
# ============================================================

KB_TOPICS = (
    "LLM-based agents",
    "loop engineering",
    "reliability and rate limiting",
)

STARTERS = (
    "What is loop engineering?",
    "What are the main components of an LLM-based agent?",
    "What does the documentation say about rate limiting?",
)


def render_empty_state():
    stats = fetch_kb_stats(st.session_state.backend_url) or {}
    documents, chunks = stats.get("documents"), stats.get("chunks")

    scale = (
        f"{documents} documents and {chunks} indexed passages. "
        if documents and chunks
        else ""
    )

    display_html(
        f"""
        <div class="km-hero">
            <div class="km-eyebrow">Enterprise knowledge assistant</div>
            <h1>Ask the <em>knowledge base</em></h1>
            <p>
                {esc(scale)}Covers {esc(", ".join(KB_TOPICS))}. Every claim in an
                answer links to its source. When your documents fall short, the
                system searches the web and labels those sources as web evidence.
            </p>
        </div>
        <div class="km-features">
            <div class="km-feature"><div class="km-feature-icon">✓</div>
                <b>Grounded answers</b>
                <span>Every claim cites a source passage you can open and inspect.</span></div>
            <div class="km-feature"><div class="km-feature-icon">↻</div>
                <b>Self-correcting</b>
                <span>Claims are checked against the evidence and revised before you see them.</span></div>
            <div class="km-feature"><div class="km-feature-icon">◷</div>
                <b>Fully observable</b>
                <span>Per-stage timing, retrieval scores and the complete execution trace.</span></div>
        </div>
        <div class="km-section-label">Try asking</div>
        """
    )

    with st.container(key="km_starters"):
        for i, question in enumerate(STARTERS):
            if st.button(question, key=f"km_starter_{i}", **_STRETCH):
                submit_question(question)
                st.rerun()

    st.write("")

    with st.expander("How an answer is produced"):
        st.markdown(
            " → ".join(title for _, title, _ in PIPELINE_STAGES)
            + "\n\nEach answer includes a run summary with per-stage timing, "
            "sources and citations, quality checks, and the full execution trace."
        )


# ============================================================
# CHAT HISTORY
# ============================================================


def render_topbar() -> str:
    kb_stats = fetch_kb_stats(st.session_state.backend_url) or {}

    pills = [
        f'<span class="km-pill"><span class="km-dot {esc(system_dot_class)}"></span>'
        f"API · {esc(system_text)}</span>"
    ]

    if kb_stats.get("chunks"):
        pills.append(
            f'<span class="km-pill">Index · {esc(kb_stats["chunks"])} chunks</span>'
        )

    pills.append(
        f'<span class="km-pill mono">Thread {esc(st.session_state.session_id[:8].upper())}</span>'
    )

    return (
        '<div class="km-topbar"><div>'
        '<div class="km-topbar-title">Ask KnowledgeMesh</div>'
        '<div class="km-topbar-sub">Grounded answers with visible evidence and validation</div>'
        f'</div><div class="km-topbar-pills">{"".join(pills)}</div></div>'
    )


def render_message(index: int, message: dict, last_index: int):
    if message.get("role") == "user":
        display_html(
            '<div class="km-msg-row user">'
            f'<div class="km-msg-avatar">{icon("user", 14, "#c5ccda")}</div>'
            '<div class="km-msg-label">You</div></div>'
        )

        with st.container(key=f"km_user_bubble_{index}"):
            st.markdown(message.get("content", ""))

        return

    answer_latency = safe_float((message.get("trace") or {}).get("latency"))
    meta_html = (
        f'<span class="km-msg-meta">{esc(format_seconds(answer_latency))}</span>'
        if answer_latency is not None
        else ""
    )

    display_html(
        '<div class="km-msg-row"><div class="km-msg-avatar">◈</div>'
        f'<div class="km-msg-label">KnowledgeMesh</div>{meta_html}</div>'
    )

    with st.container(key=f"km_bubble_{index}"):
        st.markdown(message.get("content", "No response."))

    error_detail = message.get("error_detail")

    if error_detail:
        with st.expander("Technical error details", expanded=False):
            if error_detail.get("status_code") is not None:
                st.write(f"HTTP status: `{error_detail['status_code']}`")

            st.code(error_detail.get("body", ""), language="text")

        return

    trace = message.get("trace")

    if not trace:
        return

    private_list = trace.get("private_sources", trace.get("sources", []))
    web_list = trace.get("web_sources", [])
    answer_list = trace.get("answer_sources", [])
    citation_provenance = trace.get("citation_provenance", [])
    query_type = trace.get("query_type", "technical")

    # Interactive citations -> evidence dialog, then the answer health panel.
    render_citation_row(index, message.get("content", ""), trace)

    health_html = render_answer_health(trace)

    if health_html:
        display_html(health_html)

    display_html(render_run_banner(trace))

    with st.expander("Execution details", expanded=False):
        display_html(render_latency_panel(trace))
        display_html(render_pipeline_rail(trace))

        stage_keys = [key for key, _, _ in PIPELINE_STAGES]
        picked_stage = st.radio(
            "Inspect stage",
            stage_keys,
            index=stage_keys.index(bottleneck_key(trace) or "retrieval"),
            format_func=lambda key: STAGE_TITLES[key],
            horizontal=True,
            key=f"km_stage_{index}",
            label_visibility="collapsed",
        )
        display_html(render_stage_detail(trace, picked_stage))

        recovery_notice = render_recovery_notice(trace)

        if recovery_notice:
            display_html(recovery_notice)

        overview_tab, evidence_tab, quality_tab, trace_tab = st.tabs(
            [
                "Overview",
                f"Evidence ({len(private_list) + len(web_list)})",
                "Quality",
                "Trace",
            ]
        )

        with overview_tab:
            chips = [
                ("Mode", query_type.title()),
                (
                    "Private retrieval",
                    trace.get("private_retrieval_status", "Not reported"),
                ),
                (
                    "Web fallback",
                    "Used" if trace.get("web_search_used") else "Not used",
                ),
                ("Private sources", len(private_list)),
                ("Answer sources", len(answer_list)),
            ]
            chips_html = "".join(
                f'<span class="km-chip">{esc(label)} <span class="num">{esc(value)}</span></span>'
                for label, value in chips
            )
            display_html(
                '<div class="km-tab-panel"><div class="km-tab-heading">Run overview</div>'
                f'<div class="km-signal-row">{chips_html}</div></div>'
            )

            if trace.get("context_reason"):
                display_html(
                    '<div class="km-info-card"><div class="km-info-card-title">'
                    "Context evaluation</div>"
                    f'<div class="km-info-card-body">{esc(trace["context_reason"])}</div></div>'
                )

            if trace.get("search_query"):
                display_html(
                    '<div class="km-info-card"><div class="km-info-card-title">'
                    "Planner search query</div>"
                    f'<div class="km-info-card-body"><code>{esc(trace["search_query"])}</code></div></div>'
                )

            if trace.get("status"):
                display_html(
                    f'<div class="km-status-note">Status: {esc(trace["status"])}</div>'
                )

        with evidence_tab:
            for heading, items in (
                ("Private knowledge sources", private_list),
                ("Web fallback sources", web_list),
                ("Sources used in the answer", answer_list),
            ):
                if items:
                    st.markdown(f"#### {heading}")
                    display_html(
                        f'<div class="km-sources">{render_source_cards(items)}</div>'
                    )

            if not answer_list and (private_list or web_list):
                display_html(
                    '<div class="km-empty-note">Retrieved evidence is listed above, '
                    "but the backend did not record any source as directly used in "
                    "the final answer.</div>"
                )

            provenance_html = render_citation_provenance_table(citation_provenance)

            if provenance_html:
                st.markdown("#### Citation provenance")
                display_html(provenance_html)

            if not (private_list or web_list or answer_list or citation_provenance):
                st.info("No source evidence was reported for this response.")

        with quality_tab:
            quality_html = render_quality_summary(trace)

            if quality_html:
                display_html(
                    '<div class="km-tab-panel"><div class="km-tab-heading">Validation signals</div>'
                    f'<div class="km-signal-row">{quality_html}</div></div>'
                )
            else:
                st.info("No validation results were reported for this response.")

            claims_html = render_claims(
                trace.get("claims") or [], trace.get("grounding_details")
            )

            if claims_html:
                st.markdown("#### Claim-level grounding review")
                display_html(claims_html)

            recovery_html = render_recovery_summary(trace)

            if recovery_html:
                display_html(
                    '<div class="km-tab-panel"><div class="km-tab-heading">Self-correction and recovery</div>'
                    f'<div class="km-signal-row">{recovery_html}</div></div>'
                )

            feedback = trace.get("grounding_feedback")

            if feedback:
                items = feedback if isinstance(feedback, list) else [feedback]
                feedback_html = "".join(
                    f"<li>{esc(item)}</li>" for item in items if str(item).strip()
                )

                if feedback_html:
                    display_html(
                        '<div class="km-info-card"><div class="km-info-card-title">'
                        "Grounding feedback</div>"
                        f'<div class="km-info-card-body"><ul class="km-trace-list">{feedback_html}</ul></div></div>'
                    )

        with trace_tab:
            steps = trace.get("steps", [])

            if steps:
                items = "".join(
                    f"<li><b>{esc(code)}</b> — {esc(detail)}</li>"
                    for _, code, detail in (classify_step(step) for step in steps)
                )
                display_html(
                    '<div class="km-tab-panel"><div class="km-tab-heading">Execution trace</div>'
                    f'<ol class="km-trace-list">{items}</ol></div>'
                )
            else:
                st.info("No reasoning trace was reported for this response.")


display_html(render_topbar())

if not st.session_state.messages:
    render_empty_state()
else:
    _last_index = len(st.session_state.messages) - 1

    for _index, _message in enumerate(st.session_state.messages):
        render_message(_index, _message, _last_index)

    # The user's message is already on screen; now run the request below it.
    if st.session_state.pending:
        run_pending_query()
        st.rerun()


# ============================================================
# CHAT INPUT + FOOTER
# ============================================================

prompt = st.chat_input("Ask about your documentation…")

if prompt:
    submit_question(prompt)
    st.rerun()

display_html(
    '<div class="km-status-note" style="margin-top:14px">'
    "KnowledgeMesh · self-correcting agentic RAG over your local FastAPI backend."
    "</div>"
)

