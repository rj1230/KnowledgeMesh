import os
import streamlit as st
import requests
import time
import uuid
import logfire
from dotenv import load_dotenv


# Load environment variables explicitly from the root directory
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(dotenv_path=env_path)


# Initialize Logfire
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
    LOGFIRE_STATUS = f"Standby (Error: {e})"
    LOGFIRE_OK = False


# --- PAGE CONFIG ---
st.set_page_config(
    page_title="KnowledgeMesh",
    page_icon="🕸️",
    layout="wide",
)

# --- AVATARS ---
AI_AVATAR = "🤖"
USER_AVATAR = "👤"


# --- SESSION MANAGEMENT ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    logfire.info(f"✨ New User Session Created: {st.session_state.session_id}")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_started_at" not in st.session_state:
    st.session_state.session_started_at = time.strftime("%H:%M:%S")


# =========================================================
# STYLING — dark "KnowledgeMesh" theme (visual only, no
# behavioral changes below this block)
# =========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --km-bg: #0a0e14;
        --km-panel: #111722;
        --km-panel-alt: #151c29;
        --km-border: #232b38;
        --km-accent: #5eead4;
        --km-accent-dim: #2dd4bf55;
        --km-text: #e6edf3;
        --km-muted: #7d8ba1;
        --km-danger: #f87171;
        --km-warn: #fbbf24;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: var(--km-bg);
        color: var(--km-text);
    }

    /* ===================== SIDEBAR ===================== */
    section[data-testid="stSidebar"] {
        background: var(--km-panel);
        border-right: 1px solid var(--km-border);
    }
    section[data-testid="stSidebar"] * {
        color: var(--km-text) !important;
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 0.5rem;
    }

    /* Brand block at top of sidebar */
    .km-sidebar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 4px 0 16px 0;
        border-bottom: 1px solid var(--km-border);
        margin-bottom: 18px;
    }
    .km-sidebar-brand .mark {
        width: 36px;
        height: 36px;
        border-radius: 9px;
        background: linear-gradient(135deg, var(--km-accent), #22a89a);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.15rem;
        flex-shrink: 0;
        box-shadow: 0 0 16px var(--km-accent-dim);
    }
    .km-sidebar-brand .name {
        font-weight: 700;
        font-size: 1.05rem;
        line-height: 1.1;
        color: #ffffff !important;
    }
    .km-sidebar-brand .tag {
        font-size: 0.72rem;
        color: var(--km-muted) !important;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    /* Section labels */
    .km-sidebar-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--km-muted) !important;
        margin: 18px 0 8px 0;
        font-weight: 500;
    }

    /* Status row */
    .km-status-row {
        display: flex;
        align-items: center;
        gap: 8px;
        background: var(--km-panel-alt);
        border: 1px solid var(--km-border);
        border-radius: 8px;
        padding: 10px 12px;
        font-size: 0.85rem;
    }
    .km-status-row .dot {
        width: 8px; height: 8px; border-radius: 50%;
        flex-shrink: 0;
    }
    .km-status-row .dot.ok { background: var(--km-accent); box-shadow: 0 0 6px var(--km-accent); }
    .km-status-row .dot.warn { background: var(--km-warn); box-shadow: 0 0 6px var(--km-warn); }

    /* Session info card */
    .km-session-card {
        background: var(--km-panel-alt);
        border: 1px solid var(--km-border);
        border-radius: 8px;
        padding: 10px 12px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: var(--km-muted) !important;
        line-height: 1.7;
    }
    .km-session-card b {
        color: var(--km-text) !important;
        font-weight: 500;
    }

    /* Sidebar footer */
    .km-sidebar-footer {
        margin-top: 28px;
        padding-top: 14px;
        border-top: 1px solid var(--km-border);
        font-size: 0.72rem;
        color: var(--km-muted) !important;
        text-align: center;
    }

    /* Clear-history button — make it a clean outline/danger style */
    section[data-testid="stSidebar"] button {
        background: var(--km-panel-alt) !important;
        border: 1px solid var(--km-border) !important;
        color: var(--km-text) !important;
        font-weight: 500 !important;
        transition: border-color 0.15s ease, color 0.15s ease;
    }
    section[data-testid="stSidebar"] button:hover {
        border-color: var(--km-danger) !important;
        color: var(--km-danger) !important;
    }
    /* ===================================================== */

    /* Hero header */
    .km-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        border-bottom: 1px solid var(--km-border);
        padding-bottom: 18px;
        margin-bottom: 28px;
    }
    .km-title {
        font-size: 2rem;
        font-weight: 700;
        color: var(--km-text);
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 0;
    }
    .km-title .spark { color: var(--km-accent); }
    .km-subtitle {
        color: var(--km-accent);
        font-size: 0.95rem;
        margin-top: 6px;
    }
    .km-badge {
        background: var(--km-panel);
        border: 1px solid var(--km-border);
        color: var(--km-muted);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        padding: 6px 14px;
        border-radius: 999px;
        display: flex;
        align-items: center;
        gap: 8px;
        white-space: nowrap;
    }
    .km-badge .dot {
        width: 8px; height: 8px; border-radius: 50%;
        background: var(--km-accent);
        box-shadow: 0 0 8px var(--km-accent);
    }

    /* Hero code panel (shown before the first message) */
    .km-hero-panel {
        background: var(--km-panel);
        border: 1px solid var(--km-border);
        border-radius: 12px;
        padding: 32px;
        margin-bottom: 24px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.92rem;
        line-height: 1.9;
        color: #c9d5e3;
        text-align: center;
    }
    .km-hero-panel .tag { color: #7dd3fc; }
    .km-hero-panel .eyebrow { color: var(--km-accent); }
    .km-hero-panel .title-line { color: #ffffff; font-weight: 600; }

    /* Chat bubbles */
    div[data-testid="stChatMessage"] {
        background: var(--km-panel);
        border: 1px solid var(--km-border);
        border-radius: 12px;
    }

    /* Chat input */
    div[data-testid="stChatInput"] textarea {
        background: var(--km-panel) !important;
        color: var(--km-text) !important;
        border: 1px solid var(--km-border) !important;
    }

    /* Expanders (sources) */
    div[data-testid="stExpander"] {
        background: var(--km-panel);
        border: 1px solid var(--km-border);
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- SIDEBAR ---
with st.sidebar:
    # Brand block
    st.markdown(
        """
        <div class="km-sidebar-brand">
            <div class="mark">🕸️</div>
            <div>
                <div class="name">KnowledgeMesh</div>
                <div class="tag">Agentic RAG Console</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # System status
    st.markdown(
        '<div class="km-sidebar-label">System Status</div>', unsafe_allow_html=True
    )
    status_dot_class = "ok" if LOGFIRE_OK else "warn"
    status_text = "Tracing Connected" if LOGFIRE_OK else "Tracing in Standby"
    st.markdown(
        f"""
        <div class="km-status-row">
            <span class="dot {status_dot_class}"></span>
            <span>{status_text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Session info
    st.markdown('<div class="km-sidebar-label">Session</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="km-session-card">
            <b>ID</b> &nbsp;{st.session_state.session_id[:8]}<br/>
            <b>Started</b> &nbsp;{st.session_state.session_started_at}<br/>
            <b>Messages</b> &nbsp;{len(st.session_state.messages)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Actions
    st.markdown('<div class="km-sidebar-label">Actions</div>', unsafe_allow_html=True)
    if st.button("🗑️  Clear History & Memory", width="stretch"):
        logfire.warn(
            f"🗑️ Memory Wipe Triggered for session: {st.session_state.session_id}"
        )
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.session_started_at = time.strftime("%H:%M:%S")
        st.rerun()

    # Footer
    st.markdown(
        """
        <div class="km-sidebar-footer">
            KnowledgeMesh &middot; Internal Docs Assistant
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- MAIN CHAT ---
st.markdown(
    f"""
    <div class="km-header">
        <div>
            <p class="km-title"><span class="spark">✦</span> KnowledgeMesh</p>
            <p class="km-subtitle">Ask questions grounded in your internal documentation</p>
        </div>
        <div class="km-badge"><span class="dot"></span> Trace active · {time.strftime("%H:%M")}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Hero panel — only shown before the conversation starts, purely decorative
if not st.session_state.messages:
    st.markdown(
        """
        <div class="km-hero-panel">
            <div class="eyebrow">Agentic RAG · Reasoning Steps · Retrieved Sources</div>
            <div class="title-line">Ask a question, watch it think, check its work.</div>
            <br/>
            KnowledgeMesh sends your question to a retrieval-augmented backend,
            streams back its reasoning as it goes, and cites every chunk it
            pulled from your documents in an expandable source list — with
            each session traced end-to-end in Logfire.
        </div>
        """,
        unsafe_allow_html=True,
    )

# Display history
for message in st.session_state.messages:
    avatar = AI_AVATAR if message["role"] == "assistant" else USER_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask about your documentation..."):
    # START TRACE: User Interaction
    with logfire.span(
        "💬 User Chat Interaction",
        user_query=prompt,
        session_id=st.session_state.session_id,
    ):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)

        # Assistant Response
        with st.chat_message("assistant", avatar=AI_AVATAR):
            with st.status("🔍 Agent is thinking...", expanded=True) as status:
                try:
                    # DISTRIBUTED TRACE: Calling Backend
                    with logfire.span("📡 Calling RAG Backend"):
                        # Get backend URL from env, or default to local if not set
                        base_url = os.getenv("BACKEND_URL", "http://localhost:8000")
                        url = f"{base_url}/query"
                        payload = {
                            "q": prompt,
                            "thread_id": st.session_state.session_id,
                        }
                        response = requests.post(url, json=payload, timeout=60)
                        data = response.json()

                    # Show Reasoning Steps from Backend
                    steps = data.get("thought_process", [])
                    for step in steps:
                        st.write(f"⚙️ {step}")

                    status.update(
                        label="✅ Answer Synthesized", state="complete", expanded=False
                    )

                    # --- SHOW SOURCES (NESTED EXPANDABLES) ---
                    sources = data.get("sources", [])
                    if sources:
                        with st.expander("📄 View Retrieved Context (Sources)"):
                            for i, source in enumerate(sources):
                                # Create a preview title for each chunk
                                preview = source[:100].replace("\n", " ") + "..."
                                with st.expander(f"Chunk {i + 1}: {preview}"):
                                    st.info(source)
                except Exception as e:
                    logfire.error(f"❌ UI-Backend Connection Failed: {e}")
                    status.update(label="❌ Connection Failed", state="error")
                    st.error("Backend Offline.")
                    st.stop()

            # Final Answer Streaming
            answer_placeholder = st.empty()
            full_answer = data.get("answer", "No response.")

            curr_text = ""
            for char in full_answer:
                curr_text += char
                answer_placeholder.markdown(curr_text + "▌")
                time.sleep(0.005)

            answer_placeholder.markdown(full_answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": full_answer}
            )
            logfire.info("✅ Chat cycle completed successfully.")
