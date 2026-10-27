"""Part 2 of the KnowledgeMesh redesign. Run AFTER redesign_ui.py.

    python redesign_ui.py       (skip if you already ran it)
    python redesign_ui_v2.py

Adds: route-tagged starters, evidence side panel (replaces the dialog),
"Route taken" strip, timeline grouped by phase, grader-score chart.
A backup is written to ui.py.bak3 first; nothing is written if any step fails.
"""

import shutil
from pathlib import Path

EXTRA_CSS = r"""
/* v2 */
.km-tl-group{margin:14px 0 2px;padding:0 8px;color:var(--km-muted);font-size:11px;font-weight:500;letter-spacing:.08em;text-transform:uppercase}
.km-route{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:6px 0 18px}
.km-route-chip{display:inline-flex;align-items:center;gap:7px;padding:4px 11px;border:1px solid var(--km-line-strong);border-radius:999px;color:var(--km-ink-2);font-size:12px}
.km-route-chip.success{border-color:var(--km-success-border);color:var(--km-success)}
.km-route-chip.warning{border-color:var(--km-warning-border);color:var(--km-warning)}
.km-route-chip.danger{border-color:var(--km-danger-border);background:var(--km-danger-soft);color:var(--km-danger)}
.km-route-arrow{color:var(--km-muted);font-size:12px}
.km-gs{margin:8px 0 18px}
.km-gs-head{display:flex;justify-content:space-between;color:var(--km-muted);font-size:11px;font-weight:500;letter-spacing:.08em;text-transform:uppercase}
.km-gs-head span:last-child{font-family:var(--km-mono);letter-spacing:0;text-transform:none}
.km-gs-group{margin:10px 0 2px;color:var(--km-muted);font-size:12px}
.km-gs-row{display:grid;grid-template-columns:26px 1fr 38px;align-items:center;gap:10px;padding:3px 0;color:var(--km-ink-2);font:400 12px var(--km-mono)}
.km-gs-track{position:relative;height:6px;border-radius:3px;background:var(--km-surface-2)}
.km-gs-fill{position:absolute;top:0;left:0;display:block;height:6px;border-radius:3px;background:#5d6678}
.km-gs-fill.ok{background:var(--km-success)}
.km-gs-th{position:absolute;top:-3px;width:1px;height:12px;background:var(--km-muted)}
.km-gs-score{text-align:right;color:var(--km-ink)}
.km-starter-tag{color:var(--km-accent);font:500 11px var(--km-mono)}
.km-rail{height:0}
[data-testid="stColumn"]:has(.km-rail),[data-testid="column"]:has(.km-rail){position:sticky;top:1rem;align-self:flex-start}
"""

PHASE_TIMELINE = r'''
_STAGE_GROUPS = {
    "guardrails": "Plan",
    "planner": "Plan",
    "retrieval": "Private retrieval",
    "reranking": "Private retrieval",
    "grader": "Private retrieval",
    "web_search": "Web fallback",
    "responder": "Answer",
    "validator": "Answer",
    "relevance": "Answer",
    "critic": "Answer",
}


def render_pipeline_rail(trace=None):
    """Timeline grouped by phase. Bars lay the timed stages out in order."""
    trace = trace or {}
    statuses = stage_statuses(trace)
    slow_key = bottleneck_key(trace)
    known = [stage_latency_ms(trace, key) or 0.0 for key, _, _ in PIPELINE_STAGES]
    scale = max(backend_total_ms(trace) or 0.0, sum(known)) or 1.0
    cursor = 0.0
    group = None
    rows = []

    for (key, title, default_detail), ms in zip(PIPELINE_STAGES, known):
        label = _STAGE_GROUPS.get(key, "Other")

        if label != group:
            group = label
            rows.append(f'<div class="km-tl-group">{esc(label)}</div>')

        stage = statuses.get(key, {})
        state = normalize_stage_state(stage.get("state"))
        detail = stage.get("detail") or default_detail
        slow = " slow" if key == slow_key else ""

        if ms:
            left = cursor / scale * 100
            width = max(ms / scale * 100, 0.5)
            cursor += ms
            bar = f'<i class="km-tl-bar" style="left:{left:.2f}%;width:{width:.2f}%"></i>'
            time_html = esc(format_ms(ms))
        else:
            bar = ""
            time_html = "—"

        rows.append(
            f'<div class="km-tl-row {esc(state)}{slow}">'
            f'<span class="km-tl-name"><i class="km-tl-dot"></i>{esc(title)}</span>'
            f'<span class="km-tl-detail">{esc(detail)}</span>'
            f'<span class="km-tl-track">{bar}</span>'
            f'<span class="km-tl-ms">{time_html}</span></div>'
        )

    return f'<div class="km-tl">{"".join(rows)}</div>'
'''

NEW_FUNCS = r'''
GRADER_THRESHOLD = 0.6  # keep in sync with the backend's generation threshold


def render_route_strip(trace: dict) -> str:
    """The path the graph took, derived from the fields already in the payload."""
    trace = trace or {}
    chips = [("Planner", "")]

    if trace.get("query_type") == "conversational":
        chips.append(("Conversation path", ""))
    else:
        web = bool(trace.get("web_search_used"))
        quality = str(trace.get("context_quality") or "").upper()
        chips.append(("Private retrieval", "warning" if web else "success"))

        if web:
            chips.append(("Web fallback", "warning"))

        if quality:
            chips.append((f"Grader · {quality.lower()}", _CONTEXT_TONE.get(quality, "")))

    chips.append(
        ("Responder", "danger" if _generation_failed(trace.get("status")) else "")
    )

    if trace.get("citation_valid") is False:
        chips.append(("Citations · failed", "danger"))

    if trace.get("answer_useful") is False:
        chips.append(("Relevance · failed", "danger"))

    if trace.get("is_grounded") is False:
        chips.append(("Grounding · failed", "danger"))

    chips.append(
        ("Abstained", "danger") if is_abstained(trace) else ("Answered", "success")
    )

    html_chips = [
        f'<span class="km-route-chip {esc(tone)}"><i class="km-dot {esc(tone)}"></i>'
        f"{esc(label)}</span>"
        for label, tone in chips
    ]
    joined = '<span class="km-route-arrow">→</span>'.join(html_chips)

    return (
        '<div class="km-section-label">Route taken</div>'
        f'<div class="km-route">{joined}</div>'
    )


def render_grader_chart(private_list, web_list) -> str:
    def rows(items, prefix):
        out = []

        for i, source in enumerate(items or [], start=1):
            score = safe_float(source.get("grader_score"))

            if score is None:
                continue

            width = max(0.0, min(score, 1.0)) * 100
            ok = " ok" if source.get("grader_relevant") else ""
            marker = f"left:{GRADER_THRESHOLD * 100:.0f}%"
            out.append(
                f'<div class="km-gs-row"><span>{prefix}{i}</span>'
                f'<span class="km-gs-track"><i class="km-gs-fill{ok}" '
                f'style="width:{width:.0f}%"></i><i class="km-gs-th" style="{marker}"></i></span>'
                f'<span class="km-gs-score">{score:.2f}</span></div>'
            )

        return "".join(out)

    private_rows = rows(private_list, "P")
    web_rows = rows(web_list, "W")

    if not (private_rows or web_rows):
        return ""

    private_block = (
        f'<div class="km-gs-group">Private</div>{private_rows}' if private_rows else ""
    )
    web_block = f'<div class="km-gs-group">Web</div>{web_rows}' if web_rows else ""

    return (
        '<div class="km-gs"><div class="km-gs-head"><span>Grader scores</span>'
        f"<span>threshold {GRADER_THRESHOLD:.2f}</span></div>"
        f"{private_block}{web_block}</div>"
    )


def render_evidence_rail():
    """Side panel for the citation picked in the conversation."""
    selection = st.session_state.get("evidence")

    if not selection:
        return

    messages = st.session_state.messages
    index = selection.get("index", -1)
    trace = (messages[index].get("trace") or {}) if 0 <= index < len(messages) else {}
    source = _sources_by_citation(trace).get(str(selection.get("cid")))

    if source is None:
        st.session_state.evidence = None
        return

    display_html('<div class="km-rail"></div><div class="km-section-label">Evidence</div>')
    render_evidence_body(source, trace)

    if st.button("Close evidence", key="km_rail_close", **_STRETCH):
        st.session_state.evidence = None
        st.rerun()
'''

STARTERS_BLOCK = r"""STARTERS = (
    # Knowledge base only
    "Explain loop engineering for LLM-based agents: how the agent loop works step by step, what happens in each iteration, and what makes a loop reliable.",
    # KB + web: harness engineering from the KB, current tooling from the web
    "How does harness engineering improve the reliability of LLM-based agents, and how do current tools like LangGraph and the OpenAI Agents SDK put those ideas into practice?",
    # Off-topic: the router skips retrieval
    "Give me a detailed step-by-step plan to brew pour-over coffee at home, with ratios, grind size, and timing.",
)

STARTER_ROUTES = ("Knowledge base", "KB + web", "Off-topic")
"""

STARTER_LOOP_OLD = """    with st.container(key="km_starters"):
        for i, question in enumerate(STARTERS):
            if st.button(question, key=f"km_starter_{i}", **_STRETCH):
                submit_question(question)
                st.rerun()
"""

STARTER_LOOP_NEW = """    with st.container(key="km_starters"):
        for i, question in enumerate(STARTERS):
            tag_col, button_col = st.columns([1, 7], vertical_alignment="center")
            route = STARTER_ROUTES[i] if i < len(STARTER_ROUTES) else ""

            with tag_col:
                display_html(f'<span class="km-starter-tag">{esc(route)}</span>')

            if button_col.button(question, key=f"km_starter_{i}", **_STRETCH):
                submit_question(question)
                st.rerun()
"""

MAIN_OLD = """display_html(render_topbar())

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
"""

MAIN_NEW = """display_html(render_topbar())

if not st.session_state.messages:
    render_empty_state()
else:
    if st.session_state.get("evidence"):
        # Widen the page and split it: conversation left, evidence right.
        st.markdown(
            "<style>[data-testid='stMainBlockContainer'],.block-container"
            "{max-width:1280px!important}</style>",
            unsafe_allow_html=True,
        )
        _left, _right = st.columns([5, 3], gap="large")
    else:
        _left, _right = st.container(), None

    _last_index = len(st.session_state.messages) - 1

    with _left:
        for _index, _message in enumerate(st.session_state.messages):
            render_message(_index, _message, _last_index)

        # The user's message is already on screen; now run the request below it.
        if st.session_state.pending:
            run_pending_query()
            st.rerun()

    if _right is not None:
        with _right:
            render_evidence_rail()
"""


def replace_once(text, old, new):
    assert text.count(old) == 1, f"expected exactly one match for: {old[:70]!r}"
    return text.replace(old, new)


def replace_func(text, name, new):
    marker = f"\ndef {name}("
    assert text.count(marker) == 1, f"function {name} not found once"
    start = text.index(marker) + 1
    end = text.index("\n\n\n", start)
    return text[:start] + new.strip("\n") + text[end:]


def apply(text):
    assert "km-verify" in text, "run redesign_ui.py first"
    assert "render_route_strip" not in text, "part 2 is already applied"

    # CSS
    end = text.index('""" + "".join(f".km-w{i}')
    text = text[:end] + EXTRA_CSS + text[end:]

    # Grouped timeline
    text = replace_func(text, "render_pipeline_rail", PHASE_TIMELINE)

    # New helper functions, placed before the source cards
    text = replace_once(
        text,
        "def render_source_cards(source_list):",
        NEW_FUNCS.strip("\n") + "\n\n\ndef render_source_cards(source_list):",
    )

    # Citation click selects evidence instead of opening a dialog
    text = replace_once(
        text,
        "                    open_evidence(by_id[cid], trace)",
        '                    st.session_state.evidence = {"index": index, "cid": cid}\n'
        "                    st.rerun()",
    )

    # Session state
    text = replace_once(
        text,
        '    ("backend_url", lambda: DEFAULT_BACKEND_URL),\n):',
        '    ("backend_url", lambda: DEFAULT_BACKEND_URL),\n    ("evidence", lambda: None),\n):',
    )
    text = replace_once(
        text,
        "    st.session_state.messages = []\n    st.session_state.latencies = []\n"
        "    st.session_state.pending = False",
        "    st.session_state.messages = []\n    st.session_state.latencies = []\n"
        "    st.session_state.pending = False\n    st.session_state.evidence = None",
    )
    text = replace_once(
        text,
        '    st.session_state.session_started_at = entry["started"]\n'
        "    st.session_state.pending = False",
        '    st.session_state.session_started_at = entry["started"]\n'
        "    st.session_state.pending = False\n    st.session_state.evidence = None",
    )

    # Starters + empty state
    start = text.index("STARTERS = (")
    stop = text.index("\n)\n", start) + 3
    text = text[:start] + STARTERS_BLOCK + text[stop:]
    text = replace_once(text, STARTER_LOOP_OLD, STARTER_LOOP_NEW)

    # Execution details: route strip above the timeline, grader chart in Evidence
    text = replace_once(
        text,
        "        display_html(render_latency_panel(trace))\n"
        "        display_html(render_pipeline_rail(trace))",
        "        display_html(render_route_strip(trace))\n"
        "        display_html(render_latency_panel(trace))\n"
        "        display_html(render_pipeline_rail(trace))",
    )
    text = replace_once(
        text,
        "        with evidence_tab:\n            for heading, items in (",
        "        with evidence_tab:\n"
        "            chart_html = render_grader_chart(private_list, web_list)\n\n"
        "            if chart_html:\n"
        "                display_html(chart_html)\n\n"
        "            for heading, items in (",
    )

    # Main layout with the evidence side panel
    text = replace_once(text, MAIN_OLD, MAIN_NEW)

    return text


def main():
    path = Path("ui.py")
    original = path.read_text(encoding="utf-8")
    patched = apply(original)
    shutil.copy(path, "ui.py.bak3")
    path.write_text(patched, encoding="utf-8")
    print("part 2 applied. backup: ui.py.bak3")


if __name__ == "__main__":
    main()
