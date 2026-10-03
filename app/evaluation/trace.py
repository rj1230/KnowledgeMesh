from __future__ import annotations

from typing import Any, Dict


def append_trace_event(
    state: Dict[str, Any],
    *,
    step: str,
    status: str,
    hop: int | None = None,
    **details: Any,
) -> Dict[str, Any]:
    """
    Append one meaningful agent trajectory event.

    A LangGraph node execution is not automatically a reasoning hop.
    The caller explicitly supplies hop when the event represents
    meaningful reasoning/evidence progress.
    """

    trace = list(state.get("evaluation_trace") or [])

    event: Dict[str, Any] = {
        "step": step,
        "status": status,
    }

    if hop is not None:
        event["hop"] = hop

    if details:
        event["details"] = details

    trace.append(event)

    return {
        "evaluation_trace": trace,
        "hop_count": max(
            int(state.get("hop_count") or 0),
            hop or 0,
        ),
    }
