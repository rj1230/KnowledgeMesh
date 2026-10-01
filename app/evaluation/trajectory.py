from __future__ import annotations

from typing import Any, Dict, List


# ============================================================
# Failure taxonomy
# ============================================================

PLANNING_FAILURE = "PLANNING_FAILURE"
RETRIEVAL_FAILURE = "RETRIEVAL_FAILURE"
EVIDENCE_SELECTION_FAILURE = "EVIDENCE_SELECTION_FAILURE"
CITATION_FAILURE = "CITATION_FAILURE"
GROUNDING_FAILURE = "GROUNDING_FAILURE"
GENERATION_FAILURE = "GENERATION_FAILURE"
REVISION_FAILURE = "REVISION_FAILURE"


# ============================================================
# Recovery taxonomy
# ============================================================

QUERY_REWRITE = "QUERY_REWRITE"
PRIVATE_RETRY = "PRIVATE_RETRY"
WEB_ESCALATION = "WEB_ESCALATION"
ANSWER_REVISION = "ANSWER_REVISION"


# ============================================================
# Helpers
# ============================================================

def _events(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(state.get("evaluation_trace") or [])


def _events_by_step(
    state: Dict[str, Any],
    step: str,
) -> List[Dict[str, Any]]:
    return [
        event
        for event in _events(state)
        if str(event.get("step", "")).strip() == step
    ]


def _event_details(event: Dict[str, Any]) -> Dict[str, Any]:
    details = event.get("details")
    return details if isinstance(details, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ============================================================
# Required-hop inference
# ============================================================

def infer_required_hops(state: Dict[str, Any]) -> int:
    """
    Infer the minimum meaningful evidence-acquisition requirement.

    Current KnowledgeMesh semantics:

        conversational query -> 0
        technical retrieval  -> 1

    Multi-hop requirements are intentionally not guessed from query
    wording. They should become explicit once the planner supports
    decomposition into multiple evidence-acquisition objectives.
    """

    if not bool(state.get("retrieval_required", False)):
        return 0

    return 1


# ============================================================
# Evidence acquisition
# ============================================================

def _private_retrieval_events(
    state: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return _events_by_step(
        state,
        "private_retrieval",
    )


def count_retrieval_attempts(
    state: Dict[str, Any],
) -> int:
    """
    Count every private retrieval attempt.

    An empty retrieval is still an attempt, but not a successful
    evidence-acquisition hop.
    """

    return len(_private_retrieval_events(state))


def count_successful_acquisitions(
    state: Dict[str, Any],
) -> int:
    """
    Count private retrieval events that actually acquired evidence.

    This is the current operational definition of meaningful
    evidence-acquisition hops.
    """

    return sum(
        1
        for event in _private_retrieval_events(state)
        if str(event.get("status", "")).strip().lower()
        == "success"
        and int(
            _event_details(event).get("selected_count", 0) or 0
        ) > 0
    )


# ============================================================
# Recovery analysis
# ============================================================

def collect_recovery_events(
    state: Dict[str, Any],
) -> List[str]:
    recoveries: List[str] = []

    for event in _events(state):
        step = str(event.get("step", "")).strip()
        status = str(event.get("status", "")).strip().lower()

        if step == "query_rewriter" and status in {
            "rewritten",
            "retry_same_query",
        }:
            recoveries.append(QUERY_REWRITE)

        elif step == "private_retry":
            recoveries.append(PRIVATE_RETRY)

        elif step == "web_search":
            details = _event_details(event)

            if details.get("attempted") or details.get("used"):
                recoveries.append(WEB_ESCALATION)

        elif step == "revision" and status == "requested":
            recoveries.append(ANSWER_REVISION)

    return recoveries


# ============================================================
# Failure taxonomy
# ============================================================

def infer_failure_taxonomy(
    state: Dict[str, Any],
) -> List[str]:
    failures: List[str] = []

    # --------------------------------------------------------
    # Planning
    # --------------------------------------------------------

    planner_events = _events_by_step(
        state,
        "planner",
    )

    if not planner_events:
        failures.append(PLANNING_FAILURE)

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    retrieval_events = _private_retrieval_events(state)

    if bool(state.get("retrieval_failed", False)):
        failures.append(RETRIEVAL_FAILURE)
    elif any(
        str(event.get("status", "")).strip().lower() == "empty"
        for event in retrieval_events
    ):
        failures.append(RETRIEVAL_FAILURE)

    # --------------------------------------------------------
    # Evidence selection / context
    # --------------------------------------------------------

    context_quality = str(
        state.get("context_quality") or ""
    ).strip().lower()

    if context_quality in {
        "weak",
        "needs_web",
        "unverified",
    }:
        if not bool(state.get("web_search_used", False)):
            failures.append(EVIDENCE_SELECTION_FAILURE)

    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------

    if bool(state.get("generation_failed", False)):
        failures.append(GENERATION_FAILURE)

    # --------------------------------------------------------
    # Citation
    # --------------------------------------------------------

    citation_events = _events_by_step(
        state,
        "citation_check",
    )

    if any(
        str(event.get("status", "")).strip().lower() == "failed"
        for event in citation_events
    ):
        failures.append(CITATION_FAILURE)

    # --------------------------------------------------------
    # Grounding
    # --------------------------------------------------------

    grounding_events = _events_by_step(
        state,
        "grounding_critic",
    )

    if any(
        str(event.get("status", "")).strip().lower() == "failed"
        for event in grounding_events
    ):
        failures.append(GROUNDING_FAILURE)

    # --------------------------------------------------------
    # Revision
    # --------------------------------------------------------

    revision_events = _events_by_step(
        state,
        "revision",
    )

    if any(
        str(event.get("status", "")).strip().lower() == "blocked"
        for event in revision_events
    ):
        failures.append(REVISION_FAILURE)

    return list(dict.fromkeys(failures))


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    state: Dict[str, Any],
    *,
    required_hops: int,
    successful_acquisitions: int,
    retrieval_attempts: int,
) -> Dict[str, float]:

    web_events = _events_by_step(
        state,
        "web_search",
    )

    web_fallbacks = sum(
        1
        for event in web_events
        if _event_details(event).get("used")
    )

    citation_events = _events_by_step(
        state,
        "citation_check",
    )

    citation_failures = sum(
        1
        for event in citation_events
        if str(event.get("status", "")).strip().lower()
        == "failed"
    )

    grounding_events = _events_by_step(
        state,
        "grounding_critic",
    )

    grounding_failures = sum(
        1
        for event in grounding_events
        if str(event.get("status", "")).strip().lower()
        == "failed"
    )

    revision_events = _events_by_step(
        state,
        "revision",
    )

    revision_requests = sum(
        1
        for event in revision_events
        if str(event.get("status", "")).strip().lower()
        == "requested"
    )

    grounding_value = state.get("is_grounded")

    if grounding_value is None:
        grounding_value = state.get("grounding_valid")

    if grounding_value is None:
        grounding_value = state.get("answer_supported")

    final_valid = (
        bool(state.get("citation_valid", False))
        and bool(grounding_value)
        and not bool(state.get("generation_failed", False))
    )

    revision_success = (
        revision_requests > 0
        and final_valid
    )

    if required_hops == 0:
        required_hop_completion = 1.0
    else:
        required_hop_completion = min(
            successful_acquisitions / required_hops,
            1.0,
        )

    retrieval_success_rate = (
        successful_acquisitions / retrieval_attempts
        if retrieval_attempts
        else (
            1.0
            if required_hops == 0
            else 0.0
        )
    )

    citation_support_rate = (
        1.0
        if not citation_events
        else (
            1.0
            if citation_failures == 0
            else 0.0
        )
    )

    # ------------------------------------------------------------
    # Trajectory-level unsupported-claim rate
    #
    # Use historical grounding evaluations rather than final state.
    # A successful revision can clear `unsupported_atomic_claims`
    # from state, but the trajectory still experienced the failure.
    #
    # Use the worst observed rate rather than summing evaluations,
    # because the same claim may be re-evaluated after revision.
    # ------------------------------------------------------------
    grounding_claim_rates = []

    for event in grounding_events:
        details = _event_details(event)

        atomic_claim_count = int(
            details.get("atomic_claim_count", 0) or 0
        )

        unsupported_atomic_count = int(
            details.get("unsupported_atomic_count", 0) or 0
        )

        if atomic_claim_count > 0:
            grounding_claim_rates.append(
                min(
                    unsupported_atomic_count / atomic_claim_count,
                    1.0,
                )
            )

    unsupported_claim_rate = (
        max(grounding_claim_rates)
        if grounding_claim_rates
        else 0.0
    )
    return {
        "hop_success_rate": required_hop_completion,
        "required_hop_completion": required_hop_completion,
        "retrieval_success_rate": retrieval_success_rate,
        "evidence_acquisition_rate": (
            1.0
            if successful_acquisitions > 0
            else 0.0
        ),
        "web_fallback_rate": (
            1.0
            if web_fallbacks > 0
            else 0.0
        ),
        "citation_support_rate": citation_support_rate,
        "unsupported_claim_rate": unsupported_claim_rate,
        "revision_rate": (
            1.0
            if revision_requests > 0
            else 0.0
        ),
        "revision_success_rate": (
            1.0
            if revision_success
            else 0.0
            if revision_requests
            else 1.0
        ),
        "recovery_after_grounding_failure": (
            1.0
            if grounding_failures > 0 and revision_success
            else 0.0
            if grounding_failures > 0
            else 1.0
        ),
        "recovery_after_citation_failure": (
            1.0
            if citation_failures > 0 and revision_success
            else 0.0
            if citation_failures > 0
            else 1.0
        ),
    }


# ============================================================
# Main evaluator
# ============================================================

def evaluate_trajectory(
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Deterministically evaluate one completed KnowledgeMesh trajectory.

    The evaluator interprets raw trajectory events and final state.
    It does not mutate the graph state.
    """

    required_hops = int(
        state.get("required_hops")
        or infer_required_hops(state)
    )

    retrieval_attempts = count_retrieval_attempts(state)

    successful_acquisitions = count_successful_acquisitions(
        state
    )

    failures = infer_failure_taxonomy(state)

    recoveries = collect_recovery_events(state)

    grounding_value = state.get("is_grounded")

    if grounding_value is None:
        grounding_value = state.get("grounding_valid")

    if grounding_value is None:
        grounding_value = state.get("answer_supported")

    final_success = (
        bool(state.get("citation_valid", False))
        and bool(grounding_value)
        and not bool(state.get("generation_failed", False))
    )

    if final_success:
        trajectory_status = (
            "recovered"
            if failures
            else "success"
        )
    else:
        trajectory_status = "failed"

    metrics = calculate_metrics(
        state,
        required_hops=required_hops,
        successful_acquisitions=successful_acquisitions,
        retrieval_attempts=retrieval_attempts,
    )

    return {
        "trajectory_status": trajectory_status,
        "hop_count": successful_acquisitions,
        "required_hops": required_hops,
        "retrieval_attempts": retrieval_attempts,
        "failure_taxonomy": failures,
        "recovery_events": recoveries,
        "metrics": metrics,
        "evaluation_trace": _events(state),
    }

