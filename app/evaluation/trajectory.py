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
NO_APPROVED_EVIDENCE = "NO_APPROVED_EVIDENCE"
REVISION_FAILURE = "REVISION_FAILURE"
ANSWER_RELEVANCE_FAILURE = "ANSWER_RELEVANCE_FAILURE"

# Infrastructure failure of an evaluator (e.g. the relevance judge's LLM
# gateway was down). Distinct from the answer itself being irrelevant.
EVALUATOR_FAILURE = "EVALUATOR_FAILURE"

# answer_relevance trace reasons that mean "the evaluator could not run",
# not "the answer was judged irrelevant".
_EVALUATOR_FAILURE_REASONS = {
    "llm_gateway_failure",
    "evaluator_failure",
}

# Failures that describe the ANSWER stages (generate -> citation ->
# relevance -> grounding). Whether one of these is terminal is decided
# from the LAST event of its stage; upstream failures (planning,
# retrieval, evidence selection) are terminal whenever the trajectory
# did not end in a valid answer.
_ANSWER_STAGE_FAILURES = {
    CITATION_FAILURE,
    GROUNDING_FAILURE,
    ANSWER_RELEVANCE_FAILURE,
    EVALUATOR_FAILURE,
    GENERATION_FAILURE,
    REVISION_FAILURE,
}

# answer_abstention reasons -> failure category.
_ABSTENTION_REASON_CATEGORIES = {
    "citation_validation_revision_budget_exhausted": CITATION_FAILURE,
    "grounding_validation_revision_budget_exhausted": GROUNDING_FAILURE,
    "answer_relevance_revision_budget_exhausted": ANSWER_RELEVANCE_FAILURE,
    "answer_relevance_evaluator_unavailable": EVALUATOR_FAILURE,
    "generation_failed_during_revision": GENERATION_FAILURE,
    "named_document_has_no_evidence": NO_APPROVED_EVIDENCE,
}


# ============================================================
# Recovery taxonomy
# ============================================================

QUERY_REWRITE = "QUERY_REWRITE"
PRIVATE_RETRY = "PRIVATE_RETRY"
WEB_ESCALATION = "WEB_ESCALATION"
ANSWER_REVISION = "ANSWER_REVISION"
ANSWER_PRUNED = "ANSWER_PRUNED"

# A relevance failure sent back to retrieval (prepare_reretrieval_node).
RE_RETRIEVAL = "RE_RETRIEVAL"


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
        event for event in _events(state) if str(event.get("step", "")).strip() == step
    ]


def _last_event(
    state: Dict[str, Any],
    step: str,
) -> Dict[str, Any] | None:
    events = _events_by_step(state, step)

    return events[-1] if events else None


def _event_details(event: Dict[str, Any]) -> Dict[str, Any]:
    details = event.get("details")
    return details if isinstance(details, dict) else {}


def _event_status(event: Dict[str, Any]) -> str:
    return str(event.get("status", "")).strip().lower()


def _event_reason(event: Dict[str, Any]) -> str:
    return str(_event_details(event).get("reason") or event.get("reason") or "").strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or default)
    except (TypeError, ValueError):
        return default


def _final_success(state: Dict[str, Any]) -> bool:
    """
    Single definition of "this trajectory ended in a valid answer".

    Shared by trajectory_status and the metrics so they can never disagree.

    An abstention is never a success, even if stale validity flags from an
    earlier pass happen to still be True in state.
    """

    if bool(state.get("answer_abstained", False)):
        return False

    grounding_value = state.get("is_grounded")

    if grounding_value is None:
        grounding_value = state.get("grounding_valid")

    if grounding_value is None:
        grounding_value = state.get("answer_supported")

    return (
        bool(state.get("citation_valid", False))
        and bool(grounding_value)
        and bool(state.get("answer_useful", False))
        and not bool(state.get("generation_failed", False))
    )


def _classify_grounding_event(event: Dict[str, Any]) -> List[str]:
    """
    Classify ONE failed grounding_critic event.

    The critic fails a pass for two different reasons:

      * unsupported atomic claims   -> GROUNDING_FAILURE
      * citation problems (uncited, invalid, or citations that do not
        support their claim)        -> CITATION_FAILURE

    Events written before the citation counts existed carry neither
    signal, so they fall back to GROUNDING_FAILURE.
    """

    details = _event_details(event)

    unsupported = _safe_int(details.get("unsupported_atomic_count"))
    citation_issues = (
        _safe_int(details.get("citation_violation_count"))
        + _safe_int(details.get("uncited_claim_count"))
        + _safe_int(details.get("invalid_citation_count"))
    )

    categories: List[str] = []

    if unsupported > 0:
        categories.append(GROUNDING_FAILURE)

    if citation_issues > 0:
        categories.append(CITATION_FAILURE)

    if not categories:
        categories.append(GROUNDING_FAILURE)

    return categories


def _classify_relevance_event(event: Dict[str, Any]) -> str:
    """An evaluator outage is not an irrelevant answer."""

    if _event_reason(event) in _EVALUATOR_FAILURE_REASONS:
        return EVALUATOR_FAILURE

    return ANSWER_RELEVANCE_FAILURE


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
        if _event_status(event) == "success"
        and int(_event_details(event).get("selected_count", 0) or 0) > 0
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
        status = _event_status(event)

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

        elif step == "re_retrieval" and status == "requested":
            recoveries.append(RE_RETRIEVAL)

        elif step == "answer_pruned" and status == "applied":
            recoveries.append(ANSWER_PRUNED)

    return recoveries


# ============================================================
# Failure taxonomy
# ============================================================


def infer_failure_taxonomy(
    state: Dict[str, Any],
) -> List[str]:
    """
    Every failure the trajectory EXPERIENCED, including ones that a
    revision, re-retrieval or salvage later recovered from.

    Use infer_terminal_failures() for what the user actually got.
    """

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
    elif any(_event_status(event) == "empty" for event in retrieval_events):
        failures.append(RETRIEVAL_FAILURE)

    # --------------------------------------------------------
    # Evidence selection / context
    # --------------------------------------------------------

    context_quality = str(state.get("context_quality") or "").strip().lower()

    if context_quality in {
        "weak",
        "insufficient",
        "needs_web",
        "unverified",
    }:
        if not bool(state.get("web_search_used", False)):
            failures.append(EVIDENCE_SELECTION_FAILURE)

    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------

    responder_events = _events_by_step(state, "responder")
    no_approved_evidence = any(
        _event_status(event) == "blocked"
        and _event_details(event).get("reason") == "no_approved_evidence"
        for event in responder_events
    )

    if no_approved_evidence:
        failures.append(NO_APPROVED_EVIDENCE)

    web_search_events = _events_by_step(state, "web_search")
    web_search_failed = any(
        _event_status(event) == "failed" for event in web_search_events
    )

    # Preserve the existing failure classification when evidence-gated
    # abstention follows a failed web-search recovery attempt.
    if bool(state.get("generation_failed", False)) and (
        not no_approved_evidence or web_search_failed
    ):
        failures.append(GENERATION_FAILURE)

    # --------------------------------------------------------
    # Grounding critic events
    # --------------------------------------------------------

    grounding_events = _events_by_step(
        state,
        "grounding_critic",
    )

    grounding_failed_events = [
        event for event in grounding_events if _event_status(event) == "failed"
    ]

    for event in grounding_failed_events:
        failures.extend(_classify_grounding_event(event))

    # --------------------------------------------------------
    # Final abstention classification
    # --------------------------------------------------------

    for event in _events_by_step(state, "answer_abstention"):
        category = _ABSTENTION_REASON_CATEGORIES.get(_event_reason(event))

        if not category:
            continue

        # The grounding events above already classified the real cause;
        # do not re-add a generic GROUNDING_FAILURE from the abstention
        # reason alone.
        if category == GROUNDING_FAILURE and grounding_failed_events:
            continue

        failures.append(category)

    # --------------------------------------------------------
    # Citation
    # --------------------------------------------------------

    citation_events = _events_by_step(
        state,
        "citation_check",
    )

    if any(_event_status(event) == "failed" for event in citation_events):
        failures.append(CITATION_FAILURE)

    # --------------------------------------------------------
    # Answer relevance
    # --------------------------------------------------------

    for event in _events_by_step(state, "answer_relevance"):
        if _event_status(event) != "failed":
            continue

        failures.append(_classify_relevance_event(event))

    # --------------------------------------------------------
    # Revision
    # --------------------------------------------------------

    revision_events = _events_by_step(
        state,
        "revision",
    )

    if any(_event_status(event) == "blocked" for event in revision_events):
        failures.append(REVISION_FAILURE)

    return list(dict.fromkeys(failures))


def infer_terminal_failures(
    state: Dict[str, Any],
    failures: List[str],
) -> List[str]:
    """
    The failures that still hold for the FINAL outcome.

    A trajectory that ends in a valid answer has none: everything it
    experienced was recovered. Otherwise:

      * upstream failures (planning, retrieval, evidence selection, no
        approved evidence) are terminal as-is, and
      * answer-stage failures are terminal only if the LAST event of
        that stage failed, so a grounding failure fixed by a revision
        or salvage is reported as recovered, not terminal.
    """

    if _final_success(state):
        return []

    terminal: List[str] = [
        failure for failure in failures if failure not in _ANSWER_STAGE_FAILURES
    ]

    if bool(state.get("generation_failed", False)):
        terminal.append(GENERATION_FAILURE)

    last_citation = _last_event(state, "citation_check")

    if last_citation and _event_status(last_citation) == "failed":
        terminal.append(CITATION_FAILURE)

    last_grounding = _last_event(state, "grounding_critic")

    if last_grounding and _event_status(last_grounding) == "failed":
        terminal.extend(_classify_grounding_event(last_grounding))

    last_relevance = _last_event(state, "answer_relevance")

    if last_relevance and _event_status(last_relevance) == "failed":
        terminal.append(_classify_relevance_event(last_relevance))

    for event in _events_by_step(state, "answer_abstention"):
        category = _ABSTENTION_REASON_CATEGORIES.get(_event_reason(event))

        if category:
            terminal.append(category)

    if any(
        _event_status(event) == "blocked"
        for event in _events_by_step(state, "revision")
    ):
        terminal.append(REVISION_FAILURE)

    return list(dict.fromkeys(terminal))


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

    web_fallbacks = sum(1 for event in web_events if _event_details(event).get("used"))

    citation_events = _events_by_step(
        state,
        "citation_check",
    )

    citation_failures = sum(
        1 for event in citation_events if _event_status(event) == "failed"
    )

    grounding_events = _events_by_step(
        state,
        "grounding_critic",
    )

    grounding_failures = sum(
        1 for event in grounding_events if _event_status(event) == "failed"
    )

    revision_events = _events_by_step(
        state,
        "revision",
    )

    revision_requests = sum(
        1 for event in revision_events if _event_status(event) == "requested"
    )

    re_retrieval_requests = sum(
        1
        for event in _events_by_step(state, "re_retrieval")
        if _event_status(event) == "requested"
    )

    final_valid = _final_success(state)

    revision_success = revision_requests > 0 and final_valid

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
        else (1.0 if required_hops == 0 else 0.0)
    )

    citation_support_rate = (
        1.0 if not citation_events else (1.0 if citation_failures == 0 else 0.0)
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

        atomic_claim_count = details.get("atomic_claim_count")
        unsupported_atomic_count = details.get("unsupported_atomic_count")

        # Ignore malformed diagnostics instead of crashing or treating
        # invalid values as evidence of zero unsupported claims.
        valid_atomic_count = (
            isinstance(atomic_claim_count, int)
            and not isinstance(atomic_claim_count, bool)
            and atomic_claim_count > 0
        )
        valid_unsupported_count = (
            isinstance(unsupported_atomic_count, int)
            and not isinstance(unsupported_atomic_count, bool)
            and unsupported_atomic_count >= 0
        )

        if valid_atomic_count and valid_unsupported_count:
            grounding_claim_rates.append(
                min(unsupported_atomic_count / atomic_claim_count, 1.0)
            )

    unsupported_claim_rate = (
        max(grounding_claim_rates) if grounding_claim_rates else 0.0
    )
    unsupported_claim_rate_available = bool(grounding_claim_rates)

    return {
        "hop_success_rate": required_hop_completion,
        "required_hop_completion": required_hop_completion,
        "retrieval_success_rate": retrieval_success_rate,
        "evidence_acquisition_rate": (1.0 if successful_acquisitions > 0 else 0.0),
        "web_fallback_rate": (1.0 if web_fallbacks > 0 else 0.0),
        "citation_support_rate": citation_support_rate,
        "unsupported_claim_rate": unsupported_claim_rate,
        "unsupported_claim_rate_available": unsupported_claim_rate_available,
        "revision_rate": (1.0 if revision_requests > 0 else 0.0),
        "revision_success_rate": (
            1.0 if revision_success else 0.0 if revision_requests else 1.0
        ),
        # 1.0 when a relevance failure triggered a broader retrieval.
        "re_retrieval_rate": (1.0 if re_retrieval_requests > 0 else 0.0),
        "recovery_after_grounding_failure": (
            1.0
            if grounding_failures > 0 and final_valid
            else 0.0
            if grounding_failures > 0
            else 1.0
        ),
        "recovery_after_citation_failure": (
            1.0
            if citation_failures > 0 and final_valid
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

    failure_taxonomy   every failure the run experienced (unchanged).
    recovered_failures experienced but fixed by a revision, re-retrieval
                       or salvage; the run still ended in a valid answer.
    terminal_failures  what the user actually got; empty on success.
    """

    required_hops = int(state.get("required_hops") or infer_required_hops(state))

    retrieval_attempts = count_retrieval_attempts(state)

    successful_acquisitions = count_successful_acquisitions(state)

    failures = infer_failure_taxonomy(state)

    terminal_failures = infer_terminal_failures(state, failures)

    recovered_failures = [
        failure for failure in failures if failure not in terminal_failures
    ]

    recoveries = collect_recovery_events(state)

    final_success = _final_success(state)

    if final_success:
        trajectory_status = "recovered" if failures else "success"
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
        "recovered_failures": recovered_failures,
        "terminal_failures": terminal_failures,
        "recovery_events": recoveries,
        "metrics": metrics,
        "abstained": bool(state.get("answer_abstained", False)),
        "pruned": bool(state.get("answer_pruned", False)),
        "pruned_claims": list(state.get("pruned_claims") or []),
        "abstention_reason": state.get("abstention_reason"),
        "evaluation_trace": _events(state),
    }
