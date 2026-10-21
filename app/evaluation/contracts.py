"""Shared allowlists for trajectory storage and candidate mining."""

FAILURE_CATEGORIES = {
    "PLANNING_FAILURE",
    "RETRIEVAL_FAILURE",
    "EVIDENCE_SELECTION_FAILURE",
    "CITATION_FAILURE",
    "GROUNDING_FAILURE",
    "GENERATION_FAILURE",
    "NO_APPROVED_EVIDENCE",
    "REVISION_FAILURE",
    "ANSWER_RELEVANCE_FAILURE",
}

SAFE_STEPS = {
    "planner",
    "private_retrieval",
    "query_rewriter",
    "grader",
    "grounding_critic",
    "citation_check",
    "responder",
    "revision",
    "answer_relevance",
    "answer_abstention",
    "context_evaluator",
    "web_search",
}

SAFE_STATUSES = {
    "blocked",
    "requested",
    "failed",
    "passed",
    "abstained",
    "skipped",
    "success",
    "empty",
    "rewritten",
    "retry_same_query",
    "empty_input",
    "conversational",
    "technical",
    "fallback",
    "generated",
    "revised_generated",
    "recovered",
    "empty_query",
    "good",
    "weak",
    "unverified",
    "unavailable",
    "unknown",
}

SAFE_METRIC_NAMES = {
    # Evidence acquisition
    "hop_success_rate",
    "required_hop_completion",
    "retrieval_success_rate",
    "evidence_acquisition_rate",
    # Fallback and evidence quality
    "web_fallback_rate",
    "citation_support_rate",
    "unsupported_claim_rate",
    # Revision and recovery
    "revision_rate",
    "revision_success_rate",
    "recovery_after_grounding_failure",
    "recovery_after_citation_failure",
}
