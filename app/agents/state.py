"""
KnowledgeMesh · Agent State
===========================

Shared LangGraph state for the Agentic RAG pipeline.

The state intentionally separates:

    private retrieval
    web retrieval
    grading
    generation-approved evidence
    citation provenance
    grounding/evaluation
    bounded answer revision
    salvage (pruning) and abstention

The state is intentionally explicit because multiple LangGraph nodes
read/write the same fields during:

    planner
        ↓
    retrieval
        ↓
    grading
        ↓
    context evaluation
        ↓
    optional web fallback / query rewrite
        ↓
    generation
        ↓
    citation validation
        ↓
    grounding validation
        ↓
    answer relevance
        ↓
    bounded revision / salvage / abstention

IMPORTANT
---------

LangGraph silently DROPS any key a node returns that is not declared
here. No error is raised; the value simply never reaches state. Every
field a node writes must be declared in AgentState.
"""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class AgentState(TypedDict, total=False):
    # ============================================================
    # Request / query
    # ============================================================

    message: str

    original_query: str

    current_query: str

    search_query: str

    rewritten_query: str

    # ============================================================
    # Planner
    # ============================================================

    plan: List[Any]

    route: str

    retrieval_required: bool

    web_search_required: bool

    should_search_web: bool

    # ============================================================
    # Retrieval
    # ============================================================

    # Current working document collection used by the pipeline.
    documents: List[Any]

    # Private / enterprise retrieval evidence.
    private_documents: List[Any]

    # External web evidence.
    # External web evidence.
    web_documents: List[Any]

    # Combined evidence.
    all_documents: List[Any]

    # Evidence explicitly approved for answer generation.
    generation_documents: List[Any]

    # Legacy / compatibility field used by older retrieval nodes.
    retrieved_documents: List[Any]

    # ============================================================
    # Retrieval metadata
    # ============================================================

    retrieval_count: int

    retrieval_latency_ms: float

    retrieval_failed: bool

    retrieval_error: str

    # ============================================================
    # Retrieval control / retry metadata
    # ============================================================

    retrieval_rewrite_count: int

    max_retrieval_rewrites: int

    support_retry_count: int

    max_support_retries: int

    web_rewrite_count: int

    max_web_rewrites: int

    # ============================================================
    # Web search
    # ============================================================

    web_search_attempted: bool

    web_search_used: bool

    web_search_error: str

    web_search_query: str

    # ============================================================
    # Grading
    # ============================================================

    graded_documents: List[Any]

    grader_status: str

    grading_mode: str

    grader_error_type: str

    grader_error: str

    context_quality: str

    context_reason: str

    # ============================================================
    # Answer generation
    # ============================================================

    final_answer: str

    candidate_answer: str

    merged_context: str

    technical_context: str

    generation_failed: bool

    generation_rate_limited: bool

    generation_latency_ms: float

    generation_evidence_scope: str

    # ============================================================
    # Citation system
    # ============================================================

    citation_provenance: Dict[str, Any]

    citation_valid: bool

    citation_feedback: str

    citation_errors: List[str]

    citation_enforcement: Dict[str, Any]

    # Citation-consistency audit written by the grounding critic:
    # passed, violation_count/violations (hard), warning_count/warnings
    # (soft), results. The UI's "Citation consistency" panel reads this.
    citation_consistency: Dict[str, Any]

    # ============================================================
    # Grounding / answer evaluation
    #
    # These fields are intentionally explicit.
    #
    # The grounding critic produces structured output such as:
    #
    #     answer_supported
    #     support_score
    #     grounding_scores
    #     grounding_feedback
    #     unsupported_atomic_claims
    #     revision_prompt
    #
    # Keeping these fields in shared state prevents the critic's
    # results from being lost or represented only by a text message.
    # ============================================================

    # Final strict grounding result.
    is_grounded: bool

    # Alias used by grounding/evaluation code.
    grounding_valid: bool

    # Strict semantic support result.
    answer_supported: bool

    # Whether the answer is considered useful by the evaluator.
    answer_useful: bool

    # True when the answer-relevance EVALUATOR itself could not run
    # (gateway down / unparseable output), as opposed to the evaluator
    # judging the answer irrelevant. Routes to abstention without
    # spending a revision.
    relevance_evaluator_failed: bool

    # Overall support score.
    support_score: float

    # Overall usefulness score.
    usefulness_score: float

    # Backward-compatible singular grounding score.
    grounding_score: float

    # Structured grounding metrics.
    grounding_scores: Dict[str, Any]

    # HHEMv2 / grounding performance diagnostics.
    # These fields are observational only and do not affect validation.
    grounding_ms: float
    hhem_call_count: int
    hhem_total_ms: float

    # Request-scoped HHEMv2 / grounding diagnostics.
    #
    # hhem_call_count / hhem_total_ms above describe the current
    # grounding pass. These fields aggregate across every grounding
    # pass executed during this request, including revisions.
    request_hhem_call_count: int
    request_hhem_total_ms: float
    grounding_pass_count: int
    # Per-pass cache diagnostics; cache contents remain internal.
    entailment_cache_hits: int
    entailment_cache_misses: int
    cache_entries_received: int
    cache_entries_returned: int
    # JSON-compatible cache carried across grounding passes.
    entailment_cache: List[Dict[str, Any]]

    # Detailed per-claim / per-evidence grounding information.
    grounding_details: Dict[str, Any]

    # Extracted answer claims.
    claims: List[Any]

    # Human-readable grounding feedback.
    #
    # The current grounding critic returns a string. Older nodes may
    # have treated this as a list, so Any keeps compatibility while
    # allowing the current implementation to pass the string through.
    grounding_feedback: Any

    # Structured grounding errors.
    grounding_errors: List[str]

    # Claims that failed strict grounding validation.
    unsupported_atomic_claims: List[Any]

    # ============================================================
    # Revision control
    #
    # answer_revision_count
    #     Number of answer revisions already performed.
    #
    # revision_requested
    #     Whether the next responder call is specifically a
    #     revision rather than the initial generation.
    #
    # revision_prompt
    #     Explicit instructions generated by the grounding critic
    #     (or by prepare_revision_node for relevance failures).
    #     failures no longer use revisions; they trigger re-retrieval.
    # ============================================================

    answer_revision_count: int

    revision_requested: bool

    revision_prompt: str

    # ============================================================
    # Legacy revision compatibility
    #
    # Keep revision_count temporarily because older nodes may still
    # read/write it. The active graph should prefer
    # answer_revision_count.
    # ============================================================

    revision_count: int

    # Maximum allowed answer revisions.
    max_revisions: int

    # ============================================================
    # Salvage (pruning)
    #
    # When the revision budget is spent and grounding failed ONLY
    # because some lines are unsupported, the critic proposes the same
    # answer with just those lines deleted.
    #
    # pruned_answer / prunable_claims
    #     Written by the grounding critic on every failing pass: the
    #     proposed cut and the claims it would remove. Empty otherwise.
    #
    # answer_pruned / pruned_claims / answer_before_pruning
    #     Written ONLY by apply_pruned_answer_node: whether salvage was
    #     applied, which claims were actually removed (for UI
    #     disclosure), and the original answer for debugging.
    # ============================================================

    pruned_answer: str

    prunable_claims: List[str]

    answer_pruned: bool

    pruned_claims: List[str]

    answer_before_pruning: str

    # ============================================================
    # Abstention
    #
    # Written ONLY by answer_abstention_node. Consumers (API/UI,
    # trajectory evaluator) must check answer_abstained before
    # presenting or scoring the answer. The abstention node does not
    # overwrite the validity flags above; they describe the last
    # answer that was actually evaluated.
    # ============================================================

    answer_abstained: bool

    abstention_reason: str

    # ============================================================
    # Conversation
    # ============================================================

    messages: List[Any]

    # ============================================================
    # Status / observability
    # ============================================================

    status: str

    error: str

    # ============================================================
    # Checkpoint / thread metadata
    # ============================================================

    thread_id: str

    # ============================================================
    # Agent trajectory / evaluation
    #
    # Records meaningful agent execution events without treating
    # every LangGraph node execution as a reasoning hop.
    # ============================================================

    evaluation_trace: List[Dict[str, Any]]

    hop_count: int

    required_hops: int

    trajectory_status: str

    failure_taxonomy: List[str]
    trajectory_metrics: Dict[str, float]

    # Re-retrieval after an answer-relevance failure (graph.py).

    relevance_retrieval_count: int
    expand_retrieval: bool

    # Trajectory evaluation (split of failure_taxonomy).
    recovered_failures: List[str]
    terminal_failures: List[str]
