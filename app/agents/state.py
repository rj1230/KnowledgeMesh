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
    bounded revision
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

    # Overall support score.
    support_score: float

    # Overall usefulness score.
    usefulness_score: float

    # Backward-compatible singular grounding score.
    grounding_score: float

    # Structured grounding metrics.
    grounding_scores: Dict[str, Any]

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
    #     Explicit instructions generated by the grounding critic.
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
