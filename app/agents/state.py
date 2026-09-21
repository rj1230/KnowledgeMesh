from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict


class RetrievedDocument(TypedDict, total=False):
    """
    Canonical document contract shared by retrieval, reranking,
    grading, web search, generation, citation validation, and
    grounding evaluation.
    """

    # CORE DOCUMENT IDENTITY
    id: str
    document_id: str
    chunk_id: int
    total_chunks: int
    content: str
    source: str
    source_type: str

    # RETRIEVAL / RANKING
    score: Optional[float]
    rerank_score: Optional[float]
    grader_score: Optional[float]

    # DOCUMENT GRADING
    grader_relevant: bool
    grader_reason: str

    # SOURCE / WEB METADATA
    url: Optional[str]

    # EVIDENCE QUALITY
    evidence_type: str
    evidence_score: float
    reference_dominated: bool
    evidence_reason: str
    evidence_metrics: dict


class CitationProvenance(TypedDict, total=False):
    """
    Canonical provenance mapping for an LLM-facing citation.
    """

    citation_id: str
    point_id: Optional[str]
    document_id: Optional[str]
    chunk_id: Optional[int]
    total_chunks: Optional[int]
    source: str
    source_type: str


class AgentState(TypedDict, total=False):
    # ============================================================
    # CONVERSATION
    # ============================================================

    messages: Annotated[List[dict], operator.add]

    # ============================================================
    # QUERY / INTENT
    # ============================================================

    original_query: str
    current_query: str
    rewritten_query: str
    search_query: str
    route: str
    intent: str
    query_type: str
    conversational: bool

    # ============================================================
    # PRIVATE RETRIEVAL
    # ============================================================

    documents: List[RetrievedDocument]
    graded_documents: List[RetrievedDocument]

    # ============================================================
    # DOCUMENT GRADING
    # ============================================================

    # Grading lifecycle:
    #
    # success -> documents were successfully graded
    # failed  -> grader/provider failed; evidence is unverified
    # skipped -> nothing was available to grade
    grader_status: str

    # Runtime/provider exception type when grading fails.
    # Example: "RateLimitError"
    grader_error_type: Optional[str]

    # Downstream interpretation of the evidence.
    #
    # graded       -> relevance/evidence quality was evaluated
    # unavailable  -> grader/provider unavailable
    # no_documents -> nothing available to grade
    grading_mode: str

    retrieval_required: bool

    retrieval_latency_ms: float
    rerank_latency_ms: float
    grader_latency_ms: float

    # ============================================================
    # CONTEXT EVALUATION
    # ============================================================

    context_quality: str
    context_reason: str
    should_search_web: bool

    # ============================================================
    # WEB SEARCH
    # ============================================================

    web_documents: List[RetrievedDocument]
    web_search_required: bool
    web_search_used: bool

    # Complete historical evidence retained for audit/debugging.
    all_documents: List[RetrievedDocument]

    # Exact documents used by the current responder generation.
    generation_documents: List[RetrievedDocument]

    # ============================================================
    # GENERATION CONTEXT
    # ============================================================

    merged_context: str
    technical_context: str

    # Candidate before final citation/grounding processing.
    candidate_answer: str

    # ============================================================
    # CITATION PROVENANCE
    # ============================================================

    citation_provenance: Dict[str, CitationProvenance]

    citation_enforcement: dict

    # ============================================================
    # ANSWER REFLECTION / SELF-RAG
    # ============================================================

    claims: List[dict]

    # IMPORTANT:
    # grounding_scores is a dictionary/object, not a list.
    #
    # Example:
    # {
    #     "atomic_claims": [...],
    #     "claim_support_score": 0.82,
    #     "citation_valid": True,
    #     ...
    # }
    grounding_scores: Dict[str, Any]

    grounding_feedback: List[str]

    # Claims specifically identified by the grounding critic
    # as unsupported and passed to the revision node.
    unsupported_atomic_claims: List[Dict[str, Any]]

    # Exact evidence-constrained revision instructions.
    revision_prompt: str

    is_grounded: bool
    citation_valid: bool

    # This MUST remain boolean.
    answer_supported: bool

    answer_useful: bool

    support_score: float
    usefulness_score: float

    # ============================================================
    # GENERATION
    # ============================================================

    final_answer: str
    generation_latency_ms: float

    # ============================================================
    # SELF-RAG RETRIES / REVISION
    # ============================================================

    retrieval_rewrite_count: int
    web_rewrite_count: int
    support_retry_count: int

    revision_count: int
    max_revisions: int

    # ============================================================
    # EXECUTION TRACE
    # ============================================================

    plan: List[str]
    status: str

    # Optional execution flags used by nodes.
    generation_failed: bool
    simple_response: bool
