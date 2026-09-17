from __future__ import annotations

import operator
from typing import Annotated, List, Optional, TypedDict


class RetrievedDocument(TypedDict, total=False):
    """
    Canonical document contract shared by:

    - Qdrant retrieval
    - FlashRank reranking
    - document grading
    - evidence-quality filtering
    - Tavily web search
    - responder
    - grounding/citation evaluation

    Evidence-quality fields distinguish:
        CONTENT    -> explanatory/evidentiary material
        REFERENCE  -> bibliography/reference-dominated material
        NAVIGATION -> structural/navigation noise

    This is intentionally metadata-only. The original document content
    and retrieval scores remain available.
    """

    # ========================================================
    # CORE DOCUMENT IDENTITY
    # ========================================================

    id: str
    content: str
    source: str
    source_type: str

    # ========================================================
    # RETRIEVAL / RANKING
    # ========================================================

    score: Optional[float]
    rerank_score: Optional[float]
    grader_score: Optional[float]

    # ========================================================
    # DOCUMENT GRADING
    # ========================================================

    grader_relevant: bool
    grader_reason: str

    # ========================================================
    # SOURCE / WEB METADATA
    # ========================================================

    url: Optional[str]

    # ========================================================
    # EVIDENCE QUALITY
    # ========================================================
    #
    # These fields are added after retrieval/grading.
    #
    # IMPORTANT:
    # citation-heavy academic content is still allowed to be CONTENT.
    # A chunk becomes REFERENCE only when it is dominated by
    # bibliographic/reference signals such as CoRR, Proceedings,
    # DOI/arXiv metadata, page numbers, etc.
    # ========================================================

    evidence_type: str
    evidence_score: float

    reference_dominated: bool

    evidence_reason: str

    evidence_metrics: dict


class AgentState(TypedDict, total=False):
    # ========================================================
    # CONVERSATION
    # ========================================================

    messages: Annotated[List[dict], operator.add]

    # ========================================================
    # QUERY / INTENT
    # ========================================================

    original_query: str
    current_query: str
    rewritten_query: str
    search_query: str
    route: str

    # ========================================================
    # PRIVATE RETRIEVAL
    # ========================================================

    documents: List[RetrievedDocument]
    graded_documents: List[RetrievedDocument]

    retrieval_required: bool

    retrieval_latency_ms: float
    rerank_latency_ms: float
    grader_latency_ms: float

    # ========================================================
    # CONTEXT EVALUATION
    # ========================================================

    context_quality: str
    context_reason: str
    should_search_web: bool

    # ========================================================
    # WEB SEARCH
    # ========================================================

    web_documents: List[RetrievedDocument]
    web_search_required: bool
    web_search_used: bool

    all_documents: List[RetrievedDocument]

    # ========================================================
    # GENERATION CONTEXT
    # ========================================================

    merged_context: str

    # ========================================================
    # ANSWER REFLECTION
    # ========================================================

    claims: List[dict]

    grounding_scores: List[dict]

    grounding_feedback: List[str]

    is_grounded: bool
    citation_valid: bool

    answer_supported: bool
    answer_useful: bool

    support_score: float
    usefulness_score: float

    # ========================================================
    # GENERATION
    # ========================================================

    final_answer: str
    generation_latency_ms: float

    # ========================================================
    # SELF-RAG RETRIES
    # ========================================================

    retrieval_rewrite_count: int
    web_rewrite_count: int

    support_retry_count: int

    revision_count: int
    max_revisions: int

    # ========================================================
    # EXECUTION TRACE
    # ========================================================

    plan: List[str]
    status: str
