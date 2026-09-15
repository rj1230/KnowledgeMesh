from __future__ import annotations

import operator

from typing import Annotated, List, Optional, TypedDict


class RetrievedDocument(TypedDict, total=False):
    id: str
    content: str
    source: str
    source_type: str

    score: float
    rerank_score: Optional[float]

    grader_score: Optional[float]
    grader_relevant: bool
    grader_reason: str

    url: Optional[str]


class AgentState(TypedDict, total=False):
    # --------------------------------------------------------
    # Conversation
    # --------------------------------------------------------

    messages: Annotated[List[dict], operator.add]

    original_query: str
    current_query: str
    rewritten_query: str

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    documents: List[RetrievedDocument]
    web_documents: List[RetrievedDocument]

    search_query: str

    retrieval_required: bool
    web_search_required: bool
    web_search_used: bool

    # --------------------------------------------------------
    # Context evaluation
    # --------------------------------------------------------

    context_quality: str

    # --------------------------------------------------------
    # Answer evaluation
    # --------------------------------------------------------

    answer_supported: bool
    answer_useful: bool

    support_score: float
    usefulness_score: float

    # --------------------------------------------------------
    # Self-RAG control
    # --------------------------------------------------------

    retrieval_rewrite_count: int
    web_rewrite_count: int
    support_retry_count: int

    revision_count: int
    max_revisions: int

    # --------------------------------------------------------
    # Execution trace
    # --------------------------------------------------------

    plan: List[str]

    status: str

    # --------------------------------------------------------
    # Final answer
    # --------------------------------------------------------

    final_answer: str
