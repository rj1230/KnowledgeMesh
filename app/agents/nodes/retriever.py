from __future__ import annotations

import re
import time

import logfire

from app.agents.state import AgentState
from app.services.retrieval.qdrant_service import (
    search_enterprise_knowledge,
)
from app.services.retrieval.ranking_service import (
    rerank_documents,
)


# ============================================================
# RETRIEVAL ARCHITECTURE
# ============================================================
#
# Dense candidate retrieval:
#     Qdrant + BGE-small-en-v1.5 → 50 candidates
#
# Semantic reranking:
#     FlashRank → scores all 50 candidates
#
# Evidence selection:
#     Strong semantic pool
#           +
#     small dense-recovery pool
#           +
#     query coverage
#           +
#     evidence diversity
#           ↓
#     final 5
#
# Generation context:
#     Exactly 5 documents maximum.
#
# Important:
#
#     RERANK_CANDIDATE_K controls how many candidates receive
#     semantic cross-encoder scores.
#
#     FINAL_CONTEXT_K controls how many documents are exposed
#     to downstream grading / generation.
#
#     DENSE_RECOVERY_K and DENSE_RECOVERY_MIN_SCORE provide a
#     controlled recovery path for dense-retrieved evidence that
#     FlashRank strongly demotes.
# ============================================================

DENSE_CANDIDATE_K = 50
RERANK_CANDIDATE_K = 50
FINAL_CONTEXT_K = 5

# Dense recovery is intentionally small and conservative.
DENSE_RECOVERY_K = 20
DENSE_RECOVERY_MIN_SCORE = 0.68
DENSE_RECOVERY_MIN_COVERAGE = 0.20

# Normal FlashRank candidates must clear this threshold.
MIN_SEMANTIC_SCORE = 0.45


# ============================================================
# TOKENIZATION
# ============================================================

_TOKEN_PATTERN = re.compile(r"\b[a-zA-Z0-9][a-zA-Z0-9_-]*\b")


_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "for",
    "in",
    "on",
    "with",
    "by",
    "from",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "how",
    "does",
    "do",
    "did",
    "what",
    "why",
    "which",
    "who",
    "when",
    "where",
    "this",
    "that",
    "these",
    "those",
    "as",
    "at",
    "it",
    "its",
    "can",
    "could",
    "would",
    "should",
    "into",
    "than",
    "then",
    "through",
    "about",
    "over",
    "under",
    "using",
}


def _tokenize(text: str) -> set[str]:
    """
    Lightweight lexical representation used only as a
    diversity/supporting signal.

    Semantic relevance comes from FlashRank.
    """

    return {
        token.lower()
        for token in _TOKEN_PATTERN.findall(str(text))
        if len(token) >= 3 and token.lower() not in _STOPWORDS
    }


# ============================================================
# DOCUMENT IDENTITY
# ============================================================


def _document_identity(
    document: dict,
) -> tuple[str, str]:
    """
    Return the canonical corpus identity.

    chunk_id alone is NOT globally unique.

    Canonical identity:

        (document_id, chunk_id)
    """

    document_id = str(
        document.get(
            "document_id",
            "",
        )
    ).strip()

    chunk_id = document.get("chunk_id")

    if chunk_id is None:
        normalized_chunk_id = ""
    else:
        normalized_chunk_id = str(chunk_id).strip()

    return (
        document_id,
        normalized_chunk_id,
    )


# ============================================================
# DENSE CANDIDATE HELPERS
# ============================================================


def _find_dense_candidates_outside_selected(
    dense_documents: list[dict],
    selected_documents: list[dict],
) -> list[dict]:
    """
    Return dense candidates that were not selected for the
    final evidence context.

    Identity is based on:

        (document_id, chunk_id)

    rather than chunk_id alone.
    """

    selected_ids = {_document_identity(document) for document in selected_documents}

    return [
        document
        for document in dense_documents
        if _document_identity(document) not in selected_ids
    ]


# ============================================================
# TEXT / CONTENT HELPERS
# ============================================================


def _content(document: dict) -> str:
    return str(
        document.get(
            "content",
            "",
        )
    ).strip()


def _dense_score(document: dict) -> float:
    try:
        return float(
            document.get(
                "score",
                0.0,
            )
            or 0.0
        )
    except (TypeError, ValueError):
        return 0.0


def _rerank_score(document: dict) -> float:
    try:
        value = document.get("rerank_score")

        if value is None:
            return 0.0

        return float(value)

    except (TypeError, ValueError):
        return 0.0


def _jaccard_similarity(
    left: set[str],
    right: set[str],
) -> float:
    """
    Lightweight lexical redundancy estimate.

    This is NOT used as the primary relevance signal.
    """

    if not left or not right:
        return 0.0

    union = left | right

    if not union:
        return 0.0

    return len(left & right) / len(union)


def _max_redundancy(
    candidate_terms: set[str],
    selected_terms: list[set[str]],
) -> float:
    if not candidate_terms or not selected_terms:
        return 0.0

    return max(
        (
            _jaccard_similarity(
                candidate_terms,
                selected_document_terms,
            )
            for selected_document_terms in selected_terms
        ),
        default=0.0,
    )


def _query_coverage(
    query_terms: set[str],
    content_terms: set[str],
) -> float:
    """
    Fraction of meaningful query terms represented in the
    candidate.

    This is a supporting feature only. FlashRank remains the
    primary semantic relevance signal for normal candidates.
    """

    if not query_terms:
        return 0.0

    return len(query_terms & content_terms) / len(query_terms)


# ============================================================
# COMPLEMENTARY EVIDENCE SELECTION
# ============================================================


def _marginal_query_coverage(
    query_terms: set[str],
    candidate_terms: set[str],
    selected_terms: list[set[str]],
) -> float:
    """
    Measure the fraction of query terms contributed by a
    candidate that are not already covered by selected evidence.

    This is a supporting signal for dense recovery. It does not
    replace semantic relevance or dense retrieval relevance.
    """

    if not query_terms:
        return 0.0

    selected_union: set[str] = set()

    for terms in selected_terms:
        selected_union.update(terms)

    new_terms = query_terms & candidate_terms - selected_union

    return len(new_terms) / len(query_terms)


def _select_complementary_evidence(
    query: str,
    dense_documents: list[dict],
    reranked_documents: list[dict],
    final_k: int,
) -> list[dict]:
    """
    Select the final evidence set using a two-pool strategy.

    Retrieval architecture:

        Qdrant Dense Top-50
                |
                +-----------------------------+
                |                             |
                v                             v
        FlashRank semantic pool       Dense recovery pool
        (all 50 scored)               (top-N dense candidates)
                |                             |
                +-------------+---------------+
                              |
                              v
                    Complementary selector
                              |
                              v
                         Final Top-K

    FlashRank remains the primary relevance signal.

    The dense-recovery pool exists for an important failure mode:
    a document can be strongly retrieved by the dense encoder but
    receive an unexpectedly low FlashRank score because the query
    requires mechanism-level or terminology-specific evidence.

    Dense recovery does NOT automatically insert documents.
    Recovery candidates must still demonstrate:

        - sufficient dense retrieval support
        - meaningful query coverage
        - useful evidence diversity

    The selector does not use corpus-specific IDs, filenames,
    golden-test knowledge, or hard-coded document rules.
    """

    if final_k <= 0:
        return []

    if not reranked_documents and not dense_documents:
        return []

    # ========================================================
    # SEMANTIC CANDIDATE POOL
    # ========================================================

    semantic_candidates = [
        dict(document)
        for document in reranked_documents
        if _rerank_score(document) >= MIN_SEMANTIC_SCORE
    ]

    # ========================================================
    # DENSE RECOVERY POOL
    #
    # These are dense candidates which are NOT already present
    # in the semantic pool.
    #
    # This creates the recovery path missing from the previous
    # implementation.
    # ========================================================

    # FlashRank has already scored the full dense pool.
    # Use the scored documents so recovery sees the real rerank_score.
    dense_candidates = sorted(
        reranked_documents,
        key=lambda document: _dense_score(document),
        reverse=True,
    )

    dense_recovery_candidates: list[dict] = []

    for document in dense_candidates:
        dense_score = _dense_score(document)

        # Dense candidates are sorted descending, therefore once
        # this threshold is crossed all remaining candidates are
        # below the threshold.
        if dense_score < DENSE_RECOVERY_MIN_SCORE:
            break

        # FlashRank scores the full dense pool. Therefore a document
        # can already exist in semantic_candidates while still being
        # eligible for dense recovery if FlashRank considers it weak.
        rerank_score = _rerank_score(document)

        if rerank_score >= MIN_SEMANTIC_SCORE:
            continue

        coverage = _query_coverage(
            query_terms=_tokenize(query),
            content_terms=_tokenize(_content(document)),
        )

        if coverage < DENSE_RECOVERY_MIN_COVERAGE:
            continue

        recovery_document = dict(document)

        # Internal metadata only. Removed before returning.
        recovery_document["_dense_recovery"] = True

        dense_recovery_candidates.append(recovery_document)

        print(
            "\nDENSE RECOVERY CANDIDATE | "
            f"chunk={document.get('chunk_id')} "
            f"document_id={document.get('document_id')} "
            f"dense={dense_score:.6f} "
            f"flashrank={rerank_score:.6f} "
            f"coverage={coverage:.6f}"
        )

        if len(dense_recovery_candidates) >= DENSE_RECOVERY_K:
            break
    # ========================================================
    # MERGE SEMANTIC + RECOVERY POOLS
    # ========================================================

    candidates: list[dict] = []
    seen_ids: set[tuple[str, str]] = set()

    for document in semantic_candidates:
        identity = _document_identity(document)

        if identity in seen_ids:
            continue

        seen_ids.add(identity)
        candidates.append(dict(document))

    for document in dense_recovery_candidates:
        identity = _document_identity(document)

        if identity in seen_ids:
            continue

        seen_ids.add(identity)
        candidates.append(dict(document))

    if not candidates:
        return []

    query_terms = _tokenize(query)

    # ========================================================
    # INITIAL SELECTION
    #
    # Always start with the strongest actual FlashRank result.
    #
    # A dense-recovery candidate should not displace the best
    # semantic candidate before complementary selection begins.
    # ========================================================

    candidates.sort(
        key=lambda document: (
            _rerank_score(document),
            _dense_score(document),
        ),
        reverse=True,
    )

    selected: list[dict] = [dict(candidates[0])]

    selected_ids = {_document_identity(selected[0])}

    # ========================================================
    # COMPLEMENTARY SELECTION
    # ========================================================

    while len(selected) < min(
        final_k,
        len(candidates),
    ):
        selected_term_sets = [_tokenize(_content(document)) for document in selected]

        remaining = [
            document
            for document in candidates
            if _document_identity(document) not in selected_ids
        ]

        if not remaining:
            break

        scored_candidates = []

        for document in remaining:
            content_terms = _tokenize(_content(document))

            semantic_score = _rerank_score(document)

            dense_score = _dense_score(document)

            coverage = _query_coverage(
                query_terms=query_terms,
                content_terms=content_terms,
            )

            marginal_coverage = _marginal_query_coverage(
                query_terms=query_terms,
                candidate_terms=content_terms,
                selected_terms=selected_term_sets,
            )

            redundancy = _max_redundancy(
                candidate_terms=content_terms,
                selected_terms=selected_term_sets,
            )

            is_dense_recovery = bool(
                document.get(
                    "_dense_recovery",
                    False,
                )
            )

            # =================================================
            # NORMAL SEMANTIC CANDIDATES
            #
            # FlashRank remains dominant.
            # =================================================

            if not is_dense_recovery:
                selection_score = (
                    (semantic_score * 0.70)
                    + (coverage * 0.15)
                    + (dense_score * 0.10)
                    + ((1.0 - redundancy) * 0.05)
                )

            # =================================================
            # DENSE RECOVERY CANDIDATES
            #
            # Dense retrieval provides the recovery signal, but
            # lexical coverage and diversity are also required.
            #
            # Dense similarity is intentionally NOT treated as
            # equivalent to FlashRank semantic relevance.
            # =================================================

            else:
                selection_score = (
                    (dense_score * 0.45)
                    + (coverage * 0.30)
                    + ((1.0 - redundancy) * 0.15)
                    + (semantic_score * 0.10)
                )

            scored_candidates.append(
                (
                    selection_score,
                    semantic_score,
                    coverage,
                    dense_score,
                    redundancy,
                    marginal_coverage,
                    is_dense_recovery,
                    document,
                )
            )

        scored_candidates.sort(
            key=lambda item: (
                item[0],
                item[1],
                item[2],
                item[3],
            ),
            reverse=True,
        )

        (
            best_selection_score,
            best_semantic_score,
            best_coverage,
            best_dense_score,
            best_redundancy,
            best_marginal_coverage,
            best_is_dense_recovery,
            best_candidate,
        ) = scored_candidates[0]

        # ====================================================
        # ELIGIBILITY GATE
        # ====================================================

        if best_is_dense_recovery:
            if best_dense_score < DENSE_RECOVERY_MIN_SCORE:
                break

            if best_coverage < DENSE_RECOVERY_MIN_COVERAGE:
                break

        else:
            if best_semantic_score < MIN_SEMANTIC_SCORE:
                break

        # ====================================================
        # DIAGNOSTIC
        # ====================================================

        print(
            "\nSELECTOR DECISION | "
            f"chunk={best_candidate.get('chunk_id')} "
            f"document_id={best_candidate.get('document_id')} "
            f"selection_score={best_selection_score:.6f} "
            f"semantic={best_semantic_score:.6f} "
            f"dense={best_dense_score:.6f} "
            f"coverage={best_coverage:.6f} "
            f"redundancy={best_redundancy:.6f} "
            f"marginal_coverage={best_marginal_coverage:.6f} "
            f"dense_recovery={best_is_dense_recovery}"
        )

        selected.append(dict(best_candidate))

        selected_ids.add(_document_identity(best_candidate))

        logfire.info(
            "🔀 Evidence diversity selection",
            document_id=best_candidate.get("document_id"),
            chunk_id=best_candidate.get("chunk_id"),
            semantic_score=round(
                best_semantic_score,
                4,
            ),
            dense_score=round(
                best_dense_score,
                4,
            ),
            query_coverage=round(
                best_coverage,
                4,
            ),
            redundancy=round(
                best_redundancy,
                4,
            ),
            selection_score=round(
                best_selection_score,
                4,
            ),
            dense_recovery=best_is_dense_recovery,
        )

    # ========================================================
    # REMOVE INTERNAL METADATA
    # ========================================================

    for document in selected:
        dense_recovery_selected = sum(
            1 for document in selected if document.get("_dense_recovery", False)
        )

    document.pop(
        "_dense_recovery",
        None,
    )

    # ========================================================
    # FINAL SEMANTIC ORDERING
    #
    # Keep FlashRank as the primary display/order signal.
    # ========================================================

    selected.sort(
        key=lambda document: (
            _rerank_score(document),
            _dense_score(document),
        ),
        reverse=True,
    )

    return selected[:final_k]


# ============================================================
# RETRIEVER NODE
# ============================================================


def retrieve_node(state: AgentState):
    """
    Retrieve enterprise documents from Qdrant and select the
    final evidence context.

    Retrieval architecture:

        Query
          ↓
        BGE-small-en-v1.5
          ↓
        Qdrant dense retrieval
        Top 50
          ↓
        FlashRank cross-encoder
        Score all 50
          ↓
        Semantic pool
             +
        Dense recovery pool
          ↓
        Complementary evidence selection
        Top 5
          ↓
        Downstream evidence grading / generation

    Important:

        50 documents are scored by FlashRank, but only 5 are
        passed downstream as the retrieval context.
    """

    query = state["current_query"]

    total_start = time.perf_counter()

    # ========================================================
    # QDRANT DENSE RETRIEVAL
    # ========================================================

    retrieval_start = time.perf_counter()

    with logfire.span(
        "🔍 Internal Knowledge Retrieval",
        query=query,
        candidate_k=DENSE_CANDIDATE_K,
        embedding_model="BAAI/bge-small-en-v1.5",
        embedding_dimension=384,
    ):
        raw_results = search_enterprise_knowledge(
            query=query,
            limit=DENSE_CANDIDATE_K,
        )

    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

    # ========================================================
    # FLASHRANK SEMANTIC RERANKING
    # ========================================================

    rerank_start = time.perf_counter()

    with logfire.span(
        "⚖️ Internal Semantic Reranking",
        candidate_count=len(raw_results),
        rerank_candidate_k=RERANK_CANDIDATE_K,
        final_context_k=FINAL_CONTEXT_K,
    ):
        reranked_candidates = rerank_documents(
            query=query,
            documents=raw_results,
            top_n=RERANK_CANDIDATE_K,
            text_key="content",
        )

    # ========================================================
    # TEMPORARY GP005 RETRIEVAL DIAGNOSTIC
    # ========================================================

    target_document_id = "89e40c0de3ce2081b9d726b9"

    if target_document_id in {
        _document_identity(document)[0] for document in reranked_candidates
    }:
        print("\n" + "=" * 80)
        print("GP005 FLASHRANK DIAGNOSTIC")
        print(f"Query: {query}")
        print(f"Candidates scored: {len(reranked_candidates)}")
        print("=" * 80)

        target_candidates = [
            document
            for document in reranked_candidates
            if _document_identity(document)[0] == target_document_id
        ]

        target_candidates.sort(
            key=lambda document: (
                _rerank_score(document),
                _dense_score(document),
            ),
            reverse=True,
        )

        for document in target_candidates:
            print(
                "chunk="
                f"{document.get('chunk_id')} "
                f"dense="
                f"{_dense_score(document):.6f} "
                f"flashrank="
                f"{_rerank_score(document):.6f}"
            )

        print("=" * 80)

    # ========================================================
    # FINAL EVIDENCE SELECTION
    # ========================================================

    selected_documents = _select_complementary_evidence(
        query=query,
        dense_documents=raw_results,
        reranked_documents=reranked_candidates,
        final_k=FINAL_CONTEXT_K,
    )

    rerank_ms = (time.perf_counter() - rerank_start) * 1000

    # ========================================================
    # OBSERVABILITY
    # ========================================================

    total_ms = (time.perf_counter() - total_start) * 1000

    excluded_dense_documents = _find_dense_candidates_outside_selected(
        dense_documents=raw_results,
        selected_documents=selected_documents,
    )

    dense_recovery_selected = sum(
        1
        for document in selected_documents
        if document.get(
            "_dense_recovery",
            False,
        )
    )

    logfire.info(
        "📊 Retrieval pipeline timing",
        retrieval_ms=round(
            retrieval_ms,
            2,
        ),
        rerank_ms=round(
            rerank_ms,
            2,
        ),
        total_ms=round(
            total_ms,
            2,
        ),
        dense_candidate_k=DENSE_CANDIDATE_K,
        dense_candidates_returned=len(raw_results),
        rerank_candidate_k=RERANK_CANDIDATE_K,
        rerank_candidates_scored=len(reranked_candidates),
        dense_recovery_k=DENSE_RECOVERY_K,
        dense_recovery_min_score=(DENSE_RECOVERY_MIN_SCORE),
        dense_recovery_selected=(dense_recovery_selected),
        final_context_k=FINAL_CONTEXT_K,
        final_documents=len(selected_documents),
        excluded_documents=len(excluded_dense_documents),
    )

    # ========================================================
    # NORMALIZE FINAL DOCUMENTS
    # ========================================================

    documents = [
        {
            # ------------------------------------------------
            # Qdrant point identity
            # ------------------------------------------------
            "id": str(
                document.get(
                    "id",
                    "",
                )
            ),
            # ------------------------------------------------
            # Corpus identity
            # ------------------------------------------------
            "document_id": str(
                document.get(
                    "document_id",
                    "",
                )
            ),
            "chunk_id": int(
                document.get(
                    "chunk_id",
                    -1,
                )
                if document.get("chunk_id") is not None
                else -1
            ),
            "total_chunks": int(
                document.get(
                    "total_chunks",
                    0,
                )
                if document.get("total_chunks") is not None
                else 0
            ),
            # ------------------------------------------------
            # Content / source
            # ------------------------------------------------
            "content": str(
                document.get(
                    "content",
                    "",
                )
            ),
            "source": str(
                document.get(
                    "source",
                    "Unknown",
                )
            ),
            "source_type": str(
                document.get(
                    "source_type",
                    "internal",
                )
            ),
            # ------------------------------------------------
            # Retrieval / semantic scores
            # ------------------------------------------------
            "score": (
                float(document["score"]) if document.get("score") is not None else 0.0
            ),
            "rerank_score": (
                float(document["rerank_score"])
                if document.get("rerank_score") is not None
                else None
            ),
            # ------------------------------------------------
            # Optional compatibility field
            # ------------------------------------------------
            "url": document.get("url"),
        }
        for document in selected_documents
    ]

    # ========================================================
    # RETURN
    # ========================================================

    return {
        "documents": documents,
        # Existing graph/state compatibility.
        "all_documents": documents,
        "search_query": query,
        "retrieval_latency_ms": (retrieval_ms),
        "rerank_latency_ms": (rerank_ms),
        "status": (
            f"Retrieved "
            f"{len(raw_results)} dense candidates, "
            f"semantically scored "
            f"{len(reranked_candidates)}, "
            f"and selected "
            f"{len(documents)} final internal sources."
        ),
        "plan": list(
            state.get(
                "plan",
                [],
            )
        )
        + [
            (
                f"Internal Dense Retrieval: "
                f"{len(raw_results)}/"
                f"{DENSE_CANDIDATE_K} candidates"
            ),
            (
                f"Internal Semantic Reranking: "
                f"{len(reranked_candidates)}/"
                f"{RERANK_CANDIDATE_K} scored"
            ),
            (
                f"Internal Dense Recovery: "
                f"top {DENSE_RECOVERY_K}, "
                f"threshold "
                f"{DENSE_RECOVERY_MIN_SCORE:.2f}"
            ),
            (
                f"Internal Evidence Selection: "
                f"{len(documents)}/"
                f"{FINAL_CONTEXT_K} documents"
            ),
            (f"Retrieval Time: {retrieval_ms:.0f} ms"),
            (f"Rerank Time: {rerank_ms:.0f} ms"),
        ],
    }
