import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.ranking_service import rerank_documents
from app.agents.nodes.retriever import (
    _document_identity,
    _tokenize,
    _query_coverage,
    _rerank_score,
    _dense_score,
    _content,
    DENSE_RECOVERY_MIN_SCORE,
    DENSE_RECOVERY_MIN_COVERAGE,
    MIN_SEMANTIC_SCORE,
)

DATASET = (
    ROOT_DIR
    / "evals"
    / "datasets"
    / "inventory_retrieval_probes.json"
)

TARGETS = {
    "inv_llm_agents_001",
    "inv_llm_agents_002",
}

with DATASET.open(encoding="utf-8") as f:
    probes = json.load(f)

for probe in probes:
    if probe["id"] not in TARGETS:
        continue

    query = probe["question"]
    query_terms = _tokenize(query)

    gold = {
        (
            item["document_id"],
            str(item["chunk_id"]),
        )
        for item in probe["relevant_chunks"]
    }

    dense = search_enterprise_knowledge(query, limit=50)
    reranked = rerank_documents(query=query, documents=dense, top_n=50, text_key="content")

    print()
    print("=" * 100)
    print(probe["id"])
    print("=" * 100)

    print(
        "thresholds:",
        "MIN_SEMANTIC=",
        MIN_SEMANTIC_SCORE,
        "DENSE_RECOVERY=",
        DENSE_RECOVERY_MIN_SCORE,
        "COVERAGE=",
        DENSE_RECOVERY_MIN_COVERAGE,
    )

    print()
    print("GOLD DIAGNOSTIC")
    print("-" * 100)

    found = False

    for rank, document in enumerate(reranked, start=1):

        identity = _document_identity(document)

        if identity not in gold:
            continue

        found = True

        dense_score = _dense_score(document)
        rerank_score = _rerank_score(document)
        coverage = _query_coverage(
            query_terms=query_terms,
            content_terms=_tokenize(_content(document)),
        )

        print("rank              :", rank)
        print("identity          :", identity)
        print("dense_score       :", dense_score)
        print("rerank_score      :", rerank_score)
        print("coverage          :", coverage)
        print(
            "rerank_ok         :",
            rerank_score < MIN_SEMANTIC_SCORE,
        )
        print(
            "dense_ok          :",
            dense_score >= DENSE_RECOVERY_MIN_SCORE,
        )
        print(
            "coverage_ok       :",
            coverage >= DENSE_RECOVERY_MIN_COVERAGE,
        )
        print(
            "RECOVERY ELIGIBLE:",
            (
                rerank_score < MIN_SEMANTIC_SCORE
                and dense_score >= DENSE_RECOVERY_MIN_SCORE
                and coverage >= DENSE_RECOVERY_MIN_COVERAGE
            ),
        )

    if not found:
        print("GOLD NOT FOUND IN RERANKED TOP-50")

    print()
    print("ALL RECOVERY-ELIGIBLE CANDIDATES")
    print("-" * 100)

    recovery = []

    for rank, document in enumerate(reranked, start=1):

        dense_score = _dense_score(document)
        rerank_score = _rerank_score(document)
        coverage = _query_coverage(
            query_terms=query_terms,
            content_terms=_tokenize(_content(document)),
        )

        if (
            rerank_score < MIN_SEMANTIC_SCORE
            and dense_score >= DENSE_RECOVERY_MIN_SCORE
            and coverage >= DENSE_RECOVERY_MIN_COVERAGE
        ):
            recovery.append(
                (
                    rank,
                    _document_identity(document),
                    dense_score,
                    rerank_score,
                    coverage,
                )
            )

    recovery.sort(
        key=lambda item: item[2],
        reverse=True,
    )

    print("count:", len(recovery))

    for row in recovery[:20]:
        print(
            f"rank={row[0]:2d}",
            f"id={row[1]}",
            f"dense={row[2]:.6f}",
            f"rerank={row[3]:.6f}",
            f"coverage={row[4]:.6f}",
            "GOLD" if row[1] in gold else "",
        )

