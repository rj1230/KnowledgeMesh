from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.ranking_service import rerank_documents

from app.agents.nodes.retriever import (
    _content,
    _dense_score,
    _document_identity,
    _max_redundancy,
    _query_coverage,
    _rerank_score,
    _tokenize,
    DENSE_RECOVERY_K,
    DENSE_RECOVERY_MIN_COVERAGE,
    DENSE_RECOVERY_MIN_SCORE,
    MIN_SEMANTIC_SCORE,
    SEMANTIC_WEIGHT_COVERAGE,
    SEMANTIC_WEIGHT_DENSE,
    SEMANTIC_WEIGHT_DIVERSITY,
    SEMANTIC_WEIGHT_RERANK,
)

DATASET = ROOT / "evals" / "datasets" / "inventory_retrieval_probes_70.json"

FINAL_K = 5
DENSE_K = 50
RERANK_K = 50

WEIGHTS = {
    "production": (0.45, 0.30, 0.15, 0.10),
    "dense_more": (0.70, 0.15, 0.10, 0.05),
    "dense_dominant": (0.80, 0.10, 0.05, 0.05),
}

TARGETS = {
    "inv_loop_001",
    "inv_rag_008",
    "inv_llm_agents_003",
}


def identity(document):
    return _document_identity(document)


def prepare_candidates(query, reranked_documents):
    query_terms = _tokenize(query)

    semantic_candidates = [
        dict(document)
        for document in reranked_documents
        if _rerank_score(document) >= MIN_SEMANTIC_SCORE
    ]

    dense_candidates = sorted(
        reranked_documents,
        key=_dense_score,
        reverse=True,
    )

    recovery_candidates = []

    for document in dense_candidates:
        dense_score = _dense_score(document)

        if dense_score < DENSE_RECOVERY_MIN_SCORE:
            break

        semantic_score = _rerank_score(document)

        if semantic_score >= MIN_SEMANTIC_SCORE:
            continue

        coverage = _query_coverage(
            query_terms,
            _tokenize(_content(document)),
        )

        if coverage < DENSE_RECOVERY_MIN_COVERAGE:
            continue

        recovery_document = dict(document)
        recovery_document["_dense_recovery"] = True
        recovery_candidates.append(recovery_document)

        if len(recovery_candidates) >= DENSE_RECOVERY_K:
            break

    candidates = []
    seen_ids = set()

    for document in semantic_candidates + recovery_candidates:
        document_id = identity(document)

        if document_id in seen_ids:
            continue

        seen_ids.add(document_id)
        candidates.append(dict(document))

    candidates.sort(
        key=lambda document: (
            _rerank_score(document),
            _dense_score(document),
        ),
        reverse=True,
    )

    return candidates, query_terms


def score_candidates(
    candidates,
    query_terms,
    selected,
    recovery_weights,
):
    (
        recovery_dense_weight,
        recovery_coverage_weight,
        recovery_diversity_weight,
        recovery_rerank_weight,
    ) = recovery_weights

    content_terms = {
        identity(document): _tokenize(_content(document))
        for document in candidates
    }

    selected_term_sets = [
        content_terms[identity(document)]
        for document in selected
    ]

    rows = []

    for document in candidates:
        if identity(document) in {
            identity(item) for item in selected
        }:
            continue

        document_id = identity(document)
        terms = content_terms[document_id]

        semantic_score = _rerank_score(document)
        dense_score = _dense_score(document)

        coverage = _query_coverage(
            query_terms,
            terms,
        )

        redundancy = _max_redundancy(
            terms,
            selected_term_sets,
        )

        is_recovery = bool(
            document.get("_dense_recovery", False)
        )

        if is_recovery:
            selection_score = (
                dense_score * recovery_dense_weight
                + coverage * recovery_coverage_weight
                + (1.0 - redundancy) * recovery_diversity_weight
                + semantic_score * recovery_rerank_weight
            )
        else:
            selection_score = (
                semantic_score * SEMANTIC_WEIGHT_RERANK
                + coverage * SEMANTIC_WEIGHT_COVERAGE
                + dense_score * SEMANTIC_WEIGHT_DENSE
                + (1.0 - redundancy) * SEMANTIC_WEIGHT_DIVERSITY
            )

        rows.append(
            {
                "id": identity(document),
                "selection_score": selection_score,
                "rerank": semantic_score,
                "dense": dense_score,
                "coverage": coverage,
                "redundancy": redundancy,
                "recovery": is_recovery,
            }
        )

    rows.sort(
        key=lambda row: (
            row["selection_score"],
            row["rerank"],
            row["coverage"],
            row["dense"],
        ),
        reverse=True,
    )

    return rows


def run_selector(query, reranked_documents, recovery_weights):
    candidates, query_terms = prepare_candidates(
        query,
        reranked_documents,
    )

    if not candidates:
        return [], []

    selected = [dict(candidates[0])]

    trace = [
        {
            "step": 1,
            "chosen": identity(selected[0]),
            "chosen_dense": _dense_score(selected[0]),
            "chosen_rerank": _rerank_score(selected[0]),
        }
    ]

    while len(selected) < min(FINAL_K, len(candidates)):
        rows = score_candidates(
            candidates,
            query_terms,
            selected,
            recovery_weights,
        )

        if not rows:
            break

        winner = rows[0]

        chosen = next(
            document
            for document in candidates
            if identity(document) == winner["id"]
        )

        selected.append(dict(chosen))

        trace.append(
            {
                "step": len(selected),
                "chosen": winner["id"],
                "chosen_dense": winner["dense"],
                "chosen_rerank": winner["rerank"],
                "chosen_coverage": winner["coverage"],
                "chosen_redundancy": winner["redundancy"],
                "chosen_recovery": winner["recovery"],
                "top_competitors": rows[:8],
            }
        )

    selected.sort(
        key=lambda document: (
            _rerank_score(document),
            _dense_score(document),
        ),
        reverse=True,
    )

    return selected[:FINAL_K], trace


def main():
    dataset = json.loads(
        DATASET.read_text(encoding="utf-8")
    )

    probes = dataset if isinstance(dataset, list) else dataset["probes"]

    targets = [
        probe
        for probe in probes
        if probe["id"] in TARGETS
    ]

    print("=" * 100)
    print("WEIGHT FORENSIC DIAGNOSTIC")
    print("=" * 100)

    for probe in targets:
        probe_id = probe["id"]
        query = probe["question"]

        print()
        print("=" * 100)
        print(probe_id)
        print("=" * 100)
        print("QUERY:", query)

        dense = search_enterprise_knowledge(
            query=query,
            limit=DENSE_K,
        )

        reranked = rerank_documents(
            query=query,
            documents=dense,
            top_n=RERANK_K,
            text_key="content",
        )

        print(f"Retrieved={len(dense)} Reranked={len(reranked)}")

        for variant, weights in WEIGHTS.items():
            selected, trace = run_selector(
                query,
                reranked,
                weights,
            )

            print()
            print("-" * 100)
            print(
                f"{variant}: "
                f"weights={weights}"
            )
            print("FINAL TOP-5:")

            for rank, document in enumerate(selected, start=1):
                print(
                    f"  {rank}. {identity(document)} "
                    f"dense={_dense_score(document):.4f} "
                    f"rerank={_rerank_score(document):.4f}"
                )

            print()
            print("SELECTION TRACE:")

            for item in trace:
                print(
                    f"  STEP {item['step']} "
                    f"chosen={item['chosen']} "
                    f"dense={item.get('chosen_dense', 0):.4f} "
                    f"rerank={item.get('chosen_rerank', 0):.4f}"
                )

                competitors = item.get(
                    "top_competitors",
                    [],
                )

                if competitors:
                    print("    TOP COMPETITORS:")

                    for row in competitors[:5]:
                        print(
                            f"      {row['id']} "
                            f"score={row['selection_score']:.4f} "
                            f"dense={row['dense']:.4f} "
                            f"rerank={row['rerank']:.4f} "
                            f"coverage={row['coverage']:.4f} "
                            f"redundancy={row['redundancy']:.4f} "
                            f"recovery={row['recovery']}"
                        )


if __name__ == "__main__":
    main()
