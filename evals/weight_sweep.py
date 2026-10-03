from __future__ import annotations

import json
import math
import sys
import time
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
    "dense_heavy": (0.60, 0.20, 0.10, 0.10),
    "dense_more": (0.70, 0.15, 0.10, 0.05),
    "dense_dominant": (0.80, 0.10, 0.05, 0.05),
    "dense_plus_coverage": (0.65, 0.25, 0.05, 0.05),
}


def identity(document):
    return _document_identity(document)


def neighbor_hit(selected, gold):
    selected_ids = {identity(document) for document in selected}

    for document_id, chunk_id in gold:
        try:
            target = int(chunk_id)
        except (TypeError, ValueError):
            continue

        for selected_document_id, selected_chunk_id in selected_ids:
            if selected_document_id != document_id:
                continue

            try:
                selected_chunk = int(selected_chunk_id)
            except (TypeError, ValueError):
                continue

            if abs(selected_chunk - target) <= 1:
                return True

    return False


def reciprocal_rank(selected, gold):
    for rank, document in enumerate(selected, start=1):
        if identity(document) in gold:
            return 1.0 / rank

    return 0.0


def ndcg_at_5(selected, gold):
    if not gold:
        return 0.0

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, document in enumerate(selected, start=1)
        if identity(document) in gold
    )

    ideal_hits = min(len(gold), FINAL_K)

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    return dcg / idcg if idcg else 0.0


def select_with_weights(
    query,
    reranked_documents,
    recovery_weights,
):
    (
        recovery_dense_weight,
        recovery_coverage_weight,
        recovery_diversity_weight,
        recovery_rerank_weight,
    ) = recovery_weights

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

    if not candidates:
        return []

    content_terms = {
        identity(document): _tokenize(_content(document))
        for document in candidates
    }

    # Exact production initial admission/order.
    candidates.sort(
        key=lambda document: (
            _rerank_score(document),
            _dense_score(document),
        ),
        reverse=True,
    )

    selected = [dict(candidates[0])]
    selected_ids = {identity(selected[0])}

    while len(selected) < min(FINAL_K, len(candidates)):
        selected_term_sets = [
            content_terms[identity(document)]
            for document in selected
        ]

        remaining = [
            document
            for document in candidates
            if identity(document) not in selected_ids
        ]

        scored = []

        for document in remaining:
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

            scored.append(
                (
                    selection_score,
                    semantic_score,
                    coverage,
                    dense_score,
                    redundancy,
                    is_recovery,
                    document,
                )
            )

        scored.sort(
            key=lambda item: (
                item[0],
                item[1],
                item[2],
                item[3],
            ),
            reverse=True,
        )

        (
            _selection_score,
            semantic_score,
            coverage,
            dense_score,
            _redundancy,
            is_recovery,
            chosen,
        ) = scored[0]

        if is_recovery:
            if dense_score < DENSE_RECOVERY_MIN_SCORE:
                break

            if coverage < DENSE_RECOVERY_MIN_COVERAGE:
                break

        elif semantic_score < MIN_SEMANTIC_SCORE:
            break

        selected.append(dict(chosen))
        selected_ids.add(identity(chosen))

    # Exact production final ordering.
    selected.sort(
        key=lambda document: (
            _rerank_score(document),
            _dense_score(document),
        ),
        reverse=True,
    )

    for document in selected:
        document.pop("_dense_recovery", None)

    return selected[:FINAL_K]



def select_with_protection(
    query,
    reranked_documents,
    recovery_weights,
    protection_margin,
    max_recovery=None,
):
    (
        recovery_dense_weight,
        recovery_coverage_weight,
        recovery_diversity_weight,
        recovery_rerank_weight,
    ) = recovery_weights

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

    if not candidates:
        return []

    content_terms = {
        identity(document): _tokenize(_content(document))
        for document in candidates
    }

    candidates.sort(
        key=lambda document: (
            _rerank_score(document),
            _dense_score(document),
        ),
        reverse=True,
    )

    selected = [dict(candidates[0])]
    selected_ids = {identity(selected[0])}

    while len(selected) < min(FINAL_K, len(candidates)):
        selected_term_sets = [
            content_terms[identity(document)]
            for document in selected
        ]

        remaining = [
            document
            for document in candidates
            if identity(document) not in selected_ids
        ]

        scored = []

        for document in remaining:
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

            scored.append(
                (
                    selection_score,
                    semantic_score,
                    coverage,
                    dense_score,
                    redundancy,
                    is_recovery,
                    document,
                )
            )

        scored.sort(
            key=lambda item: (
                item[0],
                item[1],
                item[2],
                item[3],
            ),
            reverse=True,
        )

        # Semantic-protection gate:
        # a recovery candidate must beat the strongest remaining semantic
        # candidate by at least `protection_margin`.
        chosen = None

        strongest_semantic = next(
            (
                item
                for item in scored
                if not item[5]
            ),
            None,
        )

        for item in scored:
            is_recovery = item[5]

            if is_recovery and max_recovery is not None:
                recovery_count = sum(
                    bool(document.get("_dense_recovery", False))
                    for document in selected
                )

                if recovery_count >= max_recovery:
                    continue

            if not is_recovery:
                chosen = item
                break

            if (
                strongest_semantic is None
                or item[0] >= strongest_semantic[0] + protection_margin
            ):
                chosen = item
                break

        if chosen is None:
            break

        (
            _selection_score,
            semantic_score,
            coverage,
            dense_score,
            _redundancy,
            is_recovery,
            document,
        ) = chosen

        if is_recovery:
            if dense_score < DENSE_RECOVERY_MIN_SCORE:
                break

            if coverage < DENSE_RECOVERY_MIN_COVERAGE:
                break

        elif semantic_score < MIN_SEMANTIC_SCORE:
            break

        selected.append(dict(document))
        selected_ids.add(identity(document))

    selected.sort(
        key=lambda document: (
            _rerank_score(document),
            _dense_score(document),
        ),
        reverse=True,
    )

    for document in selected:
        document.pop("_dense_recovery", None)

    return selected[:FINAL_K]


def main():
    dataset = json.loads(
        DATASET.read_text(encoding="utf-8")
    )

    probes = dataset if isinstance(dataset, list) else dataset["probes"]

    print("=" * 100)
    print("RECOVERY WEIGHT SWEEP — LIVE 70-PROBE COUNTERFACTUAL")
    print("=" * 100)
    print(f"Dataset: {DATASET}")
    print(f"Probes: {len(probes)}")
    print()

    print("Thresholds held constant:")
    print(f"  MIN_SEMANTIC_SCORE           = {MIN_SEMANTIC_SCORE}")
    print(f"  DENSE_RECOVERY_MIN_SCORE    = {DENSE_RECOVERY_MIN_SCORE}")
    print(f"  DENSE_RECOVERY_MIN_COVERAGE = {DENSE_RECOVERY_MIN_COVERAGE}")
    print(f"  DENSE_RECOVERY_K             = {DENSE_RECOVERY_K}")
    print()

    cached = []

    print("=" * 100)
    print("RETRIEVAL / RERANK CACHE")
    print("=" * 100)

    for index, probe in enumerate(probes, start=1):
        question = probe["question"]

        started = time.perf_counter()

        dense_results = search_enterprise_knowledge(
            query=question,
            limit=DENSE_K,
        )

        reranked_results = rerank_documents(
            query=question,
            documents=dense_results,
            top_n=RERANK_K,
            text_key="content",
        )

        elapsed_ms = (time.perf_counter() - started) * 1000

        gold = {
            tuple(
                item.split("::", 1)
            )
            for item in [
                f"{chunk['document_id']}::{int(chunk['chunk_id'])}"
                for chunk in probe["relevant_chunks"]
            ]
        }

        cached.append(
            {
                "id": probe["id"],
                "question": question,
                "gold": gold,
                "reranked": reranked_results,
            }
        )

        print(
            f"[{index:02d}/{len(probes)}] "
            f"{probe['id']:<28} "
            f"reranked={len(reranked_results):>2} "
            f"{elapsed_ms:>7.1f} ms"
        )

    print()
    print("Retrieval/rerank cache complete.")
    print()

    all_results = {}

    for name, weights in WEIGHTS.items():
        rows = []
        selected_by_probe = {}

        for item in cached:
            selected = select_with_weights(
                item["question"],
                item["reranked"],
                weights,
            )

            gold = item["gold"]

            rows.append(
                {
                    "hit": any(
                        identity(document) in gold
                        for document in selected
                    ),
                    "neighbor": neighbor_hit(
                        selected,
                        gold,
                    ),
                    "rr": reciprocal_rank(
                        selected,
                        gold,
                    ),
                    "ndcg": ndcg_at_5(
                        selected,
                        gold,
                    ),
                }
            )

            selected_by_probe[item["id"]] = selected

        n = len(rows)

        all_results[name] = {
            "weights": weights,
            "metrics": {
                "hit": sum(
                    row["hit"] for row in rows
                ) / n,
                "neighbor": sum(
                    row["neighbor"] for row in rows
                ) / n,
                "mrr": sum(
                    row["rr"] for row in rows
                ) / n,
                "ndcg": sum(
                    row["ndcg"] for row in rows
                ) / n,
            },
            "selected": selected_by_probe,
        }

    print()
    print("=" * 100)
    print("AGGREGATE")
    print("=" * 100)
    print(
        f"{'Variant':<24}"
        f"{'Hit@5':>10}"
        f"{'Neighbor':>10}"
        f"{'MRR':>10}"
        f"{'nDCG@5':>10}"
    )
    print("-" * 100)

    for name, result in all_results.items():
        metrics = result["metrics"]

        print(
            f"{name:<24}"
            f"{metrics['hit']:>10.3f}"
            f"{metrics['neighbor']:>10.3f}"
            f"{metrics['mrr']:>10.3f}"
            f"{metrics['ndcg']:>10.3f}"
        )

    baseline = all_results["production"]["selected"]

    print()
    print("=" * 100)
    print("RESCUES / REGRESSIONS VS PRODUCTION")
    print("=" * 100)

    for name, result in all_results.items():
        if name == "production":
            continue

        rescued = []
        regressed = []

        for item in cached:
            probe_id = item["id"]
            gold = item["gold"]

            base_hit = any(
                identity(document) in gold
                for document in baseline[probe_id]
            )

            variant_hit = any(
                identity(document) in gold
                for document in result["selected"][probe_id]
            )

            if not base_hit and variant_hit:
                rescued.append(probe_id)

            if base_hit and not variant_hit:
                regressed.append(probe_id)

        print()
        print(name)
        print(f"  RESCUED   ({len(rescued)}): {rescued}")
        print(f"  REGRESSED ({len(regressed)}): {regressed}")

    targets = [
        "inv_loop_001",
        "inv_harness_011",
        "inv_rag_008",
        "inv_rag_009",
        "inv_training_005",
    ]

    print()
    print("=" * 100)
    print("KNOWN A_EXACT FAILURE CHECK")
    print("=" * 100)

    for probe_id in targets:
        print()
        print(probe_id)

        item = next(
            item
            for item in cached
            if item["id"] == probe_id
        )

        gold = item["gold"]

        for name, result in all_results.items():
            selected = result["selected"][probe_id]

            hit = any(
                identity(document) in gold
                for document in selected
            )

            print(
                f"  {name:<24}"
                f" hit={int(hit)}"
                f" top5={[identity(d) for d in selected]}"
            )


if __name__ == "__main__":
    main()

