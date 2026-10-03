from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.services.retrieval.qdrant_service import (
    search_enterprise_knowledge,
)
from app.services.retrieval.ranking_service import rerank_documents
from app.agents.nodes.retriever import (
    _document_identity,
    _tokenize,
    _query_coverage,
    _marginal_query_coverage,
    _max_redundancy,
    _rerank_score,
    _dense_score,
    _content,
    _select_complementary_evidence,
    DENSE_RECOVERY_MIN_SCORE,
    DENSE_RECOVERY_MIN_COVERAGE,
    DENSE_RECOVERY_K,
    MIN_SEMANTIC_SCORE,
    SEMANTIC_WEIGHT_RERANK,
    SEMANTIC_WEIGHT_COVERAGE,
    SEMANTIC_WEIGHT_DENSE,
    SEMANTIC_WEIGHT_DIVERSITY,
    RECOVERY_WEIGHT_DENSE,
    RECOVERY_WEIGHT_COVERAGE,
    RECOVERY_WEIGHT_DIVERSITY,
    RECOVERY_WEIGHT_RERANK,
)

DATASET = (
    ROOT_DIR
    / "evals"
    / "datasets"
    / "inventory_retrieval_probes.json"
)

FINAL_K = 5
NEIGHBOR_WINDOW = 1


def identity(document):
    return _document_identity(document)


def gold_ids(probe):
    return {
        (
            item["document_id"],
            str(item["chunk_id"]),
        )
        for item in probe["relevant_chunks"]
    }


def neighbor_hit(selected, gold):
    selected_ids = {
        identity(document)
        for document in selected
    }

    for document_id, chunk_id in gold:
        try:
            target = int(chunk_id)
        except ValueError:
            continue

        for selected_document_id, selected_chunk_id in selected_ids:
            if selected_document_id != document_id:
                continue

            try:
                selected_chunk = int(selected_chunk_id)
            except ValueError:
                continue

            if abs(selected_chunk - target) <= NEIGHBOR_WINDOW:
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

    dcg = 0.0

    for rank, document in enumerate(selected, start=1):
        if identity(document) in gold:
            dcg += 1.0 / __import__("math").log2(rank + 1)

    ideal_hits = min(len(gold), FINAL_K)
    idcg = sum(
        1.0 / __import__("math").log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    return dcg / idcg if idcg else 0.0


def minmax_normalize(values):
    if not values:
        return {}

    low = min(values.values())
    high = max(values.values())

    if high <= low:
        return {id_: 1.0 for id_ in values}

    return {
        id_: (value - low) / (high - low)
        for id_, value in values.items()
    }



def production_selector_diagnostic(query, reranked, gold_ids):
    query_terms = _tokenize(query)

    semantic_candidates = []
    recovery_candidates = []

    for raw_document in reranked:
        document = dict(raw_document)

        semantic_score = _rerank_score(document)
        dense_score = _dense_score(document)

        coverage = _query_coverage(
            query_terms,
            _tokenize(_content(document)),
        )

        if semantic_score >= MIN_SEMANTIC_SCORE:
            semantic_candidates.append(document)

        elif (
            dense_score >= DENSE_RECOVERY_MIN_SCORE
            and coverage >= DENSE_RECOVERY_MIN_COVERAGE
        ):
            document["_dense_recovery"] = True
            recovery_candidates.append(document)

    recovery_candidates = recovery_candidates[:DENSE_RECOVERY_K]

    candidates = []
    seen = set()

    for document in semantic_candidates + recovery_candidates:
        doc_id = identity(document)

        if doc_id in seen:
            continue

        seen.add(doc_id)
        candidates.append(document)

    gold_candidates = [
        document for document in candidates
        if identity(document) in gold_ids
    ]

    diagnostics = {
        "gold_in_rerank": any(
            identity(document) in gold_ids
            for document in reranked
        ),
        "gold_in_candidates": bool(gold_candidates),
        "gold_candidate_details": [],
        "selection_steps": [],
        "gold_selected_before_final_sort": False,
        "gold_in_final": False,
    }

    for document in gold_candidates:
        diagnostics["gold_candidate_details"].append({
            "id": str(identity(document)),
            "rerank_score": _rerank_score(document),
            "dense_score": _dense_score(document),
            "coverage": _query_coverage(
                query_terms,
                _tokenize(_content(document)),
            ),
            "recovery": bool(document.get("_dense_recovery", False)),
        })

    if not candidates:
        return [], diagnostics

    candidates.sort(
        key=lambda document: (
            _rerank_score(document),
            _dense_score(document),
        ),
        reverse=True,
    )

    selected = [dict(candidates[0])]
    selected_ids = {identity(selected[0])}

    step = 1
    while len(selected) < min(FINAL_K, len(candidates)):
        selected_terms = [
            _tokenize(_content(document))
            for document in selected
        ]

        remaining = [
            document for document in candidates
            if identity(document) not in selected_ids
        ]

        scored = []

        for document in remaining:
            content_terms = _tokenize(_content(document))

            semantic_score = _rerank_score(document)
            dense_score = _dense_score(document)

            coverage = _query_coverage(
                query_terms,
                content_terms,
            )

            marginal_coverage = _marginal_query_coverage(
                query_terms,
                content_terms,
                selected_terms,
            )

            redundancy = _max_redundancy(
                content_terms,
                selected_terms,
            )

            recovery = bool(
                document.get("_dense_recovery", False)
            )

            if not recovery:
                score = (
                    semantic_score * SEMANTIC_WEIGHT_RERANK
                    + coverage * SEMANTIC_WEIGHT_COVERAGE
                    + dense_score * SEMANTIC_WEIGHT_DENSE
                    + (1.0 - redundancy)
                    * SEMANTIC_WEIGHT_DIVERSITY
                )
            else:
                score = (
                    dense_score * RECOVERY_WEIGHT_DENSE
                    + coverage * RECOVERY_WEIGHT_COVERAGE
                    + (1.0 - redundancy)
                    * RECOVERY_WEIGHT_DIVERSITY
                    + semantic_score * RECOVERY_WEIGHT_RERANK
                )

            scored.append((score, document))

        if not scored:
            break

        scored.sort(
            key=lambda item: (
                item[0],
                _rerank_score(item[1]),
                _dense_score(item[1]),
            ),
            reverse=True,
        )

        gold_scored = [
            item for item in scored
            if identity(item[1]) in gold_ids
        ]

        chosen = scored[0][1]

        diagnostics["selection_steps"].append({
            "step": step,
            "chosen": str(identity(chosen)),
            "chosen_score": scored[0][0],
            "gold": [
                {
                    "id": str(identity(document)),
                    "selection_score": score,
                    "rerank_score": _rerank_score(document),
                    "dense_score": _dense_score(document),
                    "coverage": _query_coverage(
                        query_terms,
                        _tokenize(_content(document)),
                    ),
                    "marginal_coverage": _marginal_query_coverage(
                        query_terms,
                        _tokenize(_content(document)),
                        selected_terms,
                    ),
                    "redundancy": _max_redundancy(
                        _tokenize(_content(document)),
                        selected_terms,
                    ),
                    "recovery": bool(
                        document.get("_dense_recovery", False)
                    ),
                    "rank_among_remaining": rank + 1,
                }
                for rank, (score, document) in enumerate(scored)
                if identity(document) in gold_ids
            ],
        })

        if chosen.get("_dense_recovery"):
            if _dense_score(chosen) < DENSE_RECOVERY_MIN_SCORE:
                break

            coverage = _query_coverage(
                query_terms,
                _tokenize(_content(chosen)),
            )

            if coverage < DENSE_RECOVERY_MIN_COVERAGE:
                break
        else:
            if _rerank_score(chosen) < MIN_SEMANTIC_SCORE:
                break

        selected.append(dict(chosen))
        selected_ids.add(identity(chosen))
        step += 1

    diagnostics["gold_selected_before_final_sort"] = any(
        identity(document) in gold_ids
        for document in selected
    )

    pre_sort_selected = [
        str(identity(document))
        for document in selected
    ]

    selected.sort(
        key=lambda document: (
            _rerank_score(document),
            _dense_score(document),
        ),
        reverse=True,
    )

    final = selected[:FINAL_K]

    diagnostics["gold_in_final"] = any(
        identity(document) in gold_ids
        for document in final
    )

    diagnostics["pre_sort_selected"] = pre_sort_selected
    diagnostics["final_selected"] = [
        str(identity(document))
        for document in final
    ]

    return final, diagnostics

def dense_only_selector(dense):
    return [
        dict(document)
        for document in dense[:FINAL_K]
    ]


def hybrid_selector(query, reranked):
    query_terms = _tokenize(query)

    documents = [
        dict(document)
        for document in reranked
    ]

    dense_values = {
        identity(document): _dense_score(document)
        for document in documents
    }

    rerank_values = {
        identity(document): _rerank_score(document)
        for document in documents
    }

    dense_norm = minmax_normalize(dense_values)
    rerank_norm = minmax_normalize(rerank_values)

    scored = []

    for document in documents:
        doc_id = identity(document)

        coverage = _query_coverage(
            query_terms,
            _tokenize(_content(document)),
        )

        score = (
            0.50 * rerank_norm[doc_id]
            + 0.35 * dense_norm[doc_id]
            + 0.15 * coverage
        )

        scored.append((score, document))

    scored.sort(
        key=lambda item: (
            item[0],
            _rerank_score(item[1]),
            _dense_score(item[1]),
        ),
        reverse=True,
    )

    return [
        dict(document)
        for _, document in scored[:FINAL_K]
    ]


def thresholded_production_selector(
    query,
    reranked_documents,
    semantic_threshold,
    recovery_min_score=0.68,
    recovery_min_coverage=0.20,
):
    query_terms = _tokenize(query)

    semantic_candidates = [
        dict(document)
        for document in reranked_documents
        if _rerank_score(document) >= semantic_threshold
    ]

    dense_candidates = sorted(
        reranked_documents,
        key=_dense_score,
        reverse=True,
    )

    dense_recovery_candidates = []

    for document in dense_candidates[:DENSE_RECOVERY_K]:
        dense_score = _dense_score(document)

        if dense_score < recovery_min_score:
            break

        rerank_score = _rerank_score(document)

        if rerank_score >= semantic_threshold:
            continue

        coverage = _query_coverage(
            query_terms,
            _tokenize(_content(document)),
        )

        if coverage < recovery_min_coverage:
            continue

        recovery_document = dict(document)
        recovery_document["_dense_recovery"] = True
        dense_recovery_candidates.append(recovery_document)

    candidates = []
    seen = set()

    for document in semantic_candidates + dense_recovery_candidates:
        doc_id = identity(document)

        if doc_id in seen:
            continue

        seen.add(doc_id)
        candidates.append(document)

    return _select_complementary_evidence(
        query=query,
        reranked_documents=candidates,
        final_k=FINAL_K,
    )



def selection_order_production_selector(
    query,
    reranked_documents,
    semantic_threshold=0.45,
    recovery_min_score=0.68,
    recovery_min_coverage=0.20,
):
    query_terms = _tokenize(query)

    semantic_candidates = [
        dict(document)
        for document in reranked_documents
        if _rerank_score(document) >= semantic_threshold
    ]

    dense_candidates = sorted(
        reranked_documents,
        key=_dense_score,
        reverse=True,
    )

    dense_recovery_candidates = []

    for document in dense_candidates[:DENSE_RECOVERY_K]:
        dense_score = _dense_score(document)

        if dense_score < recovery_min_score:
            break

        rerank_score = _rerank_score(document)

        if rerank_score >= semantic_threshold:
            continue

        coverage = _query_coverage(
            query_terms,
            _tokenize(_content(document)),
        )

        if coverage < recovery_min_coverage:
            continue

        recovery_document = dict(document)
        recovery_document["_dense_recovery"] = True
        dense_recovery_candidates.append(recovery_document)

    candidates = []
    seen = set()

    for document in semantic_candidates + dense_recovery_candidates:
        doc_id = identity(document)

        if doc_id in seen:
            continue

        seen.add(doc_id)
        candidates.append(document)

    selected = []
    selected_ids = set()

    candidate_terms = {
        identity(document): _tokenize(_content(document))
        for document in candidates
    }

    while len(selected) < FINAL_K and candidates:
        scored_candidates = []

        for document in candidates:
            document_id = identity(document)

            if document_id in selected_ids:
                continue

            semantic_score = _rerank_score(document)
            dense_score = _dense_score(document)

            coverage = _query_coverage(
                query_terms,
                candidate_terms[document_id],
            )

            redundancy = 0.0

            if selected:
                redundancy = max(
                    _query_coverage(
                        candidate_terms[identity(selected_document)],
                        candidate_terms[document_id],
                    )
                    for selected_document in selected
                )

            marginal_coverage = max(
                0.0,
                coverage * (1.0 - redundancy),
            )

            is_dense_recovery = bool(
                document.get("_dense_recovery", False)
            )

            if is_dense_recovery:
                selection_score = (
                    0.45 * dense_score
                    + 0.30 * coverage
                    + 0.15 * (1.0 - redundancy)
                    + 0.10 * semantic_score
                )
            else:
                selection_score = (
                    0.70 * semantic_score
                    + 0.15 * coverage
                    + 0.10 * dense_score
                    + 0.05 * (1.0 - redundancy)
                )

            scored_candidates.append(
                (
                    selection_score,
                    semantic_score,
                    dense_score,
                    coverage,
                    redundancy,
                    marginal_coverage,
                    is_dense_recovery,
                    document,
                )
            )

        if not scored_candidates:
            break

        scored_candidates.sort(
            key=lambda item: (
                item[0],
                item[1],
                item[3],
                item[2],
            ),
            reverse=True,
        )

        (
            best_selection_score,
            best_semantic_score,
            best_dense_score,
            best_coverage,
            best_redundancy,
            best_marginal_coverage,
            best_is_dense_recovery,
            best_candidate,
        ) = scored_candidates[0]

        if best_is_dense_recovery:
            if best_dense_score < recovery_min_score:
                break
            if best_coverage < recovery_min_coverage:
                break
        else:
            if best_semantic_score < semantic_threshold:
                break

        selected.append(dict(best_candidate))
        selected_ids.add(identity(best_candidate))

    for document in selected:
        document.pop("_dense_recovery", None)

    # Intentionally preserve complementary-selection order.
    return selected[:FINAL_K]


def evaluate(selected, gold):
    hit = any(
        identity(document) in gold
        for document in selected
    )

    return {
        "hit": hit,
        "neighbor": neighbor_hit(selected, gold),
        "rr": reciprocal_rank(selected, gold),
        "ndcg": ndcg_at_5(selected, gold),
    }


def main():
    with DATASET.open(encoding="utf-8") as f:
        probes = json.load(f)

    methods = {
        "current": [],
        "dense": [],
        "hybrid": [],
    }

    disagreements = []

    print("=" * 100)
    print("SELECTOR A/B/C SHADOW EXPERIMENT")
    print("=" * 100)
    print("Probes:", len(probes))
    print()

    for index, probe in enumerate(probes, start=1):
        query = probe["question"]
        gold = gold_ids(probe)

        dense = search_enterprise_knowledge(
            query,
            limit=50,
        )

        reranked = rerank_documents(
            query=query,
            documents=dense,
            top_n=50,
            text_key="content",
        )

        current = production_selector(
            query,
            reranked,
        )

        dense_selected = dense_only_selector(
            dense,
        )

        hybrid = hybrid_selector(
            query,
            reranked,
        )

        results = {
            "current": evaluate(current, gold),
            "dense": evaluate(dense_selected, gold),
            "hybrid": evaluate(hybrid, gold),
        }

        for method, result in results.items():
            methods[method].append(result)

        hits = {
            method: result["hit"]
            for method, result in results.items()
        }

        if len(set(hits.values())) > 1:
            disagreements.append(
                (
                    probe["id"],
                    hits,
                    current,
                    dense_selected,
                    hybrid,
                    gold,
                )
            )

        print(
            f"[{index:02d}/{len(probes)}] "
            f"{probe['id']:<28} "
            f"C={int(hits['current'])} "
            f"D={int(hits['dense'])} "
            f"H={int(hits['hybrid'])}"
        )

    print()
    print("=" * 100)
    print("AGGREGATE")
    print("=" * 100)

    for method, rows in methods.items():
        n = len(rows)

        print()
        print(method.upper())
        print("-" * 100)
        print(
            "Hit@5           :",
            f"{sum(r['hit'] for r in rows) / n:.3f}",
        )
        print(
            "Neighbor Hit@5  :",
            f"{sum(r['neighbor'] for r in rows) / n:.3f}",
        )
        print(
            "MRR             :",
            f"{sum(r['rr'] for r in rows) / n:.3f}",
        )
        print(
            "nDCG@5          :",
            f"{sum(r['ndcg'] for r in rows) / n:.3f}",
        )

    print()
    print("=" * 100)
    print("DISAGREEMENTS")
    print("=" * 100)

    for (
        probe_id,
        hits,
        current,
        dense_selected,
        hybrid,
        gold,
    ) in disagreements:

        print()
        print(probe_id)
        print("GOLD:", gold)
        print(
            "CURRENT:",
            [identity(d) for d in current],
        )
        print(
            "DENSE:",
            [identity(d) for d in dense_selected],
        )
        print(
            "HYBRID:",
            [identity(d) for d in hybrid],
        )
        print("HITS:", hits)


if __name__ == "__main__":
    main()
