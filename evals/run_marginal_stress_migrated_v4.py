import json
import math
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.ranking_service import rerank_documents

from evals.selector_shadow import (
    identity,
    _tokenize,
    _query_coverage,
    _marginal_query_coverage,
    _max_redundancy,
    _rerank_score,
    _dense_score,
    _content,
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

FINAL_K = 5
NEIGHBOR_WINDOW = 1

DATASET = (
    ROOT_DIR
    / "evals"
    / "datasets"
    / "golden_questions_migrated_candidate_20261018.json"
)


def gold_ids(probe):
    return {
        (
            item["document_id"],
            str(item["chunk_id"]),
        )
        for item in probe["relevant_chunks"]
    }


def neighbor_hit(selected, gold):
    selected_ids = {identity(x) for x in selected}

    for document_id, chunk_id in gold:
        try:
            chunk = int(chunk_id)
        except ValueError:
            continue

        for selected_id in selected_ids:
            selected_doc, selected_chunk = selected_id
            if selected_doc != document_id:
                continue
            try:
                if abs(int(selected_chunk) - chunk) <= NEIGHBOR_WINDOW:
                    return True
            except ValueError:
                pass

    return False


def reciprocal_rank(selected, gold):
    for rank, document in enumerate(selected, 1):
        if identity(document) in gold:
            return 1.0 / rank
    return 0.0


def ndcg(selected, gold):
    if not gold:
        return 0.0

    dcg = 0.0

    for rank, document in enumerate(selected, 1):
        if identity(document) in gold:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(gold), FINAL_K)

    if ideal_hits == 0:
        return 0.0

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    return dcg / idcg if idcg else 0.0


def select_with_marginal(query, reranked, marginal_weight):
    query_terms = _tokenize(query)

    semantic_candidates = []
    recovery_candidates = []

    semantic_candidates = [
        dict(document)
        for document in reranked
        if _rerank_score(document) >= MIN_SEMANTIC_SCORE
    ]

    # Match production: recovery candidates are considered in
    # descending dense-score order, not reranker order.
    dense_candidates = sorted(
        reranked,
        key=_dense_score,
        reverse=True,
    )

    recovery_candidates = []

    for document in dense_candidates:
        dense_score = _dense_score(document)

        if dense_score < DENSE_RECOVERY_MIN_SCORE:
            break

        if _rerank_score(document) >= MIN_SEMANTIC_SCORE:
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
    seen = set()

    for document in semantic_candidates + recovery_candidates:
        doc_id = identity(document)

        if doc_id in seen:
            continue

        seen.add(doc_id)
        candidates.append(document)

    if not candidates:
        return []

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
        selected_terms = [
            _tokenize(_content(document))
            for document in selected
        ]

        remaining = [
            document
            for document in candidates
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

            marginal = _marginal_query_coverage(
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
                base = (
                    semantic_score * SEMANTIC_WEIGHT_RERANK
                    + coverage * SEMANTIC_WEIGHT_COVERAGE
                    + dense_score * SEMANTIC_WEIGHT_DENSE
                    + (1.0 - redundancy)
                    * SEMANTIC_WEIGHT_DIVERSITY
                )
            else:
                base = (
                    dense_score * RECOVERY_WEIGHT_DENSE
                    + coverage * RECOVERY_WEIGHT_COVERAGE
                    + (1.0 - redundancy)
                    * RECOVERY_WEIGHT_DIVERSITY
                    + semantic_score * RECOVERY_WEIGHT_RERANK
                )

            # Preserve the original score at M=0.
            # Scale the original score so adding marginal coverage
            # does not change the total weight budget.
            score = (
                base * (1.0 - marginal_weight)
                + marginal * marginal_weight
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

        chosen = scored[0][1]

        if chosen.get("_dense_recovery"):
            if _dense_score(chosen) < DENSE_RECOVERY_MIN_SCORE:
                break

            chosen_coverage = _query_coverage(
                query_terms,
                _tokenize(_content(chosen)),
            )

            if chosen_coverage < DENSE_RECOVERY_MIN_COVERAGE:
                break

        elif _rerank_score(chosen) < MIN_SEMANTIC_SCORE:
            break

        selected.append(dict(chosen))
        selected_ids.add(identity(chosen))

    # Keep the production final ordering exactly unchanged.
    selected.sort(
        key=lambda document: (
            _rerank_score(document),
            _dense_score(document),
        ),
        reverse=True,
    )

    return selected[:FINAL_K]


def run():
    probes = json.loads(
        DATASET.read_text(encoding="utf-8")
    )

    weights = [0.25, 0.50, 0.75, 1.00]

    stats = {
        weight: {
            "hit": 0,
            "neighbor": 0,
            "rr": 0.0,
            "ndcg": 0.0,
        }
        for weight in weights
    }

    per_probe = {
        weight: []
        for weight in weights
    }

    for index, probe in enumerate(probes, 1):
        question = probe["question"]
        gold = gold_ids(probe)

        dense = search_enterprise_knowledge(
            question,
            limit=50,
        )

        reranked = rerank_documents(
            question,
            dense,
            top_n=50,
        )

        for weight in weights:
            selected = select_with_marginal(
                question,
                reranked,
                weight,
            )

            hit = any(
                identity(document) in gold
                for document in selected
            )

            neighbor = neighbor_hit(selected, gold)
            rr = reciprocal_rank(selected, gold)
            score = ndcg(selected, gold)

            stats[weight]["hit"] += int(hit)
            stats[weight]["neighbor"] += int(neighbor)
            stats[weight]["rr"] += rr
            stats[weight]["ndcg"] += score

            per_probe[weight].append({
                "id": probe["id"],
                "hit": hit,
                "neighbor": neighbor,
                "rr": rr,
                "ndcg": score,
                "selected_ids": [
                    f"{identity(document)[0]}::{identity(document)[1]}"
                    for document in selected
                ],
            })

        print(
            f"[{index:02d}/{len(probes)}] {probe['id']}",
            flush=True,
        )

    # Compare the corrected M=0 selector with the saved live baseline.
    baseline_path = (
        ROOT_DIR / "DATA" / "retrieval_baseline_migrated_20261020.json"
    )

    if baseline_path.exists():
        baseline_data = json.loads(
            baseline_path.read_text(encoding="utf-8")
        )
        baseline_by_id = {
            item["id"]: item.get("final", {}).get("top_ids", [])
            for item in baseline_data.get("results", [])
        }
        # Stress-test weights intentionally omit M=0.00.
        # No shadow baseline exists for this stress-only weight grid.
        shadow_by_id = {}

        comparable = [
            probe_id for probe_id in shadow_by_id
            if probe_id in baseline_by_id
        ]
        matches = [
            probe_id for probe_id in comparable
            if shadow_by_id[probe_id] == baseline_by_id[probe_id]
        ]

        print(
            f"\\nM=0.00 production-baseline parity: "
            f"{len(matches)}/{len(comparable)} exact five-chunk matches"
        )

        for probe_id in comparable:
            if shadow_by_id[probe_id] != baseline_by_id[probe_id]:
                print(f"  MISMATCH {probe_id}")
                print(f"    baseline: {baseline_by_id[probe_id]}")
                print(f"    shadow:   {shadow_by_id[probe_id]}")
    else:
        print(f"\\nBaseline file unavailable: {baseline_path}")

    total = len(probes)

    print("\n" + "=" * 90)
    print("MARGINAL-COVERAGE COUNTERFACTUAL")
    print("=" * 90)

    print(
        f"{'Weight':<10}"
        f"{'Hit@5':<12}"
        f"{'Neighbor@5':<15}"
        f"{'MRR':<12}"
        f"{'nDCG@5':<12}"
    )

    for weight in weights:
        s = stats[weight]

        print(
            f"M={weight:<7.2f}"
            f"{s['hit'] / total:<12.3f}"
            f"{s['neighbor'] / total:<15.3f}"
            f"{s['rr'] / total:<12.3f}"
            f"{s['ndcg'] / total:<12.3f}"
        )

    baseline_path = (
        ROOT_DIR / "DATA" / "retrieval_baseline_migrated_20261020.json"
    )
    if not baseline_path.exists():
        print("WARNING: saved production baseline not found; "
              "skipping per-probe comparison.")
        return

    baseline_data = json.loads(
        baseline_path.read_text(encoding="utf-8")
    )
    baseline_by_id = {
        item["id"]: item
        for item in baseline_data.get("results", [])
    }

    for weight in weights:
        changed = []

        for candidate in per_probe[weight]:
            base = baseline_by_id.get(candidate["id"])
            if base is None:
                continue

            base_final = base.get("final", {}).get("top_ids", [])
            candidate_ids = candidate.get("selected_ids", [])

            base_hit = any(
                gold_id in base_final
                for gold_id in gold_ids(
                    next(p for p in probes if p["id"] == candidate["id"])
                )
            )
            candidate_hit = candidate["hit"]

            base_neighbor = neighbor_hit(
                [{"document_id": doc_id.split("::")[0],
                  "chunk_id": doc_id.split("::")[1]}
                 for doc_id in base_final],
                gold_ids(
                    next(p for p in probes if p["id"] == candidate["id"])
                ),
            )
            candidate_neighbor = candidate["neighbor"]

            if base_hit != candidate_hit or base_neighbor != candidate_neighbor:
                changed.append((
                    candidate["id"],
                    base_hit,
                    candidate_hit,
                    base_neighbor,
                    candidate_neighbor,
                ))
            if (
                base["hit"] != candidate["hit"]
                or base["neighbor"] != candidate["neighbor"]
            ):
                changed.append(
                    (
                        candidate["id"],
                        base["hit"],
                        candidate["hit"],
                        base["neighbor"],
                        candidate["neighbor"],
                    )
                )

        print(
            f"\nM={weight:.2f} changed {len(changed)} probes"
        )

        for row in changed:
            print(
                f"  {row[0]}"
                f"  hit {row[1]} -> {row[2]}"
                f"  neighbor {row[3]} -> {row[4]}"
            )


if __name__ == "__main__":
    run()
