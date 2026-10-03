from __future__ import annotations

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
    production_selector,
    identity,
    _tokenize,
    _query_coverage,
    _dense_score,
    _rerank_score,
    _content,
)


DATASET = (
    ROOT_DIR
    / "evals"
    / "datasets"
    / "inventory_retrieval_probes.json"
)

FINAL_K = 5
NEIGHBOR_WINDOW = 1


POLICIES = {
    "P0": None,
    "P1": {"dense": 0.78, "coverage": 0.35},
    "P2": {"dense": 0.80, "coverage": 0.35},
    "P3": {"dense": 0.78, "coverage": 0.40},
}


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
            target = int(chunk_id)
        except ValueError:
            continue

        for selected_id in selected_ids:
            selected_doc, selected_chunk = selected_id

            if selected_doc != document_id:
                continue

            try:
                candidate = int(selected_chunk)
            except ValueError:
                continue

            if abs(candidate - target) <= NEIGHBOR_WINDOW:
                return True

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


def protect_candidate(query, reranked, current, policy):
    if policy is None:
        return current, None

    query_terms = _tokenize(query)

    selected_ids = {
        identity(document)
        for document in current
    }

    eligible = []

    for document in reranked:
        doc_id = identity(document)

        if doc_id in selected_ids:
            continue

        dense = _dense_score(document)

        coverage = _query_coverage(
            query_terms,
            _tokenize(_content(document)),
        )

        if (
            dense >= policy["dense"]
            and coverage >= policy["coverage"]
        ):
            eligible.append(
                (
                    dense,
                    coverage,
                    _rerank_score(document),
                    document,
                )
            )

    if not eligible:
        return current, None

    # Strongest protected candidate:
    # dense first, then coverage, then rerank.
    eligible.sort(
        key=lambda item: (
            item[0],
            item[1],
            item[2],
        ),
        reverse=True,
    )

    protected = eligible[0][3]

    if not current:
        return [protected], {
            "protected": True,
            "protected_id": identity(protected),
            "replaced_id": None,
        }

    # Replace the weakest current selected item by rerank score.
    weakest_index = min(
        range(len(current)),
        key=lambda index: (
            _rerank_score(current[index]),
            _dense_score(current[index]),
        ),
    )

    replaced = current[weakest_index]

    updated = list(current)
    updated[weakest_index] = dict(protected)

    # Preserve production final ordering.
    updated.sort(
        key=lambda document: (
            _rerank_score(document),
            _dense_score(document),
        ),
        reverse=True,
    )

    return updated[:FINAL_K], {
        "protected": True,
        "protected_id": identity(protected),
        "replaced_id": identity(replaced),
        "protected_dense": _dense_score(protected),
        "protected_rerank": _rerank_score(protected),
        "protected_coverage": _query_coverage(
            query_terms,
            _tokenize(_content(protected)),
        ),
    }


def main():
    probes = json.loads(
        DATASET.read_text(encoding="utf-8")
    )

    results = {
        policy: {
            "hit": 0,
            "neighbor": 0,
            "rr": 0.0,
            "ndcg": 0.0,
            "protected": 0,
            "rescued": 0,
            "regressed": 0,
            "changes": [],
        }
        for policy in POLICIES
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

        baseline = production_selector(
            question,
            reranked,
        )

        baseline_hit = any(
            identity(document) in gold
            for document in baseline
        )

        for policy_name, policy in POLICIES.items():
            selected, diagnostic = protect_candidate(
                question,
                reranked,
                baseline,
                policy,
            )

            hit = any(
                identity(document) in gold
                for document in selected
            )

            neighbor = neighbor_hit(selected, gold)
            rr = reciprocal_rank(selected, gold)
            score = ndcg(selected, gold)

            results[policy_name]["hit"] += int(hit)
            results[policy_name]["neighbor"] += int(neighbor)
            results[policy_name]["rr"] += rr
            results[policy_name]["ndcg"] += score

            if diagnostic and diagnostic.get("protected"):
                results[policy_name]["protected"] += 1

            if not baseline_hit and hit:
                results[policy_name]["rescued"] += 1

            if baseline_hit and not hit:
                results[policy_name]["regressed"] += 1

            if policy_name != "P0":
                if baseline_hit != hit or (
                    neighbor_hit(baseline, gold) != neighbor
                ):
                    results[policy_name]["changes"].append(
                        {
                            "id": probe["id"],
                            "baseline_hit": baseline_hit,
                            "protected_hit": hit,
                            "baseline_neighbor": neighbor_hit(
                                baseline,
                                gold,
                            ),
                            "protected_neighbor": neighbor,
                            "diagnostic": diagnostic,
                        }
                    )

        print(
            f"[{index:02d}/{len(probes)}] {probe['id']}",
            flush=True,
        )

    total = len(probes)

    print("\n" + "=" * 95)
    print("PROTECTED-EVIDENCE COUNTERFACTUAL")
    print("=" * 95)

    print(
        f"{'Policy':<10}"
        f"{'Hit@5':<12}"
        f"{'Neighbor@5':<15}"
        f"{'MRR':<12}"
        f"{'nDCG@5':<12}"
        f"{'Protected':<12}"
        f"{'Rescued':<10}"
        f"{'Regressed':<10}"
    )

    for policy_name in POLICIES:
        result = results[policy_name]

        print(
            f"{policy_name:<10}"
            f"{result['hit'] / total:<12.3f}"
            f"{result['neighbor'] / total:<15.3f}"
            f"{result['rr'] / total:<12.3f}"
            f"{result['ndcg'] / total:<12.3f}"
            f"{result['protected']:<12}"
            f"{result['rescued']:<10}"
            f"{result['regressed']:<10}"
        )

    for policy_name in ("P1", "P2", "P3"):
        result = results[policy_name]

        print(
            f"\n{policy_name} changed "
            f"{len(result['changes'])} probes"
        )

        for change in result["changes"]:
            print(
                f"  {change['id']}"
                f"  hit {change['baseline_hit']}"
                f" -> {change['protected_hit']}"
                f"  neighbor "
                f"{change['baseline_neighbor']}"
                f" -> {change['protected_neighbor']}"
            )

            diagnostic = change["diagnostic"]

            if diagnostic:
                print(
                    f"    protected="
                    f"{diagnostic.get('protected_id')}"
                    f" dense="
                    f"{diagnostic.get('protected_dense', 0):.3f}"
                    f" coverage="
                    f"{diagnostic.get('protected_coverage', 0):.3f}"
                    f" rerank="
                    f"{diagnostic.get('protected_rerank', 0):.3f}"
                    f" replaced="
                    f"{diagnostic.get('replaced_id')}"
                )


if __name__ == "__main__":
    main()
