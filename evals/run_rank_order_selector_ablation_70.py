import importlib.util
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.WARNING)

dataset_path = Path(
    "evals/datasets/"
    "inventory_retrieval_probes_70_semantically_approved_20261020.json"
)
output_path = Path(
    "evals/results/"
    "inventory70_rank_order_selector_ablation_20261023.json"
)

spec = importlib.util.spec_from_file_location(
    "km_retrieval_benchmark",
    "evals/inventory_retrieval.py",
)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

import app.agents.nodes.retriever as r

dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
original = bench._select_complementary_evidence


def rank_order_selector(query, reranked_documents, final_k):
    """Keep production eligibility gates; remove greedy score selection."""
    if final_k <= 0 or not reranked_documents:
        return []

    query_terms = r._tokenize(query)

    # Preserve reranker order for semantic candidates.
    semantic = [
        dict(d)
        for d in reranked_documents
        if r._rerank_score(d) >= r.MIN_SEMANTIC_SCORE
    ]

    # Preserve the existing dense-recovery eligibility rules.
    recovery = []
    for d in sorted(
        reranked_documents,
        key=r._dense_score,
        reverse=True,
    ):
        dense = r._dense_score(d)
        if dense < r.DENSE_RECOVERY_MIN_SCORE:
            break
        if r._rerank_score(d) >= r.MIN_SEMANTIC_SCORE:
            continue

        coverage = r._query_coverage(
            query_terms,
            r._tokenize(r._content(d)),
        )
        if coverage < r.DENSE_RECOVERY_MIN_COVERAGE:
            continue

        copy = dict(d)
        copy["_dense_recovery"] = True
        recovery.append(copy)

        if len(recovery) >= r.DENSE_RECOVERY_K:
            break

    candidates = []
    seen = set()

    # Semantic results retain reranker order; recovery candidates follow.
    for d in semantic + recovery:
        identity = r._document_identity(d)
        if identity in seen:
            continue
        seen.add(identity)
        candidates.append(dict(d))

    selected = candidates[:final_k]

    for d in selected:
        d.pop("_dense_recovery", None)

    return selected


def evaluate(label, selector):
    results = []
    captured = {}
    errors = []

    for probe in dataset:
        probe_id = probe["id"]

        def capture(query, reranked_documents, final_k=5, **kwargs):
            chosen = selector(
                query=query,
                reranked_documents=reranked_documents,
                final_k=final_k,
            )
            captured[probe_id] = [
                {
                    "document_id": d.get("document_id"),
                    "chunk_id": d.get("chunk_id"),
                }
                for d in chosen
            ]
            return chosen

        bench._select_complementary_evidence = capture
        try:
            result = bench.evaluate_probe(probe)
            err = result.get("error")
            row = {
                "id": probe_id,
                "hit_at_5": result["final"]["hit_at_5"],
                "neighbor_hit_at_5": result["final"].get(
                    "neighbor_hit_at_5"
                ),
                "selected_chunks": captured.get(probe_id, []),
                "error": err,
            }
            results.append(row)
            if err:
                errors.append(row)
        except Exception as exc:
            row = {
                "id": probe_id,
                "error": f"{type(exc).__name__}: {exc}",
            }
            results.append(row)
            errors.append(row)

    valid = [x for x in results if not x.get("error")]
    neighbor_rows = [
        x for x in valid if x.get("neighbor_hit_at_5") is not None
    ]

    return {
        "variant": label,
        "probes": len(results),
        "successful": len(valid),
        "errors": errors,
        "exact_hit_at_5": (
            sum(bool(x["hit_at_5"]) for x in valid) / len(valid)
            if valid else None
        ),
        "neighbor_hit_at_5": (
            sum(bool(x["neighbor_hit_at_5"]) for x in neighbor_rows)
            / len(neighbor_rows)
            if neighbor_rows else None
        ),
        "per_probe": results,
    }


try:
    baseline = evaluate("current_production_selector", original)
    rank_order = evaluate("reranker_order_with_existing_gates", rank_order_selector)
finally:
    bench._select_complementary_evidence = original

output = {
    "dataset": str(dataset_path),
    "baseline": baseline,
    "rank_order_variant": rank_order,
}
output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

for name, result in (
    ("BASELINE", baseline),
    ("RANK-ORDER VARIANT", rank_order),
):
    print(f"\n{name}")
    print("Successful:", result["successful"], "/", result["probes"])
    print("Errors:", len(result["errors"]))
    print("Exact Hit@5:", result["exact_hit_at_5"])
    print("Neighbor Hit@5:", result["neighbor_hit_at_5"])

b = {x["id"]: x for x in baseline["per_probe"] if not x.get("error")}
v = {x["id"]: x for x in rank_order["per_probe"] if not x.get("error")}

print("\nCHANGED PROBES")
for pid in sorted(b.keys() & v.keys()):
    if b[pid]["hit_at_5"] != v[pid]["hit_at_5"]:
        print(
            pid,
            "baseline=", b[pid]["hit_at_5"],
            "rank_order=", v[pid]["hit_at_5"],
        )

print("\nSaved:", output_path)
