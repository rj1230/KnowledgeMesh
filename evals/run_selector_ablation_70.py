import importlib.util
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.WARNING)

dataset_path = Path("evals/datasets/inventory_retrieval_probes_70_semantically_approved_20261020.json")
output_path = Path("evals/results/inventory_retrieval_selector_ablation_70_20261023.json")
script_path = Path("evals/inventory_retrieval.py")

spec = importlib.util.spec_from_file_location("km_retrieval_benchmark", script_path)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

import app.agents.nodes.retriever as r

dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
original = bench._select_complementary_evidence

def variant_selector(query, reranked_documents, final_k):
    # Preserve the production selector's exact behavior as the starting point,
    # then test the marginal-coverage scoring signal in isolation.
    query_terms = r._tokenize(query)
    if final_k <= 0 or not reranked_documents:
        return []

    # First obtain the production result and its candidate pool.
    # We only use the variant when the normal selector would have candidates.
    # Candidate preparation mirrors the production pool eligibility rules.
    semantic = [
        dict(d) for d in reranked_documents
        if r._rerank_score(d) >= r.MIN_SEMANTIC_SCORE
    ]

    recovery = []
    for d in sorted(reranked_documents, key=r._dense_score, reverse=True):
        dense = r._dense_score(d)
        if dense < r.DENSE_RECOVERY_MIN_SCORE:
            break
        if r._rerank_score(d) >= r.MIN_SEMANTIC_SCORE:
            continue
        terms = r._tokenize(r._content(d))
        if r._query_coverage(query_terms, terms) < r.DENSE_RECOVERY_MIN_COVERAGE:
            continue
        copy = dict(d)
        copy["_dense_recovery"] = True
        recovery.append(copy)
        if len(recovery) >= r.DENSE_RECOVERY_K:
            break

    candidates, seen = [], set()
    for d in semantic + recovery:
        ident = r._document_identity(d)
        if ident not in seen:
            seen.add(ident)
            candidates.append(dict(d))

    if not candidates:
        return []

    terms_by_id = {
        r._document_identity(d): r._tokenize(r._content(d))
        for d in candidates
    }
    candidates.sort(
        key=lambda d: (r._rerank_score(d), r._dense_score(d)),
        reverse=True,
    )
    selected = [dict(candidates[0])]
    selected_ids = {r._document_identity(selected[0])}

    while len(selected) < min(final_k, len(candidates)):
        selected_terms = [
            terms_by_id[r._document_identity(d)] for d in selected
        ]
        scored = []

        for d in candidates:
            ident = r._document_identity(d)
            if ident in selected_ids:
                continue

            terms = terms_by_id[ident]
            sem = r._rerank_score(d)
            dense = r._dense_score(d)
            coverage = r._query_coverage(query_terms, terms)
            marginal = r._marginal_query_coverage(
                query_terms, terms, selected_terms
            )
            redundancy = r._max_redundancy(terms, selected_terms)
            is_recovery = bool(d.get("_dense_recovery", False))

            if is_recovery:
                score = (
                    dense * r.RECOVERY_WEIGHT_DENSE
                    + coverage * r.RECOVERY_WEIGHT_COVERAGE
                    + (1.0 - redundancy) * r.RECOVERY_WEIGHT_DIVERSITY
                    + sem * r.RECOVERY_WEIGHT_RERANK
                    + marginal * 0.10
                )
            else:
                score = (
                    sem * r.SEMANTIC_WEIGHT_RERANK
                    + coverage * r.SEMANTIC_WEIGHT_COVERAGE
                    + dense * r.SEMANTIC_WEIGHT_DENSE
                    + (1.0 - redundancy) * r.SEMANTIC_WEIGHT_DIVERSITY
                    + marginal * 0.10
                )

            scored.append((score, sem, coverage, dense, d, is_recovery))

        if not scored:
            break
        scored.sort(key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)
        _, sem, coverage, dense, best, is_recovery = scored[0]

        if is_recovery:
            if dense < r.DENSE_RECOVERY_MIN_SCORE or coverage < r.DENSE_RECOVERY_MIN_COVERAGE:
                break
        elif sem < r.MIN_SEMANTIC_SCORE:
            break

        selected.append(dict(best))
        selected_ids.add(r._document_identity(best))

    for d in selected:
        d.pop("_dense_recovery", None)

    selected.sort(
        key=lambda d: (r._rerank_score(d), r._dense_score(d)),
        reverse=True,
    )
    return selected[:final_k]

def evaluate(label, selector):
    bench._select_complementary_evidence = selector
    results = []
    selections = {}
    try:
        for probe in dataset:
            current_probe = probe["id"]

            def capture_selector(query=None, reranked_documents=None, final_k=5, **kwargs):
                if query is None:
                    query = kwargs.pop("query")
                if reranked_documents is None:
                    reranked_documents = kwargs.pop("reranked_documents")
                if "final_k" in kwargs:
                    final_k = kwargs.pop("final_k")

                chosen = selector(
                    query=query,
                    reranked_documents=reranked_documents,
                    final_k=final_k,
                )
                selections[current_probe] = [
                    {
                        "document_id": d.get("document_id"),
                        "chunk_id": d.get("chunk_id"),
                    }
                    for d in chosen
                ]
                return chosen

            bench._select_complementary_evidence = capture_selector
            result = bench.evaluate_probe(probe)
            results.append({
                "id": probe["id"],
                "hit_at_5": result["final"]["hit_at_5"],
                "neighbor_hit_at_5": result["final"].get("neighbor_hit_at_5"),
                "selected_chunks": selections.get(probe["id"], []),
                "error": result.get("error"),
            })
    finally:
        bench._select_complementary_evidence = original

    valid = [x for x in results if not x["error"]]
    summary = {
        "variant": label,
        "probes": len(results),
        "successful": len(valid),
        "errors": [x for x in results if x["error"]],
        "exact_hit_at_5": (
            sum(bool(x["hit_at_5"]) for x in valid) / len(valid)
            if valid else None
        ),
        "neighbor_hit_at_5": (
            sum(bool(x["neighbor_hit_at_5"]) for x in valid
                if x["neighbor_hit_at_5"] is not None)
            / sum(x["neighbor_hit_at_5"] is not None for x in valid)
            if any(x["neighbor_hit_at_5"] is not None for x in valid)
            else None
        ),
        "per_probe": results,
    }
    print(json.dumps(summary, indent=2))
    return summary

baseline = evaluate("current_selector", original)
variant = evaluate("marginal_coverage_bonus_0.10", variant_selector)

output = {"baseline": baseline, "variant": variant}
output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
print(f"\nSaved comparison: {output_path}")
