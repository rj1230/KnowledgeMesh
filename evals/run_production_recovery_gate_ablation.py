from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from pathlib import Path

from evals import selector_shadow as metrics
from app.agents.nodes import retriever as production
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.ranking_service import rerank_documents

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evals/datasets/inventory_retrieval_probes_70_semantically_approved_20261020.json"
CACHE = ROOT / "evals/results/inventory70_production_candidate_cache_20261020.json"
OUTPUT = ROOT / "evals/results/inventory70_production_recovery_gate_ablation_20261020.json"

CONFIGS = {
    "A_baseline": {"dense_threshold": 0.68, "coverage_threshold": 0.20},
    "B_dense_only": {"dense_threshold": 0.62, "coverage_threshold": 0.20},
    "C_coverage_only": {"dense_threshold": 0.68, "coverage_threshold": 0.10},
    "D_combined": {"dense_threshold": 0.62, "coverage_threshold": 0.10},
}


def summarize(rows):
    n = len(rows)
    if not n:
        return {"n": 0, "hit_at_5": None, "neighbor_hit_at_5": None,
                "mrr": None, "ndcg_at_5": None}
    return {
        "n": n,
        "hit_at_5": sum(r["hit_at_5"] for r in rows) / n,
        "neighbor_hit_at_5": sum(r["neighbor_hit_at_5"] for r in rows) / n,
        "mrr": statistics.mean(r["mrr"] for r in rows),
        "ndcg_at_5": statistics.mean(r["ndcg_at_5"] for r in rows),
    }


def canonical(document):
    identity = production._document_identity(document)
    return f"{identity[0]}::{identity[1]}"


def paired_significance(base_rows, candidate_rows):
    common = sorted(set(base_rows) & set(candidate_rows))
    wins = [
        pid for pid in common
        if not base_rows[pid]["hit_at_5"] and candidate_rows[pid]["hit_at_5"]
    ]
    losses = [
        pid for pid in common
        if base_rows[pid]["hit_at_5"] and not candidate_rows[pid]["hit_at_5"]
    ]

    # Exact two-sided McNemar test via the binomial distribution.
    discordant = len(wins) + len(losses)
    if discordant == 0:
        p_value = 1.0
    else:
        k = min(len(wins), len(losses))
        p_value = min(
            1.0,
            2.0 * sum(
                math.comb(discordant, i) for i in range(k + 1)
            ) / (2 ** discordant),
        )

    return {
        "paired_n": len(common),
        "miss_to_hit": wins,
        "hit_to_miss": losses,
        "net_hit_change": len(wins) - len(losses),
        "mcnemar_exact_two_sided_p": p_value,
    }


def main():
    # Refuse accidental overwrite, including on reruns.
    for path in (CACHE, OUTPUT):
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing artifact: {path}"
            )

    if not DATASET.exists():
        raise FileNotFoundError(DATASET)

    probes = json.loads(DATASET.read_text(encoding="utf-8"))
    if not isinstance(probes, list) or not probes:
        raise ValueError("Expected a non-empty list of probes.")

    cache_rows = []
    errors = []
    started = time.perf_counter()

    print("=== PHASE 1: RETRIEVE, RERANK, CACHE ===")
    for index, probe in enumerate(probes, start=1):
        probe_id = probe["id"]
        query = probe["question"]
        t0 = time.perf_counter()
        try:
            dense = search_enterprise_knowledge(query=query, limit=50)
            reranked = rerank_documents(
                query=query, documents=dense, top_n=50, text_key="content"
            )
            # Ensure JSON serialization is possible before proceeding.
            json.dumps(reranked, ensure_ascii=False)
            cache_rows.append({
                "id": probe_id,
                "question": query,
                "relevant_chunks": probe["relevant_chunks"],
                "reranked_documents": reranked,
            })
            print(
                f"[{index}/{len(probes)}] {probe_id}: "
                f"{len(reranked)} candidates, {time.perf_counter()-t0:.2f}s"
            )
        except Exception as exc:
            errors.append({
                "id": probe_id,
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"[{index}/{len(probes)}] ERROR {probe_id}: {exc}")

    if errors or len(cache_rows) != len(probes):
        raise RuntimeError(
            f"Cache incomplete ({len(cache_rows)}/{len(probes)}); "
            f"{len(errors)} errors. No ablation run. Fix errors before retrying."
        )

    cache_artifact = {
        "experiment": "production_selector_candidate_cache",
        "dataset": str(DATASET.relative_to(ROOT)),
        "dataset_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
        "retrieval": {"dense_k": 50, "rerank_k": 50},
        "results": cache_rows,
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("x", encoding="utf-8") as f:
        json.dump(cache_artifact, f, indent=2, ensure_ascii=False)

    print(f"\nCandidate cache saved: {CACHE.relative_to(ROOT)}")
    print("\n=== PHASE 2: ACTUAL PRODUCTION SELECTOR ===")

    original_dense = production.DENSE_RECOVERY_MIN_SCORE
    original_coverage = production.DENSE_RECOVERY_MIN_COVERAGE
    all_rows = []

    try:
        for config_name, config in CONFIGS.items():
            production.DENSE_RECOVERY_MIN_SCORE = config["dense_threshold"]
            production.DENSE_RECOVERY_MIN_COVERAGE = config["coverage_threshold"]

            for item in cache_rows:
                selected = production._select_complementary_evidence(
                    query=item["question"],
                    reranked_documents=item["reranked_documents"],
                    final_k=5,
                )
                gold = {
                    (str(x["document_id"]), str(x["chunk_id"]))
                    for x in item["relevant_chunks"]
                }
                evaluation = metrics.evaluate(selected, gold)
                all_rows.append({
                    "id": item["id"],
                    "config": config_name,
                    "hit_at_5": evaluation["hit"],
                    "neighbor_hit_at_5": evaluation["neighbor"],
                    "mrr": evaluation["rr"],
                    "ndcg_at_5": evaluation["ndcg"],
                    "gold_ids": sorted(f"{a}::{b}" for a, b in gold),
                    "selected_ids": [canonical(d) for d in selected],
                    "selected_count": len(selected),
                    "candidate_count": len(item["reranked_documents"]),
                })
    finally:
        production.DENSE_RECOVERY_MIN_SCORE = original_dense
        production.DENSE_RECOVERY_MIN_COVERAGE = original_coverage

    results_by_config = {
        name: [r for r in all_rows if r["config"] == name]
        for name in CONFIGS
    }
    summary = {
        name: summarize(rows) for name, rows in results_by_config.items()
    }
    by_config_probe = {
        name: {r["id"]: r for r in rows}
        for name, rows in results_by_config.items()
    }
    baseline = by_config_probe["A_baseline"]
    paired = {
        name: paired_significance(
            baseline, by_config_probe[name]
        )
        for name in list(CONFIGS)[1:]
    }

    expected_rows = len(probes) * len(CONFIGS)
    if len(all_rows) != expected_rows:
        raise RuntimeError(
            f"Unexpected row count: {len(all_rows)} != {expected_rows}"
        )
    if any(r["selected_count"] > 5 for r in all_rows):
        raise RuntimeError("Selector returned more than five documents.")

    artifact = {
        "experiment": "production_recovery_threshold_2x2_ablation",
        "selector": "app.agents.nodes.retriever._select_complementary_evidence",
        "dataset": str(DATASET.relative_to(ROOT)),
        "dataset_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
        "candidate_cache": str(CACHE.relative_to(ROOT)),
        "candidate_cache_sha256": hashlib.sha256(CACHE.read_bytes()).hexdigest(),
        "candidate_pool_policy": (
            "One retrieval and rerank per probe; identical cached candidate "
            "documents reused for every configuration."
        ),
        "pipeline": {"dense_k": 50, "rerank_k": 50, "final_k": 5},
        "fixed_weights": {
            name: getattr(production, name)
            for name in (
                "SEMANTIC_WEIGHT_RERANK", "SEMANTIC_WEIGHT_COVERAGE",
                "SEMANTIC_WEIGHT_DENSE", "SEMANTIC_WEIGHT_DIVERSITY",
                "RECOVERY_WEIGHT_DENSE", "RECOVERY_WEIGHT_COVERAGE",
                "RECOVERY_WEIGHT_DIVERSITY", "RECOVERY_WEIGHT_RERANK",
                "MIN_SEMANTIC_SCORE", "DENSE_RECOVERY_K",
            )
        },
        "configurations": CONFIGS,
        "summary": summary,
        "paired_vs_baseline": paired,
        "errors": errors,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "results": all_rows,
    }

    with OUTPUT.open("x", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)

    print("\n=== PRODUCTION ABLATION SUMMARY ===")
    for name, result in summary.items():
        print(name, json.dumps(result))
    print("\n=== PAIRED RESULTS VS BASELINE ===")
    for name, result in paired.items():
        print(name, json.dumps(result))
    print(f"\nCache: {CACHE.relative_to(ROOT)}")
    print(f"Results: {OUTPUT.relative_to(ROOT)}")
    print(f"Completed: {len(all_rows)} selector evaluations")
    print(f"Elapsed: {artifact['elapsed_seconds']} seconds")


if __name__ == "__main__":
    main()
