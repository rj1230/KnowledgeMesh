from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.ranking_service import rerank_documents
from evals import selector_shadow as selector

DATASET = ROOT / "evals/datasets/inventory_retrieval_probes_70_semantically_approved_20261020.json"
OUTPUT = ROOT / "evals/results/inventory70_recovery_gate_ablation_20261020.json"

CONFIGS = {
    "A_baseline": {"dense_threshold": 0.68, "coverage_threshold": 0.20},
    "B_dense_only": {"dense_threshold": 0.62, "coverage_threshold": 0.20},
    "C_coverage_only": {"dense_threshold": 0.68, "coverage_threshold": 0.10},
    "D_combined": {"dense_threshold": 0.62, "coverage_threshold": 0.10},
}

def summarize(rows):
    if not rows:
        return {"n": 0, "hit_at_5": None, "neighbor_hit_at_5": None,
                "mrr": None, "ndcg_at_5": None}

    n = len(rows)
    return {
        "n": n,
        "hit_at_5": sum(r["hit_at_5"] for r in rows) / n,
        "neighbor_hit_at_5": sum(r["neighbor_hit_at_5"] for r in rows) / n,
        "mrr": statistics.mean(r["mrr"] for r in rows),
        "ndcg_at_5": statistics.mean(r["ndcg_at_5"] for r in rows),
    }

def canonical(identity):
    if identity is None:
        return None
    document_id, chunk_id = identity
    return f"{document_id}::{chunk_id}"

def main():
    if not DATASET.exists():
        raise FileNotFoundError(f"Approved dataset not found: {DATASET}")

    probes = json.loads(DATASET.read_text(encoding="utf-8"))
    if not isinstance(probes, list) or not probes:
        raise ValueError("Expected a non-empty JSON list of probes.")

    original_dense = selector.DENSE_RECOVERY_MIN_SCORE
    original_coverage = selector.DENSE_RECOVERY_MIN_COVERAGE

    all_rows = []
    errors = []
    started_all = time.perf_counter()

    try:
        for index, probe in enumerate(probes, start=1):
            probe_id = probe["id"]
            query = probe["question"]
            probe_started = time.perf_counter()

            try:
                dense = search_enterprise_knowledge(query=query, limit=50)
                reranked = rerank_documents(
                    query=query,
                    documents=dense,
                    top_n=50,
                    text_key="content",
                )
                gold = selector.gold_ids(probe)

                # Every configuration receives the same reranked candidate pool.
                for config_name, config in CONFIGS.items():
                    selector.DENSE_RECOVERY_MIN_SCORE = config["dense_threshold"]
                    selector.DENSE_RECOVERY_MIN_COVERAGE = config["coverage_threshold"]

                    selected = selector.production_selector(query, reranked)
                    metrics = selector.evaluate(selected, gold)

                    all_rows.append({
                        "id": probe_id,
                        "config": config_name,
                        "hit_at_5": metrics["hit"],
                        "neighbor_hit_at_5": metrics["neighbor"],
                        "mrr": metrics["rr"],
                        "ndcg_at_5": metrics["ndcg"],
                        "gold_ids": sorted(canonical(x) for x in gold),
                        "selected_ids": [
                            canonical(selector.identity(x)) for x in selected
                        ],
                        "candidate_count": len(reranked),
                    })

                elapsed = time.perf_counter() - probe_started
                print(f"[{index}/{len(probes)}] {probe_id}: "
                      f"{len(reranked)} candidates; {elapsed:.2f}s")

            except Exception as exc:
                errors.append({
                    "id": probe_id,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                print(f"[{index}/{len(probes)}] ERROR {probe_id}: {exc}")

    finally:
        selector.DENSE_RECOVERY_MIN_SCORE = original_dense
        selector.DENSE_RECOVERY_MIN_COVERAGE = original_coverage

    results_by_config = {
        name: [r for r in all_rows if r["config"] == name]
        for name in CONFIGS
    }
    summary = {
        name: summarize(rows)
        for name, rows in results_by_config.items()
    }

    by_config_probe = {
        name: {r["id"]: r for r in rows}
        for name, rows in results_by_config.items()
    }
    paired = {}

    baseline = by_config_probe["A_baseline"]
    for name in list(CONFIGS)[1:]:
        candidate = by_config_probe[name]
        common = sorted(set(baseline) & set(candidate))
        wins = [
            pid for pid in common
            if not baseline[pid]["hit_at_5"] and candidate[pid]["hit_at_5"]
        ]
        regressions = [
            pid for pid in common
            if baseline[pid]["hit_at_5"] and not candidate[pid]["hit_at_5"]
        ]
        paired[name] = {
            "paired_n": len(common),
            "miss_to_hit": wins,
            "hit_to_miss": regressions,
            "net_hit_change": len(wins) - len(regressions),
        }

    artifact = {
        "experiment": "recovery_threshold_2x2_ablation",
        "dataset": str(DATASET.relative_to(ROOT)),
        "dataset_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
        "candidate_pool_policy": (
            "One live dense retrieval and one rerank per probe; "
            "same reranked list reused for all four threshold configurations."
        ),
        "pipeline": {"dense_k": 50, "rerank_k": 50, "final_k": 5},
        "fixed_weights": {
            "semantic_rerank": selector.SEMANTIC_WEIGHT_RERANK,
            "semantic_coverage": selector.SEMANTIC_WEIGHT_COVERAGE,
            "semantic_dense": selector.SEMANTIC_WEIGHT_DENSE,
            "semantic_diversity": selector.SEMANTIC_WEIGHT_DIVERSITY,
            "recovery_dense": selector.RECOVERY_WEIGHT_DENSE,
            "recovery_coverage": selector.RECOVERY_WEIGHT_COVERAGE,
            "recovery_diversity": selector.RECOVERY_WEIGHT_DIVERSITY,
            "recovery_rerank": selector.RECOVERY_WEIGHT_RERANK,
            "min_semantic_score": selector.MIN_SEMANTIC_SCORE,
            "dense_recovery_k": selector.DENSE_RECOVERY_K,
        },
        "configurations": CONFIGS,
        "summary": summary,
        "paired_vs_baseline": paired,
        "errors": errors,
        "elapsed_seconds": round(time.perf_counter() - started_all, 2),
        "results": all_rows,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== ABLATION SUMMARY ===")
    for name, metrics in summary.items():
        print(name, json.dumps(metrics))
    print("\n=== PAIRED CHANGES VS BASELINE ===")
    for name, metrics in paired.items():
        print(name, json.dumps(metrics))
    print(f"\nErrors: {len(errors)} / {len(probes)} probes")
    print(f"Artifact: {OUTPUT.relative_to(ROOT)}")
    print(f"Elapsed: {artifact['elapsed_seconds']} seconds")

    if errors:
        print("WARNING: inspect errors before interpreting comparisons.")
    if any(summary[name]["n"] != len(probes) for name in CONFIGS):
        print("WARNING: incomplete results; do not promote thresholds.")

if __name__ == "__main__":
    main()
