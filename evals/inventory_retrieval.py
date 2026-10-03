from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.services.retrieval.qdrant_service import (
    search_enterprise_knowledge,
)
from app.services.retrieval.ranking_service import rerank_documents
from app.agents.nodes.retriever import _select_complementary_evidence


ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "evals" / "datasets" / "inventory_retrieval_probes.json"
OUTPUT = ROOT / "evals" / "results" / "inventory_retrieval_pilot.json"

DENSE_K = 50
RERANK_K = 50
FINAL_K = 5
NEIGHBOR_WINDOW = 1


def canonical_id(document_id, chunk_id):
    return f"{document_id}::{int(chunk_id)}"


def result_identity(result):
    document_id = result.get("document_id")
    chunk_id = result.get("chunk_id")

    if document_id is None or chunk_id is None:
        return None

    try:
        return canonical_id(document_id, chunk_id)
    except (TypeError, ValueError):
        return None


def unique_ids(results):
    output = []

    for result in results:
        identity = result_identity(result)

        if identity is not None and identity not in output:
            output.append(identity)

    return output


def dense_score(result):
    value = result.get("score")

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def rerank_score(result):
    value = result.get("rerank_score")

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def hit_at_k(ids, gold):
    return int(bool(set(ids[:k]) & gold))


def recall_at_k(ids, gold, k):
    if not gold:
        return 1.0

    return len(set(ids[:k]) & gold) / len(gold)


def reciprocal_rank(ids, gold):
    for rank, identity in enumerate(ids, start=1):
        if identity in gold:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(ids, gold, k):
    if not gold:
        return 1.0

    dcg = 0.0

    for rank, identity in enumerate(ids[:k], start=1):
        if identity in gold:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(gold), k)

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    return dcg / idcg if idcg else 0.0


def neighbor_hit(ids, gold, inventory_chunk_ids):
    """
    Count a retrieval as a neighbor hit when it retrieves a chunk
    immediately adjacent to a gold chunk from the same document.
    """
    if not gold:
        return False

    for identity in ids[:FINAL_K]:
        if identity in gold:
            return True

        if "::" not in identity:
            continue

        document_id, chunk_text = identity.rsplit("::", 1)

        try:
            chunk_id = int(chunk_text)
        except ValueError:
            continue

        for gold_identity in gold:
            if "::" not in gold_identity:
                continue

            gold_document, gold_chunk_text = gold_identity.rsplit("::", 1)

            if gold_document != document_id:
                continue

            try:
                gold_chunk = int(gold_chunk_text)
            except ValueError:
                continue

            if abs(chunk_id - gold_chunk) <= NEIGHBOR_WINDOW:
                return True

    return False


def first_relevant_rank(ids, gold):
    for rank, identity in enumerate(ids, start=1):
        if identity in gold:
            return rank

    return None


def summarize(values):
    if not values:
        return 0.0

    return sum(values) / len(values)


def evaluate_probe(probe):
    question = probe["question"]

    gold = {
        canonical_id(
            item["document_id"],
            item["chunk_id"],
        )
        for item in probe["relevant_chunks"]
    }

    started = time.perf_counter()

    # ==============================================================
    # Stage 1: production dense retrieval
    # ==============================================================
    dense_results = search_enterprise_knowledge(
        query=question,
        limit=DENSE_K,
    )

    dense_ms = (time.perf_counter() - started) * 1000

    dense_ids = unique_ids(dense_results)

    unknown_dense = sum(
        1
        for result in dense_results
        if result_identity(result) is None
    )

    # ==============================================================
    # Stage 2: production FlashRank reranking
    # ==============================================================
    rerank_started = time.perf_counter()

    reranked_results = rerank_documents(
        query=question,
        documents=dense_results,
        top_n=RERANK_K,
        text_key="content",
    )

    rerank_ms = (time.perf_counter() - rerank_started) * 1000

    reranked_ids = unique_ids(reranked_results)

    unknown_reranked = sum(
        1
        for result in reranked_results
        if result_identity(result) is None
    )

    # ==============================================================
    # Stage 3: production complementary evidence selector
    # ==============================================================
    selector_started = time.perf_counter()

    final_results = _select_complementary_evidence(
        query=question,
        reranked_documents=reranked_results,
        final_k=FINAL_K,
    )

    selector_ms = (time.perf_counter() - selector_started) * 1000

    final_ids = unique_ids(final_results)

    unknown_final = sum(
        1
        for result in final_results
        if result_identity(result) is None
    )

    total_ms = (time.perf_counter() - started) * 1000

    dense_rank = first_relevant_rank(dense_ids, gold)
    rerank_rank = first_relevant_rank(reranked_ids, gold)
    final_rank = first_relevant_rank(final_ids, gold)

    return {
        "id": probe["id"],
        "question": question,
        "category": probe.get("category"),
        "source_filename": probe.get("metadata", {}).get(
            "source_filename"
        ),
        "gold_ids": sorted(gold),

        "dense": {
            "count": len(dense_ids),
            "recall_at_50": recall_at_k(
                dense_ids,
                gold,
                DENSE_K,
            ),
            "hit_at_50": bool(set(dense_ids[:DENSE_K]) & gold),
            "first_relevant_rank": dense_rank,
            "unknown_id_count": unknown_dense,
            "top_ids": dense_ids[:DENSE_K],
            "scores": [
                {
                    "id": result_identity(result),
                    "score": dense_score(result),
                }
                for result in dense_results[:10]
            ],
        },

        "reranked": {
            "count": len(reranked_ids),
            "recall_at_50": recall_at_k(
                reranked_ids,
                gold,
                RERANK_K,
            ),
            "hit_at_50": bool(
                set(reranked_ids[:RERANK_K]) & gold
            ),
            "first_relevant_rank": rerank_rank,
            "unknown_id_count": unknown_reranked,
            "top_ids": reranked_ids[:RERANK_K],
            "scores": [
                {
                    "id": result_identity(result),
                    "rerank_score": rerank_score(result),
                }
                for result in reranked_results[:10]
            ],
        },

        "final": {
            "count": len(final_ids),
            "hit_at_5": bool(set(final_ids[:FINAL_K]) & gold),
            "neighbor_hit_at_5": neighbor_hit(
                final_ids,
                gold,
                None,
            ),
            "first_relevant_rank": final_rank,
            "mrr": reciprocal_rank(final_ids, gold),
            "ndcg_at_5": ndcg_at_k(
                final_ids,
                gold,
                FINAL_K,
            ),
            "unknown_id_count": unknown_final,
            "top_ids": final_ids[:FINAL_K],
            "scores": [
                {
                    "id": result_identity(result),
                    "score": dense_score(result),
                    "rerank_score": rerank_score(result),
                }
                for result in final_results
            ],
        },

        "timing_ms": {
            "dense": round(dense_ms, 2),
            "rerank": round(rerank_ms, 2),
            "selector": round(selector_ms, 2),
            "total": round(total_ms, 2),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="KnowledgeMesh inventory retrieval benchmark"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET,
        help="Probe dataset JSON path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT,
        help="Benchmark result JSON path",
    )
    args = parser.parse_args()

    dataset = (
        args.dataset
        if args.dataset.is_absolute()
        else ROOT / args.dataset
    )
    output = (
        args.output
        if args.output.is_absolute()
        else ROOT / args.output
    )

    probes = json.loads(
        dataset.read_text(encoding="utf-8")
    )

    results = []

    print()
    print("=" * 78)
    print("KnowledgeMesh Inventory Retrieval Pilot")
    print("=" * 78)
    print(f"Dataset : {dataset}")
    print(f"Probes  : {len(probes)}")
    print(
        f"Pipeline: Dense@{DENSE_K} -> "
        f"FlashRank@{RERANK_K} -> Final@{FINAL_K}"
    )
    print("=" * 78)
    print()

    for index, probe in enumerate(probes, start=1):
        print(
            f"[{index:02d}/{len(probes)}] "
            f"{probe['id']:<26}",
            end=" ",
            flush=True,
        )

        try:
            result = evaluate_probe(probe)
            results.append(result)

            dense_hit = "Y" if result["dense"]["hit_at_50"] else "N"
            rerank_hit = (
                "Y" if result["reranked"]["hit_at_50"] else "N"
            )
            final_hit = "Y" if result["final"]["hit_at_5"] else "N"
            neighbor = (
                "Y"
                if result["final"]["neighbor_hit_at_5"]
                else "N"
            )

            print(
                f"D50={dense_hit} "
                f"R50={rerank_hit} "
                f"F5={final_hit} "
                f"N5={neighbor} "
                f"RR={result['final']['mrr']:.3f} "
                f"{result['timing_ms']['total']:.0f}ms"
            )

        except Exception as exc:
            print(f"ERROR: {type(exc).__name__}: {exc}")

            results.append({
                "id": probe["id"],
                "question": probe["question"],
                "category": probe.get("category"),
                "source_filename": probe.get(
                    "metadata", {}
                ).get("source_filename"),
                "error": f"{type(exc).__name__}: {exc}",
            })

    successful = [
        result
        for result in results
        if "error" not in result
    ]

    errors = [
        result
        for result in results
        if "error" in result
    ]

    dense_recall = summarize([
        result["dense"]["recall_at_50"]
        for result in successful
    ])

    dense_hit = summarize([
        float(result["dense"]["hit_at_50"])
        for result in successful
    ])

    rerank_recall = summarize([
        result["reranked"]["recall_at_50"]
        for result in successful
    ])

    rerank_hit = summarize([
        float(result["reranked"]["hit_at_50"])
        for result in successful
    ])

    final_hit = summarize([
        float(result["final"]["hit_at_5"])
        for result in successful
    ])

    neighbor_hit_value = summarize([
        float(result["final"]["neighbor_hit_at_5"])
        for result in successful
    ])

    mrr = summarize([
        result["final"]["mrr"]
        for result in successful
    ])

    ndcg = summarize([
        result["final"]["ndcg_at_5"]
        for result in successful
    ])

    unknown_total = sum(
        result["dense"]["unknown_id_count"]
        + result["reranked"]["unknown_id_count"]
        + result["final"]["unknown_id_count"]
        for result in successful
    )

    retrieved_total = sum(
        result["dense"]["count"]
        + result["reranked"]["count"]
        + result["final"]["count"]
        for result in successful
    )

    unknown_rate = (
        unknown_total / retrieved_total
        if retrieved_total
        else 0.0
    )

    latency = [
        result["timing_ms"]["total"]
        for result in successful
    ]

    summary = {
        "dataset": str(dataset),
        "probes": len(probes),
        "successful": len(successful),
        "errors": len(errors),
        "pipeline": {
            "dense_k": DENSE_K,
            "rerank_k": RERANK_K,
            "final_k": FINAL_K,
            "neighbor_window": NEIGHBOR_WINDOW,
        },
        "metrics": {
            "dense_recall_at_50": dense_recall,
            "dense_hit_at_50": dense_hit,
            "reranked_recall_at_50": rerank_recall,
            "reranked_hit_at_50": rerank_hit,
            "final_hit_at_5": final_hit,
            "final_neighbor_hit_at_5": neighbor_hit_value,
            "final_mrr": mrr,
            "final_ndcg_at_5": ndcg,
            "unknown_id_rate": unknown_rate,
            "mean_latency_ms": summarize(latency),
            "p95_latency_ms": (
                sorted(latency)[
                    min(
                        len(latency) - 1,
                        max(0, math.ceil(len(latency) * 0.95) - 1),
                    )
                ]
                if latency
                else 0.0
            ),
        },
        "errors": errors,
        "results": results,
    }

    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("PILOT SUMMARY")
    print("=" * 78)
    print(f"Successful probes       : {len(successful)}/{len(probes)}")
    print(f"Errors                   : {len(errors)}")
    print()
    print(
        f"Dense Recall@50          : "
        f"{dense_recall:.3f}"
    )
    print(
        f"Dense Hit@50             : "
        f"{dense_hit:.3f}"
    )
    print(
        f"Reranked Recall@50       : "
        f"{rerank_recall:.3f}"
    )
    print(
        f"Reranked Hit@50          : "
        f"{rerank_hit:.3f}"
    )
    print(
        f"Final Hit@5              : "
        f"{final_hit:.3f}"
    )
    print(
        f"Final Neighbor Hit@5     : "
        f"{neighbor_hit_value:.3f}"
    )
    print(
        f"Final MRR                : "
        f"{mrr:.3f}"
    )
    print(
        f"Final nDCG@5             : "
        f"{ndcg:.3f}"
    )
    print(
        f"Unknown-ID rate         : "
        f"{unknown_rate:.4f}"
    )
    print(
        f"Mean latency             : "
        f"{summarize(latency):.1f} ms"
    )
    print(
        f"P95 latency              : "
        f"{summary['metrics']['p95_latency_ms']:.1f} ms"
    )
    print()
    print(f"Results: {output}")
    print("=" * 78)


if __name__ == "__main__":
    main()
