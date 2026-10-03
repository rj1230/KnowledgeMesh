import json
from pathlib import Path
from collections import Counter

audit = json.loads(
    Path("evals/results/inventory_probe_audit_labeled.json")
    .read_text(encoding="utf-8")
)

results = json.loads(
    Path("evals/results/inventory_retrieval_pilot.json")
    .read_text(encoding="utf-8")
)

by_id = {
    item["id"]: item
    for item in results["results"]
}

trusted = [
    item for item in audit
    if item["audit_label"] == "A_EXACT"
]

def avg(values):
    return sum(values) / len(values) if values else 0.0

dense_recall = []
dense_hit = []
rerank_recall = []
rerank_hit = []
final_hit = []
neighbor_hit = []
mrr = []
ndcg = []

print()
print("=" * 100)
print("TRUSTED A_EXACT RETRIEVAL BASELINE")
print("=" * 100)
print(f"Probes: {len(trusted)}")
print()

print(
    f"{'Probe':24s} "
    f"{'D50':>5s} "
    f"{'R50':>5s} "
    f"{'F5':>5s} "
    f"{'N5':>5s} "
    f"{'MRR':>6s} "
    f"{'nDCG':>6s}"
)
print("-" * 100)

for item in trusted:
    result = by_id[item["id"]]

    dense = result["dense"]
    reranked = result["reranked"]
    final = result["final"]

    dense_hit_value = bool(dense["hit_at_50"])
    rerank_hit_value = bool(reranked["hit_at_50"])
    final_hit_value = bool(final["hit_at_5"])
    neighbor_hit_value = bool(final["neighbor_hit_at_5"])

    dense_recall.append(float(dense["recall_at_50"]))
    dense_hit.append(float(dense_hit_value))
    rerank_recall.append(float(reranked["recall_at_50"]))
    rerank_hit.append(float(rerank_hit_value))
    final_hit.append(float(final_hit_value))
    neighbor_hit.append(float(neighbor_hit_value))
    mrr.append(float(final["mrr"]))
    ndcg.append(float(final["ndcg_at_5"]))

    print(
        f"{item['id']:24s} "
        f"{'Y' if dense_hit_value else 'N':>5s} "
        f"{'Y' if rerank_hit_value else 'N':>5s} "
        f"{'Y' if final_hit_value else 'N':>5s} "
        f"{'Y' if neighbor_hit_value else 'N':>5s} "
        f"{final['mrr']:6.3f} "
        f"{final['ndcg_at_5']:6.3f}"
    )

print()
print("=" * 100)
print("AGGREGATE")
print("=" * 100)

print(f"Dense Recall@50       : {avg(dense_recall):.3f}")
print(f"Dense Hit@50          : {avg(dense_hit):.3f}")
print(f"Reranked Recall@50    : {avg(rerank_recall):.3f}")
print(f"Reranked Hit@50       : {avg(rerank_hit):.3f}")
print(f"Final Hit@5           : {avg(final_hit):.3f}")
print(f"Final Neighbor Hit@5  : {avg(neighbor_hit):.3f}")
print(f"Final MRR             : {avg(mrr):.3f}")
print(f"Final nDCG@5          : {avg(ndcg):.3f}")

print()
print("=" * 100)
print("STAGE FAILURES")
print("=" * 100)

dense_misses = []
rerank_misses = []
selector_misses = []

for item in trusted:
    result = by_id[item["id"]]

    dense = result["dense"]
    reranked = result["reranked"]
    final = result["final"]

    if not dense["hit_at_50"]:
        dense_misses.append(item["id"])
    elif not reranked["hit_at_50"]:
        rerank_misses.append(item["id"])
    elif not final["hit_at_5"]:
        selector_misses.append(item["id"])

print(f"Dense misses       : {len(dense_misses)}")
for probe in dense_misses:
    print(f"  {probe}")

print()
print(f"Reranker losses    : {len(rerank_misses)}")
for probe in rerank_misses:
    print(f"  {probe}")

print()
print(f"Selector losses    : {len(selector_misses)}")
for probe in selector_misses:
    print(f"  {probe}")
