import json
from pathlib import Path

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

print()
print("=" * 150)
print("21-PROBE A_EXACT FAILURE MATRIX")
print("=" * 150)

print(
    f"{'Probe':24s} "
    f"{'Gold':8s} "
    f"{'D50':>5s} "
    f"{'R50':>5s} "
    f"{'F5':>5s} "
    f"{'N5':>5s} "
    f"{'D-Rank':>7s} "
    f"{'R-Rank':>7s} "
    f"{'FinalRank':>9s}"
)
print("-" * 150)

for item in trusted:
    result = by_id[item["id"]]

    gold_id = result["gold_ids"][0]
    dense = result["dense"]
    reranked = result["reranked"]
    final = result["final"]

    dense_rank = dense["first_relevant_rank"]
    rerank_rank = reranked["first_relevant_rank"]
    final_rank = final["first_relevant_rank"]

    print(
        f"{item['id']:24s} "
        f"{gold_id.split('::')[-1]:>8s} "
        f"{'Y' if dense['hit_at_50'] else 'N':>5s} "
        f"{'Y' if reranked['hit_at_50'] else 'N':>5s} "
        f"{'Y' if final['hit_at_5'] else 'N':>5s} "
        f"{'Y' if final['neighbor_hit_at_5'] else 'N':>5s} "
        f"{str(dense_rank) if dense_rank is not None else '-':>7s} "
        f"{str(rerank_rank) if rerank_rank is not None else '-':>7s} "
        f"{str(final_rank) if final_rank is not None else '-':>9s}"
    )

print()
print("=" * 150)
print("FAILURE CLASSIFICATION")
print("=" * 150)

dense_misses = []
selector_misses = []
successes = []

for item in trusted:
    result = by_id[item["id"]]

    dense = result["dense"]
    reranked = result["reranked"]
    final = result["final"]

    if not dense["hit_at_50"]:
        dense_misses.append(item["id"])
    elif dense["hit_at_50"] and not final["hit_at_5"]:
        selector_misses.append(item["id"])
    else:
        successes.append(item["id"])

print()
print(f"DENSE MISS ({len(dense_misses)}):")
for probe in dense_misses:
    result = by_id[probe]
    print(
        f"  {probe:24s} "
        f"gold={result['gold_ids'][0]} "
        f"dense_rank={result['dense']['first_relevant_rank']}"
    )

print()
print(f"SELECTOR LOSS ({len(selector_misses)}):")
for probe in selector_misses:
    result = by_id[probe]
    print(
        f"  {probe:24s} "
        f"gold={result['gold_ids'][0]} "
        f"dense_rank={result['dense']['first_relevant_rank']} "
        f"rerank_rank={result['reranked']['first_relevant_rank']} "
        f"neighbor={result['final']['neighbor_hit_at_5']}"
    )

print()
print(f"FULL SUCCESS ({len(successes)}):")
for probe in successes:
    result = by_id[probe]
    print(
        f"  {probe:24s} "
        f"dense_rank={result['dense']['first_relevant_rank']} "
        f"rerank_rank={result['reranked']['first_relevant_rank']} "
        f"final_rank={result['final']['first_relevant_rank']}"
    )

print()
print("=" * 150)
print("INTERPRETATION")
print("=" * 150)

print(
    "A_EXACT probes where dense retrieval found the gold but final@5 missed it "
    "are selector-loss candidates."
)
print(
    "A_EXACT probes where dense retrieval missed the gold are upstream dense-retrieval "
    "candidates and should not be attributed to the selector."
)
