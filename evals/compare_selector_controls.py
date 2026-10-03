import json
from pathlib import Path

path = Path("evals/results/inventory_retrieval_pilot.json")
data = json.loads(path.read_text(encoding="utf-8"))

wanted = {
    "inv_loop_001",
    "inv_training_001",
    "inv_training_003",
    "inv_llm_agents_001",
}

for result in data["results"]:
    if result["id"] not in wanted:
        continue

    print()
    print("=" * 110)
    print(result["id"])
    print("=" * 110)

    print("DENSE RANK :", result["dense"]["first_relevant_rank"])
    print("RERANK RANK:", result["reranked"]["first_relevant_rank"])
    print("FINAL RANK :", result["final"]["first_relevant_rank"])
    print("FINAL HIT  :", result["final"]["hit_at_5"])

    print()
    print("FINAL TOP IDS:")
    for i, item in enumerate(result["final"]["top_ids"], 1):
        print(f"  {i}: {item}")

