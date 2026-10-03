import json
from pathlib import Path

path = Path("evals/results/inventory_probe_audit.json")
data = json.loads(path.read_text(encoding="utf-8"))

review = {
    "inv_attention_001",
    "inv_llm_001",
    "inv_llm_003",
    "inv_llm_004",
    "inv_training_004",
    "inv_loop_001",
    "inv_loop_004",
}

for item in data:
    if item["id"] not in review:
        continue

    print("=" * 100)
    print(item["id"])
    print("QUESTION:", item["question"])

    for gold in item["gold"]:
        print(
            "\nGOLD:",
            gold.get("filename"),
            "chunk",
            gold.get("chunk_id"),
        )

        for neighbor in gold.get("neighbors", []):
            text = " ".join(
                neighbor["text"].split()
            )

            print(
                f"\n[{neighbor['relative'].upper()} "
                f"CHUNK {neighbor['chunk_id']}]"
            )
            print(text[:1200])
