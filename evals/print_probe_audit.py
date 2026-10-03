import json
from pathlib import Path

path = Path("evals/results/inventory_probe_audit.json")
data = json.loads(path.read_text(encoding="utf-8"))

for item in data:
    print("=" * 100)
    print(item["id"])
    print("Q:", item["question"])

    for gold in item["gold"]:
        print(
            f"GOLD: {gold.get('filename')} "
            f"chunk={gold.get('chunk_id')}"
        )

        for neighbor in gold.get("neighbors", []):
            text = " ".join(
                neighbor["text"].split()
            )

            print(
                f"  {neighbor['relative']:8s} "
                f"chunk={neighbor['chunk_id']:>3s}: "
                f"{text[:300]}"
            )
