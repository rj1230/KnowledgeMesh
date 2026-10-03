import json
from pathlib import Path

data = json.loads(
    Path("evals/results/inventory_retrieval_pilot.json")
    .read_text(encoding="utf-8")
)

print("TOP-LEVEL KEYS:")
print(list(data.keys()))

print()
print("FIRST RESULT KEYS:")
print(list(data["results"][0].keys()))

print()
print("FIRST RESULT:")
print(json.dumps(data["results"][0], indent=2, ensure_ascii=False))
