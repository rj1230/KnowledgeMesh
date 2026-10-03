from pathlib import Path

path = Path(".\evals\selector_diagnostic.py")
text = path.read_text(encoding="utf-8")

print("LINES:", len(text.splitlines()))

for i, line in enumerate(text.splitlines(), 1):
    if any(
        key in line
        for key in [
            "selection_steps",
            "selection_score",
            "chosen_score",
            "pre_sort_selected",
            "selected",
            "candidate_score",
        ]
    ):
        print(f"{i:4}: {line}")
