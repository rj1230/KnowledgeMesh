from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DATASET = (
    ROOT_DIR
    / "evals"
    / "datasets"
    / "inventory_retrieval_probes.json"
)

INVENTORY = (
    ROOT_DIR
    / "evals"
    / "true_chunk_inventory.csv"
)

OUTPUT = (
    ROOT_DIR
    / "evals"
    / "results"
    / "inventory_probe_audit.json"
)


def load_inventory():
    rows = {}

    with INVENTORY.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            document_id = str(row["DocumentId"]).strip()
            chunk_id = str(row["ChunkId"]).strip()

            rows[(document_id, chunk_id)] = {
                "document_id": document_id,
                "filename": row["Filename"],
                "chunk_id": chunk_id,
                "characters": row["Characters"],
                "text": row["Text"],
            }

    return rows


def surrounding_chunks(inventory, document_id, chunk_id):
    try:
        chunk_number = int(chunk_id)
    except ValueError:
        return []

    neighbors = []

    for offset in (-1, 0, 1):
        candidate = str(chunk_number + offset)

        item = inventory.get(
            (document_id, candidate)
        )

        if item:
            neighbors.append(
                {
                    "relative": (
                        "previous"
                        if offset == -1
                        else "gold"
                        if offset == 0
                        else "next"
                    ),
                    "chunk_id": candidate,
                    "text": item["text"],
                    "characters": item["characters"],
                }
            )

    return neighbors


def main():
    probes = json.loads(
        DATASET.read_text(encoding="utf-8")
    )

    inventory = load_inventory()

    output = []

    for probe in probes:
        gold_entries = []

        for gold in probe["relevant_chunks"]:
            document_id = str(
                gold["document_id"]
            ).strip()

            chunk_id = str(
                gold["chunk_id"]
            ).strip()

            item = inventory.get(
                (document_id, chunk_id)
            )

            if item is None:
                gold_entries.append(
                    {
                        "document_id": document_id,
                        "chunk_id": chunk_id,
                        "missing": True,
                    }
                )
                continue

            gold_entries.append(
                {
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "filename": item["filename"],
                    "characters": item["characters"],
                    "text": item["text"],
                    "neighbors": surrounding_chunks(
                        inventory,
                        document_id,
                        chunk_id,
                    ),
                }
            )

        output.append(
            {
                "id": probe["id"],
                "question": probe["question"],
                "category": probe.get("category"),
                "metadata": probe.get("metadata", {}),
                "gold": gold_entries,
                "audit_label": None,
                "audit_reason": None,
                "neighbor_relevant": None,
                "gold_answer_bearing": None,
                "notes": None,
            }
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"Created audit artifact: {OUTPUT}"
    )
    print(
        f"Probes: {len(output)}"
    )
    print(
        "Each probe contains gold + previous/current/next chunks."
    )


if __name__ == "__main__":
    main()
