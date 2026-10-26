import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DATASET = Path("evals/datasets/inventory_retrieval_probes_70.json")
RESULTS = Path("evals/results")
OUTPUT = RESULTS / "inventory70_semantic_review_decisions_consolidated_20261020.json"
COLLECTION = "enterprise_rag_bge_base"
EXPECTED_SHA = "21ee63518d16f2b08d8e5c1e452a4cfa8f5675b5f0322727f623c1aa2ac3e101"

def fail(message):
    raise RuntimeError(message)

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def get_probes(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("probes", "items", "data"):
            if isinstance(obj.get(key), list):
                return obj[key]
    fail("Unrecognized canonical dataset structure.")

def signature(decision):
    refs = decision.get("resolved_live_references")
    if not isinstance(refs, list) or not refs:
        fail(f"{decision.get('probe_id')}: missing live references")

    normalized = []
    for ref in refs:
        required = ("source", "document_id", "chunk_id", "point_id")
        missing = [key for key in required if ref.get(key) is None]
        if missing:
            fail(f"{decision.get('probe_id')}: reference missing {missing}")

        normalized.append((
            str(ref["source"]),
            str(ref["document_id"]),
            str(ref["chunk_id"]),
            str(ref["point_id"]),
        ))
    return tuple(sorted(normalized))

def preference(filename):
    if "revision4" in filename:
        return 4
    if "revision3" in filename:
        return 3
    return 1

def main():
    if OUTPUT.exists():
        fail(f"Refusing to overwrite existing output: {OUTPUT}")

    if not DATASET.exists():
        fail(f"Canonical dataset not found: {DATASET}")

    actual_sha = sha256(DATASET)
    if actual_sha != EXPECTED_SHA:
        fail(f"Canonical SHA mismatch: {actual_sha}")

    raw_dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    probes = get_probes(raw_dataset)
    if len(probes) != 70:
        fail(f"Expected 70 canonical probes, found {len(probes)}")

    canonical = {}
    for probe in probes:
        pid = str(probe.get("probe_id", probe.get("id", "")))
        if not pid or pid in canonical:
            fail(f"Missing or duplicate canonical probe ID: {pid!r}")
        canonical[pid] = probe

    trusted = {}
    excluded = []
    scanned = 0

    files = sorted(RESULTS.glob(
        "inventory70_semantic_review_decisions_batch*_20261020.json"
    ))

    for path in files:
        scanned += 1
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            excluded.append((path.name, f"JSON parse error: {exc}"))
            continue

        artifact_sha = (
            artifact.get("source_dataset_sha256")
            or artifact.get("dataset_sha256")
        )
        if artifact_sha != EXPECTED_SHA:
            excluded.append((path.name, "missing or mismatched dataset SHA"))
            continue

        decisions = artifact.get("decisions", [])
        if not isinstance(decisions, list):
            excluded.append((path.name, "decisions is not a list"))
            continue

        for decision in decisions:
            if not isinstance(decision, dict):
                continue

            if not (
                decision.get("semantic_approval") is True
                and decision.get("semantic_review_status")
                    == "SEMANTICALLY_APPROVED"
                and decision.get("reference_verification_status") == "PASS"
                and not decision.get("reference_issues")
            ):
                continue

            pid = str(decision.get("probe_id", ""))
            if pid not in canonical:
                continue

            expected_probe = canonical[pid]
            expected_question = expected_probe.get("question")
            expected_source = expected_probe.get(
                "source_filename",
                expected_probe.get("source")
            )

            if decision.get("question") != expected_question:
                fail(f"{path.name}: canonical question mismatch for {pid}")

            actual_source = decision.get("source_filename")
            if expected_source is not None and actual_source != expected_source:
                fail(f"{path.name}: canonical source mismatch for {pid}")

            if not decision.get("resolved_live_references"):
                continue

            sig = signature(decision)
            trusted.setdefault(pid, []).append({
                "filename": path.name,
                "decision": decision,
                "signature": sig,
                "preference": preference(path.name),
            })

    print("Review files scanned:", scanned)
    print("Excluded files:")
    for name, reason in excluded:
        print(f"  - {name}: {reason}")

    selected = {}
    duplicate_count = 0

    for pid, entries in trusted.items():
        signatures = {entry["signature"] for entry in entries}
        if len(signatures) != 1:
            origins = [entry["filename"] for entry in entries]
            fail(f"Conflicting approved references for {pid}: {origins}")

        entries.sort(
            key=lambda entry: (
                entry["preference"],
                entry["filename"]
            ),
            reverse=True,
        )
        selected[pid] = entries[0]
        duplicate_count += len(entries) - 1

    missing = sorted(set(canonical) - set(selected))
    if missing:
        fail(
            f"Coverage incomplete: {len(selected)}/70 approved. "
            f"Missing probe IDs: {missing}"
        )

    if len(selected) != 70:
        fail(f"Expected 70 selected probes, got {len(selected)}")

    # A Qdrant point can support multiple probes. Validate its identity
    # using source/document/chunk, not the probe ID.
    expected_points = {}
    for pid, entry in selected.items():
        for source, doc_id, chunk_id, point_id in entry["signature"]:
            expected = (source, doc_id, chunk_id)
            if point_id in expected_points and expected_points[point_id] != expected:
                fail(f"Point ID has conflicting reference metadata: {point_id}")
            expected_points[point_id] = expected

    from app.services.retrieval.qdrant_service import client

    found = {}
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION,
            limit=128,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            point_id = str(point.id)
            if point_id not in expected_points:
                continue
            payload = point.payload or {}
            found[point_id] = (
                str(payload.get("source")),
                str(payload.get("document_id")),
                str(payload.get("chunk_id")),
            )

        if next_offset is None:
            break
        offset = next_offset

    live_errors = []
    for point_id, expected in expected_points.items():
        actual = found.get(point_id)
        if actual != expected:
            live_errors.append(
                f"{point_id}: expected={expected}, actual={actual}"
            )

    if live_errors:
        fail("Live Qdrant validation failed:\n" + "\n".join(live_errors[:20]))

    decisions = []
    for pid in canonical:
        entry = selected[pid]
        decision = dict(entry["decision"])
        decision["consolidation_selected_from"] = entry["filename"]
        decisions.append(decision)

    template = json.loads(
        (RESULTS / selected[next(iter(selected))]["filename"])
        .read_text(encoding="utf-8")
    )

    result = {
        **template,
        "artifact_type": "inventory70_semantic_review_decisions_consolidated",
        "created_date": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(DATASET).replace("\\", "/"),
        "source_dataset_sha256": EXPECTED_SHA,
        "dataset_sha256": EXPECTED_SHA,
        "collection_name": COLLECTION,
        "candidate_count": len(decisions),
        "unique_probe_count": len(decisions),
        "semantic_approvals": len(decisions),
        "reference_issue_count": 0,
        "reference_issue_probe_ids": [],
        "benchmark_ready": False,
        "decisions": decisions,
        "consolidation_metadata": {
            "source_review_files_scanned": scanned,
            "selected_unique_probes": len(decisions),
            "duplicate_approved_decisions_deduplicated": duplicate_count,
            "unique_live_points_validated": len(expected_points),
            "live_qdrant_reference_errors": 0,
            "excluded_files": [
                {"filename": name, "reason": reason}
                for name, reason in excluded
            ],
            "canonical_dataset_sha256": EXPECTED_SHA,
            "benchmark_ready": False,
            "note": (
                "Semantic reference consolidation only. "
                "This artifact does not certify benchmark readiness "
                "or retrieval performance."
            ),
        },
    }

    # Exclusive creation: never overwrite an existing artifact.
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    # Reopen and verify the written artifact and canonical dataset.
    saved = json.loads(OUTPUT.read_text(encoding="utf-8"))
    saved_ids = [d.get("probe_id") for d in saved["decisions"]]

    if len(saved_ids) != 70 or len(set(saved_ids)) != 70:
        fail("Post-write check failed: output does not contain 70 unique probes")
    if set(saved_ids) != set(canonical):
        fail("Post-write check failed: output probe coverage differs from canonical")
    if saved.get("benchmark_ready") is not False:
        fail("Post-write check failed: benchmark_ready must remain false")
    if sha256(DATASET) != EXPECTED_SHA:
        fail("Canonical dataset hash changed during consolidation")

    print("\nCONSOLIDATION SUCCEEDED")
    print("Canonical probes:", len(canonical))
    print("Selected approvals:", len(decisions))
    print("Deduplicated repeated approvals:", duplicate_count)
    print("Unique live Qdrant points checked:", len(expected_points))
    print("Live-reference errors:", len(live_errors))
    print("Output:", OUTPUT)
    print("Output SHA-256:", sha256(OUTPUT))
    print("Canonical dataset SHA unchanged:", True)
    print("benchmark_ready:", saved["benchmark_ready"])
    print("NEXT: run the independent final audit before claiming readiness.")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nCONSOLIDATION FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
