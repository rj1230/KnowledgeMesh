"""Promote explicitly approved trajectory candidates into a new gold dataset.

This command never modifies the trusted datasets. Review labels must be
written by a human in a separate JSONL file.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
import sys

# Support direct execution from the repository's evals directory.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from app.evaluation.contracts import FAILURE_CATEGORIES
from evals.evaluator import ChunkInventory, canonical_chunk_key


ALLOWED_ROUTES = {"private", "web", "both", "abstain"}
FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}: invalid JSON on line {line_number}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"{path}: line {line_number} must contain an object"
                )
            records.append(record)
    return records


def _nonempty_strings(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{field} must contain non-empty strings only")
    return [item.strip() for item in value]


def _validate_chunks(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("relevant_chunks must be a list")
    result = []
    for index, chunk in enumerate(value):
        if not isinstance(chunk, dict):
            raise ValueError(f"relevant_chunks[{index}] must be an object")
        document_id = chunk.get("document_id")
        chunk_id = chunk.get("chunk_id")
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError(
                f"relevant_chunks[{index}].document_id must be non-empty"
            )
        if (
            isinstance(chunk_id, bool)
            or not isinstance(chunk_id, int)
            or chunk_id < 0
        ):
            raise ValueError(
                f"relevant_chunks[{index}].chunk_id must be a non-negative integer"
            )
        result.append({
            "document_id": document_id.strip(),
            "chunk_id": chunk_id,
        })
    if len({
        (item["document_id"], item["chunk_id"]) for item in result
    }) != len(result):
        raise ValueError("relevant_chunks contains duplicate references")
    return result


def _validate_web_sources(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError("expected_web_sources must be a list")
    result = []
    for index, source in enumerate(value):
        if not isinstance(source, dict):
            raise ValueError(f"expected_web_sources[{index}] must be an object")
        url = source.get("url")
        if not isinstance(url, str) or not url.strip():
            raise ValueError(
                f"expected_web_sources[{index}].url must be non-empty"
            )
        parsed = urlparse(url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(
                f"expected_web_sources[{index}].url must be an HTTP(S) URL"
            )
        result.append({"url": url.strip()})
    return result


def _validate_review(review: dict[str, Any]) -> dict[str, Any]:
    if review.get("review_status") != "approved":
        raise ValueError("candidate review_status must explicitly be 'approved'")

    case_id = review.get("id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("review id must be a non-empty string")

    question = review.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    points = _nonempty_strings(
        review.get("expected_answer_points"), "expected_answer_points"
    )

    route = review.get("expected_route")
    if not isinstance(route, str) or route.strip() not in ALLOWED_ROUTES:
        raise ValueError(
            "expected_route must be private, web, both, or abstain"
        )
    route = route.strip()

    chunks = _validate_chunks(review.get("relevant_chunks", []))
    sources = _validate_web_sources(review.get("expected_web_sources", []))
    domains = _nonempty_strings(
        review.get("expected_domains", []), "expected_domains"
    ) if review.get("expected_domains", []) else []

    if route in {"private", "both"} and not chunks:
        raise ValueError(f"{route} route requires relevant_chunks")
    if route in {"web", "both"} and not (sources or domains):
        raise ValueError(f"{route} route requires web sources or domains")

    citations = review.get("required_citation_ids", [])
    if not isinstance(citations, list) or any(
        not isinstance(item, str) or not item.strip() for item in citations
    ):
        raise ValueError("required_citation_ids must be a list of non-empty strings")

    expected_answer = review.get("expected_answer", "")
    if not isinstance(expected_answer, str):
        raise ValueError("expected_answer must be a string")

    return {
        "id": case_id.strip(),
        "question": question.strip(),
        "expected_answer": expected_answer.strip(),
        "expected_answer_points": points,
        "relevant_chunks": chunks,
        "expected_web_sources": sources,
        "expected_domains": domains,
        "expected_route": route,
        "required_citation_ids": list(dict.fromkeys(
            item.strip() for item in citations
        )),
        "category": "trajectory_regression",
    }


def promote_candidates(
    candidates_path: Path,
    reviews_path: Path,
    output_path: Path,
    existing_datasets: list[Path],
    inventory_path: Path | None = None,
) -> dict[str, int]:
    # Never overwrite a dataset, even if it is empty.
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")

    candidates = _read_jsonl(candidates_path)
    reviews = _read_jsonl(reviews_path)
    inventory = (
        ChunkInventory.from_csv(inventory_path)
        if inventory_path is not None
        else None
    )

    candidate_by_id: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        candidate_id = candidate.get("candidate_id")
        fingerprint = candidate.get("fingerprint")
        if (
            not isinstance(candidate_id, str)
            or not isinstance(fingerprint, str)
            or not FINGERPRINT_RE.fullmatch(fingerprint)
            or candidate_id != "cand_" + fingerprint[:16]
        ):
            raise ValueError("candidate has an invalid ID/fingerprint contract")
        if candidate.get("trajectory_status") not in {"failed", "recovered"}:
            raise ValueError(f"{candidate_id}: invalid trajectory status")
        if candidate.get("review_status") != "needs_review":
            raise ValueError(
                f"{candidate_id}: mined candidate must remain needs_review"
            )

        failure_taxonomy = candidate.get("failure_taxonomy")
        if not isinstance(failure_taxonomy, list) or not failure_taxonomy:
            raise ValueError(
                f"{candidate_id}: failure_taxonomy must be a non-empty list"
            )
        if any(
            not isinstance(item, str) or item not in FAILURE_CATEGORIES
            for item in failure_taxonomy
        ):
            raise ValueError(
                f"{candidate_id}: failure_taxonomy contains invalid categories"
            )
        if len(set(failure_taxonomy)) != len(failure_taxonomy):
            raise ValueError(
                f"{candidate_id}: failure_taxonomy contains duplicate categories"
            )

        if candidate_id in candidate_by_id:
            raise ValueError(f"duplicate candidate_id: {candidate_id}")
        candidate_by_id[candidate_id] = candidate

    review_by_candidate: dict[str, dict[str, Any]] = {}
    for review in reviews:
        candidate_id = review.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError("each review needs a non-empty candidate_id")
        if candidate_id not in candidate_by_id:
            raise ValueError(f"review references unknown candidate: {candidate_id}")
        if candidate_id in review_by_candidate:
            raise ValueError(f"duplicate review for candidate: {candidate_id}")
        review_by_candidate[candidate_id] = review

    # Validate all reviews before writing anything.
    promoted: list[dict[str, Any]] = []
    seen_new_ids: set[str] = set()
    for candidate_id in sorted(review_by_candidate):
        review = review_by_candidate[candidate_id]
        case = _validate_review(review)
        chunks = case["relevant_chunks"]
        if chunks and inventory is None:
            raise ValueError(
                f"{case['id']}: private gold references require --inventory"
            )
        for chunk in chunks:
            key = canonical_chunk_key(
                chunk["document_id"], chunk["chunk_id"]
            )
            if key is None or key not in inventory:
                raise ValueError(
                    f"{case['id']}: relevant chunk is absent from inventory: "
                    f"{chunk['document_id']}::{chunk['chunk_id']}"
                )
        if case["id"] in seen_new_ids:
            raise ValueError(f"duplicate promoted case ID: {case['id']}")
        seen_new_ids.add(case["id"])

        # Include only sanitized provenance, never the original query/answer.
        candidate = candidate_by_id[candidate_id]
        case["metadata"] = {
            "source": "human_approved_trajectory_candidate",
            "candidate_id": candidate_id,
            "fingerprint": candidate["fingerprint"],
            "trajectory_status": candidate["trajectory_status"],
            "failure_taxonomy": candidate.get("failure_taxonomy", []),
        }
        promoted.append(case)

    if not promoted:
        raise ValueError("no reviewed candidates to promote")

    existing_ids: set[str] = set()
    for dataset_path in existing_datasets:
        with dataset_path.open("r", encoding="utf-8-sig") as handle:
            dataset = json.load(handle)
        if not isinstance(dataset, list):
            raise ValueError(f"{dataset_path}: expected a JSON array")
        for item in dataset:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                existing_ids.add(item["id"])

    collisions = sorted(seen_new_ids & existing_ids)
    if collisions:
        raise ValueError(
            "promoted IDs collide with an existing benchmark: "
            + ", ".join(collisions)
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=output_path.name + ".", suffix=".tmp",
        dir=str(output_path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(promoted, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
        # Exclusive creation protects against a file appearing after preflight.
        with open(temp_name, "rb") as source:
            payload = source.read()
        with output_path.open("xb") as target:
            target.write(payload)
        os.unlink(temp_name)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise

    return {
        "candidate_records": len(candidates),
        "reviewed_candidates": len(reviews),
        "promoted_cases": len(promoted),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Promote explicitly approved candidates to a new dataset."
    )
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--existing-dataset", type=Path, action="append", default=[]
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        help="CSV inventory used to validate private gold chunk references",
    )
    args = parser.parse_args()

    try:
        counts = promote_candidates(
            args.candidates,
            args.reviews,
            args.output,
            args.existing_dataset,
            inventory_path=args.inventory,
        )
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Promotion refused: {exc}\n")

    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


