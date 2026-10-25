"""Mine privacy-conscious regression-test candidates from trajectory JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from app.evaluation.contracts import (
    FAILURE_CATEGORIES,
    SAFE_METRIC_NAMES,
    SAFE_STATUSES,
    SAFE_STEPS,
)


SCHEMA_VERSION = "1.0"


def _safe_event_summary(trajectory: dict[str, Any]) -> list[dict[str, str]]:
    """Export canonical event labels only; never copy arbitrary trace text."""
    trace = trajectory.get("evaluation_trace", [])
    if not isinstance(trace, list):
        raise ValueError("evaluation_trace must be a list")

    summary = []
    for event in trace:
        if not isinstance(event, dict):
            raise ValueError("each evaluation_trace event must be an object")

        step = event.get("step", "")
        status = event.get("status", "")
        if not isinstance(step, str) or not isinstance(status, str):
            raise ValueError("event step/status must be strings")

        summary.append({
            "step": step if step in SAFE_STEPS else "unknown",
            "status": status if status in SAFE_STATUSES else "unknown",
        })

    return summary


def candidate_from_record(record: Any) -> dict[str, Any] | None:
    """Convert a trajectory record into a safe, review-only candidate."""
    if not isinstance(record, dict):
        raise ValueError("each JSONL record must be an object")

    trajectory = record.get("trajectory", record)
    if not isinstance(trajectory, dict):
        raise ValueError("trajectory must be an object")

    status = trajectory.get("trajectory_status")
    if not isinstance(status, str):
        raise ValueError("trajectory_status must be a string")
    if status not in {"failed", "recovered"}:
        return None

    failures = trajectory.get("failure_taxonomy")
    if not isinstance(failures, list) or not failures:
        return None
    if any(not isinstance(item, str) for item in failures):
        raise ValueError("failure_taxonomy must contain strings")

    unknown = set(failures) - FAILURE_CATEGORIES
    if unknown:
        raise ValueError(
            "unrecognized failure categories: " + ", ".join(sorted(unknown))
        )

    failures = sorted(set(failures))
    events = _safe_event_summary(trajectory)

    # Hash only the sanitized signature, never raw user content.
    signature = {
        "schema_version": SCHEMA_VERSION,
        "trajectory_status": status,
        "failure_taxonomy": failures,
        "events": events,
    }
    canonical = json.dumps(
        signature, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    metrics = trajectory.get("metrics", {})
    if not isinstance(metrics, dict):
        raise ValueError("metrics must be an object")

    safe_metrics = {}
    for name in SAFE_METRIC_NAMES:
        if name not in metrics or metrics[name] is None:
            continue

        value = metrics[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"metric {name!r} must be a number in [0, 1]")

        if (
            (isinstance(value, float) and not math.isfinite(value))
            or not 0 <= value <= 1
        ):
            raise ValueError(
                f"metric {name!r} must be a finite number in [0, 1]"
            )

        safe_metrics[name] = value

    return {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": "cand_" + fingerprint[:16],
        "fingerprint": fingerprint,
        "trajectory_status": status,
        "failure_taxonomy": failures,
        "event_summary": events,
        "metrics": safe_metrics,
        "review_status": "needs_review",
        "gold_answer": None,
        "promotion_status": "not_promoted",
    }


def mine_candidates(input_path: Path, output_path: Path) -> dict[str, int]:
    """Aggregate duplicate failure patterns into deterministic review candidates."""
    candidates: dict[str, dict[str, Any]] = {}
    metric_values: dict[str, dict[str, list[float]]] = {}
    input_count = 0
    skipped_count = 0

    with input_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue

            input_count += 1
            try:
                record = json.loads(line)
                candidate = candidate_from_record(record)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid input at line {line_number}: {exc}"
                ) from exc

            if candidate is None:
                skipped_count += 1
                continue

            fingerprint = candidate["fingerprint"]
            observed_metrics = candidate["metrics"]

            if fingerprint not in candidates:
                aggregated = dict(candidate)
                aggregated["occurrence_count"] = 0
                candidates[fingerprint] = aggregated
                metric_values[fingerprint] = {}

            aggregated = candidates[fingerprint]
            aggregated["occurrence_count"] += 1

            # Collect only observed values. Missing metrics are not zeroes.
            for name, value in observed_metrics.items():
                metric_values[fingerprint].setdefault(name, []).append(
                    float(value)
                )

    # Finalize aggregates only after all records have been validated.
    # Sorting each value list makes floating-point aggregation independent
    # of the input record order.
    for fingerprint, candidate in candidates.items():
        observations = metric_values[fingerprint]
        candidate["metric_observation_counts"] = {
            name: len(values)
            for name, values in sorted(observations.items())
        }
        candidate["metrics"] = {
            name: math.fsum(sorted(values)) / len(values)
            for name, values in sorted(observations.items())
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=output_path.name + ".",
        suffix=".tmp",
        dir=str(output_path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as target:
            for fingerprint in sorted(candidates):
                target.write(
                    json.dumps(
                        candidates[fingerprint],
                        sort_keys=True,
                        ensure_ascii=True,
                    ) + "\n"
                )
        os.replace(temporary_name, output_path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise

    return {
        "input_records": input_count,
        "skipped_non_candidates": skipped_count,
        "unique_candidates": len(candidates),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mine review-only regression candidates from trajectory JSONL."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        counts = mine_candidates(args.input, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Error: {exc}\n")

    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())