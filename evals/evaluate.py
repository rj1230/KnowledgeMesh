"""
KnowledgeMesh Evaluation Runner
===============================

Connects the evaluation framework to the live KnowledgeMesh API.

Canonical retrieval identity:
    document_id::chunk_id

Example:
    89e40c0de3ce2081b9d726b9::0

Qdrant point IDs are preserved by the API for provenance/debugging,
but are not used as benchmark retrieval IDs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import httpx

from .evaluator import (
    EvaluationCase,
    KnowledgeMeshEvaluator,
    canonical_chunk_key,
)


# ============================================================
# Paths
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_DATASET = ROOT_DIR / "evals" / "datasets" / "golden_questions.json"

DEFAULT_OUTPUT = ROOT_DIR / "evals" / "results" / "latest.json"

DEFAULT_BASE_URL = "http://127.0.0.1:8000"


# ============================================================
# Dataset loading
# ============================================================


def load_dataset(
    path: Path,
) -> list[EvaluationCase]:
    """
    Load an evaluation dataset.

    Supported retrieval formats:

    New canonical format:

        "relevant_chunks": [
            {
                "document_id": "...",
                "chunk_id": 0
            }
        ]

    Legacy format:

        "relevant_chunk_ids": [
            "some-id"
        ]
    """

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Evaluation dataset must contain a JSON array.")

    cases: list[EvaluationCase] = []

    for index, item in enumerate(
        data,
        start=1,
    ):
        if not isinstance(item, dict):
            raise ValueError(f"Dataset item {index} must be an object.")

        case_id = str(item.get("id") or f"case_{index}")

        question = str(item.get("question") or "").strip()

        if not question:
            raise ValueError(f"Dataset case '{case_id}' has an empty question.")

        # ----------------------------------------------------
        # Answer points
        # ----------------------------------------------------

        keypoints = item.get(
            "expected_answer_points",
            item.get(
                "expected_answer_keypoints",
                [],
            ),
        )

        if not isinstance(
            keypoints,
            list,
        ):
            keypoints = []

        expected_answer_points = [
            str(value).strip() for value in keypoints if str(value).strip()
        ]

        # ----------------------------------------------------
        # Retrieval targets
        # ----------------------------------------------------

        relevant_ids_raw = item.get("relevant_chunk_ids")

        if relevant_ids_raw is None:
            relevant_ids_raw = item.get(
                "relevant_chunks",
                [],
            )

        if not isinstance(
            relevant_ids_raw,
            list,
        ):
            relevant_ids_raw = []

        relevant_ids: list[str] = []

        for value in relevant_ids_raw:
            # New structured format.
            if isinstance(
                value,
                dict,
            ):
                key = canonical_chunk_key(
                    value.get("document_id"),
                    value.get("chunk_id"),
                )

                if key is not None:
                    relevant_ids.append(key)

                continue

            # Legacy string format.
            text = str(value).strip()

            if text:
                relevant_ids.append(text)

        relevant_ids = list(dict.fromkeys(relevant_ids))

        # ----------------------------------------------------
        # Required citation IDs
        # ----------------------------------------------------

        required_citations = item.get(
            "required_citation_ids",
            [],
        )

        if not isinstance(
            required_citations,
            list,
        ):
            required_citations = []

        required_citations = list(
            dict.fromkeys(
                str(value).strip() for value in required_citations if str(value).strip()
            )
        )

        # ----------------------------------------------------
        # Build case
        # ----------------------------------------------------

        metadata = item.get(
            "metadata",
            {},
        )

        if not isinstance(
            metadata,
            dict,
        ):
            metadata = {}

        cases.append(
            EvaluationCase(
                id=case_id,
                question=question,
                expected_answer=str(
                    item.get(
                        "expected_answer",
                        "",
                    )
                    or ""
                ),
                expected_answer_points=(expected_answer_points),
                relevant_chunk_ids=(relevant_ids),
                required_citation_ids=(required_citations),
                category=str(
                    item.get(
                        "category",
                        "golden",
                    )
                ),
                metadata=metadata,
            )
        )

    return cases


# ============================================================
# API client
# ============================================================


class KnowledgeMeshAPIClient:
    def __init__(
        self,
        base_url: str,
        timeout: float = 180.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict[str, Any]:
        response = httpx.get(
            f"{self.base_url}/health",
            timeout=10.0,
        )

        response.raise_for_status()

        data = response.json()

        if not isinstance(data, dict):
            raise ValueError("Health endpoint returned invalid JSON.")

        return data

    def query(
        self,
        question: str,
        thread_id: str,
    ) -> dict[str, Any]:

        response = httpx.post(
            f"{self.base_url}/query",
            json={
                "q": question,
                "thread_id": thread_id,
            },
            timeout=self.timeout,
        )

        response.raise_for_status()

        data = response.json()

        if not isinstance(data, dict):
            raise ValueError("Query endpoint returned invalid JSON.")

        return data


# ============================================================
# Safe conversion helpers
# ============================================================


def _safe_int(
    value: Any,
    default: int = 0,
) -> int:
    try:
        return int(value)
    except (
        TypeError,
        ValueError,
    ):
        return default


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        return float(value)
    except (
        TypeError,
        ValueError,
    ):
        return default


# ============================================================
# API response extraction
# ============================================================


def _extract_private_source_ids(
    response: dict[str, Any],
) -> list[str]:
    """
    Extract canonical retrieval identities.

    API source:

        {
            "document_id": "...",
            "chunk_id": 74,
            "id": "QDRANT-UUID"
        }

    Evaluation identity:

        document_id::chunk_id
    """

    sources = response.get(
        "private_sources",
        [],
    )

    if not isinstance(
        sources,
        list,
    ):
        return []

    ids: list[str] = []

    for source in sources:
        if not isinstance(
            source,
            dict,
        ):
            continue

        key = canonical_chunk_key(
            source.get("document_id"),
            source.get("chunk_id"),
        )

        if key is not None:
            ids.append(key)

    return list(dict.fromkeys(ids))


def _extract_reranked_ids(
    response: dict[str, Any],
) -> list[str]:
    """
    Extract reranked canonical chunk identities.

    Ranking uses rerank_score first, then score.

    Identity uses:

        document_id::chunk_id
    """

    sources = response.get(
        "private_sources",
        [],
    )

    if not isinstance(
        sources,
        list,
    ):
        return []

    valid_sources = [
        source
        for source in sources
        if isinstance(
            source,
            dict,
        )
        and source.get("document_id") is not None
        and source.get("chunk_id") is not None
    ]

    ranked = sorted(
        valid_sources,
        key=lambda source: _safe_float(
            source.get("rerank_score"),
            _safe_float(
                source.get("score"),
                0.0,
            ),
        ),
        reverse=True,
    )

    ids: list[str] = []

    for source in ranked:
        key = canonical_chunk_key(
            source.get("document_id"),
            source.get("chunk_id"),
        )

        if key is not None:
            ids.append(key)

    return list(dict.fromkeys(ids))


def _extract_cited_chunk_ids(
    response: dict[str, Any],
) -> list[str]:
    """
    Extract LLM citation IDs.

    Citation metrics intentionally operate in the LLM citation
    namespace:

        chunk_1
        chunk_2
        ...

    They must NOT use canonical private identities such as:

        document_id::chunk_id
    """

    claims = response.get(
        "claims",
        [],
    )

    if not isinstance(
        claims,
        list,
    ):
        return []

    ids: list[str] = []

    for claim in claims:
        if not isinstance(
            claim,
            dict,
        ):
            continue

        # ----------------------------------------------------
        # Preferred source:
        #
        # Explicit LLM citation list.
        # ----------------------------------------------------

        citations = claim.get(
            "citations",
            [],
        )

        if isinstance(
            citations,
            str,
        ):
            citations = [citations]

        if isinstance(
            citations,
            list,
        ):
            for citation_id in citations:
                if citation_id is None:
                    continue

                citation_id = str(citation_id).strip()

                if not citation_id:
                    continue

                if citation_id not in ids:
                    ids.append(citation_id)

        # ----------------------------------------------------
        # Backward-compatible fallback.
        #
        # Only accept cited_chunk_id if it is already in the
        # LLM citation namespace.
        #
        # NEVER prepend "chunk_" here.
        # ----------------------------------------------------

        cited_chunk_id = claim.get("cited_chunk_id")

        if cited_chunk_id is not None:
            cited_chunk_id = str(cited_chunk_id).strip()

            if cited_chunk_id.startswith("chunk_"):
                if cited_chunk_id not in ids:
                    ids.append(cited_chunk_id)

    return list(dict.fromkeys(ids))


def _extract_supported_chunk_ids(
    response: dict[str, Any],
) -> list[str]:
    """
    Extract grounded/supported LLM citation IDs.

    Only citations attached to grounding results whose `supported`
    value is exactly True are returned.

    Citation namespace:

        chunk_1
        chunk_2
        ...

    Canonical document/chunk identities are intentionally not used
    here.
    """

    grounding_scores = response.get(
        "grounding_scores",
        [],
    )

    if not isinstance(
        grounding_scores,
        list,
    ):
        return []

    ids: list[str] = []

    for item in grounding_scores:
        if not isinstance(
            item,
            dict,
        ):
            continue

        if item.get("supported") is not True:
            continue

        # ----------------------------------------------------
        # Preferred source:
        #
        # Explicit LLM citation list.
        # ----------------------------------------------------

        citations = item.get(
            "citations",
            [],
        )

        if isinstance(
            citations,
            str,
        ):
            citations = [citations]

        if isinstance(
            citations,
            list,
        ):
            for citation_id in citations:
                if citation_id is None:
                    continue

                citation_id = str(citation_id).strip()

                if not citation_id:
                    continue

                if citation_id not in ids:
                    ids.append(citation_id)

        # ----------------------------------------------------
        # Backward-compatible fallback.
        #
        # Again, NEVER prepend "chunk_".
        # ----------------------------------------------------

        cited_chunk_id = item.get("cited_chunk_id")

        if cited_chunk_id is not None:
            cited_chunk_id = str(cited_chunk_id).strip()

            if cited_chunk_id.startswith("chunk_"):
                if cited_chunk_id not in ids:
                    ids.append(cited_chunk_id)

    return list(dict.fromkeys(ids))


def _count_claims(
    response: dict[str, Any],
) -> int:

    claims = response.get("claims")

    if isinstance(
        claims,
        list,
    ):
        return len(claims)

    return 0


def _count_unsupported_claims(
    response: dict[str, Any],
) -> int:

    grounding_scores = response.get("grounding_scores")

    if not isinstance(
        grounding_scores,
        list,
    ):
        return 0

    return sum(
        1
        for item in grounding_scores
        if (
            isinstance(
                item,
                dict,
            )
            and item.get("supported") is False
        )
    )


# ============================================================
# API normalization
# ============================================================


def normalize_api_response(
    response: dict[str, Any],
) -> dict[str, Any]:

    retrieved_ids = _extract_private_source_ids(response)

    reranked_ids = _extract_reranked_ids(response)

    cited_ids = _extract_cited_chunk_ids(response)

    supported_ids = _extract_supported_chunk_ids(response)

    total_claims = _count_claims(response)

    unsupported_claims = _count_unsupported_claims(response)

    validation_passed = response.get("validation_passed")

    if validation_passed is None:
        validation_passed = (
            response.get("citation_valid") is True
            and response.get("is_grounded") is True
        )

    return {
        "answer": str(
            response.get(
                "answer",
                "",
            )
            or ""
        ),
        "retrieved_ids": retrieved_ids,
        "reranked_ids": reranked_ids,
        "cited_ids": cited_ids,
        "supported_citation_ids": (supported_ids),
        "total_claims": total_claims,
        "unsupported_claims": (unsupported_claims),
        "revision_count": _safe_int(
            response.get("revision_count"),
            0,
        ),
        "final_gate_passed": bool(validation_passed),
        "latency_ms": _safe_float(
            response.get("latency_ms"),
            0.0,
        ),
        # Keep the complete API response.
        "raw": response,
    }


# ============================================================
# Pipeline callback
# ============================================================


def build_pipeline_callback(
    client: KnowledgeMeshAPIClient,
):
    def pipeline(
        question: str,
    ) -> dict[str, Any]:

        thread_id = f"eval-{int(time.time() * 1000)}"

        response = client.query(
            question=question,
            thread_id=thread_id,
        )

        return normalize_api_response(response)

    return pipeline


# ============================================================
# Printing
# ============================================================


def _get_result_metric(
    result: dict[str, Any],
    name: str,
) -> float:

    metrics = result.get(
        "metrics",
        {},
    )

    if not isinstance(
        metrics,
        dict,
    ):
        return 0.0

    return _safe_float(
        metrics.get(name),
        0.0,
    )


def print_case_result(
    result: dict[str, Any],
) -> None:

    metrics = result.get("metrics", {})

    self_rag = result.get("self_rag", {})

    print("-" * 72)

    print(f"Case: {result.get('case_id')}")

    print(f"Question: {result.get('question')}")

    if result.get("error"):
        print(f"ERROR: {result['error']}")
        return

    print(
        "Retrieval:"
        f" R@5="
        f"{_safe_float(metrics.get('retrieval_recall_at_5')):.3f}"
        f" P@5="
        f"{_safe_float(metrics.get('retrieval_precision_at_5')):.3f}"
        f" NDCG@5="
        f"{_safe_float(metrics.get('ndcg_at_5')):.3f}"
    )

    print(
        "Answer:"
        f" relevancy="
        f"{_safe_float(metrics.get('answer_relevancy')):.3f}"
        f" coverage="
        f"{_safe_float(metrics.get('answer_point_coverage')):.3f}"
    )

    print(
        "Citations:"
        f" precision="
        f"{_safe_float(metrics.get('citation_precision')):.3f}"
        f" recall="
        f"{_safe_float(metrics.get('citation_recall')):.3f}"
    )

    print(
        "Grounding:"
        f" score="
        f"{_safe_float(metrics.get('grounding_score')):.3f}"
        f" unsupported_rate="
        f"{_safe_float(metrics.get('unsupported_claim_rate')):.3f}"
    )

    print(
        "Self-RAG:"
        f" revisions="
        f"{self_rag.get('revision_count', 0)}"
        f" claims="
        f"{self_rag.get('total_claims', 0)}"
        f" unsupported="
        f"{self_rag.get('unsupported_claims', 0)}"
        f" final_gate="
        f"{self_rag.get('final_gate_passed', False)}"
    )

    print(f"Latency: {_safe_float(result.get('latency_ms')) / 1000:.2f}s")


def print_summary(
    summary: Any,
) -> None:

    print()
    print("=" * 72)
    print("Evaluation Summary")
    print("=" * 72)

    print(f"Total cases:      {summary.total_cases}")

    print(f"Successful:       {summary.successful_cases}")

    print(f"Failed:           {summary.failed_cases}")

    print()
    print("Metrics")

    for name, value in summary.metrics.items():
        print(f"  {name:<32} {value:.4f}")

    print()
    print("Latency")

    for name, value in summary.latency.items():
        print(f"  {name:<16} {value:.2f} ms")

    print()
    print("Quality Gate")

    gate = summary.quality_gate

    if isinstance(
        gate,
        dict,
    ):
        for key, value in gate.items():
            print(f"  {key:<24} {value}")

    print()

    if summary.errors:
        print("Errors")

        for error in summary.errors:
            print(f"  {error.get('case_id')}: {error.get('error')}")

    print("=" * 72)


# ============================================================
# Save results
# ============================================================


def save_results(
    summary: Any,
    output_path: Path,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = summary.to_dict()

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description=("KnowledgeMesh evaluation runner."))

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================


def main() -> int:

    args = parse_args()

    print()
    print("=" * 72)
    print("KnowledgeMesh Evaluation")
    print("=" * 72)

    print(f"API:     {args.base_url}")

    print(f"Dataset: {args.dataset.resolve()}")

    print(f"Output:  {args.output.resolve()}")

    print()

    client = KnowledgeMeshAPIClient(
        base_url=args.base_url,
        timeout=args.timeout,
    )

    # --------------------------------------------------------
    # Health
    # --------------------------------------------------------

    try:
        health = client.health()

        if health.get("status") != "ok":
            raise RuntimeError(f"Unexpected API health response: {health}")

    except Exception as exc:
        print("ERROR: KnowledgeMesh API health check failed.")

        print(f"Type:    {type(exc).__name__}")

        print(f"Details: {exc}")

        print()

        traceback.print_exc()

        return 2

    print(
        "API Health:"
        f" {health.get('status')} "
        f"{health.get('service')} "
        f"v{health.get('version')}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    try:
        cases = load_dataset(args.dataset)

    except Exception as exc:
        print()

        print("ERROR: Could not load evaluation dataset.")

        print(f"Type:    {type(exc).__name__}")

        print(f"Details: {exc}")

        return 2

    if args.limit > 0:
        cases = cases[: args.limit]

    print(f"Cases:   {len(cases)}")

    print()

    # --------------------------------------------------------
    # Show benchmark identity
    # --------------------------------------------------------

    print("Retrieval identity: document_id::chunk_id")

    print()

    # --------------------------------------------------------
    # Evaluator
    # --------------------------------------------------------

    evaluator = KnowledgeMeshEvaluator(pipeline=build_pipeline_callback(client))

    started = time.perf_counter()

    try:
        summary = evaluator.evaluate(
            cases,
            metadata={
                "api_base_url": args.base_url,
                "dataset": str(args.dataset.resolve()),
                "case_count": len(cases),
                "retrieval_identity": ("document_id::chunk_id"),
                "evaluation_runtime_ms": (time.perf_counter() - started) * 1000,
            },
        )

    except Exception as exc:
        print()

        print("ERROR: Evaluation failed.")

        print(f"Type:    {type(exc).__name__}")

        print(f"Details: {exc}")

        print()

        traceback.print_exc()

        return 2

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    for result in summary.cases:
        print_case_result(result)

    print_summary(summary)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    try:
        save_results(
            summary,
            args.output,
        )

    except Exception as exc:
        print()

        print("ERROR: Could not save evaluation results.")

        print(f"Type:    {type(exc).__name__}")

        print(f"Details: {exc}")

        return 2

    print()

    print(f"Results saved to: {args.output.resolve()}")

    gate_passed = bool(
        summary.quality_gate.get(
            "passed",
            False,
        )
        if isinstance(
            summary.quality_gate,
            dict,
        )
        else False
    )

    print(f"Quality Gate: {'PASSED' if gate_passed else 'FAILED'}")

    return 0 if gate_passed else 1


if __name__ == "__main__":
    sys.exit(main())
