"""
KnowledgeMesh Evaluator
=======================

Evaluation contract:

Retrieval identity:
    document_id::chunk_id

Citation identity:
    chunk_1
    chunk_2
    ...

Vector-store point IDs:
    retained inside raw API responses for provenance/debugging,
    but NOT used as retrieval benchmark IDs.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Sequence

from .metrics import (
    answer_relevancy,
    average,
    citation_precision,
    citation_recall,
    evaluate_quality_gate,
    grounding_score,
    keyword_coverage,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    unsupported_claim_rate,
)


# ============================================================
# Data models
# ============================================================


@dataclass
class EvaluationCase:
    id: str
    question: str

    expected_answer: str = ""

    expected_answer_points: list[str] = field(
        default_factory=list,
    )

    # Canonical retrieval IDs:
    #
    # document_id::chunk_id
    #
    # Example:
    # 89e40c0de3ce2081b9d726b9::0
    relevant_chunk_ids: list[str] = field(
        default_factory=list,
    )

    # LLM citation namespace:
    #
    # chunk_1
    # chunk_2
    #
    required_citation_ids: list[str] = field(
        default_factory=list,
    )

    category: str = "golden"

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )


@dataclass
class PipelineEvaluationResult:
    case_id: str

    answer: str = ""

    # Canonical retrieval identities.
    retrieved_ids: list[str] = field(
        default_factory=list,
    )

    reranked_ids: list[str] = field(
        default_factory=list,
    )

    # LLM citation identities.
    cited_ids: list[str] = field(
        default_factory=list,
    )

    supported_citation_ids: list[str] = field(
        default_factory=list,
    )

    total_claims: int = 0

    unsupported_claims: int = 0

    revision_count: int = 0

    final_gate_passed: bool = False

    latency_ms: float = 0.0

    error: str | None = None

    raw: dict[str, Any] = field(
        default_factory=dict,
    )


@dataclass
class EvaluationSummary:
    total_cases: int
    successful_cases: int
    failed_cases: int

    metrics: dict[str, float]

    quality_gate: dict[str, Any]

    cases: list[dict[str, Any]]

    latency: dict[str, float]

    errors: list[dict[str, Any]]

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================
# Canonical ID helpers
# ============================================================


def canonical_chunk_key(
    document_id: Any,
    chunk_id: Any,
) -> str | None:
    """
    Convert corpus metadata into the canonical benchmark identity.

    Example:

        document_id = "abc123"
        chunk_id = 17

    becomes:

        "abc123::17"
    """
    if document_id is None or chunk_id is None:
        return None

    document = str(document_id).strip()

    if not document:
        return None

    try:
        chunk = int(chunk_id)
    except (TypeError, ValueError):
        return None

    return f"{document}::{chunk}"


# ============================================================
# Evaluator
# ============================================================


class KnowledgeMeshEvaluator:
    """
    Evaluation runner for KnowledgeMesh.
    """

    def __init__(
        self,
        pipeline: Callable[[str], Any],
        thresholds: dict[str, float] | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.thresholds = thresholds

    # --------------------------------------------------------
    # Pipeline normalization
    # --------------------------------------------------------

    @staticmethod
    def _get_value(
        result: Any,
        key: str,
        default: Any = None,
    ) -> Any:
        if isinstance(result, dict):
            return result.get(key, default)

        return getattr(result, key, default)

    @staticmethod
    def _string_list(value: Any) -> list[str]:
        if not isinstance(value, (list, tuple)):
            return []

        result: list[str] = []

        for item in value:
            text = str(item).strip()

            if text:
                result.append(text)

        return list(dict.fromkeys(result))

    def _normalize_result(
        self,
        case: EvaluationCase,
        result: Any,
        latency_ms: float,
    ) -> PipelineEvaluationResult:
        return PipelineEvaluationResult(
            case_id=case.id,

            answer=str(
                self._get_value(
                    result,
                    "answer",
                    "",
                )
                or ""
            ),

            retrieved_ids=self._string_list(
                self._get_value(
                    result,
                    "retrieved_ids",
                    [],
                )
            ),

            reranked_ids=self._string_list(
                self._get_value(
                    result,
                    "reranked_ids",
                    [],
                )
            ),

            cited_ids=self._string_list(
                self._get_value(
                    result,
                    "cited_ids",
                    [],
                )
            ),

            supported_citation_ids=self._string_list(
                self._get_value(
                    result,
                    "supported_citation_ids",
                    [],
                )
            ),

            total_claims=int(
                self._get_value(
                    result,
                    "total_claims",
                    0,
                )
                or 0
            ),

            unsupported_claims=int(
                self._get_value(
                    result,
                    "unsupported_claims",
                    0,
                )
                or 0
            ),

            revision_count=int(
                self._get_value(
                    result,
                    "revision_count",
                    0,
                )
                or 0
            ),

            final_gate_passed=bool(
                self._get_value(
                    result,
                    "final_gate_passed",
                    False,
                )
            ),

            latency_ms=latency_ms,

            raw=(
                result
                if isinstance(result, dict)
                else {}
            ),
        )

    # --------------------------------------------------------
    # Per-case evaluation
    # --------------------------------------------------------

    def evaluate_case(
        self,
        case: EvaluationCase,
    ) -> dict[str, Any]:

        started = time.perf_counter()

        try:
            raw_result = self.pipeline(case.question)

            latency_ms = (
                time.perf_counter() - started
            ) * 1000

            result = self._normalize_result(
                case,
                raw_result,
                latency_ms,
            )

        except Exception as exc:
            latency_ms = (
                time.perf_counter() - started
            ) * 1000

            result = PipelineEvaluationResult(
                case_id=case.id,
                latency_ms=latency_ms,
                error=f"{type(exc).__name__}: {exc}",
            )

        # ----------------------------------------------------
        # Retrieval
        # ----------------------------------------------------

        retrieved_ids = (
            result.reranked_ids
            if result.reranked_ids
            else result.retrieved_ids
        )

        recall5 = recall_at_k(
            retrieved_ids,
            case.relevant_chunk_ids,
            5,
        )

        precision5 = precision_at_k(
            retrieved_ids,
            case.relevant_chunk_ids,
            5,
        )

        ndcg5 = ndcg_at_k(
            retrieved_ids,
            case.relevant_chunk_ids,
            5,
        )

        rr = reciprocal_rank(
            retrieved_ids,
            case.relevant_chunk_ids,
        )

        # ----------------------------------------------------
        # Generation
        # ----------------------------------------------------

        relevancy = answer_relevancy(
            case.question,
            result.answer,
        )

        coverage = keyword_coverage(
            result.answer,
            case.expected_answer_points,
        )

        # ----------------------------------------------------
        # Citations
        # ----------------------------------------------------

        citation_prec = citation_precision(
            result.cited_ids,
            result.supported_citation_ids,
        )

        citation_rec = citation_recall(
            result.cited_ids,
            case.required_citation_ids,
        )

        # ----------------------------------------------------
        # Grounding
        # ----------------------------------------------------

        grounding = grounding_score(
            result.total_claims,
            result.unsupported_claims,
        )

        unsupported_rate = unsupported_claim_rate(
            result.total_claims,
            result.unsupported_claims,
        )

        # ----------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------

        print()
        print("DEBUG RETRIEVAL")
        print(
            f"  Expected:  "
            f"{case.relevant_chunk_ids}"
        )
        print(
            f"  Retrieved: "
            f"{retrieved_ids[:5]}"
        )

        return {
            "case_id": case.id,
            "category": case.category,
            "question": case.question,
            "answer": result.answer,

            "metrics": {
                "retrieval_recall_at_5": recall5,
                "retrieval_precision_at_5": precision5,
                "ndcg_at_5": ndcg5,
                "reciprocal_rank": rr,
                "answer_relevancy": relevancy,
                "answer_point_coverage": coverage,
                "citation_precision": citation_prec,
                "citation_recall": citation_rec,
                "grounding_score": grounding,
                "unsupported_claim_rate": unsupported_rate,
            },

            "retrieval": {
                "retrieved_ids": result.retrieved_ids,
                "reranked_ids": result.reranked_ids,
                "relevant_ids": case.relevant_chunk_ids,
            },

            "citations": {
                "cited_ids": result.cited_ids,
                "supported_ids": result.supported_citation_ids,
                "required_ids": case.required_citation_ids,
            },

            "self_rag": {
                "revision_count": result.revision_count,
                "total_claims": result.total_claims,
                "unsupported_claims": result.unsupported_claims,
                "final_gate_passed": result.final_gate_passed,
            },

            "latency_ms": result.latency_ms,

            "error": result.error,
        }

    # --------------------------------------------------------
    # Full evaluation
    # --------------------------------------------------------

    def evaluate(
        self,
        cases: Sequence[EvaluationCase],
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationSummary:

        case_results: list[dict[str, Any]] = []

        for case in cases:
            case_results.append(
                self.evaluate_case(case)
            )

        total = len(case_results)

        successful = sum(
            1
            for result in case_results
            if not result["error"]
        )

        failed = total - successful

        metric_names = [
            "retrieval_recall_at_5",
            "retrieval_precision_at_5",
            "ndcg_at_5",
            "reciprocal_rank",
            "answer_relevancy",
            "answer_point_coverage",
            "citation_precision",
            "citation_recall",
            "grounding_score",
            "unsupported_claim_rate",
        ]

        aggregated: dict[str, float] = {}

        for name in metric_names:
            values = [
                result["metrics"][name]
                for result in case_results
                if not result["error"]
            ]

            aggregated[name] = average(values)

        final_gate_values = [
            result["self_rag"]["final_gate_passed"]
            for result in case_results
            if not result["error"]
        ]

        aggregated["final_gate_pass_rate"] = (
            sum(
                bool(value)
                for value in final_gate_values
            )
            / len(final_gate_values)
            if final_gate_values
            else 0.0
        )

        # ----------------------------------------------------
        # Latency
        # ----------------------------------------------------

        latencies = [
            result["latency_ms"]
            for result in case_results
            if not result["error"]
        ]

        latency = {
            "average_ms": average(latencies),
            "p50_ms": percentile(latencies, 50),
            "p95_ms": percentile(latencies, 95),
            "p99_ms": percentile(latencies, 99),
            "max_ms": (
                max(latencies)
                if latencies
                else 0.0
            ),
        }

        # ----------------------------------------------------
        # Quality gate
        # ----------------------------------------------------

        quality_gate = evaluate_quality_gate(
            aggregated,
            self.thresholds,
        )

        errors = [
            {
                "case_id": result["case_id"],
                "error": result["error"],
            }
            for result in case_results
            if result["error"]
        ]

        return EvaluationSummary(
            total_cases=total,
            successful_cases=successful,
            failed_cases=failed,
            metrics=aggregated,
            quality_gate=quality_gate,
            cases=case_results,
            latency=latency,
            errors=errors,
            metadata=metadata or {},
        )