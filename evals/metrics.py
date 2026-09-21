"""
KnowledgeMesh Evaluation Metrics
================================

Metrics are deliberately independent of the vector-store implementation.

Canonical retrieval identity:
    document_id::chunk_id

Example:
    7ed18d7a17efb3ce8ec56478::74

Qdrant point UUIDs are storage-level identifiers and should NOT be used
as the primary retrieval benchmark identity.

Citation identity remains separate:
    chunk_1
    chunk_2
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from typing import Any


# ============================================================
# Tokenization
# ============================================================

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def _tokens(text: Any) -> set[str]:
    """
    Normalize text into a small lexical token set.

    This is intentionally lightweight. These metrics are deterministic
    benchmark signals, not substitutes for semantic/LLM judging.
    """
    if text is None:
        return set()

    return {
        token.lower()
        for token in _TOKEN_PATTERN.findall(str(text))
        if token.strip()
    }


def _normalize_id(value: Any) -> str:
    return str(value).strip()


def _unique(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for value in values:
        normalized = _normalize_id(value)

        if not normalized or normalized in seen:
            continue

        seen.add(normalized)
        result.append(normalized)

    return result


# ============================================================
# Basic aggregation
# ============================================================


def average(values: Sequence[float]) -> float:
    if not values:
        return 0.0

    return sum(float(value) for value in values) / len(values)


def percentile(
    values: Sequence[float],
    percentile_value: float,
) -> float:
    """
    Linear-interpolated percentile.

    Returns 0 for an empty sequence.
    """
    if not values:
        return 0.0

    ordered = sorted(float(value) for value in values)

    if len(ordered) == 1:
        return ordered[0]

    p = max(0.0, min(100.0, float(percentile_value)))

    rank = (p / 100.0) * (len(ordered) - 1)

    lower = math.floor(rank)
    upper = math.ceil(rank)

    if lower == upper:
        return ordered[lower]

    weight = rank - lower

    return (
        ordered[lower] * (1.0 - weight)
        + ordered[upper] * weight
    )


# ============================================================
# Retrieval metrics
# ============================================================


def recall_at_k(
    retrieved_ids: Sequence[Any],
    relevant_ids: Sequence[Any],
    k: int,
) -> float:
    """
    Recall@K:

        relevant retrieved within K
        ---------------------------
             total relevant

    IDs are compared as canonical strings.
    """
    relevant = set(_unique(relevant_ids))

    if not relevant:
        return 0.0

    retrieved = set(_unique(retrieved_ids)[: max(0, k)])

    return len(retrieved & relevant) / len(relevant)


def precision_at_k(
    retrieved_ids: Sequence[Any],
    relevant_ids: Sequence[Any],
    k: int,
) -> float:
    """
    Precision@K:

        relevant retrieved within K
        ---------------------------
        retrieved items considered
    """
    retrieved = _unique(retrieved_ids)[: max(0, k)]

    if not retrieved:
        return 0.0

    relevant = set(_unique(relevant_ids))

    return sum(
        1
        for item in retrieved
        if item in relevant
    ) / len(retrieved)


def reciprocal_rank(
    retrieved_ids: Sequence[Any],
    relevant_ids: Sequence[Any],
) -> float:
    """
    Reciprocal rank of the first relevant retrieved document.
    """
    relevant = set(_unique(relevant_ids))

    if not relevant:
        return 0.0

    for rank, item in enumerate(_unique(retrieved_ids), start=1):
        if item in relevant:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    retrieved_ids: Sequence[Any],
    relevant_ids: Sequence[Any],
    k: int,
) -> float:
    """
    Binary-relevance nDCG@K.

    Relevant chunks receive gain 1.
    Non-relevant chunks receive gain 0.
    """
    retrieved = _unique(retrieved_ids)[: max(0, k)]
    relevant = set(_unique(relevant_ids))

    if not relevant:
        return 0.0

    dcg = 0.0

    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_count = min(len(relevant), max(0, k))

    if ideal_count == 0:
        return 0.0

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_count + 1)
    )

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


# ============================================================
# Answer metrics
# ============================================================


def answer_relevancy(
    question: str,
    answer: str,
) -> float:
    """
    Deterministic lexical answer relevancy.

    This is intentionally conservative:
    it measures overlap between meaningful question tokens and
    answer tokens.

    It should not be interpreted as semantic correctness.
    """
    question_tokens = _tokens(question)
    answer_tokens = _tokens(answer)

    if not question_tokens or not answer_tokens:
        return 0.0

    overlap = question_tokens & answer_tokens

    return len(overlap) / len(question_tokens)


def keyword_coverage(
    answer: str,
    expected_points: Sequence[Any],
) -> float:
    """
    Measures how many expected answer points have lexical evidence
    in the generated answer.

    Each expected point is treated independently.
    """
    if not expected_points:
        return 0.0

    answer_tokens = _tokens(answer)

    if not answer_tokens:
        return 0.0

    matched = 0

    for point in expected_points:
        point_tokens = _tokens(point)

        if not point_tokens:
            continue

        # A point is considered covered when at least one meaningful
        # token from the point occurs in the answer.
        if point_tokens & answer_tokens:
            matched += 1

    return matched / len(expected_points)


# ============================================================
# Citation metrics
# ============================================================


def citation_precision(
    cited_ids: Sequence[Any],
    supported_ids: Sequence[Any],
) -> float:
    """
    Citation precision:

        supported citations
        -------------------
        all cited citations

    Citation IDs use the LLM context namespace:

        chunk_1
        chunk_2
        ...
    """
    cited = _unique(cited_ids)

    if not cited:
        return 0.0

    supported = set(_unique(supported_ids))

    return sum(
        1
        for citation_id in cited
        if citation_id in supported
    ) / len(cited)


def citation_recall(
    cited_ids: Sequence[Any],
    required_ids: Sequence[Any],
) -> float:
    """
    Citation recall.

    If a dataset does not specify required citation IDs, the metric
    is treated as not applicable and returns 1.0.

    This prevents an unspecified citation requirement from being
    interpreted as a citation failure.
    """
    required = set(_unique(required_ids))

    if not required:
        return 1.0

    cited = set(_unique(cited_ids))

    return len(cited & required) / len(required)


# ============================================================
# Grounding metrics
# ============================================================


def grounding_score(
    total_claims: int,
    unsupported_claims: int,
) -> float:
    """
    Grounding score:

        supported claims
        ----------------
        total claims

    If there are no claims, return 1.0 because there is no
    unsupported claim evidence.
    """
    total = max(0, int(total_claims))
    unsupported = max(0, min(int(unsupported_claims), total))

    if total == 0:
        return 1.0

    supported = total - unsupported

    return supported / total


def unsupported_claim_rate(
    total_claims: int,
    unsupported_claims: int,
) -> float:
    """
    Fraction of generated claims that were unsupported.

    If there are no claims, return 0.0.
    """
    total = max(0, int(total_claims))
    unsupported = max(0, min(int(unsupported_claims), total))

    if total == 0:
        return 0.0

    return unsupported / total


# ============================================================
# Quality gate
# ============================================================


DEFAULT_QUALITY_THRESHOLDS: dict[str, float] = {
    "retrieval_recall_at_5": 0.70,
    "answer_relevancy": 0.70,
    "grounding_score": 0.90,
    "citation_precision": 0.90,
    "final_gate_pass_rate": 0.90,
    "unsupported_claim_rate": 0.10,
}


def evaluate_quality_gate(
    metrics: dict[str, float],
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Evaluate deterministic quality-gate thresholds.

    Higher-is-better:
        retrieval_recall_at_5
        answer_relevancy
        grounding_score
        citation_precision
        final_gate_pass_rate

    Lower-is-better:
        unsupported_claim_rate
    """
    effective_thresholds = dict(DEFAULT_QUALITY_THRESHOLDS)

    if thresholds:
        effective_thresholds.update(thresholds)

    checks: dict[str, dict[str, Any]] = {}

    higher_is_better = {
        "retrieval_recall_at_5",
        "answer_relevancy",
        "grounding_score",
        "citation_precision",
        "final_gate_pass_rate",
    }

    lower_is_better = {
        "unsupported_claim_rate",
    }

    for name, threshold in effective_thresholds.items():
        value = float(metrics.get(name, 0.0))
        threshold_value = float(threshold)

        if name in higher_is_better:
            passed = value >= threshold_value
        elif name in lower_is_better:
            passed = value <= threshold_value
        else:
            passed = False

        checks[name] = {
            "value": value,
            "threshold": threshold_value,
            "passed": passed,
        }

    return {
        "passed": all(
            check["passed"]
            for check in checks.values()
        ),
        "checks": checks,
    }