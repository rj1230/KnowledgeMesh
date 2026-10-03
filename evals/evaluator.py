"""
KnowledgeMesh Evaluator
=======================

Evaluates an agentic RAG system that can answer from two sources:

    private knowledge base   retrieval identity  document_id::chunk_id
    web search                retrieval identity  normalized URL / domain

and that is expected to decline ("abstain") when neither source has the
answer.

Ground truth:
    true_chunk_inventory.csv (DocumentId, Filename, ChunkId, ...) lists
    every private chunk that exists. ChunkInventory loads it so gold
    labels and retrieved chunk IDs can both be checked against what
    really exists. Web gold has no such inventory — it is whatever URLs
    or domains the dataset author specifies per case.

Per-case gold is independent per source:
    relevant_chunk_ids     private gold (may be empty)
    expected_urls /
    expected_domains       web gold (may be empty)
    expected_route         "private" | "web" | "both" | "abstain" (optional;
                            drives routing/abstention scoring only)

A metric that does not apply to a case (e.g. web hit-rate on a
private-only case) is None for that case and is left out of the
average, not counted as zero — see metrics.average_applicable.
"""

from __future__ import annotations

import csv
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .metrics import (
    ROUTES,
    SAFETY_METRIC_NAMES,
    actual_route,
    answer_relevancy,
    average,
    average_applicable,
    citation_precision,
    citation_recall,
    detect_abstention,
    detect_leak,
    detect_refusal,
    document_hit_at_k,
    evaluate_quality_gate,
    first_relevant_rank,
    grounding_score,
    hit_at_k,
    keyword_coverage,
    ndcg_at_k,
    neighbor_hit_at_k,
    normalize_route,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    routing_metrics,
    safety_metrics,
    split_chunk_key,
    unknown_id_rate,
    unsupported_claim_rate,
    web_domain_precision_at_k,
    web_hit_at_k,
    web_reciprocal_rank,
)


# Retrieval metrics are reported at this cutoff for both private chunks
# and web results. The metric names ("..._at_5") are part of the result
# schema, so change RETRIEVAL_K and the names together if you need a
# different cutoff.
RETRIEVAL_K = 5


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

    # Private gold. Canonical IDs: document_id::chunk_id
    # Example: 89e40c0de3ce2081b9d726b9::0
    # Empty for web-only or abstain cases.
    relevant_chunk_ids: list[str] = field(
        default_factory=list,
    )

    # Web gold. A case matches if a retrieved URL is one of expected_urls
    # (or a page under one) or is hosted on one of expected_domains.
    # Empty for private-only or abstain cases.
    expected_urls: list[str] = field(
        default_factory=list,
    )

    expected_domains: list[str] = field(
        default_factory=list,
    )

    # "private" | "web" | "both" | "abstain"; None if the dataset does
    # not label routes (routing/abstention metrics are skipped then).
    expected_route: str | None = None

    # LLM citation namespace: chunk_1, chunk_2, ...
    required_citation_ids: list[str] = field(
        default_factory=list,
    )

    category: str = "golden"

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )

    @property
    def has_private_gold(self) -> bool:
        return bool(self.relevant_chunk_ids)

    @property
    def has_web_gold(self) -> bool:
        return bool(self.expected_urls or self.expected_domains)

    @property
    def is_abstain(self) -> bool:
        return self.expected_route == "abstain"


@dataclass
class PipelineEvaluationResult:
    """
    The agent's output, in a transport-agnostic shape. evaluate.py's
    normalize_api_response() is responsible for mapping your actual API
    response onto these fields — adjust its extraction functions to your
    endpoint's real field names.
    """

    case_id: str

    answer: str = ""

    # Private retrieval. Canonical identities: document_id::chunk_id
    retrieved_ids: list[str] = field(
        default_factory=list,
    )

    reranked_ids: list[str] = field(
        default_factory=list,
    )

    # Web retrieval. Raw URLs; normalized for comparison inside the
    # metrics functions.
    web_urls: list[str] = field(
        default_factory=list,
    )

    # Whether the agent actually consulted each source this turn.
    # Defaults to "did we get any results back", but should be set from
    # an explicit API flag when your agent exposes one (a private lookup
    # that legitimately returned zero hits is not the same as never
    # trying).
    used_private: bool = False

    used_web: bool = False

    # Whether the agent declined to answer. Prefer an explicit API flag;
    # evaluate.py falls back to metrics.detect_abstention(answer) when
    # the API does not report this.
    abstained: bool = False

    # LLM citation identities: chunk_1, chunk_2, ...
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

    # metric name -> mean over the cases where it applied; None if no
    # case in the run was eligible for it.
    metrics: dict[str, float | None]

    # metric name -> how many cases contributed to its mean. Metrics
    # differ in applicability (private-only cases skip web metrics and
    # vice versa), so this is how you tell "0.0 on 40 cases" from
    # "0.0 on 2 cases".
    metrics_n: dict[str, int]

    quality_gate: dict[str, Any]

    cases: list[dict[str, Any]]

    latency: dict[str, float]

    errors: list[dict[str, Any]]

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )

    # Private retrieval metrics grouped by the (first) gold document.
    by_document: dict[str, dict[str, float]] = field(
        default_factory=dict,
    )

    # Routing/abstention metrics grouped by expected_route.
    by_route: dict[str, dict[str, float]] = field(
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
# True chunk inventory
# ============================================================

INVENTORY_REQUIRED_COLUMNS = (
    "DocumentId",
    "Filename",
    "ChunkId",
)


class ChunkInventory:
    """
    The set of private chunks that really exist, loaded from
    true_chunk_inventory.csv.

    Every chunk is identified by the canonical key
    ``document_id::chunk_id``. Data-quality findings that do not make
    the inventory unusable (empty text, Characters mismatch, gaps in
    ChunkId, ambiguous filenames) are collected in ``issues``. A
    duplicate key is a hard error because it makes the truth ambiguous.
    """

    def __init__(
        self,
        keys: Iterable[str],
        filename_by_document: dict[str, str],
        issues: list[str] | None = None,
        source: str = "",
    ) -> None:
        self.keys: frozenset[str] = frozenset(keys)
        self.filename_by_document = dict(filename_by_document)
        self.issues = list(issues or [])
        self.source = source

        self._documents_by_filename: dict[str, set[str]] = defaultdict(set)

        for document, filename in self.filename_by_document.items():
            self._documents_by_filename[filename].add(document)

    # ----------------------------------------------------
    # Loading
    # ----------------------------------------------------

    @classmethod
    def from_csv(cls, path: Path | str) -> "ChunkInventory":
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Chunk inventory not found: {path}")

        csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

        keys: list[str] = []
        seen: set[str] = set()
        filename_by_document: dict[str, str] = {}
        chunk_ids_by_document: dict[str, list[int]] = defaultdict(list)
        empty_text: list[str] = []
        length_mismatch: list[str] = []

        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)

            missing = [
                column
                for column in INVENTORY_REQUIRED_COLUMNS
                if column not in (reader.fieldnames or [])
            ]

            if missing:
                raise ValueError(
                    f"{path}: missing required columns {missing}; "
                    f"found {reader.fieldnames}"
                )

            for line, row in enumerate(reader, start=2):
                key = canonical_chunk_key(
                    row.get("DocumentId"),
                    row.get("ChunkId"),
                )

                if key is None:
                    raise ValueError(
                        f"{path}:{line}: invalid DocumentId/ChunkId "
                        f"({row.get('DocumentId')!r}, {row.get('ChunkId')!r})"
                    )

                if key in seen:
                    raise ValueError(f"{path}:{line}: duplicate chunk key {key}")

                seen.add(key)
                keys.append(key)

                document, chunk = split_chunk_key(key)  # type: ignore[misc]

                filename_by_document[document] = row["Filename"]
                chunk_ids_by_document[document].append(chunk)

                label = f"{row['Filename']}#{chunk}"
                text = row.get("Text")

                if text is not None and not text.strip():
                    empty_text.append(label)

                declared = row.get("Characters")

                if text is not None and declared not in (None, ""):
                    try:
                        if int(declared) != len(text):
                            length_mismatch.append(label)
                    except ValueError:
                        length_mismatch.append(label)

        if not keys:
            raise ValueError(f"{path}: inventory is empty")

        issues: list[str] = []

        if empty_text:
            issues.append(
                f"{len(empty_text)} chunks have empty text, e.g. {empty_text[:3]}"
            )

        if length_mismatch:
            issues.append(
                f"{len(length_mismatch)} chunks where Characters != len(Text), "
                f"e.g. {length_mismatch[:3]}"
            )

        for document, chunk_ids in chunk_ids_by_document.items():
            ordered = sorted(chunk_ids)

            if ordered != list(range(ordered[0], ordered[0] + len(ordered))):
                issues.append(
                    f"{filename_by_document[document]}: ChunkId values are not contiguous"
                )

        names = Counter(filename_by_document.values())

        for name, count in names.items():
            if count > 1:
                issues.append(
                    f"filename {name!r} belongs to {count} DocumentIds; "
                    "filename-based gold labels for it are ambiguous"
                )

        return cls(
            keys,
            filename_by_document,
            issues,
            source=str(path),
        )

    # ----------------------------------------------------
    # Lookup
    # ----------------------------------------------------

    def __len__(self) -> int:
        return len(self.keys)

    def __contains__(self, key: object) -> bool:
        return key in self.keys

    def document_for_filename(self, filename: Any) -> str | None:
        """
        DocumentId for a filename, or None if the filename is unknown or
        maps to more than one document.
        """
        documents = self._documents_by_filename.get(Path(str(filename)).name)

        if documents and len(documents) == 1:
            return next(iter(documents))

        return None

    def document_label(self, document_id: str) -> str:
        return self.filename_by_document.get(document_id, document_id)

    def label(self, key: str) -> str:
        """
        Human-readable form of a canonical key: "RAG.pdf#3".

        Keys that are not in the inventory are returned unchanged.
        """
        parsed = split_chunk_key(key)

        if parsed is None or key not in self.keys:
            return key

        return f"{self.document_label(parsed[0])}#{parsed[1]}"

    def chunks_per_document(self) -> dict[str, int]:
        counts: Counter[str] = Counter()

        for key in self.keys:
            parsed = split_chunk_key(key)

            if parsed is not None:
                counts[self.document_label(parsed[0])] += 1

        return dict(sorted(counts.items()))

    def summary(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "n_chunks": len(self.keys),
            "n_documents": len(self.filename_by_document),
            "chunks_per_document": self.chunks_per_document(),
            "issues": self.issues,
        }


# ============================================================
# Dataset audit
# ============================================================


@dataclass
class DatasetAudit:
    """
    How the golden dataset relates to the true chunk inventory, plus a
    route-label breakdown when the dataset uses expected_route.
    """

    total_cases: int

    # case_id -> gold private IDs that do not exist in the inventory.
    unknown_gold: dict[str, list[str]]

    # Cases with no private gold, no web gold, and not labeled "abstain"
    # always score 0 on every retrieval metric — there is nothing they
    # could possibly hit.
    cases_without_gold: list[str]

    # filename -> number of cases with private gold in that document.
    cases_per_document: dict[str, int]

    documents_without_cases: list[str]

    # distinct private gold chunks / chunks in the inventory
    gold_chunk_coverage: float

    # route label -> case count (only routes present in the dataset).
    route_counts: dict[str, int]

    # Cases with an expected_route the dataset used but that
    # normalize_route() did not recognize.
    unrecognized_routes: dict[str, str]

    @property
    def ok(self) -> bool:
        return not self.unknown_gold

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "ok": self.ok}


def audit_dataset(
    cases: Sequence[EvaluationCase],
    inventory: ChunkInventory,
) -> DatasetAudit:
    unknown_gold: dict[str, list[str]] = {}
    without_gold: list[str] = []
    per_document: Counter[str] = Counter()
    covered: set[str] = set()
    route_counts: Counter[str] = Counter()
    unrecognized_routes: dict[str, str] = {}

    for case in cases:
        if case.expected_route is not None:
            if case.expected_route in ROUTES:
                route_counts[case.expected_route] += 1
            else:
                unrecognized_routes[case.id] = case.expected_route

        if not case.has_private_gold and not case.has_web_gold and not case.is_abstain:
            without_gold.append(case.id)

        if not case.relevant_chunk_ids:
            continue

        unknown = [
            chunk_id
            for chunk_id in case.relevant_chunk_ids
            if chunk_id not in inventory
        ]

        if unknown:
            unknown_gold[case.id] = unknown

        known = [
            chunk_id for chunk_id in case.relevant_chunk_ids if chunk_id in inventory
        ]

        covered.update(known)

        documents = {
            inventory.document_label(parsed[0])
            for parsed in map(split_chunk_key, known)
            if parsed is not None
        }

        per_document.update(documents)

    all_documents = set(inventory.filename_by_document.values())

    return DatasetAudit(
        total_cases=len(cases),
        unknown_gold=unknown_gold,
        cases_without_gold=without_gold,
        cases_per_document=dict(sorted(per_document.items())),
        documents_without_cases=sorted(all_documents - set(per_document)),
        gold_chunk_coverage=len(covered) / len(inventory),
        route_counts=dict(route_counts),
        unrecognized_routes=unrecognized_routes,
    )


# ============================================================
# Evaluator
# ============================================================


# Metrics computed for every case that has the relevant kind of gold.
# None entries in a case's metrics dict for these names mean "not
# applicable to this case", not "scored zero".
PRIVATE_RETRIEVAL_METRICS = (
    "retrieval_recall_at_5",
    "retrieval_precision_at_5",
    "retrieval_hit_at_5",
    "retrieval_document_hit_at_5",
    "retrieval_neighbor_hit_at_5",
    "ndcg_at_5",
    "reciprocal_rank",
)

WEB_RETRIEVAL_METRICS = (
    "web_hit_at_5",
    "web_reciprocal_rank",
    "web_domain_precision_at_5",
)

ANSWER_METRICS = (
    "answer_relevancy",
    "answer_point_coverage",
    "citation_precision",
    "citation_recall",
    "grounding_score",
    "unsupported_claim_rate",
)


class KnowledgeMeshEvaluator:
    """
    Evaluation runner for an agentic RAG over a private KB + web search.

    When an inventory is supplied, retrieved private IDs are also
    checked against it (metric ``retrieved_unknown_id_rate``) and case
    results carry readable chunk labels.
    """

    def __init__(
        self,
        pipeline: Callable[[str], Any],
        thresholds: dict[str, float] | None = None,
        inventory: ChunkInventory | None = None,
        verbose: bool = False,
        abstention_fallback: Callable[[str], bool] | None = detect_abstention,
    ) -> None:
        self.pipeline = pipeline
        self.thresholds = thresholds
        self.inventory = inventory
        self.verbose = verbose
        # Used only if a case's PipelineEvaluationResult doesn't already
        # set `abstained` some other way (evaluate.py normally resolves
        # this before the evaluator ever sees the result; this is a
        # second safety net for callers that construct results directly).
        self.abstention_fallback = abstention_fallback

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
        answer = str(self._get_value(result, "answer", "") or "")

        retrieved_ids = self._string_list(self._get_value(result, "retrieved_ids", []))

        reranked_ids = self._string_list(self._get_value(result, "reranked_ids", []))

        web_urls = self._string_list(self._get_value(result, "web_urls", []))

        used_private = self._get_value(result, "used_private", None)

        if used_private is None:
            used_private = bool(retrieved_ids or reranked_ids)

        used_web = self._get_value(result, "used_web", None)

        if used_web is None:
            used_web = bool(web_urls)

        abstained = self._get_value(result, "abstained", None)

        if abstained is None:
            abstained = (
                self.abstention_fallback(answer)
                if self.abstention_fallback is not None
                else False
            )

        return PipelineEvaluationResult(
            case_id=case.id,
            answer=answer,
            retrieved_ids=retrieved_ids,
            reranked_ids=reranked_ids,
            web_urls=web_urls,
            used_private=bool(used_private),
            used_web=bool(used_web),
            abstained=bool(abstained),
            cited_ids=self._string_list(self._get_value(result, "cited_ids", [])),
            supported_citation_ids=self._string_list(
                self._get_value(result, "supported_citation_ids", [])
            ),
            total_claims=int(self._get_value(result, "total_claims", 0) or 0),
            unsupported_claims=int(
                self._get_value(result, "unsupported_claims", 0) or 0
            ),
            revision_count=int(self._get_value(result, "revision_count", 0) or 0),
            final_gate_passed=bool(self._get_value(result, "final_gate_passed", False)),
            latency_ms=latency_ms,
            raw=result if isinstance(result, dict) else {},
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

            latency_ms = (time.perf_counter() - started) * 1000

            result = self._normalize_result(case, raw_result, latency_ms)

        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000

            result = PipelineEvaluationResult(
                case_id=case.id,
                latency_ms=latency_ms,
                error=f"{type(exc).__name__}: {exc}",
            )

        metrics: dict[str, float | None] = {}

        # ----------------------------------------------------
        # Private retrieval (only meaningful if the case has private gold)
        # ----------------------------------------------------

        private_ids = (
            result.reranked_ids if result.reranked_ids else result.retrieved_ids
        )

        if case.has_private_gold:
            metrics["retrieval_recall_at_5"] = recall_at_k(
                private_ids, case.relevant_chunk_ids, RETRIEVAL_K
            )
            metrics["retrieval_precision_at_5"] = precision_at_k(
                private_ids, case.relevant_chunk_ids, RETRIEVAL_K
            )
            metrics["retrieval_hit_at_5"] = hit_at_k(
                private_ids, case.relevant_chunk_ids, RETRIEVAL_K
            )
            metrics["retrieval_document_hit_at_5"] = document_hit_at_k(
                private_ids, case.relevant_chunk_ids, RETRIEVAL_K
            )
            metrics["retrieval_neighbor_hit_at_5"] = neighbor_hit_at_k(
                private_ids, case.relevant_chunk_ids, RETRIEVAL_K
            )
            metrics["ndcg_at_5"] = ndcg_at_k(
                private_ids, case.relevant_chunk_ids, RETRIEVAL_K
            )
            metrics["reciprocal_rank"] = reciprocal_rank(
                private_ids, case.relevant_chunk_ids
            )
        else:
            for name in PRIVATE_RETRIEVAL_METRICS:
                metrics[name] = None

        # ----------------------------------------------------
        # Web retrieval (the metric functions already return None when
        # the case has no web gold, so no branching needed here)
        # ----------------------------------------------------

        metrics["web_hit_at_5"] = web_hit_at_k(
            result.web_urls, case.expected_urls, case.expected_domains, RETRIEVAL_K
        )
        metrics["web_reciprocal_rank"] = web_reciprocal_rank(
            result.web_urls, case.expected_urls, case.expected_domains
        )
        metrics["web_domain_precision_at_5"] = web_domain_precision_at_k(
            result.web_urls, case.expected_domains, RETRIEVAL_K
        )

        # ----------------------------------------------------
        # Routing and abstention (only when the dataset labels routes)
        # ----------------------------------------------------

        route_metrics = routing_metrics(
            normalize_route(case.expected_route),
            result.used_private,
            result.used_web,
            result.abstained,
        )
        metrics.update(route_metrics)

        # ----------------------------------------------------
        # Generation, citations, grounding (always computed; a case
        # with no expected_answer_points simply gets 0.0 coverage, same
        # as the original single-source evaluator)
        # ----------------------------------------------------

        metrics["answer_relevancy"] = answer_relevancy(case.question, result.answer)
        metrics["answer_point_coverage"] = keyword_coverage(
            result.answer, case.expected_answer_points
        )
        metrics["citation_precision"] = citation_precision(
            result.cited_ids, result.supported_citation_ids
        )
        metrics["citation_recall"] = citation_recall(
            result.cited_ids, case.required_citation_ids
        )
        metrics["grounding_score"] = grounding_score(
            result.total_claims, result.unsupported_claims
        )
        metrics["unsupported_claim_rate"] = unsupported_claim_rate(
            result.total_claims, result.unsupported_claims
        )

        retrieval: dict[str, Any] = {
            "retrieved_ids": result.retrieved_ids,
            "reranked_ids": result.reranked_ids,
            "relevant_ids": case.relevant_chunk_ids,
            "web_urls": result.web_urls,
            "expected_urls": case.expected_urls,
            "expected_domains": case.expected_domains,
            "first_relevant_rank": (
                first_relevant_rank(private_ids, case.relevant_chunk_ids)
                if case.has_private_gold
                else None
            ),
            "expected_route": case.expected_route,
            "actual_route": actual_route(result.used_private, result.used_web),
            "abstained": result.abstained,
        }

        if self.inventory is not None and case.has_private_gold:
            top_ids = list(dict.fromkeys(private_ids))[:RETRIEVAL_K]

            metrics["retrieved_unknown_id_rate"] = unknown_id_rate(
                private_ids, self.inventory, RETRIEVAL_K
            )

            retrieval["relevant_labels"] = [
                self.inventory.label(chunk_id) for chunk_id in case.relevant_chunk_ids
            ]
            retrieval["retrieved_labels"] = [
                self.inventory.label(chunk_id) for chunk_id in top_ids
            ]
            retrieval["unknown_retrieved_ids"] = [
                chunk_id for chunk_id in top_ids if chunk_id not in self.inventory
            ]
        else:
            metrics["retrieved_unknown_id_rate"] = None

        if self.verbose:
            print()
            print(
                f"[{case.id}] route expected={case.expected_route!r} "
                f"actual={retrieval['actual_route']!r} abstained={result.abstained}"
            )

            if case.has_private_gold:
                print(
                    f"  private expected: {retrieval.get('relevant_labels', case.relevant_chunk_ids)}"
                )
                print(
                    f"  private retrieved: {retrieval.get('retrieved_labels', private_ids[:RETRIEVAL_K])}"
                )

            if case.has_web_gold:
                print(
                    f"  web expected: urls={case.expected_urls} domains={case.expected_domains}"
                )
                print(f"  web retrieved: {result.web_urls[:RETRIEVAL_K]}")

        return {
            "case_id": case.id,
            "category": case.category,
            "question": case.question,
            "answer": result.answer,
            "metrics": metrics,
            "retrieval": retrieval,
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
    # Breakdowns
    # --------------------------------------------------------

    def _by_document(
        self,
        cases: Sequence[EvaluationCase],
        case_results: Sequence[dict[str, Any]],
    ) -> dict[str, dict[str, float]]:

        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for case, result in zip(cases, case_results):
            if result["error"] or not case.relevant_chunk_ids:
                continue

            parsed = split_chunk_key(case.relevant_chunk_ids[0])

            name = (
                self.inventory.document_label(parsed[0])
                if parsed is not None and self.inventory is not None
                else (parsed[0] if parsed is not None else "(no canonical gold)")
            )

            groups[name].append(result)

        breakdown: dict[str, dict[str, float]] = {}

        for name, results in sorted(groups.items()):
            breakdown[name] = {
                "n": float(len(results)),
                **{
                    metric: average([item["metrics"][metric] for item in results])
                    for metric in (
                        "retrieval_recall_at_5",
                        "retrieval_hit_at_5",
                        "retrieval_neighbor_hit_at_5",
                        "reciprocal_rank",
                    )
                },
            }

        return breakdown

    def _by_route(
        self,
        cases: Sequence[EvaluationCase],
        case_results: Sequence[dict[str, Any]],
    ) -> dict[str, dict[str, float]]:

        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for case, result in zip(cases, case_results):
            if result["error"]:
                continue

            route = normalize_route(case.expected_route)

            if route is not None:
                groups[route].append(result)

        breakdown: dict[str, dict[str, float]] = {}

        for route, results in sorted(groups.items()):
            available = [
                name
                for name in (
                    "routing_accuracy",
                    "web_decision_accuracy",
                    "web_use_recall",
                    "unnecessary_web_rate",
                    "correct_abstention_rate",
                    "false_abstention_rate",
                )
                if any(item["metrics"][name] is not None for item in results)
            ]

            breakdown[route] = {
                "n": float(len(results)),
                **{
                    name: average_applicable(item["metrics"][name] for item in results)
                    for name in available
                },
            }

        return breakdown

    # --------------------------------------------------------
    # Full evaluation
    # --------------------------------------------------------

    def evaluate(
        self,
        cases: Sequence[EvaluationCase],
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationSummary:

        started = time.perf_counter()

        case_results: list[dict[str, Any]] = [
            self.evaluate_case(case) for case in cases
        ]

        total = len(case_results)
        successful = sum(1 for r in case_results if not r["error"])
        failed = total - successful

        metric_names = list(
            PRIVATE_RETRIEVAL_METRICS + WEB_RETRIEVAL_METRICS + ANSWER_METRICS
        ) + [
            "retrieved_unknown_id_rate",
            "routing_accuracy",
            "web_decision_accuracy",
            "web_use_recall",
            "unnecessary_web_rate",
            "correct_abstention_rate",
            "false_abstention_rate",
        ]

        aggregated: dict[str, float | None] = {}
        counts: dict[str, int] = {}

        for name in metric_names:
            values = [r["metrics"][name] for r in case_results if not r["error"]]
            present = [v for v in values if v is not None]

            aggregated[name] = average_applicable(values)
            counts[name] = len(present)

        final_gate_values = [
            r["self_rag"]["final_gate_passed"] for r in case_results if not r["error"]
        ]

        aggregated["final_gate_pass_rate"] = (
            sum(bool(v) for v in final_gate_values) / len(final_gate_values)
            if final_gate_values
            else None
        )
        counts["final_gate_pass_rate"] = len(final_gate_values)

        latencies = [r["latency_ms"] for r in case_results if not r["error"]]

        latency = {
            "average_ms": average(latencies),
            "p50_ms": percentile(latencies, 50),
            "p95_ms": percentile(latencies, 95),
            "p99_ms": percentile(latencies, 99),
            "max_ms": max(latencies) if latencies else 0.0,
        }

        quality_gate = evaluate_quality_gate(aggregated, self.thresholds)

        errors = [
            {"case_id": r["case_id"], "error": r["error"]}
            for r in case_results
            if r["error"]
        ]

        summary_metadata = dict(metadata or {})
        summary_metadata["evaluation_runtime_ms"] = (
            time.perf_counter() - started
        ) * 1000

        return EvaluationSummary(
            total_cases=total,
            successful_cases=successful,
            failed_cases=failed,
            metrics=aggregated,
            metrics_n=counts,
            quality_gate=quality_gate,
            cases=case_results,
            latency=latency,
            errors=errors,
            metadata=summary_metadata,
            by_document=self._by_document(cases, case_results),
            by_route=self._by_route(cases, case_results),
        )


# ============================================================
# Adversarial / safety cases
# ============================================================
#
# A different shape from EvaluationCase on purpose: there's no
# retrieval gold and no route to score, just a question and a label
# for whether the correct behavior is to decline. This runs through
# the same `pipeline` callable as the golden/unanswerable suites (same
# agent, same API), just scored differently.


@dataclass
class AdversarialCase:
    id: str
    question: str

    # True: the correct behavior is to decline (attack, jailbreak,
    # out-of-scope ask, etc). False: a benign control the agent should
    # actually answer — used to measure over-refusal.
    should_refuse: bool

    # Free-text label, ideally one of metrics.ATTACK_TYPES, but not
    # enforced — an unrecognized value just won't get its own
    # breakdown row.
    attack_type: str = "unspecified"

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AdversarialSummary:
    total_cases: int
    successful_cases: int
    failed_cases: int

    # metric name -> mean over the cases where it applied; None if no
    # case was eligible (e.g. attack_resisted needs should_refuse=True
    # cases; a dataset of only benign controls would leave it None).
    metrics: dict[str, float | None]

    metrics_n: dict[str, int]

    quality_gate: dict[str, Any]

    by_attack_type: dict[str, dict[str, Any]]

    cases: list[dict[str, Any]]

    errors: list[dict[str, Any]]

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_adversarial_case(
    pipeline: Callable[[str], Any],
    case: AdversarialCase,
    refusal_fallback: Callable[[str], bool] | None = detect_refusal,
    leak_fallback: Callable[[str], bool] | None = detect_leak,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Run one adversarial case through `pipeline` and score it.

    `pipeline` is expected to return a dict (as evaluate.py's
    normalize_api_response does) that may already contain "refused"
    (bool). When it doesn't, `refusal_fallback(answer)` is used —
    pass None to disable the fallback and treat a missing signal as
    "not refused".
    """

    def get(result: Any, key: str, default: Any = None) -> Any:
        if isinstance(result, dict):
            return result.get(key, default)

        return getattr(result, key, default)

    started = time.perf_counter()

    try:
        raw = pipeline(case.question)
        latency_ms = (time.perf_counter() - started) * 1000

        answer = str(get(raw, "answer", "") or "")

        explicit_refused = get(raw, "refused", None)
        refused = (
            bool(explicit_refused)
            if explicit_refused is not None
            else (refusal_fallback(answer) if refusal_fallback is not None else False)
        )

        leaked = bool(leak_fallback(answer)) if leak_fallback is not None else False

        error = None

    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000
        answer, refused, leaked = "", False, False
        error = f"{type(exc).__name__}: {exc}"

    metrics: dict[str, float | None] = (
        safety_metrics(case.should_refuse, refused, leaked) if error is None else {}
    )

    if verbose:
        verdict = (
            "n/a (error)"
            if error
            else ("CORRECT" if metrics.get("safety_correct") == 1.0 else "FAILED")
        )

        print(
            f"[{case.id}] attack_type={case.attack_type!r} should_refuse={case.should_refuse} "
            f"refused={refused} leaked={leaked} -> {verdict}"
        )

    return {
        "case_id": case.id,
        "question": case.question,
        "attack_type": case.attack_type,
        "should_refuse": case.should_refuse,
        "answer": answer,
        "refused": refused,
        "leaked_sensitive_content": leaked,
        "metrics": metrics,
        "latency_ms": latency_ms,
        "error": error,
    }


def evaluate_adversarial(
    pipeline: Callable[[str], Any],
    cases: Sequence[AdversarialCase],
    thresholds: dict[str, float] | None = None,
    verbose: bool = False,
) -> AdversarialSummary:
    if not cases:
        raise ValueError("no adversarial cases to evaluate")

    results = [
        evaluate_adversarial_case(pipeline, case, verbose=verbose) for case in cases
    ]

    ok = [r for r in results if not r["error"]]

    aggregated: dict[str, float | None] = {}
    counts: dict[str, int] = {}

    for name in SAFETY_METRIC_NAMES:
        values = [r["metrics"][name] for r in ok]
        present = [v for v in values if v is not None]

        aggregated[name] = average_applicable(values)
        counts[name] = len(present)

    aggregated["attack_success_rate"] = (
        1.0 - aggregated["attack_resisted"]
        if aggregated["attack_resisted"] is not None
        else None
    )
    counts["attack_success_rate"] = counts["attack_resisted"]

    by_attack_type: dict[str, dict[str, Any]] = {}

    for attack_type in sorted({case.attack_type for case in cases}):
        rows = [
            r
            for r, case in zip(results, cases)
            if case.attack_type == attack_type and not r["error"]
        ]

        entry: dict[str, Any] = {"n": float(len(rows))}

        for name in SAFETY_METRIC_NAMES:
            entry[name] = average_applicable(r["metrics"][name] for r in rows)

        entry["attack_success_rate"] = (
            1.0 - entry["attack_resisted"]
            if entry["attack_resisted"] is not None
            else None
        )

        by_attack_type[attack_type] = entry

    quality_gate = evaluate_quality_gate(aggregated, thresholds)

    errors = [
        {"case_id": r["case_id"], "error": r["error"]} for r in results if r["error"]
    ]

    return AdversarialSummary(
        total_cases=len(results),
        successful_cases=len(ok),
        failed_cases=len(results) - len(ok),
        metrics=aggregated,
        metrics_n=counts,
        quality_gate=quality_gate,
        by_attack_type=by_attack_type,
        cases=results,
        errors=errors,
    )
