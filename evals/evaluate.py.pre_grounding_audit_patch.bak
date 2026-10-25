"""
KnowledgeMesh Evaluation Runner
===============================

Evaluates an agentic RAG that answers from a private knowledge base, web
search, or both, and that should decline when neither source has the
answer.

Identities
----------
Private retrieval:  document_id::chunk_id   (from true_chunk_inventory.csv)
Web retrieval:       URL / domain
Citations:           chunk_1, chunk_2, ...  (LLM context namespace)

*** ADJUST THIS FILE TO YOUR API ***
The extraction functions below (_extract_private_source_ids,
_extract_web_urls, _extract_used_private, _extract_used_web,
_extract_abstained, and friends) assume a response shape that may not
match your actual /query endpoint. They are the ONLY place that knows
about your API's JSON; evaluator.py is transport-agnostic. Edit the
field names inside normalize_api_response() to match what your agent
actually returns, then everything downstream (metrics, reporting,
gating) works unchanged.

Ground truth:
    evals/datasets/true_chunk_inventory.csv lists every private chunk
    that exists. Gold labels in the dataset are checked against it
    before the run, and retrieved private IDs are checked against it
    during the run. Web gold has no such inventory; it is whatever URLs
    or domains you put in the dataset.

Dataset schema (see evals/datasets/golden_questions.json and
evals/README.md for the full field reference and worked examples):

    {
      "id": "gp001",
      "question": "...",
      "relevant_chunks": [{"document_id": "...", "chunk_id": 0}],   # private gold, optional
      "expected_web_sources": [{"url": "https://arxiv.org/abs/..."}], # web gold, optional
      "expected_domains": ["arxiv.org"],                              # web gold, optional
      "expected_route": "private",   # "private" | "web" | "both" | "abstain", optional
      "expected_answer_points": ["..."],
      "required_citation_ids": ["chunk_1"]
    }

Usage:
    python -m evals.evaluate                                    # full run
    python -m evals.evaluate --validate-only                    # offline, no API needed
    python -m evals.evaluate --threshold web_decision_accuracy=0.9
    python -m evals.evaluate --k 10 --verbose
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
    AdversarialCase,
    ChunkInventory,
    DatasetAudit,
    EvaluationCase,
    KnowledgeMeshEvaluator,
    audit_dataset,
    canonical_chunk_key,
    evaluate_adversarial,
)
from .metrics import ATTACK_TYPES, detect_abstention, detect_refusal, normalize_route


# ============================================================
# Paths
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_DATASET = ROOT_DIR / "evals" / "datasets" / "golden_questions.json"

DEFAULT_ADVERSARIAL_DATASET = ROOT_DIR / "evals" / "datasets" / "adversarial_cases.json"

DEFAULT_UNANSWERABLE_DATASET = (
    ROOT_DIR / "evals" / "datasets" / "unanswerable_questions.json"
)

DEFAULT_INVENTORY = ROOT_DIR / "evals" / "datasets" / "true_chunk_inventory.csv"

DEFAULT_OUTPUT = ROOT_DIR / "evals" / "results" / "latest.json"

DEFAULT_BASE_URL = "http://127.0.0.1:8000"


# ============================================================
# Dataset loading
# ============================================================


def _parse_web_gold(item: dict, where: str) -> list[str]:
    raw = item.get("expected_web_sources", item.get("expected_urls", []))

    if not isinstance(raw, list):
        raise ValueError(f"{where}: expected_web_sources must be a list")

    urls: list[str] = []

    for entry in raw:
        if isinstance(entry, dict):
            url = entry.get("url")
        else:
            url = entry

        url = str(url or "").strip()

        if not url:
            raise ValueError(
                f"{where}: expected_web_sources entry {entry!r} has no url"
            )

        urls.append(url)

    return list(dict.fromkeys(urls))


def load_dataset(
    path: Path,
    inventory: ChunkInventory | None = None,
) -> list[EvaluationCase]:
    """
    Load an evaluation dataset.

    Private retrieval targets (any one document must resolve; a
    structured entry that cannot be resolved is an error, not a silent
    skip — a dropped gold chunk would inflate recall):

        canonical: "relevant_chunks": [{"document_id": "...", "chunk_id": 0}]
        filename:  "relevant_chunks": [{"filename": "RAG.pdf", "chunk_id": 0}]
        legacy:    "relevant_chunk_ids": ["some-id"]

    Web retrieval targets (either or both; at least one URL/domain
    across the two is enough to make the case web-scoreable):

        "expected_web_sources": [{"url": "https://arxiv.org/abs/1706.03762"}]
        "expected_domains": ["arxiv.org"]

    Routing label (optional; enables routing_metrics for the case):

        "expected_route": "private" | "web" | "both" | "abstain"

    A case needs relevant_chunks, expected_web_sources/expected_domains,
    or "expected_route": "abstain" — at least one, so every case has
    something it could be scored against. A case with expected_route
    "abstain" legitimately has no retrieval gold: the correct behavior
    is declining to answer.
    """

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Evaluation dataset must contain a JSON array.")

    cases: list[EvaluationCase] = []

    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Dataset item {index} must be an object.")

        case_id = str(item.get("id") or f"case_{index}")
        where = f"{path}#{case_id}"

        question = str(item.get("question") or "").strip()

        if not question:
            raise ValueError(f"{where}: empty question")

        # ----------------------------------------------------
        # Answer points
        # ----------------------------------------------------

        keypoints = item.get(
            "expected_answer_points",
            item.get("expected_answer_keypoints", []),
        )

        if not isinstance(keypoints, list):
            keypoints = []

        expected_answer_points = [str(v).strip() for v in keypoints if str(v).strip()]

        # ----------------------------------------------------
        # Private retrieval targets
        # ----------------------------------------------------

        relevant_ids_raw = item.get("relevant_chunk_ids")

        if relevant_ids_raw is None:
            relevant_ids_raw = item.get("relevant_chunks", [])

        if not isinstance(relevant_ids_raw, list):
            relevant_ids_raw = []

        relevant_ids: list[str] = []

        for value in relevant_ids_raw:
            if isinstance(value, dict):
                document_id = value.get("document_id")

                if (
                    document_id is None
                    and value.get("filename")
                    and inventory is not None
                ):
                    document_id = inventory.document_for_filename(value["filename"])

                key = canonical_chunk_key(document_id, value.get("chunk_id"))

                if key is None:
                    raise ValueError(
                        f"{where}: relevant_chunks entry {value} needs a document_id "
                        "(or a filename that is unique in the chunk inventory) and an "
                        "integer chunk_id."
                    )

                relevant_ids.append(key)

                continue

            text = str(value).strip()

            if text:
                relevant_ids.append(text)

        relevant_ids = list(dict.fromkeys(relevant_ids))

        # ----------------------------------------------------
        # Web retrieval targets
        # ----------------------------------------------------

        expected_urls = _parse_web_gold(item, where)

        expected_domains = item.get("expected_domains", [])

        if not isinstance(expected_domains, list):
            expected_domains = []

        expected_domains = list(
            dict.fromkeys(str(d).strip() for d in expected_domains if str(d).strip())
        )

        # ----------------------------------------------------
        # Route label
        # ----------------------------------------------------

        expected_route = item.get("expected_route")

        if expected_route is not None:
            expected_route = str(expected_route).strip()

            if normalize_route(expected_route) is None:
                raise ValueError(
                    f"{where}: unrecognized expected_route {expected_route!r}; "
                    "use private, web, both, or abstain"
                )

        is_abstain = normalize_route(expected_route) == "abstain"

        if (
            not relevant_ids
            and not expected_urls
            and not expected_domains
            and not is_abstain
        ):
            raise ValueError(
                f"{where}: case has no private gold, no web gold, and is not "
                '"expected_route": "abstain" — nothing to score it against.'
            )

        # ----------------------------------------------------
        # Required citation IDs
        # ----------------------------------------------------

        required_citations = item.get("required_citation_ids", [])

        if not isinstance(required_citations, list):
            required_citations = []

        required_citations = list(
            dict.fromkeys(str(v).strip() for v in required_citations if str(v).strip())
        )

        metadata = item.get("metadata", {})

        if not isinstance(metadata, dict):
            metadata = {}

        cases.append(
            EvaluationCase(
                id=case_id,
                question=question,
                expected_answer=str(item.get("expected_answer", "") or ""),
                expected_answer_points=expected_answer_points,
                relevant_chunk_ids=relevant_ids,
                expected_urls=expected_urls,
                expected_domains=expected_domains,
                expected_route=expected_route,
                required_citation_ids=required_citations,
                category=str(item.get("category", "golden")),
                metadata=metadata,
            )
        )

    if not cases:
        raise ValueError(f"{path}: no cases found")

    ids = [c.id for c in cases]
    dupes = {i for i in ids if ids.count(i) > 1}

    if dupes:
        raise ValueError(f"{path}: duplicate case ids {sorted(dupes)}")

    return cases


def load_adversarial_dataset(path: Path) -> list[AdversarialCase]:
    """
    Load adversarial/safety test cases.

        [{"id": "a001", "question": "...", "should_refuse": true, "attack_type": "jailbreak"}]

    should_refuse is required and must be a real boolean (not a
    truthy string) — getting this backwards silently flips every
    metric. attack_type is free text; values outside metrics.ATTACK_TYPES
    are accepted but flagged in the printed audit and won't get their
    own row in the by-attack-type breakdown label list (they still
    count toward the overall totals).
    """
    if not path.exists():
        raise FileNotFoundError(f"Adversarial dataset not found: {path}")

    with path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Adversarial dataset must contain a JSON array.")

    cases: list[AdversarialCase] = []

    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Adversarial dataset item {index} must be an object.")

        case_id = str(item.get("id") or f"a{index}")
        where = f"{path}#{case_id}"

        question = str(item.get("question") or "").strip()

        if not question:
            raise ValueError(f"{where}: empty question")

        should_refuse = item.get("should_refuse")

        if not isinstance(should_refuse, bool):
            raise ValueError(
                f"{where}: should_refuse must be a JSON boolean (true/false), "
                f"got {should_refuse!r}"
            )

        attack_type = str(item.get("attack_type") or "unspecified").strip()

        metadata = item.get("metadata", {})

        if not isinstance(metadata, dict):
            metadata = {}

        cases.append(
            AdversarialCase(
                id=case_id,
                question=question,
                should_refuse=should_refuse,
                attack_type=attack_type,
                metadata=metadata,
            )
        )

    if not cases:
        raise ValueError(f"{path}: no cases found")

    ids = [c.id for c in cases]
    dupes = {i for i in ids if ids.count(i) > 1}

    if dupes:
        raise ValueError(f"{path}: duplicate case ids {sorted(dupes)}")

    return cases


def load_unanswerable_dataset(path: Path) -> list[EvaluationCase]:
    """
    Load hallucination-resistance cases: questions the agent has no
    business answering (nothing in the KB, nothing a web search should
    responsibly resolve). The correct behavior is declining, not
    fabricating.

        [{"id": "u001", "question": "...", "note": "why this is unanswerable (informational only)"}]

    These are just EvaluationCase objects with expected_route="abstain"
    under the hood, so they run through the SAME evaluator as the
    golden dataset and reuse correct_abstention_rate /
    false_abstention_rate — no new scoring machinery needed. "note" is
    carried into metadata for your own reference; it isn't scored.
    """
    if not path.exists():
        raise FileNotFoundError(f"Unanswerable dataset not found: {path}")

    with path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Unanswerable dataset must contain a JSON array.")

    cases: list[EvaluationCase] = []

    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Unanswerable dataset item {index} must be an object.")

        case_id = str(item.get("id") or f"u{index}")
        question = str(item.get("question") or "").strip()

        if not question:
            raise ValueError(f"{path}#{case_id}: empty question")

        cases.append(
            EvaluationCase(
                id=case_id,
                question=question,
                expected_route="abstain",
                category="unanswerable",
                metadata={"note": str(item.get("note", "") or "")},
            )
        )

    if not cases:
        raise ValueError(f"{path}: no cases found")

    ids = [c.id for c in cases]
    dupes = {i for i in ids if ids.count(i) > 1}

    if dupes:
        raise ValueError(f"{path}: duplicate case ids {sorted(dupes)}")

    return cases


# ============================================================
# API client
# ============================================================


class KnowledgeMeshAPIClient:
    def __init__(self, base_url: str, timeout: float = 180.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict[str, Any]:
        response = httpx.get(f"{self.base_url}/health", timeout=10.0)
        response.raise_for_status()
        data = response.json()

        if not isinstance(data, dict):
            raise ValueError("Health endpoint returned invalid JSON.")

        return data

    def query(self, question: str, thread_id: str) -> dict[str, Any]:
        response = httpx.post(
            f"{self.base_url}/query",
            json={"q": question, "thread_id": thread_id},
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


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ============================================================
# API response extraction — EDIT THIS SECTION FOR YOUR API
# ============================================================


def _extract_private_source_ids(response: dict[str, Any]) -> list[str]:
    """
    Private chunks the agent's KB retriever returned.

    Assumed API shape:

        "private_sources": [{"document_id": "...", "chunk_id": 74, "id": "QDRANT-UUID"}]

    Evaluation identity: document_id::chunk_id. Qdrant point UUIDs are
    kept in `raw` for debugging but are not used as the benchmark ID.
    """
    sources = response.get("private_sources", [])

    if not isinstance(sources, list):
        return []

    ids: list[str] = []

    for source in sources:
        if not isinstance(source, dict):
            continue

        key = canonical_chunk_key(source.get("document_id"), source.get("chunk_id"))

        if key is not None:
            ids.append(key)

    return list(dict.fromkeys(ids))


def _extract_reranked_ids(response: dict[str, Any]) -> list[str]:
    """Same private_sources list, ordered by rerank_score (falls back to score)."""
    sources = response.get("private_sources", [])

    if not isinstance(sources, list):
        return []

    valid = [
        s
        for s in sources
        if isinstance(s, dict)
        and s.get("document_id") is not None
        and s.get("chunk_id") is not None
    ]

    ranked = sorted(
        valid,
        key=lambda s: _safe_float(
            s.get("rerank_score"), _safe_float(s.get("score"), 0.0)
        ),
        reverse=True,
    )

    ids = [canonical_chunk_key(s.get("document_id"), s.get("chunk_id")) for s in ranked]

    return list(dict.fromkeys(i for i in ids if i is not None))


def _extract_web_urls(response: dict[str, Any]) -> list[str]:
    """
    Web search results the agent used.

    Assumed API shape:

        "web_sources": [{"url": "https://...", "title": "...", "score": 0.8}]

    If your web tool returns results under a different key (e.g.
    "tavily_results", "search_results"), change the key below — that is
    the only edit needed for web_hit_at_k / web_reciprocal_rank /
    web_domain_precision_at_k to work.
    """
    sources = response.get("web_sources", [])

    if not isinstance(sources, list):
        return []

    ranked = sorted(
        (s for s in sources if isinstance(s, dict) and s.get("url")),
        key=lambda s: _safe_float(
            s.get("rerank_score"), _safe_float(s.get("score"), 0.0)
        ),
        reverse=True,
    )

    urls = [str(s["url"]).strip() for s in ranked]

    return list(dict.fromkeys(u for u in urls if u))


def _extract_used_private(response: dict[str, Any], retrieved_ids: list[str]) -> bool:
    """
    Whether the agent actually queried the private KB this turn.

    Prefers an explicit flag if your API sets one (e.g. a routing/plan
    trace field); falls back to "did private_sources come back non-empty".
    The fallback under-counts a KB lookup that legitimately found
    nothing, so add an explicit flag if your agent logs one — e.g.
    response.get("used_private_kb") or a step name in
    response.get("trace", []).
    """
    explicit = response.get("used_private_kb")

    if explicit is not None:
        return bool(explicit)

    return bool(retrieved_ids)


def _extract_used_web(response: dict[str, Any], web_urls: list[str]) -> bool:
    """Whether the agent actually called web search this turn. Same caveat as above."""
    explicit = response.get("used_web_search")

    if explicit is not None:
        return bool(explicit)

    return bool(web_urls)


def _extract_refused(response: dict[str, Any], answer: str) -> bool:
    """
    Whether the agent declined on policy grounds (jailbreak/injection
    resistance, scope enforcement) — distinct from `_extract_abstained`,
    which is about admitting a knowledge gap. The phrasing differs
    enough between the two that they use separate pattern sets; see
    metrics.DEFAULT_REFUSAL_PATTERNS vs DEFAULT_ABSTENTION_PATTERNS.

    Prefers an explicit "refused" flag if your API sets one; falls back
    to the lexical heuristic otherwise.
    """
    explicit = response.get("refused")

    if explicit is not None:
        return bool(explicit)

    return detect_refusal(answer)


def _extract_abstained(response: dict[str, Any], answer: str) -> bool:
    """
    Whether the agent declined to answer.

    Prefers an explicit flag (e.g. response.get("abstained") or a
    validation_passed=False + empty-answer combination your agent uses).
    Falls back to a lexical heuristic (metrics.detect_abstention) over
    the answer text, which can misfire on answers that merely discuss
    uncertainty — replace this with your API's real signal as soon as
    it has one.
    """
    explicit = response.get("abstained")

    if explicit is not None:
        return bool(explicit)

    return detect_abstention(answer)


def _extract_cited_chunk_ids(response: dict[str, Any]) -> list[str]:
    """
    LLM citation IDs, in the LLM context namespace (chunk_1, chunk_2, ...).
    Must NOT be canonical document_id::chunk_id identities.
    """
    claims = response.get("claims", [])

    if not isinstance(claims, list):
        return []

    ids: list[str] = []

    for claim in claims:
        if not isinstance(claim, dict):
            continue

        citations = claim.get("citations", [])

        if isinstance(citations, str):
            citations = [citations]

        if isinstance(citations, list):
            for citation_id in citations:
                if citation_id is None:
                    continue

                text = str(citation_id).strip()

                if text and text not in ids:
                    ids.append(text)

        cited_chunk_id = claim.get("cited_chunk_id")

        if cited_chunk_id is not None:
            text = str(cited_chunk_id).strip()

            if text.startswith("chunk_") and text not in ids:
                ids.append(text)

    return list(dict.fromkeys(ids))


def _extract_supported_chunk_ids(response: dict[str, Any]) -> list[str]:
    """Citation IDs attached to grounding results whose supported value is exactly True."""
    grounding_scores = response.get("grounding_scores", [])

    if not isinstance(grounding_scores, list):
        return []

    ids: list[str] = []

    for item in grounding_scores:
        if not isinstance(item, dict) or item.get("supported") is not True:
            continue

        citations = item.get("citations", [])

        if isinstance(citations, str):
            citations = [citations]

        if isinstance(citations, list):
            for citation_id in citations:
                if citation_id is None:
                    continue

                text = str(citation_id).strip()

                if text and text not in ids:
                    ids.append(text)

        cited_chunk_id = item.get("cited_chunk_id")

        if cited_chunk_id is not None:
            text = str(cited_chunk_id).strip()

            if text.startswith("chunk_") and text not in ids:
                ids.append(text)

    return list(dict.fromkeys(ids))


def _count_claims(response: dict[str, Any]) -> int:
    claims = response.get("claims")

    return len(claims) if isinstance(claims, list) else 0


def _count_unsupported_claims(response: dict[str, Any]) -> int:
    grounding_scores = response.get("grounding_scores")

    if not isinstance(grounding_scores, list):
        return 0

    return sum(
        1
        for item in grounding_scores
        if isinstance(item, dict) and item.get("supported") is False
    )


# ============================================================
# API normalization
# ============================================================


def normalize_api_response(response: dict[str, Any]) -> dict[str, Any]:
    """
    Maps your live API's JSON onto evaluator.PipelineEvaluationResult's
    generic contract. This is the single seam between your API and the
    evaluator — everything above in this section exists to fill it in.
    """
    answer = str(response.get("answer", "") or "")

    retrieved_ids = _extract_private_source_ids(response)
    reranked_ids = _extract_reranked_ids(response)
    web_urls = _extract_web_urls(response)

    total_claims = _count_claims(response)
    unsupported_claims = _count_unsupported_claims(response)

    validation_passed = response.get("validation_passed")

    if validation_passed is None:
        validation_passed = (
            response.get("citation_valid") is True
            and response.get("is_grounded") is True
        )

    return {
        "answer": answer,
        "retrieved_ids": retrieved_ids,
        "reranked_ids": reranked_ids,
        "web_urls": web_urls,
        "used_private": _extract_used_private(response, retrieved_ids),
        "used_web": _extract_used_web(response, web_urls),
        "abstained": _extract_abstained(response, answer),
        "refused": _extract_refused(response, answer),
        "cited_ids": _extract_cited_chunk_ids(response),
        "supported_citation_ids": _extract_supported_chunk_ids(response),
        "total_claims": total_claims,
        "unsupported_claims": unsupported_claims,
        "revision_count": _safe_int(response.get("revision_count"), 0),
        "final_gate_passed": bool(validation_passed),
        "latency_ms": _safe_float(response.get("latency_ms"), 0.0),
        "raw": response,
    }


def build_pipeline_callback(client: KnowledgeMeshAPIClient):
    def pipeline(question: str) -> dict[str, Any]:
        thread_id = f"eval-{int(time.time() * 1000)}"
        response = client.query(question=question, thread_id=thread_id)

        return normalize_api_response(response)

    return pipeline


# ============================================================
# Printing
# ============================================================


def print_inventory(inventory: ChunkInventory) -> None:
    print(
        f"Inventory: {len(inventory)} chunks across "
        f"{len(inventory.filename_by_document)} documents"
    )

    for name, count in inventory.chunks_per_document().items():
        print(f"  {name:<28} {count:>5} chunks")

    for issue in inventory.issues:
        print(f"  WARNING: {issue}")

    print()


def print_dataset_audit(audit: DatasetAudit) -> None:
    print("Dataset audit")

    if audit.route_counts:
        routes = ", ".join(
            f"{route}={count}" for route, count in sorted(audit.route_counts.items())
        )
        print(f"  Routes: {routes}")

    print(
        f"  Private gold chunk coverage: {audit.gold_chunk_coverage:.1%} "
        "of inventory chunks are a private gold target"
    )

    for name, count in audit.cases_per_document.items():
        print(f"  {name:<28} {count:>5} cases")

    if audit.documents_without_cases:
        print(
            f"  WARNING: no private cases target: {', '.join(audit.documents_without_cases)}"
        )

    if audit.cases_without_gold:
        print(
            f"  WARNING: {len(audit.cases_without_gold)} cases have no gold of any kind "
            f'and are not "expected_route": "abstain": {audit.cases_without_gold[:5]}'
        )

    if audit.unrecognized_routes:
        print(
            f"  WARNING: unrecognized expected_route values: {audit.unrecognized_routes}"
        )

    if audit.unknown_gold:
        print(
            f"  ERROR: {len(audit.unknown_gold)} cases reference private chunks not in the inventory:"
        )

        for case_id, ids in list(audit.unknown_gold.items())[:10]:
            print(f"    {case_id}: {ids}")

    print()


def print_case_result(result: dict[str, Any]) -> None:
    metrics = result.get("metrics", {})
    self_rag = result.get("self_rag", {})
    retrieval = result.get("retrieval", {})

    print("-" * 72)
    print(
        f"Case: {result.get('case_id')}  [{retrieval.get('expected_route') or 'unlabeled'}]"
    )
    print(f"Question: {result.get('question')}")

    if result.get("error"):
        print(f"ERROR: {result['error']}")
        return

    def fmt(name: str) -> str:
        value = metrics.get(name)

        return "n/a" if value is None else f"{value:.3f}"

    if metrics.get("retrieval_recall_at_5") is not None:
        print(
            f"Private: R@5={fmt('retrieval_recall_at_5')} P@5={fmt('retrieval_precision_at_5')} "
            f"NDCG@5={fmt('ndcg_at_5')} hit@5={fmt('retrieval_hit_at_5')} "
            f"doc_hit@5={fmt('retrieval_document_hit_at_5')} first_rank={retrieval.get('first_relevant_rank')}"
        )

        if retrieval.get("unknown_retrieved_ids"):
            print(
                f"  WARNING: retrieved IDs not in inventory: {retrieval['unknown_retrieved_ids']}"
            )

    if metrics.get("web_hit_at_5") is not None:
        print(
            f"Web: hit@5={fmt('web_hit_at_5')} RR={fmt('web_reciprocal_rank')} "
            f"domain_precision@5={fmt('web_domain_precision_at_5')}"
        )

    print(
        f"Routing: expected={retrieval.get('expected_route') or 'n/a'} "
        f"actual={retrieval.get('actual_route')} abstained={retrieval.get('abstained')}"
    )

    print(
        f"Answer: relevancy={fmt('answer_relevancy')} coverage={fmt('answer_point_coverage')}"
    )
    print(
        f"Citations: precision={fmt('citation_precision')} recall={fmt('citation_recall')}"
    )
    print(
        f"Grounding: score={fmt('grounding_score')} unsupported_rate={fmt('unsupported_claim_rate')}"
    )
    print(
        f"Self-RAG: revisions={self_rag.get('revision_count', 0)} claims={self_rag.get('total_claims', 0)} "
        f"unsupported={self_rag.get('unsupported_claims', 0)} final_gate={self_rag.get('final_gate_passed', False)}"
    )
    print(f"Latency: {_safe_float(result.get('latency_ms')) / 1000:.2f}s")


def print_summary(summary: Any) -> None:
    print()
    print("=" * 72)
    print("Evaluation Summary")
    print("=" * 72)
    print(f"Total cases:      {summary.total_cases}")
    print(f"Successful:       {summary.successful_cases}")
    print(f"Failed:           {summary.failed_cases}")

    print()
    print("Metrics (n = cases the metric applied to)")

    for name, value in summary.metrics.items():
        n = summary.metrics_n.get(name, 0)
        shown = "n/a" if value is None else f"{value:.4f}"
        print(f"  {name:<32} {shown:<10} (n={n})")

    if summary.by_document:
        print()
        print("Private retrieval by gold document")
        print(
            f"  {'document':<28}{'n':>4}{'R@5':>8}{'hit@5':>8}{'neigh@5':>9}{'RR':>8}"
        )

        for name, values in summary.by_document.items():
            print(
                f"  {name[:27]:<28}{int(values['n']):>4}"
                f"{values['retrieval_recall_at_5']:>8.3f}"
                f"{values['retrieval_hit_at_5']:>8.3f}"
                f"{values['retrieval_neighbor_hit_at_5']:>9.3f}"
                f"{values['reciprocal_rank']:>8.3f}"
            )

    if summary.by_route:
        print()
        print("Routing / abstention by expected route")

        for route, values in summary.by_route.items():
            parts = ", ".join(
                f"{k}={v:.3f}" for k, v in values.items() if k != "n" and v is not None
            )
            print(f"  {route:<10} n={int(values['n']):<4} {parts}")

    print()
    print("Latency")

    for name, value in summary.latency.items():
        print(f"  {name:<16} {value:.2f} ms")

    print()
    print("Quality Gate")

    gate = summary.quality_gate

    if isinstance(gate, dict):
        for key, value in gate.items():
            if key == "checks" and isinstance(value, dict):
                for name, check in value.items():
                    if check.get("skipped"):
                        print(f"  {name:<32} skipped (no applicable cases)")

                        continue

                    status = "PASS" if check["passed"] else "FAIL"
                    print(
                        f"  {name:<32} {check['value']:.4f} (threshold {check['threshold']}) {status}"
                    )
            else:
                print(f"  {key:<24} {value}")

    print()

    if summary.errors:
        print("Errors")

        for error in summary.errors:
            print(f"  {error.get('case_id')}: {error.get('error')}")

    print("=" * 72)


def print_adversarial_case_result(result: dict[str, Any]) -> None:
    print("-" * 72)
    print(f"Case: {result['case_id']}  [{result['attack_type']}]")
    print(f"Question: {result['question']}")

    if result["error"]:
        print(f"ERROR: {result['error']}")

        return

    verdict = "PASS" if result["metrics"].get("safety_correct") == 1.0 else "FAIL"

    print(
        f"should_refuse={result['should_refuse']} refused={result['refused']} "
        f"leaked={result['leaked_sensitive_content']}  ->  {verdict}"
    )
    print(f"Answer: {result['answer'][:200]}")


def print_adversarial_summary(summary: Any) -> None:
    print()
    print("=" * 72)
    print("Adversarial / Safety Summary")
    print("=" * 72)
    print(
        f"Total cases: {summary.total_cases}  Successful: {summary.successful_cases}  Failed: {summary.failed_cases}"
    )
    print()
    print("Metrics (n = cases the metric applied to)")

    for name, value in summary.metrics.items():
        n = summary.metrics_n.get(name, 0)
        shown = "n/a" if value is None else f"{value:.4f}"
        print(f"  {name:<28} {shown:<10} (n={n})")

    if summary.by_attack_type:
        print()
        print("By attack type")
        print(
            f"  {'attack_type':<28}{'n':>4}{'resisted':>10}{'success':>10}{'leaked':>9}"
        )

        for attack_type, values in summary.by_attack_type.items():
            unrecognized = " (unrecognized)" if attack_type not in ATTACK_TYPES else ""

            def cell(name: str) -> str:
                v = values.get(name)

                return "n/a" if v is None else f"{v:.2f}"

            print(
                f"  {attack_type[:27]:<28}{int(values['n']):>4}"
                f"{cell('attack_resisted'):>10}{cell('attack_success_rate'):>10}"
                f"{cell('leaked_sensitive_content'):>9}{unrecognized}"
            )

    print()
    print("Quality Gate")

    gate = summary.quality_gate

    for key, value in gate.items():
        if key == "checks" and isinstance(value, dict):
            for name, check in value.items():
                if check.get("skipped"):
                    print(f"  {name:<28} skipped (no applicable cases)")

                    continue

                status = "PASS" if check["passed"] else "FAIL"
                print(
                    f"  {name:<28} {check['value']:.4f} (threshold {check['threshold']}) {status}"
                )
        else:
            print(f"  {key:<24} {value}")

    if summary.errors:
        print()
        print("Errors")

        for error in summary.errors:
            print(f"  {error['case_id']}: {error['error']}")

    print("=" * 72)


# ============================================================
# Save results
# ============================================================


def save_results(summary: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary.to_dict(), file, indent=2, ensure_ascii=False)


# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KnowledgeMesh evaluation runner (private KB + web search).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--adversarial-dataset",
        type=Path,
        default=None,
        help=f"Jailbreak/injection/off-topic test cases. Defaults to {DEFAULT_ADVERSARIAL_DATASET} when it exists.",
    )
    parser.add_argument(
        "--unanswerable-dataset",
        type=Path,
        default=None,
        help=f"Hallucination-resistance test cases. Defaults to {DEFAULT_UNANSWERABLE_DATASET} when it exists.",
    )
    parser.add_argument(
        "--skip-retrieval", action="store_true", help="Skip the golden/retrieval suite."
    )
    parser.add_argument(
        "--skip-adversarial",
        action="store_true",
        help="Skip the adversarial/safety suite.",
    )
    parser.add_argument(
        "--skip-unanswerable",
        action="store_true",
        help="Skip the hallucination-resistance suite.",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=None,
        help=f"true_chunk_inventory.csv. Defaults to {DEFAULT_INVENTORY} when it exists.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check the dataset against the inventory and exit. Does not call the API.",
    )
    parser.add_argument(
        "--allow-unknown-gold",
        action="store_true",
        help="Continue even if private gold chunks are missing from the inventory (scored as misses).",
    )
    parser.add_argument(
        "--threshold",
        action="append",
        default=[],
        metavar="METRIC=VALUE",
        help="Override or add a quality-gate threshold, e.g. web_decision_accuracy=0.9. Repeatable.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-case routing and retrieval detail.",
    )

    return parser.parse_args()


def parse_thresholds(items: list[str]) -> dict[str, float]:
    thresholds: dict[str, float] = {}

    for item in items:
        name, separator, value = item.partition("=")

        try:
            if not separator or not name.strip():
                raise ValueError

            thresholds[name.strip()] = float(value)
        except ValueError:
            raise SystemExit(
                f"error: --threshold expects METRIC=VALUE, got {item!r}"
            ) from None

    return thresholds


# ============================================================
# Main
# ============================================================


def _gate_passed(gate: Any) -> bool:
    return bool(gate.get("passed", False)) if isinstance(gate, dict) else False


def main() -> int:
    args = parse_args()
    thresholds = parse_thresholds(args.threshold)

    run_retrieval = not args.skip_retrieval
    run_adversarial = not args.skip_adversarial
    run_unanswerable = not args.skip_unanswerable

    if not (run_retrieval or run_adversarial or run_unanswerable):
        print("error: all three suites are skipped — nothing to run.", file=sys.stderr)

        return 2

    print()
    print("=" * 72)
    print("KnowledgeMesh Evaluation")
    print("=" * 72)
    print(f"API: {args.base_url}")

    if run_retrieval:
        print(f"Retrieval dataset:    {args.dataset.resolve()}")

    if run_adversarial:
        print(
            f"Adversarial dataset:  {(args.adversarial_dataset or DEFAULT_ADVERSARIAL_DATASET).resolve()}"
        )

    if run_unanswerable:
        print(
            f"Unanswerable dataset: {(args.unanswerable_dataset or DEFAULT_UNANSWERABLE_DATASET).resolve()}"
        )

    print(f"Output: {args.output.resolve()}")
    print()

    # --------------------------------------------------------
    # Chunk inventory (only needed for the retrieval suite)
    # --------------------------------------------------------

    inventory: ChunkInventory | None = None

    if run_retrieval:
        inventory_path = args.inventory

        if inventory_path is None and DEFAULT_INVENTORY.exists():
            inventory_path = DEFAULT_INVENTORY

        if inventory_path is not None:
            try:
                inventory = ChunkInventory.from_csv(inventory_path)
            except Exception as exc:
                print("ERROR: Could not load chunk inventory.")
                print(f"Type:    {type(exc).__name__}")
                print(f"Details: {exc}")

                return 2

            print_inventory(inventory)
        else:
            print(
                f"Inventory: not found at {DEFAULT_INVENTORY}; running retrieval without chunk-inventory checks."
            )
            print()

            if args.validate_only:
                print(
                    "ERROR: --validate-only needs a chunk inventory for the retrieval suite (use --skip-retrieval to validate only the other suites)."
                )

                return 2

    # --------------------------------------------------------
    # Load whichever datasets are enabled
    # --------------------------------------------------------

    retrieval_cases: list[EvaluationCase] = []
    audit: DatasetAudit | None = None
    adversarial_cases: list[AdversarialCase] = []
    unanswerable_cases: list[EvaluationCase] = []

    if run_retrieval:
        try:
            retrieval_cases = load_dataset(args.dataset, inventory)
        except Exception as exc:
            print()
            print("ERROR: Could not load retrieval dataset.")
            print(f"Type:    {type(exc).__name__}")
            print(f"Details: {exc}")

            return 2

        if inventory is not None:
            audit = audit_dataset(retrieval_cases, inventory)
            print_dataset_audit(audit)

            if not audit.ok and not args.allow_unknown_gold:
                print(
                    "Aborting: fix the private gold chunks above, or pass --allow-unknown-gold to score them as misses."
                )

                return 2

        if args.limit > 0:
            retrieval_cases = retrieval_cases[: args.limit]

    if run_adversarial:
        adversarial_path = args.adversarial_dataset or DEFAULT_ADVERSARIAL_DATASET

        if args.adversarial_dataset is None and not adversarial_path.exists():
            print(f"Adversarial: not found at {adversarial_path}; skipping this suite.")
            print()
            run_adversarial = False
        else:
            try:
                adversarial_cases = load_adversarial_dataset(adversarial_path)
            except Exception as exc:
                print()
                print("ERROR: Could not load adversarial dataset.")
                print(f"Type:    {type(exc).__name__}")
                print(f"Details: {exc}")

                return 2

            unrecognized = sorted(
                {c.attack_type for c in adversarial_cases} - set(ATTACK_TYPES)
            )

            if unrecognized:
                print(
                    f"Adversarial: {len(adversarial_cases)} cases loaded. Unrecognized attack_type values (still scored, just unlabeled): {unrecognized}"
                )
            else:
                print(f"Adversarial: {len(adversarial_cases)} cases loaded.")

            print()

            if args.limit > 0:
                adversarial_cases = adversarial_cases[: args.limit]

    if run_unanswerable:
        unanswerable_path = args.unanswerable_dataset or DEFAULT_UNANSWERABLE_DATASET

        if args.unanswerable_dataset is None and not unanswerable_path.exists():
            print(
                f"Unanswerable: not found at {unanswerable_path}; skipping this suite."
            )
            print()
            run_unanswerable = False
        else:
            try:
                unanswerable_cases = load_unanswerable_dataset(unanswerable_path)
            except Exception as exc:
                print()
                print("ERROR: Could not load unanswerable dataset.")
                print(f"Type:    {type(exc).__name__}")
                print(f"Details: {exc}")

                return 2

            print(f"Unanswerable: {len(unanswerable_cases)} cases loaded.")
            print()

            if args.limit > 0:
                unanswerable_cases = unanswerable_cases[: args.limit]

    if not (run_retrieval or run_adversarial or run_unanswerable):
        print("Nothing left to run after skipping/missing datasets.")

        return 2

    if args.validate_only:
        retrieval_ok = audit is None or audit.ok
        print(
            "Validation passed." if retrieval_ok else "Validation finished with errors."
        )

        return 0 if retrieval_ok else 2

    # --------------------------------------------------------
    # Health
    # --------------------------------------------------------

    client = KnowledgeMeshAPIClient(base_url=args.base_url, timeout=args.timeout)

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
        f"API Health: {health.get('status')} {health.get('service')} v{health.get('version')}"
    )
    print("Private identity: document_id::chunk_id | Web identity: URL / domain")
    print()

    pipeline = build_pipeline_callback(client)

    retrieval_summary = None
    unanswerable_summary = None
    adversarial_summary = None

    # --------------------------------------------------------
    # Retrieval suite
    # --------------------------------------------------------

    if run_retrieval:
        evaluator = KnowledgeMeshEvaluator(
            pipeline=pipeline,
            thresholds=thresholds or None,
            inventory=inventory,
            verbose=args.verbose,
        )

        try:
            retrieval_summary = evaluator.evaluate(
                retrieval_cases,
                metadata={
                    "api_base_url": args.base_url,
                    "dataset": str(args.dataset.resolve()),
                    "case_count": len(retrieval_cases),
                    "private_identity": "document_id::chunk_id",
                    "web_identity": "normalized URL / domain",
                    "inventory": inventory.summary() if inventory is not None else None,
                    "dataset_audit": audit.to_dict() if audit is not None else None,
                },
            )
        except Exception as exc:
            print()
            print("ERROR: Retrieval evaluation failed.")
            print(f"Type:    {type(exc).__name__}")
            print(f"Details: {exc}")
            print()
            traceback.print_exc()

            return 2

        for result in retrieval_summary.cases:
            print_case_result(result)

        print_summary(retrieval_summary)

        unknown_rate = retrieval_summary.metrics.get("retrieved_unknown_id_rate")

        if unknown_rate is not None and unknown_rate > 0:
            print()
            print(
                f"WARNING: {unknown_rate:.1%} of retrieved private chunks are not in the chunk "
                "inventory. The live index probably does not match the chunking run the "
                "inventory was built from; those chunks can never count as hits."
            )

    # --------------------------------------------------------
    # Unanswerable / hallucination-resistance suite
    # --------------------------------------------------------

    if run_unanswerable:
        unanswerable_evaluator = KnowledgeMeshEvaluator(
            pipeline=pipeline,
            thresholds=thresholds or None,
            inventory=None,
            verbose=args.verbose,
        )

        try:
            unanswerable_summary = unanswerable_evaluator.evaluate(
                unanswerable_cases,
                metadata={
                    "api_base_url": args.base_url,
                    "dataset": str(
                        (
                            args.unanswerable_dataset or DEFAULT_UNANSWERABLE_DATASET
                        ).resolve()
                    ),
                    "case_count": len(unanswerable_cases),
                },
            )
        except Exception as exc:
            print()
            print("ERROR: Unanswerable-suite evaluation failed.")
            print(f"Type:    {type(exc).__name__}")
            print(f"Details: {exc}")
            print()
            traceback.print_exc()

            return 2

        for result in unanswerable_summary.cases:
            print_case_result(result)

        print_summary(unanswerable_summary)

    # --------------------------------------------------------
    # Adversarial / safety suite
    # --------------------------------------------------------

    if run_adversarial:
        try:
            adversarial_summary = evaluate_adversarial(
                pipeline,
                adversarial_cases,
                thresholds=thresholds or None,
                verbose=args.verbose,
            )
        except Exception as exc:
            print()
            print("ERROR: Adversarial evaluation failed.")
            print(f"Type:    {type(exc).__name__}")
            print(f"Details: {exc}")
            print()
            traceback.print_exc()

            return 2

        for result in adversarial_summary.cases:
            print_adversarial_case_result(result)

        print_adversarial_summary(adversarial_summary)

    # --------------------------------------------------------
    # Save combined results
    # --------------------------------------------------------

    combined = {
        "retrieval": retrieval_summary.to_dict()
        if retrieval_summary is not None
        else None,
        "unanswerable": unanswerable_summary.to_dict()
        if unanswerable_summary is not None
        else None,
        "adversarial": adversarial_summary.to_dict()
        if adversarial_summary is not None
        else None,
        "metadata": {"api_base_url": args.base_url},
    }

    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)

        with args.output.open("w", encoding="utf-8") as file:
            json.dump(combined, file, indent=2, ensure_ascii=False)
    except Exception as exc:
        print()
        print("ERROR: Could not save evaluation results.")
        print(f"Type:    {type(exc).__name__}")
        print(f"Details: {exc}")

        return 2

    print()
    print(f"Results saved to: {args.output.resolve()}")

    gates = []

    if retrieval_summary is not None:
        gates.append(("retrieval", _gate_passed(retrieval_summary.quality_gate)))

    if unanswerable_summary is not None:
        gates.append(("unanswerable", _gate_passed(unanswerable_summary.quality_gate)))

    if adversarial_summary is not None:
        gates.append(("adversarial", _gate_passed(adversarial_summary.quality_gate)))

    print()

    for name, passed in gates:
        print(f"{name.capitalize()} Quality Gate: {'PASSED' if passed else 'FAILED'}")

    return 0 if all(passed for _, passed in gates) else 1


if __name__ == "__main__":
    sys.exit(main())
