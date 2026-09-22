"""
Batch document relevance grader.

Grades all retrieved documents in a single LLM call instead of
making one LLM request per document.

Pipeline:

    Dense Retrieval TOP-50
            ↓
    FlashRank TOP-15
            ↓
    LLM Relevance Grader
            ↓
    Evidence Quality Classification
            ↓
    Deterministic Generation Selection TOP-5

Important distinction:

    semantic relevance != evidentiary quality

Reference-heavy bibliography chunks may be highly relevant to a
query because they contain exact paper names or terminology, but
they should not normally become the primary evidence for an answer.

The evidence-quality layer therefore:

    CONTENT
        Preferred as primary answer evidence.

    REFERENCE
        Retained for observability and fallback, but deprioritized
        when useful CONTENT evidence exists.

    NAVIGATION
        Structural/noise content that should not normally be used
        for answer generation.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import logfire

from app.gateway import create_chat_completion
from app.services.evidence_quality import rank_evidence

MIN_RELEVANT_DOCS = 2

# Keep this unchanged while diagnosing the grader.
MIN_CONTEXT_SCORE = 0.60

# Maximum number of documents passed to downstream generation.
#
# Retrieval deliberately keeps a larger candidate pool:
#
#     Qdrant     -> 50
#     FlashRank  -> 15
#     Grader     -> 15
#     Generation -> 5
#
# This separates retrieval recall from generation context size.
MAX_GENERATION_DOCUMENTS = 5

# Keep grading prompts smaller than the final generation context.
MAX_GRADER_DOCUMENT_CHARS = 2500


def _extract_json(text: str) -> Any:
    """Extract JSON from plain text or markdown fenced output."""

    text = (text or "").strip()

    # Remove markdown fences.
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Direct JSON.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find first JSON array.
    match = re.search(r"\[[\s\S]*\]", text)

    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Find first JSON object.
    match = re.search(r"\{[\s\S]*\}", text)

    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError("Could not extract valid JSON from grader response")


def _build_document_payload(documents):
    """
    Create compact numbered documents for the grader.

    We intentionally send less text to the grader than to the
    final answer generator. Relevance grading usually does not
    require the entire chunk.
    """

    payload = []

    for idx, doc in enumerate(documents):
        payload.append(
            {
                "index": idx,
                "content": str(doc.get("content", ""))[:MAX_GRADER_DOCUMENT_CHARS],
            }
        )

    return payload


def _batch_grade_documents(query: str, documents):
    """Grade all documents in one LLM call."""

    payload = _build_document_payload(documents)

    prompt = f"""
You are a strict retrieval relevance grader.

User query:
{query}

Below are retrieved documents.

{
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
    }

For EACH document, determine whether it is relevant to answering
the user's query.

Return ONLY a JSON array.

Required format:

[
  {{
    "index": 0,
    "relevant": true,
    "score": 0.95,
    "reason": "Directly answers the query."
  }}
]

Rules:

- score must be between 0.0 and 1.0
- relevant=true only when the document provides useful evidence
- directly relevant evidence should generally score >= 0.75
- partially useful evidence may score 0.60-0.74
- irrelevant evidence should score below 0.60
- do not reward keyword overlap alone
- judge semantic usefulness for answering the query
- bibliography/reference entries may be relevant to identifying
  papers, but they are weaker evidence for explanatory questions
- do not assume that mentioning a paper proves the paper's findings
- return exactly one result for every document
- preserve the original document index
"""

    llm_start = time.perf_counter()

    response = create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise enterprise RAG document "
                    "relevance grader. Return valid JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )

    grading_ms = (time.perf_counter() - llm_start) * 1000

    raw = response.choices[0].message.content

    print("\n" + "=" * 80)
    print("🧪 GRADER RAW RESPONSE")
    print(raw)
    print("=" * 80)

    results = _extract_json(raw)

    if not isinstance(results, list):
        raise ValueError("Grader output must be a JSON array")

    print(
        f"🧪 GRADER RESULT COUNT | documents={len(documents)} | results={len(results)}"
    )

    logfire.info(
        "📊 Batch document grading timing",
        grading_ms=round(grading_ms, 2),
        documents=len(documents),
        grader_results=len(results),
    )

    return results, grading_ms


def _evidence_counts(documents):
    """Return evidence-type counts for diagnostics."""

    content_count = sum(1 for doc in documents if doc.get("evidence_type") == "CONTENT")
    reference_count = sum(
        1 for doc in documents if doc.get("evidence_type") == "REFERENCE"
    )
    navigation_count = sum(
        1 for doc in documents if doc.get("evidence_type") == "NAVIGATION"
    )

    return content_count, reference_count, navigation_count


def _select_generation_documents(documents):
    """
    Select documents suitable for answer generation.

    Policy:

    1. Relevant CONTENT evidence is preferred.
    2. Relevant REFERENCE evidence is retained as fallback.
    3. NAVIGATION evidence is never used as primary evidence.
    4. If enough CONTENT evidence exists, reference-only chunks
       are excluded from the generation context.
    5. If CONTENT evidence is insufficient, relevant REFERENCE
       chunks may be retained rather than failing unnecessarily.
    6. Generation context is deterministically capped at
       MAX_GENERATION_DOCUMENTS.

    The complete graded document set remains available separately
    through `graded_documents` for observability and diagnostics.
    """

    relevant_documents = [
        doc
        for doc in documents
        if doc.get("grader_relevant")
        and float(doc.get("grader_score", 0.0)) >= MIN_CONTEXT_SCORE
    ]

    content_documents = [
        doc
        for doc in relevant_documents
        if doc.get("evidence_type", "CONTENT") == "CONTENT"
    ]

    reference_documents = [
        doc for doc in relevant_documents if doc.get("evidence_type") == "REFERENCE"
    ]

    navigation_documents = [
        doc for doc in relevant_documents if doc.get("evidence_type") == "NAVIGATION"
    ]

    print(
        "\n🧪 GENERATION SELECTION INPUT | "
        f"total={len(documents)} | "
        f"relevant={len(relevant_documents)} | "
        f"content={len(content_documents)} | "
        f"reference={len(reference_documents)} | "
        f"navigation={len(navigation_documents)} | "
        f"threshold={MIN_CONTEXT_SCORE}"
    )

    # Primary case: useful CONTENT exists.
    if content_documents:
        selected = content_documents[:MAX_GENERATION_DOCUMENTS]
        print(f"🧪 GENERATION SELECTION | selected_content={len(selected)}")
        return selected, 0, len(navigation_documents)

    # Fallback case: no CONTENT evidence exists.
    if reference_documents:
        selected = reference_documents[:MAX_GENERATION_DOCUMENTS]
        print(f"🧪 GENERATION SELECTION | selected_reference={len(selected)}")
        return selected, len(reference_documents), len(navigation_documents)

    print("🧪 GENERATION SELECTION | selected=0")

    return [], 0, len(navigation_documents)


def grade_documents_node(state):
    """
    Grade all retrieved documents in ONE LLM call.

    Strong:
        >= MIN_RELEVANT_DOCS relevant CONTENT documents

    Weak:
        Some usable evidence, but below the strong threshold

    Empty:
        No usable evidence

    Evidence-quality policy:

        CONTENT > REFERENCE > NAVIGATION

    Important pipeline separation:

        retrieved documents
            ↓
        graded_documents
            ↓
        evidence classification
            ↓
        generation documents MAX-5

    The full graded document set is preserved for observability.
    """

    documents = state.get("documents", [])

    query = (
        state.get("current_query")
        or state.get("rewritten_query")
        or state.get("original_query")
        or ""
    )

    print(f"\n🧪 GRADER INPUT | query={query!r} | documents={len(documents)}")

    if not documents:
        return {
            "documents": [],
            "generation_documents": [],
            "graded_documents": [],
            "grader_status": "skipped",
            "grader_error_type": None,
            "grading_mode": "no_documents",
            "context_quality": "empty",
            "grader_latency_ms": 0.0,
            "plan": state.get("plan", [])
            + [
                "Document Grade: 0/0 relevant",
                "Evidence Quality: no documents",
                "Context Quality: empty",
            ],
            "status": "No private documents retrieved.",
        }

    try:
        grades, grading_ms = _batch_grade_documents(query, documents)

        grade_map = {}

        for item in grades:
            try:
                index = int(item["index"])
                score = float(item.get("score", 0.0))
                score = max(0.0, min(1.0, score))

                grade_map[index] = {
                    "relevant": bool(item.get("relevant", False)),
                    "score": score,
                    "reason": str(item.get("reason", "")),
                }

            except (KeyError, TypeError, ValueError):
                print(f"⚠️ GRADER INVALID ITEM | item={item!r}")
                continue

        expected_indices = set(range(len(documents)))
        received_indices = set(grade_map.keys())
        missing_indices = sorted(expected_indices - received_indices)
        extra_indices = sorted(received_indices - expected_indices)

        print(
            "🧪 GRADER INDEX COVERAGE | "
            f"expected={sorted(expected_indices)} | "
            f"received={sorted(received_indices)} | "
            f"missing={missing_indices} | "
            f"extra={extra_indices}"
        )

        if missing_indices:
            logfire.warning(
                "⚠️ Grader returned incomplete document coverage",
                expected_indices=sorted(expected_indices),
                received_indices=sorted(received_indices),
                missing_indices=missing_indices,
                extra_indices=extra_indices,
            )

        graded_documents = []

        for idx, doc in enumerate(documents):
            grade = grade_map.get(
                idx,
                {"relevant": False, "score": 0.0, "reason": "No valid grader result."},
            )

            enriched = dict(doc)
            enriched["grader_score"] = grade["score"]
            enriched["grader_relevant"] = grade["relevant"]
            enriched["grader_reason"] = grade["reason"]

            graded_documents.append(enriched)

            print(
                "🧪 GRADED DOCUMENT | "
                f"idx={idx} | "
                f"grader_relevant={enriched.get('grader_relevant')} | "
                f"grader_score={enriched.get('grader_score')} | "
                f"evidence_type={enriched.get('evidence_type')} | "
                f"reason={enriched.get('grader_reason')}"
            )

        graded_documents = rank_evidence(graded_documents)

        content_count, reference_count, navigation_count = _evidence_counts(
            graded_documents
        )

        print(
            "🧪 EVIDENCE CLASSIFICATION | "
            f"content={content_count} | "
            f"reference={reference_count} | "
            f"navigation={navigation_count}"
        )

        (
            relevant_documents,
            fallback_reference_count,
            excluded_navigation_count,
        ) = _select_generation_documents(graded_documents)

        relevant_documents.sort(
            key=lambda d: (
                1 if d.get("evidence_type", "CONTENT") == "CONTENT" else 0,
                float(d.get("grader_score", 0.0)),
                float(d.get("rerank_score", 0.0) or 0.0),
            ),
            reverse=True,
        )

        relevant_documents = relevant_documents[:MAX_GENERATION_DOCUMENTS]
        relevant_count = len(relevant_documents)

        generation_content_count = sum(
            1 for doc in relevant_documents if doc.get("evidence_type") == "CONTENT"
        )

        generation_reference_count = sum(
            1 for doc in relevant_documents if doc.get("evidence_type") == "REFERENCE"
        )

        if generation_content_count >= MIN_RELEVANT_DOCS:
            context_quality = "strong"
        elif generation_content_count > 0 or generation_reference_count > 0:
            context_quality = "weak"
        else:
            context_quality = "empty"

        print(
            "🧪 GRADER FINAL | "
            f"graded={len(graded_documents)} | "
            f"generation={relevant_count} | "
            f"generation_content={generation_content_count} | "
            f"generation_reference={generation_reference_count} | "
            f"context_quality={context_quality}"
        )

        logfire.info(
            "🔎 Evidence quality filtering",
            total_documents=len(graded_documents),
            content_documents=content_count,
            reference_documents=reference_count,
            navigation_documents=navigation_count,
            generation_documents=relevant_count,
            generation_document_limit=MAX_GENERATION_DOCUMENTS,
            generation_content_documents=generation_content_count,
            generation_reference_documents=generation_reference_count,
            excluded_navigation_documents=excluded_navigation_count,
            fallback_reference_count=fallback_reference_count,
        )

        plan = list(state.get("plan", []))

        plan.extend(
            [
                f"Document Grade: {relevant_count}/{len(documents)} usable",
                (
                    f"Evidence Quality: "
                    f"{content_count} content / "
                    f"{reference_count} reference / "
                    f"{navigation_count} navigation"
                ),
                (
                    f"Generation Evidence: "
                    f"{generation_content_count} content / "
                    f"{generation_reference_count} reference"
                ),
                f"Generation Context: {relevant_count}/{MAX_GENERATION_DOCUMENTS}",
                f"Context Quality: {context_quality}",
                f"Grader Time: {grading_ms:.0f} ms",
            ]
        )

        return {
            "documents": documents,
            "graded_documents": graded_documents,
            "generation_documents": relevant_documents,
            "grader_status": "success",
            "grader_error_type": None,
            "grading_mode": "graded",
            "context_quality": context_quality,
            "grader_latency_ms": grading_ms,
            "plan": plan,
            "status": "Documents graded and evidence quality classified.",
        }

    except Exception as exc:
        # Safe degraded mode: private evidence is preserved but marked unverified.
        error_type = type(exc).__name__

        print("\n" + "=" * 80)
        print("❌ DOCUMENT GRADER FAILED")
        print(f"error_type={error_type}")
        print(f"error={exc}")
        print("=" * 80)

        plan = list(state.get("plan", []))

        plan.extend(
            [
                f"Document Grade: failed ({error_type})",
                "Evidence Quality: unavailable",
                "Context Quality: unverified",
                f"Private Evidence Preserved: {len(documents)}",
            ]
        )

        logfire.exception(
            "❌ Document grading failed; private evidence preserved "
            "but marked unverified.",
            error_type=error_type,
            retrieved_documents=len(documents),
            generation_documents=0,
        )

        return {
            "documents": documents,
            "graded_documents": [],
            "generation_documents": [],
            "grader_status": "failed",
            "grader_error_type": error_type,
            "grading_mode": "unavailable",
            "context_quality": "unverified",
            "grader_latency_ms": 0.0,
            "plan": plan,
            "status": (
                "Document grading unavailable; "
                "private evidence preserved but unverified."
            ),
        }
