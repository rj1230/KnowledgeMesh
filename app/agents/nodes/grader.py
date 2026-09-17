"""
Batch document relevance grader.

Grades all retrieved documents in a single LLM call instead of
making one LLM request per document.

Also records LLM grading latency for performance diagnostics.

Evidence quality is evaluated after relevance grading so that:

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

from app.gateway import portkey_client
from app.services.evidence_quality import rank_evidence


# ================================================================
# CONFIGURATION
# ================================================================

MIN_RELEVANT_DOCS = 2
MIN_CONTEXT_SCORE = 0.60

# Keep grading prompts smaller than the final generation context.
MAX_GRADER_DOCUMENT_CHARS = 2500


# ================================================================
# JSON EXTRACTION
# ================================================================


def _extract_json(text: str) -> Any:
    """Extract JSON from plain text or markdown fenced output."""

    text = (text or "").strip()

    # Remove markdown fences.
    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"^```\s*",
        "",
        text,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    # Direct JSON.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find first JSON array.
    match = re.search(
        r"\[[\s\S]*\]",
        text,
    )

    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Find first JSON object.
    match = re.search(
        r"\{[\s\S]*\}",
        text,
    )

    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError("Could not extract valid JSON from grader response")


# ================================================================
# GRADER PAYLOAD
# ================================================================


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


# ================================================================
# BATCH LLM GRADING
# ================================================================


def _batch_grade_documents(query: str, documents):
    """Grade all documents in one LLM call."""

    payload = _build_document_payload(documents)

    prompt = f"""
You are a strict retrieval relevance grader.

User query:
{query}

Below are retrieved documents.

{json.dumps(payload, ensure_ascii=False, indent=2)}

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

    # ============================================================
    # LLM GRADING TIMING
    # ============================================================

    llm_start = time.perf_counter()

    response = portkey_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise enterprise RAG document "
                    "relevance grader. Return valid JSON only."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.0,
    )

    grading_ms = (time.perf_counter() - llm_start) * 1000

    raw = response.choices[0].message.content

    results = _extract_json(raw)

    if not isinstance(results, list):
        raise ValueError("Grader output must be a JSON array")

    logfire.info(
        "📊 Batch document grading timing",
        grading_ms=round(grading_ms, 2),
        documents=len(documents),
    )

    return results, grading_ms


# ================================================================
# EVIDENCE QUALITY HELPERS
# ================================================================


def _evidence_counts(documents):
    """Return evidence-type counts for diagnostics."""

    content_count = sum(1 for doc in documents if doc.get("evidence_type") == "CONTENT")

    reference_count = sum(
        1 for doc in documents if doc.get("evidence_type") == "REFERENCE"
    )

    navigation_count = sum(
        1 for doc in documents if doc.get("evidence_type") == "NAVIGATION"
    )

    return (
        content_count,
        reference_count,
        navigation_count,
    )


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

    This prevents bibliography chunks such as chunk 171 from
    dominating explanatory questions while preserving their
    usefulness for queries such as:

        "Which paper introduced Memorybank?"
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

    # Navigation chunks should never become answer evidence.
    navigation_documents = [
        doc for doc in relevant_documents if doc.get("evidence_type") == "NAVIGATION"
    ]

    # ------------------------------------------------------------
    # Primary case:
    # useful CONTENT exists.
    # ------------------------------------------------------------

    if content_documents:
        selected = content_documents

        # Keep a small number of strong reference documents only
        # when they add genuine supporting metadata.
        #
        # We intentionally do not add them automatically here.
        # This is the critical protection against bibliography
        # chunks being promoted into answer evidence.
        fallback_reference_count = 0

        return (
            selected,
            fallback_reference_count,
            len(navigation_documents),
        )

    # ------------------------------------------------------------
    # Fallback case:
    # no CONTENT evidence exists.
    #
    # We allow relevant reference evidence so the system does not
    # unnecessarily declare context empty for citation/paper
    # identification questions.
    # ------------------------------------------------------------

    if reference_documents:
        return (
            reference_documents,
            len(reference_documents),
            len(navigation_documents),
        )

    return (
        [],
        0,
        len(navigation_documents),
    )


# ================================================================
# MAIN NODE
# ================================================================


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

    The full graded document set is preserved for observability.
    """

    documents = state.get("documents", [])

    query = (
        state.get("current_query")
        or state.get("rewritten_query")
        or state.get("original_query")
        or ""
    )

    # ============================================================
    # NO DOCUMENTS
    # ============================================================

    if not documents:
        return {
            "documents": [],
            "graded_documents": [],
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
        # ========================================================
        # BATCH LLM GRADING
        # ========================================================

        grades, grading_ms = _batch_grade_documents(
            query,
            documents,
        )

        grade_map = {}

        for item in grades:
            try:
                index = int(item["index"])

                score = float(item.get("score", 0.0))

                score = max(
                    0.0,
                    min(1.0, score),
                )

                grade_map[index] = {
                    "relevant": bool(
                        item.get(
                            "relevant",
                            False,
                        )
                    ),
                    "score": score,
                    "reason": str(
                        item.get(
                            "reason",
                            "",
                        )
                    ),
                }

            except (
                KeyError,
                TypeError,
                ValueError,
            ):
                continue

        # ========================================================
        # ENRICH DOCUMENTS WITH GRADER RESULTS
        # ========================================================

        graded_documents = []

        for idx, doc in enumerate(documents):
            grade = grade_map.get(
                idx,
                {
                    "relevant": False,
                    "score": 0.0,
                    "reason": ("No valid grader result."),
                },
            )

            enriched = dict(doc)

            enriched["grader_score"] = grade["score"]

            enriched["grader_relevant"] = grade["relevant"]

            enriched["grader_reason"] = grade["reason"]

            graded_documents.append(enriched)

        # ========================================================
        # EVIDENCE QUALITY CLASSIFICATION
        # ========================================================
        #
        # IMPORTANT:
        # This happens AFTER LLM relevance grading.
        #
        # Therefore:
        #
        #   relevant bibliography
        #
        # can still be identified as:
        #
        #   REFERENCE
        #
        # rather than becoming primary answer evidence.
        # ========================================================

        graded_documents = rank_evidence(graded_documents)

        (
            content_count,
            reference_count,
            navigation_count,
        ) = _evidence_counts(graded_documents)

        # ========================================================
        # SELECT GENERATION CONTEXT
        # ========================================================

        (
            relevant_documents,
            fallback_reference_count,
            excluded_navigation_count,
        ) = _select_generation_documents(graded_documents)

        # ========================================================
        # SORT GENERATION DOCUMENTS
        # ========================================================

        relevant_documents.sort(
            key=lambda d: (
                float(
                    d.get(
                        "grader_score",
                        0.0,
                    )
                ),
                float(
                    d.get(
                        "rerank_score",
                        0.0,
                    )
                    or 0.0
                ),
            ),
            reverse=True,
        )

        relevant_count = len(relevant_documents)

        # ========================================================
        # CONTENT EVIDENCE COUNT
        # ========================================================

        generation_content_count = sum(
            1 for doc in relevant_documents if doc.get("evidence_type") == "CONTENT"
        )

        generation_reference_count = sum(
            1 for doc in relevant_documents if doc.get("evidence_type") == "REFERENCE"
        )

        # ========================================================
        # CONTEXT QUALITY
        # ========================================================
        #
        # Strong context should be based primarily on CONTENT.
        #
        # A collection of bibliography chunks must not become
        # "strong" merely because an LLM grader considers them
        # semantically relevant.
        # ========================================================

        if generation_content_count >= MIN_RELEVANT_DOCS:
            context_quality = "strong"

        elif generation_content_count > 0 or generation_reference_count > 0:
            context_quality = "weak"

        else:
            context_quality = "empty"

        # ========================================================
        # LOG EVIDENCE QUALITY
        # ========================================================

        logfire.info(
            "🔎 Evidence quality filtering",
            total_documents=len(graded_documents),
            content_documents=content_count,
            reference_documents=reference_count,
            navigation_documents=navigation_count,
            generation_documents=relevant_count,
            generation_content_documents=(generation_content_count),
            generation_reference_documents=(generation_reference_count),
            excluded_navigation_documents=(excluded_navigation_count),
        )

        # ========================================================
        # PLAN
        # ========================================================

        plan = list(state.get("plan", []))

        plan.extend(
            [
                (f"Document Grade: {relevant_count}/{len(documents)} usable"),
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
                (f"Context Quality: {context_quality}"),
                (f"Grader Time: {grading_ms:.0f} ms"),
            ]
        )

        # ========================================================
        # RETURN
        # ========================================================

        return {
            # Documents selected for generation.
            #
            # CONTENT is preferred over bibliography/reference
            # chunks when actual explanatory evidence exists.
            "documents": relevant_documents,
            # Preserve EVERY graded and classified document for
            # observability, diagnostics, and future policies.
            "graded_documents": graded_documents,
            "context_quality": context_quality,
            "grader_latency_ms": grading_ms,
            "plan": plan,
            "status": ("Documents graded and evidence quality classified."),
        }

    except Exception as exc:
        # ========================================================
        # FAIL CLOSED
        # ========================================================

        plan = list(state.get("plan", []))

        plan.extend(
            [
                (f"Document Grade: failed ({type(exc).__name__})"),
                "Evidence Quality: skipped",
                "Context Quality: empty",
            ]
        )

        logfire.exception(
            "❌ Document grading failed safely",
            error_type=type(exc).__name__,
        )

        return {
            "documents": [],
            "graded_documents": [],
            "context_quality": "empty",
            "grader_latency_ms": 0.0,
            "plan": plan,
            "status": ("Document grading failed safely."),
        }
