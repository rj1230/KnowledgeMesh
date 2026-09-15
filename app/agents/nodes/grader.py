"""
Batch document relevance grader.

Grades all retrieved documents in a single LLM call instead of
making one LLM request per document.
"""

import json
import re
from typing import Any

from app.gateway import portkey_client

MIN_RELEVANT_DOCS = 2
MIN_CONTEXT_SCORE = 0.60


def _extract_json(text: str) -> Any:
    """Extract JSON from plain text or markdown fenced output."""

    text = (text or "").strip()

    # Remove markdown fences
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Direct JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find first JSON array
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Find first JSON object
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError("Could not extract valid JSON from grader response")


def _build_document_payload(documents):
    """Create compact numbered documents for the grader."""

    payload = []

    for idx, doc in enumerate(documents):
        payload.append(
            {
                "index": idx,
                "content": str(doc.get("content", ""))[:5000],
                "source": doc.get("source", ""),
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
- return exactly one result for every document
- preserve the original document index
"""

    response = portkey_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise enterprise RAG document relevance grader. "
                    "Return valid JSON only."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.0,
    )

    raw = response.choices[0].message.content
    results = _extract_json(raw)

    if not isinstance(results, list):
        raise ValueError("Grader output must be a JSON array")

    return results


def grade_documents_node(state):
    """
    Grade all retrieved documents in ONE LLM call.

    Strong:
        >= MIN_RELEVANT_DOCS relevant documents

    Weak:
        Some relevant documents, but below the strong threshold

    Empty:
        No relevant documents
    """

    documents = state.get("documents", [])
    query = (
        state.get("current_query")
        or state.get("rewritten_query")
        or state.get("original_query")
        or ""
    )

    if not documents:
        return {
            "documents": [],
            "context_quality": "empty",
            "plan": state.get("plan", [])
            + [
                "Document Grade: 0/0 relevant",
                "Context Quality: empty",
            ],
            "status": "No private documents retrieved.",
        }

    try:
        grades = _batch_grade_documents(query, documents)

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
                continue

        graded_documents = []

        for idx, doc in enumerate(documents):
            grade = grade_map.get(
                idx,
                {
                    "relevant": False,
                    "score": 0.0,
                    "reason": "No valid grader result.",
                },
            )

            enriched = dict(doc)

            enriched["grader_score"] = grade["score"]
            enriched["grader_relevant"] = grade["relevant"]
            enriched["grader_reason"] = grade["reason"]

            graded_documents.append(enriched)

        # Highest grader score first
        graded_documents.sort(
            key=lambda d: float(d.get("grader_score", 0.0)),
            reverse=True,
        )

        # Keep only relevant documents above threshold
        relevant_documents = [
            doc
            for doc in graded_documents
            if doc.get("grader_relevant")
            and float(doc.get("grader_score", 0.0)) >= MIN_CONTEXT_SCORE
        ]

        relevant_count = len(relevant_documents)

        if relevant_count >= MIN_RELEVANT_DOCS:
            context_quality = "strong"
        elif relevant_count > 0:
            context_quality = "weak"
        else:
            context_quality = "empty"

        plan = list(state.get("plan", []))

        plan.extend(
            [
                f"Document Grade: {relevant_count}/{len(documents)} relevant",
                f"Context Quality: {context_quality}",
            ]
        )

        return {
            "documents": relevant_documents,
            "context_quality": context_quality,
            "plan": plan,
            "status": "Documents graded.",
        }

    except Exception as exc:
        # Fail closed.
        # Never treat ungraded evidence as trustworthy.
        plan = list(state.get("plan", []))

        plan.extend(
            [
                f"Document Grade: failed ({type(exc).__name__})",
                "Context Quality: empty",
            ]
        )

        return {
            "documents": [],
            "context_quality": "empty",
            "plan": plan,
            "status": "Document grading failed safely.",
        }
