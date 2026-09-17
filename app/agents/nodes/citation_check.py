"""
KnowledgeMesh — Citation Validator

Validates that citations appearing in the generated answer reference
actual evidence chunks present in merged_context.

This is intentionally separate from the Grounding Critic:

Citation Check:
    "Does chunk_3 actually exist?"

Grounding Critic:
    "Does chunk_3 actually support the claim?"

The two checks therefore provide complementary validation.
"""

from __future__ import annotations

import re
from typing import Any

import logfire

from app.agents.state import AgentState


# Supports:
#   [chunk_1]
#   【chunk_1】
#   (chunk_1)
CITATION_PATTERN = re.compile(
    r"(?:"
    r"\[chunk_(\d+)\]"
    r"|【chunk_(\d+)】"
    r"|\(chunk_(\d+)\)"
    r")",
    re.IGNORECASE,
)


def _extract_citation_ids(answer: str) -> list[int]:
    """
    Extract unique cited chunk IDs from the generated answer.
    """
    if not answer:
        return []

    ids: list[int] = []

    for match in CITATION_PATTERN.finditer(answer):
        for group in match.groups():
            if group is not None:
                ids.append(int(group))
                break

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(ids))


def citation_check_node(state: AgentState):
    """
    Verify that every citation in the answer points to an existing
    evidence chunk.

    Important:
    revision_count is NOT incremented here.

    revision_count represents actual answer revision attempts, not
    evaluation passes.
    """

    route = state.get("route", "technical")

    # Simple/conversational responses do not require evidence citations.
    if route == "simple":
        return {
            "citation_valid": True,
            "status": "Citation check skipped for conversational response.",
        }

    answer = state.get("final_answer", "") or ""
    context = state.get("merged_context", "") or ""

    citation_ids = _extract_citation_ids(answer)

    with logfire.span(
        "📎 Citation Check",
        citation_count=len(citation_ids),
    ):
        if not citation_ids:
            logfire.warning("Citation check failed: no citations found in answer.")

            return {
                "citation_valid": False,
                "status": "Citation check failed: no citations found.",
            }

        valid = True
        invalid_ids: list[int] = []

        for chunk_id in citation_ids:
            marker = f"[chunk_{chunk_id}]"

            if marker not in context:
                valid = False
                invalid_ids.append(chunk_id)

                logfire.warning(
                    "Citation references nonexistent chunk",
                    chunk_id=chunk_id,
                )

        if valid:
            logfire.info(
                "Citation check passed",
                citations=citation_ids,
            )
        else:
            logfire.warning(
                "Citation check failed",
                invalid_chunk_ids=invalid_ids,
            )

    return {
        "citation_valid": valid,
        "status": (
            "Citation check passed."
            if valid
            else (f"Citation check failed. Invalid chunks: {invalid_ids}")
        ),
    }
