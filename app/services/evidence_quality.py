"""
KnowledgeMesh · Evidence Quality Filter
========================================

Classifies retrieved chunks by evidentiary usefulness.

Important distinction:

    semantic relevance != evidentiary quality

A bibliography chunk can contain the exact terminology from a question
while still being poor evidence for answering that question.

This module therefore identifies reference-dominated chunks without
rejecting normal academic content that contains many inline citations.

Classes:
    CONTENT    -> normal explanatory/taxonomy content
    REFERENCE  -> bibliography/reference dominated
    NAVIGATION -> structural/navigation noise
"""

from __future__ import annotations

import re
from typing import Any, Dict


# ---------------------------------------------------------------------------
# Reference signals
# ---------------------------------------------------------------------------

CORR_PATTERN = re.compile(r"\bCoRR\b", re.IGNORECASE)

PROCEEDINGS_PATTERN = re.compile(
    r"\bProceedings\b",
    re.IGNORECASE,
)

PAGES_PATTERN = re.compile(
    r"\bpages?\s+\d+",
    re.IGNORECASE,
)

DOI_PATTERN = re.compile(
    r"\bdoi\b|10\.\d{4,9}/\S+",
    re.IGNORECASE,
)

ARXIV_PATTERN = re.compile(
    r"\barXiv\b|abs/\d{4}\.\d{4,5}",
    re.IGNORECASE,
)

BIBLIOGRAPHIC_YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")

# Common bibliography author/title patterns.
REFERENCE_ENTRY_PATTERN = re.compile(
    r"(?:"
    r"\[\d+\]\s+[A-Z][A-Za-z\-']+"
    r"|"
    r"\b(?:et al\.|eds?\.|editor)\b"
    r"|"
    r"\b(?:Association for Computational Linguistics|ACL|NeurIPS|ICLR|"
    r"EMNLP|AAAI|IJCAI|COLING|CVPR|ICML)\b"
    r")",
    re.IGNORECASE,
)

# Markdown / document navigation artifacts.
NAVIGATION_PATTERN = re.compile(
    r"^\s*(?:"
    r"(?:table\s+of\s+contents)"
    r"|(?:contents)"
    r"|(?:references)"
    r"|(?:bibliography)"
    r"|(?:index)"
    r")\s*$",
    re.IGNORECASE,
)


def _count(pattern: re.Pattern[str], text: str) -> int:
    return len(pattern.findall(text or ""))


def classify_evidence(text: str) -> Dict[str, Any]:
    """
    Classify a chunk according to evidentiary quality.

    The classifier deliberately distinguishes between:

        CONTENT:
            academic prose or taxonomy with inline citations

        REFERENCE:
            bibliography/reference dominated content

        NAVIGATION:
            obvious structural/navigation noise

    Returns diagnostic metrics so the decision is inspectable.
    """

    text = (text or "").strip()

    if not text:
        return {
            "evidence_type": "NAVIGATION",
            "evidence_score": 0.0,
            "reference_dominated": True,
            "reason": "Empty chunk.",
            "metrics": {},
        }

    normalized = re.sub(r"\s+", " ", text)

    # ---------------------------------------------------------------
    # Navigation detection
    # ---------------------------------------------------------------

    if NAVIGATION_PATTERN.match(normalized):
        return {
            "evidence_type": "NAVIGATION",
            "evidence_score": 0.0,
            "reference_dominated": True,
            "reason": "Chunk is a navigation/structural marker.",
            "metrics": {},
        }

    # ---------------------------------------------------------------
    # Reference signals
    # ---------------------------------------------------------------

    citation_count = len(re.findall(r"\[\s*\d+\s*\]", normalized))

    corr_count = _count(CORR_PATTERN, normalized)
    proceedings_count = _count(PROCEEDINGS_PATTERN, normalized)
    pages_count = _count(PAGES_PATTERN, normalized)
    doi_count = _count(DOI_PATTERN, normalized)
    arxiv_count = _count(ARXIV_PATTERN, normalized)
    year_count = _count(BIBLIOGRAPHIC_YEAR_PATTERN, normalized)
    reference_entry_count = _count(
        REFERENCE_ENTRY_PATTERN,
        normalized,
    )

    strong_reference_signals = (
        corr_count + proceedings_count + pages_count + doi_count + arxiv_count
    )

    # ---------------------------------------------------------------
    # Reference dominance
    # ---------------------------------------------------------------

    # A bibliography chunk generally has multiple strong signals.
    #
    # Example:
    #
    #   CoRR + Proceedings + pages + paper title + year
    #
    # A normal academic chunk may have dozens of [123] citations but
    # no CoRR / Proceedings / DOI / page-number signals.
    #
    # Therefore citations alone NEVER make a chunk REFERENCE.
    reference_dominated = (
        strong_reference_signals >= 3
        or (strong_reference_signals >= 2 and reference_entry_count >= 2)
        or (corr_count >= 2 and year_count >= 2)
    )

    if reference_dominated:
        evidence_type = "REFERENCE"
        evidence_score = 0.15

        reason = (
            "Reference-dominated chunk: "
            f"CoRR={corr_count}, "
            f"Proceedings={proceedings_count}, "
            f"pages={pages_count}, "
            f"DOI={doi_count}, "
            f"arXiv={arxiv_count}, "
            f"reference_entries={reference_entry_count}."
        )

        return {
            "evidence_type": evidence_type,
            "evidence_score": evidence_score,
            "reference_dominated": True,
            "reason": reason,
            "metrics": {
                "citation_count": citation_count,
                "corr_count": corr_count,
                "proceedings_count": proceedings_count,
                "pages_count": pages_count,
                "doi_count": doi_count,
                "arxiv_count": arxiv_count,
                "year_count": year_count,
                "reference_entry_count": reference_entry_count,
                "strong_reference_signals": strong_reference_signals,
            },
        }

    # ---------------------------------------------------------------
    # Normal content
    # ---------------------------------------------------------------

    # Citation-heavy academic prose remains CONTENT.
    evidence_score = 1.0

    if citation_count >= 20:
        reason = "Citation-heavy academic content, but not reference dominated."
    elif citation_count > 0:
        reason = "Academic content with inline citations."
    else:
        reason = "Normal explanatory content."

    return {
        "evidence_type": "CONTENT",
        "evidence_score": evidence_score,
        "reference_dominated": False,
        "reason": reason,
        "metrics": {
            "citation_count": citation_count,
            "corr_count": corr_count,
            "proceedings_count": proceedings_count,
            "pages_count": pages_count,
            "doi_count": doi_count,
            "arxiv_count": arxiv_count,
            "year_count": year_count,
            "reference_entry_count": reference_entry_count,
            "strong_reference_signals": strong_reference_signals,
        },
    }


def annotate_document(document: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add evidence-quality metadata to a retrieved document.

    The original document is copied rather than mutated.
    """

    content = document.get("content") or document.get("text") or ""

    classification = classify_evidence(content)

    return {
        **document,
        "evidence_type": classification["evidence_type"],
        "evidence_score": classification["evidence_score"],
        "reference_dominated": classification["reference_dominated"],
        "evidence_reason": classification["reason"],
        "evidence_metrics": classification["metrics"],
    }


def rank_evidence(documents: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    """
    Annotate and deterministically rank documents by evidence quality.

    CONTENT documents are preferred over REFERENCE documents.

    Within the same evidence class, the existing rerank/grader score
    remains the primary signal.

    Reference chunks are NOT deleted.
    """

    annotated = [annotate_document(document) for document in documents]

    def sort_key(document: Dict[str, Any]) -> tuple:
        evidence_type = document.get("evidence_type", "CONTENT")

        evidence_priority = {
            "CONTENT": 2,
            "REFERENCE": 1,
            "NAVIGATION": 0,
        }.get(evidence_type, 0)

        rerank_score = float(
            document.get("rerank_score") or document.get("score") or 0.0
        )

        grader_score = float(document.get("grader_score") or 0.0)

        return (
            evidence_priority,
            grader_score,
            rerank_score,
        )

    annotated.sort(key=sort_key, reverse=True)

    return annotated
