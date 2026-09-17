"""
KnowledgeMesh — Grounding Critic

Validates whether factual claims in the generated answer are supported
by the evidence chunks used to generate the answer.

Supported citation formats:
    [chunk_1]
    【chunk_1】
    (chunk_1)

The critic is intentionally strict about factual grounding while
avoiding false failures caused by:

    - Markdown headings
    - standalone bold/italic titles
    - Markdown tables
    - table separators
    - repeated citations
    - formatting-only lines

Pipeline:

    Generated Answer
          ↓
    Claim Extraction
          ↓
    Citation Resolution
          ↓
    Evidence Resolution
          ↓
    Entailment Evaluation
          ↓
    Grounding Score
          ↓
    Revision Feedback
"""

from __future__ import annotations

import re
from typing import Any

import logfire

from app.agents.state import AgentState
from app.tools.entailment_tool import score_entailment


# =====================================================================
# CONFIGURATION
# =====================================================================

CITATION_PATTERN = re.compile(
    r"(?P<citation>"
    r"\[chunk_(?P<bracket>\d+)\]"
    r"|【chunk_(?P<fullwidth>\d+)】"
    r"|\(chunk_(?P<paren>\d+)\)"
    r")",
    re.IGNORECASE,
)

# Markdown headings:
#
#   # Heading
#   ## Heading
#   ### Heading
#
HEADING_PATTERN = re.compile(r"^\s{0,3}#{1,6}\s+")

# Standalone bold/italic title lines:
#
#   **Main memory-related capabilities**
#   *Main memory-related capabilities*
#   __Main memory-related capabilities__
#
# These are presentation elements, not factual claims.
STANDALONE_EMPHASIS_PATTERN = re.compile(
    r"^\s*"
    r"(?:"
    r"\*\*(?P<bold>.+?)\*\*"
    r"|"
    r"__(?P<underscore>.+?)__"
    r"|"
    r"\*(?P<italic>[^*].*?)\*"
    r"|"
    r"_(?P<italic_underscore>[^_].*?)_"
    r")"
    r"\s*$"
)

# Markdown table separators.
TABLE_SEPARATOR_PATTERN = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*"
    r"(?:\|\s*:?-{3,}:?\s*)+"
    r"\|?\s*$"
)

# Markdown table headers.
TABLE_HEADER_PATTERN = re.compile(
    r"^\s*\|?\s*"
    r"(?:"
    r"capability|"
    r"description|"
    r"evidence|"
    r"source|"
    r"claim|"
    r"answer|"
    r"explanation|"
    r"topic|"
    r"information|"
    r"feature|"
    r"details"
    r")"
    r"\s*"
    r"(?:\|\s*"
    r"(?:"
    r"capability|"
    r"description|"
    r"evidence|"
    r"source|"
    r"claim|"
    r"answer|"
    r"explanation|"
    r"topic|"
    r"information|"
    r"feature|"
    r"details"
    r")"
    r"\s*)+"
    r"\|?\s*$",
    re.IGNORECASE,
)

MARKDOWN_PREFIX_PATTERN = re.compile(r"^\s*(?:[-*•]\s+|\d+[.)]\s+)")

MIN_CLAIM_LENGTH = 8

# HHEMv2 / entailment threshold.
#
# IMPORTANT:
# Do not lower this simply to make answers pass.
ENTAILMENT_THRESHOLD = 0.50


# =====================================================================
# CITATION HELPERS
# =====================================================================


def _extract_chunk_id(
    match: re.Match[str],
) -> int:
    """Extract the numeric chunk ID from a citation."""

    for group_name in (
        "bracket",
        "fullwidth",
        "paren",
    ):
        value = match.group(group_name)

        if value is not None:
            return int(value)

    raise ValueError("Citation match did not contain a chunk ID.")


# =====================================================================
# TEXT NORMALIZATION
# =====================================================================


def _clean_claim(
    text: str,
) -> str:
    """
    Normalize Markdown and whitespace so the entailment model receives
    a clean semantic claim.
    """

    text = text.strip()

    if not text:
        return ""

    text = MARKDOWN_PREFIX_PATTERN.sub(
        "",
        text,
    )

    text = text.strip().strip("|").strip()

    text = CITATION_PATTERN.sub(
        "",
        text,
    )

    # Remove common Markdown emphasis.
    text = re.sub(
        r"\*\*(.*?)\*\*",
        r"\1",
        text,
    )

    text = re.sub(
        r"__(.*?)__",
        r"\1",
        text,
    )

    text = re.sub(
        r"`([^`]*)`",
        r"\1",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


# =====================================================================
# LINE CLASSIFICATION
# =====================================================================


def _is_standalone_emphasis(
    text: str,
) -> bool:
    """
    Detect lines that consist entirely of Markdown emphasis.

    Example:

        **Main memory-related capabilities of LLM-based agents**

    This is a title/heading and should not be treated as an uncited
    factual claim.
    """

    stripped = text.strip()

    if not stripped:
        return False

    return bool(STANDALONE_EMPHASIS_PATTERN.fullmatch(stripped))


def _is_non_claim_line(
    text: str,
) -> bool:
    """
    Return True for Markdown/formatting lines that should not be
    evaluated as factual claims.
    """

    stripped = text.strip()

    if not stripped:
        return True

    # ---------------------------------------------------------------
    # Markdown headings
    # ---------------------------------------------------------------

    if HEADING_PATTERN.match(stripped):
        return True

    # ---------------------------------------------------------------
    # Standalone bold / italic titles
    # ---------------------------------------------------------------

    if _is_standalone_emphasis(stripped):
        return True

    # ---------------------------------------------------------------
    # Markdown table separators
    # ---------------------------------------------------------------

    if TABLE_SEPARATOR_PATTERN.match(stripped):
        return True

    # ---------------------------------------------------------------
    # Markdown table headers
    # ---------------------------------------------------------------

    if TABLE_HEADER_PATTERN.match(stripped):
        return True

    # ---------------------------------------------------------------
    # A line containing only a citation
    # ---------------------------------------------------------------

    if CITATION_PATTERN.fullmatch(stripped):
        return True

    return False


# =====================================================================
# PROSE CLAIM EXTRACTION
# =====================================================================


def _split_prose_into_claims(
    text: str,
) -> list[str]:
    """
    Split prose into reasonably small factual claims.
    """

    text = text.strip()

    if not text:
        return []

    parts = re.split(
        r"(?<=[.!?])\s+(?=[A-Z0-9])",
        text,
    )

    if not parts:
        parts = [text]

    claims: list[str] = []

    for part in parts:
        cleaned = _clean_claim(part)

        if len(cleaned) >= MIN_CLAIM_LENGTH:
            claims.append(cleaned)

    return claims


# =====================================================================
# MARKDOWN TABLE HELPERS
# =====================================================================


def _split_table_cells(
    line: str,
) -> list[str]:
    """
    Split a Markdown table row into cells.
    """

    stripped = line.strip()

    if stripped.startswith("|"):
        stripped = stripped[1:]

    if stripped.endswith("|"):
        stripped = stripped[:-1]

    return [cell.strip() for cell in stripped.split("|")]


def _extract_table_claim(
    line: str,
    citation_match: re.Match[str],
) -> str:
    """
    Extract the semantic description cell from a Markdown table.

    Typical structure:

        | Capability | What it enables | [chunk_1] |

    The second cell is treated as the factual description.
    """

    cells = _split_table_cells(line)

    if len(cells) >= 2:
        description = _clean_claim(cells[1])

        if len(description) >= MIN_CLAIM_LENGTH:
            return description

    # Fallback for unusual tables.
    before_citation = line[: citation_match.start()]

    fallback_cells = _split_table_cells(before_citation)

    if len(fallback_cells) >= 2:
        fallback = _clean_claim(fallback_cells[-1])

        if len(fallback) >= MIN_CLAIM_LENGTH:
            return fallback

    return _clean_claim(before_citation)


# =====================================================================
# CLAIM DEDUPLICATION
# =====================================================================


def _deduplicate_claims(
    claims: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Remove duplicate semantic claims referring to the same evidence
    chunk.

    This prevents repeated table citations from causing duplicate
    entailment evaluations.
    """

    unique_claims: list[dict[str, Any]] = []

    seen: set[tuple[str, int]] = set()

    for claim in claims:
        normalized_text = re.sub(
            r"\s+",
            " ",
            str(
                claim.get(
                    "text",
                    "",
                )
            )
            .strip()
            .lower(),
        )

        chunk_id = int(
            claim.get(
                "cited_chunk_id",
                -1,
            )
        )

        key = (
            normalized_text,
            chunk_id,
        )

        if key in seen:
            continue

        seen.add(key)
        unique_claims.append(claim)

    return unique_claims


# =====================================================================
# CLAIM EXTRACTION
# =====================================================================


def _extract_claims(
    answer: str,
) -> list[dict[str, Any]]:
    """
    Extract individual claims and their cited chunks.
    """

    if not answer or not answer.strip():
        return []

    claims: list[dict[str, Any]] = []

    for line_number, line in enumerate(
        answer.splitlines(),
        start=1,
    ):
        if _is_non_claim_line(line):
            continue

        citations = list(CITATION_PATTERN.finditer(line))

        if not citations:
            continue

        is_table_row = "|" in line

        for citation_match in citations:
            chunk_id = _extract_chunk_id(citation_match)

            if is_table_row:
                claim_text = _extract_table_claim(
                    line=line,
                    citation_match=citation_match,
                )

            else:
                before_citation = line[: citation_match.start()]

                candidate_claims = _split_prose_into_claims(before_citation)

                if not candidate_claims:
                    continue

                claim_text = candidate_claims[-1]

            claim_text = _clean_claim(claim_text)

            if len(claim_text) < MIN_CLAIM_LENGTH:
                continue

            claims.append(
                {
                    "text": claim_text,
                    "cited_chunk_id": chunk_id,
                    "line_number": line_number,
                    "citation": citation_match.group("citation"),
                }
            )

    return _deduplicate_claims(claims)


# =====================================================================
# UNCITED CONTENT
# =====================================================================


def _extract_substantive_uncited_lines(
    answer: str,
) -> list[str]:
    """
    Find substantive answer lines without citations.

    Formatting-only Markdown is ignored.

    In particular, standalone bold/italic titles such as:

        **Main memory-related capabilities of LLM-based agents**

    are not factual claims and are therefore ignored.
    """

    if not answer:
        return []

    uncited: list[str] = []

    for line in answer.splitlines():
        stripped = line.strip()

        # ------------------------------------------------------------
        # Formatting / heading lines
        # ------------------------------------------------------------

        if _is_non_claim_line(stripped):
            continue

        # ------------------------------------------------------------
        # Citation-bearing lines
        # ------------------------------------------------------------

        if CITATION_PATTERN.search(stripped):
            continue

        cleaned = _clean_claim(stripped)

        if len(cleaned) < MIN_CLAIM_LENGTH:
            continue

        # ------------------------------------------------------------
        # Markdown table protection
        # ------------------------------------------------------------

        if "|" in stripped:
            cells = _split_table_cells(stripped)

            if len(cells) >= 2:
                normalized_cells = {cell.lower().strip() for cell in cells}

                common_header_terms = {
                    "capability",
                    "description",
                    "evidence",
                    "source",
                    "claim",
                    "answer",
                    "explanation",
                    "topic",
                    "information",
                    "feature",
                    "details",
                }

                if normalized_cells & common_header_terms:
                    continue

        uncited.append(cleaned)

    return uncited


# =====================================================================
# CONTEXT EXTRACTION
# =====================================================================


def _get_chunk_text(
    merged_context: str,
    chunk_id: int,
) -> str:
    """
    Extract canonical evidence for [chunk_N].
    """

    if not merged_context:
        return ""

    marker = f"[chunk_{chunk_id}]"

    start_marker = merged_context.find(marker)

    if start_marker == -1:
        return ""

    content_start = start_marker + len(marker)

    next_marker = re.search(
        r"\[chunk_\d+\]",
        merged_context[content_start:],
    )

    if next_marker:
        content_end = content_start + next_marker.start()
    else:
        content_end = len(merged_context)

    return merged_context[content_start:content_end].strip()


# =====================================================================
# CLAIM EVALUATION
# =====================================================================


def _evaluate_claim(
    claim: dict[str, Any],
    merged_context: str,
) -> dict[str, Any]:
    """
    Evaluate one claim against its cited evidence.
    """

    chunk_id = int(claim["cited_chunk_id"])

    claim_text = str(claim["text"])

    chunk_text = _get_chunk_text(
        merged_context=merged_context,
        chunk_id=chunk_id,
    )

    if not chunk_text:
        return {
            "claim": claim_text,
            "cited_chunk_id": chunk_id,
            "line_number": claim.get("line_number"),
            "score": 0.0,
            "supported": False,
            "reason": ("Cited chunk does not exist in merged context."),
        }

    try:
        score = float(
            score_entailment(
                premise=chunk_text,
                hypothesis=claim_text,
            )
        )

    except Exception as exc:
        logfire.exception(
            "Entailment evaluation failed",
            error_type=type(exc).__name__,
            chunk_id=chunk_id,
        )

        return {
            "claim": claim_text,
            "cited_chunk_id": chunk_id,
            "line_number": claim.get("line_number"),
            "score": 0.0,
            "supported": False,
            "reason": (f"Entailment evaluation failed: {type(exc).__name__}"),
        }

    supported = score >= ENTAILMENT_THRESHOLD

    return {
        "claim": claim_text,
        "cited_chunk_id": chunk_id,
        "line_number": claim.get("line_number"),
        "score": round(
            score,
            4,
        ),
        "supported": supported,
        "reason": (
            "Claim is supported by cited evidence."
            if supported
            else ("Claim is not sufficiently supported by cited evidence.")
        ),
    }


# =====================================================================
# REVISION FEEDBACK
# =====================================================================


def _build_revision_feedback(
    scores: list[dict[str, Any]],
    uncited_lines: list[str],
) -> list[str]:
    """
    Build compact actionable feedback for responder revisions.

    The feedback deliberately tells the generator to remove
    unsupported elaboration rather than merely asking it to
    "improve" the answer.
    """

    feedback: list[str] = []

    for item in scores:
        if item["supported"]:
            continue

        feedback.append(
            "Unsupported claim: "
            f"{item['claim']} "
            f"(chunk_{item['cited_chunk_id']}, "
            f"score={item['score']:.3f}). "
            "On revision, remove this claim or rewrite it "
            "using only information explicitly supported by "
            f"chunk_{item['cited_chunk_id']}."
        )

    for line in uncited_lines:
        feedback.append(
            "Uncited substantive statement: "
            f"{line}. "
            "On revision, either cite supporting evidence "
            "or remove the statement."
        )

    if feedback:
        feedback.append(
            "Grounding rule: do not add details, mechanisms, "
            "examples, or interpretations that are not "
            "explicitly supported by the cited evidence."
        )

    return feedback


# =====================================================================
# MAIN NODE
# =====================================================================


def grounding_critic_node(
    state: AgentState,
):
    """
    Evaluate whether the generated answer is grounded.

    Strict grounding requires:

        1. At least one valid cited claim.
        2. Every cited chunk exists.
        3. Every cited claim passes entailment.
        4. No substantive factual content is uncited.

    Markdown headings and formatting-only lines are excluded from
    factual claim evaluation.
    """

    route = (state.get("route") or "").strip().lower()

    current_query = (state.get("current_query") or "").strip()

    # ================================================================
    # CONVERSATIONAL BYPASS
    # ================================================================

    if route == "simple" or current_query.upper() == "CONVERSATIONAL":
        logfire.info("Grounding critic bypassed for conversational response.")

        return {
            "claims": [],
            "grounding_scores": [],
            "grounding_feedback": [],
            "is_grounded": True,
            "answer_supported": True,
            "support_score": 1.0,
            "status": ("Grounding bypassed: conversational response."),
        }

    answer = (state.get("final_answer") or "").strip()

    merged_context = (state.get("merged_context") or "").strip()

    with logfire.span(
        "🔬 Grounding Critic",
        answer_length=len(answer),
        context_length=len(merged_context),
    ):
        # ============================================================
        # EMPTY ANSWER
        # ============================================================

        if not answer:
            return {
                "claims": [],
                "grounding_scores": [],
                "grounding_feedback": ["Generated answer is empty."],
                "is_grounded": False,
                "answer_supported": False,
                "support_score": 0.0,
                "status": ("Grounding failed: generated answer is empty."),
            }

        # ============================================================
        # EMPTY EVIDENCE
        # ============================================================

        if not merged_context:
            return {
                "claims": [],
                "grounding_scores": [],
                "grounding_feedback": ["No evidence context is available."],
                "is_grounded": False,
                "answer_supported": False,
                "support_score": 0.0,
                "status": ("Grounding failed: no evidence context."),
            }

        # ============================================================
        # CLAIM EXTRACTION
        # ============================================================

        claims = _extract_claims(answer)

        # ============================================================
        # UNCITED CONTENT
        # ============================================================

        uncited_lines = _extract_substantive_uncited_lines(answer)

        # ============================================================
        # NO CITED CLAIMS
        # ============================================================

        if not claims:
            feedback = ["No valid citations were found in the answer."]

            feedback.extend(
                "Uncited substantive statement: " + line for line in uncited_lines
            )

            return {
                "claims": [],
                "grounding_scores": [],
                "grounding_feedback": feedback,
                "is_grounded": False,
                "answer_supported": False,
                "support_score": 0.0,
                "status": ("Grounding failed: no valid cited claims found."),
            }

        # ============================================================
        # ENTAILMENT
        # ============================================================

        scores: list[dict[str, Any]] = []

        for claim in claims:
            scores.append(
                _evaluate_claim(
                    claim=claim,
                    merged_context=merged_context,
                )
            )

        supported_count = sum(1 for item in scores if item["supported"])

        total_claims = len(scores)

        support_score = supported_count / total_claims if total_claims else 0.0

        all_claims_supported = total_claims > 0 and supported_count == total_claims

        no_uncited_content = not uncited_lines

        is_grounded = all_claims_supported and no_uncited_content

        revision_feedback = _build_revision_feedback(
            scores=scores,
            uncited_lines=uncited_lines,
        )

        logfire.info(
            "Grounding evaluation completed",
            supported_claims=supported_count,
            total_claims=total_claims,
            support_score=round(
                support_score,
                4,
            ),
            uncited_statements=len(uncited_lines),
            feedback_items=len(revision_feedback),
            is_grounded=is_grounded,
        )

    # ================================================================
    # STATUS
    # ================================================================

    if is_grounded:
        status = f"Grounding passed: {supported_count}/{total_claims} claims supported."

    else:
        status = f"Grounding failed: {supported_count}/{total_claims} claims supported."

        if uncited_lines:
            status += f"; {len(uncited_lines)} uncited statements."

        status += " Revision required."

    return {
        "claims": claims,
        "grounding_scores": scores,
        "grounding_feedback": revision_feedback,
        "is_grounded": is_grounded,
        "answer_supported": is_grounded,
        "support_score": support_score,
        "status": status,
    }
