# ============================================================
# KnowledgeMesh · Citation Check
#
# Strict citation coverage validation.
#
# Responsibilities
# ----------------
# 1. Detect substantive factual statements.
# 2. Validate citation IDs against citation_provenance.
# 3. Understand Markdown sections, bullets and tables.
# 4. Do NOT treat section-introduction lines such as:
#
#       - Modern frameworks expose modular buffers:
#
#    as independent uncited claims when they introduce a cited
#    structured list immediately below.
#
# 5. Preserve final_answer unchanged.
#
# Grounding / entailment is handled separately by
# grounding_critic.py.
# ============================================================

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import logfire

from app.agents.state import AgentState


# ============================================================
# Citation patterns
# ============================================================

CITATION_PATTERN = re.compile(
    r"""
    (?:
        \[chunk_(\d+)\]
        |
        【chunk_(\d+)】
        |
        \(chunk_(\d+)\)
    )
    """,
    re.VERBOSE,
)

CITATION_ONLY_PATTERN = re.compile(
    r"""
    ^\s*
    (?:
        \[
            chunk_\d+
            (?:\]\s*\[\s*chunk_\d+\s*)*
        \]
        |
        【
            chunk_\d+
            (?:】\s*【\s*chunk_\d+\s*)*
        】
        |
        \(
            chunk_\d+
            (?:\)\s*\(\s*chunk_\d+\s*)*
        \)
    )
    \s*$
    """,
    re.VERBOSE,
)


# ============================================================
# Markdown patterns
# ============================================================

MARKDOWN_HEADING_PATTERN = re.compile(r"^\s{0,3}#{1,6}\s+.+$")

STANDALONE_EMPHASIS_PATTERN = re.compile(
    r"""
    ^\s*
    (?:\*\*[^*]+\*\*|__[^_]+__)
    \s*:?\s*
    $
    """,
    re.VERBOSE,
)

BULLET_EMPHASIS_PATTERN = re.compile(
    r"""
    ^\s*
    (?:[-*•]|\d+[.)])
    \s*
    (?:\*\*[^*]+\*\*|__[^_]+__)
    \s*:?\s*
    $
    """,
    re.VERBOSE,
)

BULLET_PATTERN = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")

TABLE_ROW_PATTERN = re.compile(r"^\s*\|.*\|.*$")

TABLE_SEPARATOR_PATTERN = re.compile(
    r"""
    ^\s*
    \|?
    \s*:?-{2,}:?\s*
    (?:\|\s*:?-{2,}:?\s*)+
    \|?
    \s*$
    """,
    re.VERBOSE,
)

URL_ONLY_PATTERN = re.compile(
    r"^\s*https?://\S+\s*$",
    re.IGNORECASE,
)

METADATA_ONLY_PATTERN = re.compile(
    r"""
    ^\s*
    (?:
        source\s*=
        |type\s*=
        |score\s*=
        |url\s*=
        |document_id\s*=
        |chunk_id\s*=
        |title\s*=
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


# ============================================================
# Helpers
# ============================================================


def _normalize_whitespace(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def _extract_citation_ids(text: str) -> List[str]:
    result: List[str] = []

    for match in CITATION_PATTERN.finditer(text):
        number = next(
            (value for value in match.groups() if value is not None),
            None,
        )

        if number is None:
            continue

        citation_id = f"chunk_{number}"

        if citation_id not in result:
            result.append(citation_id)

    return result


def _strip_citations(text: str) -> str:
    return CITATION_PATTERN.sub(
        "",
        text,
    ).strip()


def _has_citation(text: str) -> bool:
    return bool(_extract_citation_ids(text))


def _is_markdown_heading(text: str) -> bool:
    return bool(MARKDOWN_HEADING_PATTERN.fullmatch(text.strip()))


def _is_standalone_emphasis(text: str) -> bool:
    return bool(STANDALONE_EMPHASIS_PATTERN.fullmatch(text.strip()))


def _is_bullet_emphasis_heading(text: str) -> bool:
    return bool(BULLET_EMPHASIS_PATTERN.fullmatch(text.strip()))


def _is_table_row(text: str) -> bool:
    return bool(TABLE_ROW_PATTERN.match(text.strip()))


def _is_table_separator(text: str) -> bool:
    return bool(TABLE_SEPARATOR_PATTERN.fullmatch(text.strip()))


def _is_bullet(text: str) -> bool:
    return bool(BULLET_PATTERN.match(text))


def _is_metadata_only(text: str) -> bool:
    stripped = text.strip()

    if not stripped:
        return True

    if URL_ONLY_PATTERN.fullmatch(stripped):
        return True

    if CITATION_ONLY_PATTERN.fullmatch(stripped):
        return True

    return bool(METADATA_ONLY_PATTERN.match(stripped))


# ============================================================
# Section-introduction detection
# ============================================================


def _is_section_introduction(text: str) -> bool:
    """
    Detect lines that introduce a structured list.

    Examples:

        Modern frameworks expose modular buffers:

        Emerging architectural patterns in 2026 include:

        The main changes are:

    These are not useful standalone factual claims when followed
    by a structured, cited list.
    """

    stripped = text.strip()

    if not stripped:
        return False

    # Must end with a colon.
    if not stripped.endswith(":"):
        return False

    # Remove Markdown bullet marker.
    candidate = re.sub(
        r"^\s*(?:[-*•]|\d+[.)])\s+",
        "",
        stripped,
    ).strip()

    # Pure heading-like lines are structural.
    if _is_standalone_emphasis(candidate):
        return True

    # Common list-introduction constructions.
    intro_patterns = [
        r"\binclude(?:s)?\s*:$",
        r"\bexamples?\s*:$",
        r"\bpatterns?\s*:$",
        r"\bframeworks?\s*:$",
        r"\barchitectures?\s*:$",
        r"\bapproaches?\s*:$",
        r"\btypes?\s*:$",
        r"\bstrateg(?:y|ies)\s*:$",
        r"\bchanges?\s*:$",
        r"\bdevelopments?\s*:$",
        r"\bfeatures?\s*:$",
        r"\boptions?\s*:$",
        r"\bmodels?\s*:$",
        r"\bmethods?\s*:$",
        r"\btechniques?\s*:$",
        r"\bways?\s*:$",
        r"\bpatterns?\s+include\s*:$",
    ]

    for pattern in intro_patterns:
        if re.search(
            pattern,
            candidate,
            flags=re.IGNORECASE,
        ):
            return True

    # A colon-terminated bullet that is short and clearly introduces
    # content below should also be treated as structural.
    words = candidate[:-1].split()

    if len(words) <= 12 and (
        "include" in candidate.lower()
        or "includes" in candidate.lower()
        or "include:" in candidate.lower()
    ):
        return True

    return False


# ============================================================
# Non-claim classification
# ============================================================


def _is_non_claim_line(text: str) -> bool:
    stripped = text.strip()

    if not stripped:
        return True

    if _is_markdown_heading(stripped):
        return True

    if _is_standalone_emphasis(stripped):
        return True

    if _is_bullet_emphasis_heading(stripped):
        return True

    if _is_table_separator(stripped):
        return True

    if _is_metadata_only(stripped):
        return True

    if _is_section_introduction(stripped):
        return True

    return False


# ============================================================
# Main claim extraction
# ============================================================


def _extract_claim_groups(
    answer: str,
) -> List[Dict[str, Any]]:
    """
    Group answer lines into logical factual statements.

    Structured Markdown continuation lines are grouped together
    rather than independently judged as uncited claims.
    """

    lines = answer.splitlines()

    groups: List[Dict[str, Any]] = []

    current: Dict[str, Any] | None = None

    def flush() -> None:
        nonlocal current

        if current is not None:
            if current.get("lines"):
                groups.append(current)

        current = None

    for index, raw_line in enumerate(lines):
        line = raw_line.strip()

        if not line:
            continue

        # --------------------------------------------------------
        # Structural headings
        # --------------------------------------------------------

        if _is_markdown_heading(line):
            flush()
            continue

        if _is_standalone_emphasis(line):
            flush()
            continue

        if _is_bullet_emphasis_heading(line):
            continue

        if _is_table_separator(line):
            continue

        # --------------------------------------------------------
        # Section introduction
        # --------------------------------------------------------

        if _is_section_introduction(line):
            # Do NOT create an uncited claim.
            #
            # Keep it as context only if a structured list follows.
            if current is None:
                current = {
                    "lines": [],
                    "citation_ids": [],
                    "structured": True,
                    "section_intro": line,
                }
            else:
                current["section_intro"] = line

            continue

        # --------------------------------------------------------
        # Metadata
        # --------------------------------------------------------

        if _is_metadata_only(line):
            continue

        # --------------------------------------------------------
        # Table rows
        # --------------------------------------------------------

        if _is_table_row(line):
            if current is None:
                current = {
                    "lines": [],
                    "citation_ids": [],
                    "structured": True,
                }

            current["lines"].append(line)

            current["citation_ids"].extend(_extract_citation_ids(line))

            continue

        # --------------------------------------------------------
        # Explicitly cited line
        # --------------------------------------------------------

        citations = _extract_citation_ids(line)

        if citations:
            if current is None:
                current = {
                    "lines": [],
                    "citation_ids": [],
                    "structured": _is_bullet(line),
                }

            current["lines"].append(line)

            current["citation_ids"].extend(citations)

            # Normal prose with a citation is a complete claim.
            #
            # Structured bullets/tables remain grouped so that
            # multiple adjacent facts can share section evidence.
            if not _is_bullet(line) and not _is_table_row(line):
                flush()

            continue

        # --------------------------------------------------------
        # Uncited structured continuation
        # --------------------------------------------------------

        if current is not None:
            previous_lines = current.get(
                "lines",
                [],
            )

            previous = previous_lines[-1] if previous_lines else ""

            section_intro = current.get("section_intro")

            # A bullet immediately under a section introduction is
            # part of that section. It should still need a citation,
            # but we evaluate the entire structured group rather than
            # incorrectly treating the intro as a separate claim.
            if _is_bullet(line) and (section_intro or previous):
                current["lines"].append(line)
                continue

            # A table row belongs to the current structured group.
            if _is_table_row(line):
                current["lines"].append(line)
                continue

            # If the previous line introduced a section with a colon,
            # retain the continuation in the same group.
            if previous.rstrip().endswith(":"):
                current["lines"].append(line)
                continue

            # Otherwise the current group is complete.
            flush()

        # --------------------------------------------------------
        # New substantive uncited claim
        # --------------------------------------------------------

        current = {
            "lines": [line],
            "citation_ids": [],
            "structured": _is_bullet(line),
        }

    flush()

    return groups


# ============================================================
# Group normalization
# ============================================================


def _group_text(
    group: Dict[str, Any],
) -> str:
    lines = group.get(
        "lines",
        [],
    )

    return "\n".join(str(line) for line in lines if str(line).strip()).strip()


def _group_has_substantive_content(
    group: Dict[str, Any],
) -> bool:
    lines = group.get(
        "lines",
        [],
    )

    for line in lines:
        if _is_non_claim_line(line):
            continue

        if _is_table_row(line):
            return True

        if _is_bullet(line):
            return True

        if _strip_citations(line).strip():
            return True

    return False


# ============================================================
# Citation validation
# ============================================================


def _validate_group(
    group: Dict[str, Any],
    provenance: Dict[str, Any],
) -> Tuple[bool, List[str], List[str]]:
    """
    Return:

        has_citation
        invalid_citations
        substantive_lines_without_local_citation
    """

    text = _group_text(group)

    citation_ids = list(
        dict.fromkeys(
            group.get(
                "citation_ids",
                [],
            )
        )
    )

    invalid = [
        citation_id for citation_id in citation_ids if citation_id not in provenance
    ]

    if citation_ids:
        return (
            True,
            invalid,
            [],
        )

    # No explicit citation anywhere in the group.
    #
    # A section introduction alone is not substantive. The actual
    # bullets/table rows are substantive and therefore still require
    # citation evidence.
    substantive_lines: List[str] = []

    for line in group.get(
        "lines",
        [],
    ):
        if _is_non_claim_line(line):
            continue

        if _is_table_row(line):
            substantive_lines.append(line)
            continue

        if _is_bullet(line):
            substantive_lines.append(line)
            continue

        if _strip_citations(line).strip():
            substantive_lines.append(line)

    if not substantive_lines:
        return (
            False,
            invalid,
            [],
        )

    return (
        False,
        invalid,
        substantive_lines,
    )


# ============================================================
# Public validation function
# ============================================================


def validate_citations(
    answer: str,
    provenance: Dict[str, Any],
) -> Dict[str, Any]:
    groups = _extract_claim_groups(answer)

    uncited_claims: List[str] = []
    invalid_citations: List[str] = []
    citation_count = 0

    for group in groups:
        if not _group_has_substantive_content(group):
            continue

        (
            has_citation,
            invalid,
            uncited_lines,
        ) = _validate_group(
            group=group,
            provenance=provenance,
        )

        invalid_citations.extend(invalid)

        if has_citation:
            citation_count += len(
                group.get(
                    "citation_ids",
                    [],
                )
            )
        else:
            # Section introduction itself is NOT reported.
            #
            # Only report the actual factual structured lines.
            for line in uncited_lines:
                clean = _normalize_whitespace(_strip_citations(line))

                if clean:
                    uncited_claims.append(clean)

    invalid_citations = list(dict.fromkeys(invalid_citations))

    uncited_claims = list(dict.fromkeys(uncited_claims))

    valid = not uncited_claims and not invalid_citations

    return {
        "valid": valid,
        "uncited_claims": uncited_claims,
        "invalid_citations": invalid_citations,
        "citation_count": citation_count,
    }


# ============================================================
# Main LangGraph node
# ============================================================


def citation_check_node(
    state: AgentState,
):
    final_answer = state.get("final_answer") or ""

    provenance = state.get("citation_provenance") or {}

    if not isinstance(
        provenance,
        dict,
    ):
        provenance = {}

    # --------------------------------------------------------
    # Empty answer
    # --------------------------------------------------------

    if not final_answer.strip():
        return {
            "citation_valid": True,
            "citation_feedback": "",
            "citation_errors": [],
            "status": ("Citation check skipped: empty answer."),
            "plan": state.get("plan", []) + ["Citation Check: empty answer"],
        }

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    result = validate_citations(
        answer=final_answer,
        provenance=provenance,
    )

    uncited_claims = result["uncited_claims"]

    invalid_citations = result["invalid_citations"]

    citation_valid = bool(result["valid"])

    errors: List[str] = []

    if uncited_claims:
        errors.append((f"{len(uncited_claims)} uncited substantive statements"))

    if invalid_citations:
        errors.append((f"{len(invalid_citations)} invalid citation(s)"))

    # --------------------------------------------------------
    # Feedback
    # --------------------------------------------------------

    if citation_valid:
        feedback = ""
        status = "Citation validation passed."

    else:
        feedback_lines = [
            "Citation check failed.",
        ]

        if uncited_claims:
            feedback_lines.append("Every factual statement must have a valid citation.")

            feedback_lines.append("Uncited factual statements:")

            for claim in uncited_claims[:12]:
                feedback_lines.append(f"- {claim}")

        if invalid_citations:
            feedback_lines.append("Invalid citations:")

            for citation in invalid_citations:
                feedback_lines.append(f"- {citation}")

        feedback_lines.append(
            "Markdown headings and section-introduction "
            "lines ending with ':' are structural. "
            "Citations should be attached to the factual "
            "bullet, table row, or sentence containing "
            "the supported claim."
        )

        feedback = "\n".join(feedback_lines)

        status = "Citation check failed: " + ", ".join(errors) + "."

    # --------------------------------------------------------
    # Observability
    # --------------------------------------------------------

    with logfire.span(
        "🔎 Citation Check",
        citation_valid=citation_valid,
        uncited_claims=len(uncited_claims),
        invalid_citations=len(invalid_citations),
        citation_count=result["citation_count"],
    ):
        logfire.info(
            "Citation validation completed",
            citation_valid=citation_valid,
            uncited_claims=len(uncited_claims),
            invalid_citations=len(invalid_citations),
            citation_count=result["citation_count"],
        )

    return {
        "citation_valid": citation_valid,
        "citation_feedback": feedback,
        "citation_errors": errors,
        "status": status,
        "plan": state.get("plan", [])
        + [(f"Citation Check: {'PASS' if citation_valid else 'FAIL'}")],
    }
