"""
KnowledgeMesh — Strict Grounding Critic

Responsibilities
----------------
1. Validate every substantive claim in the final answer.
2. Validate atomic claims rather than trusting whole paragraphs.
3. Validate citation IDs against actual generation evidence.
4. Use strict entailment scoring.
5. Evaluate claims against bounded evidence windows.
6. Normalize PDF-extracted evidence before sentence/window construction.
7. Allow conservative exact/near-exact evidence matches.
8. Produce a stable grounding_scores schema.
9. Export unsupported_atomic_claims for revision/UI.
10. Export revision_prompt when grounding fails.
11. Keep answer_supported strictly boolean.
12. Never weaken grounding simply because a citation exists.

Evidence architecture
---------------------

    citation
        ↓
    complete cited chunk
        ↓
    normalized evidence text
        ↓
    semantic evidence units
        ↓
    bounded contextual windows
        ↓
    lexical candidate ranking
        ↓
    HHEMv2 entailment
        ↓
    best supported window
"""

from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field

import re
from difflib import SequenceMatcher
from time import perf_counter
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from app.evaluation.trace import append_trace_event


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENTAILMENT_THRESHOLD = 0.50

EXACT_MATCH_MIN_TOKENS = 6
EXACT_MATCH_TOKEN_COVERAGE = 0.94
EXACT_MATCH_SEQUENCE_RATIO = 0.90

EVIDENCE_WINDOW_RADIUS = 1
EVIDENCE_LARGE_WINDOW_RADIUS = 2

MAX_ENTAILMENT_CANDIDATES = 8
MAX_CITED_ENTAILMENT_CANDIDATES = 3
MAX_DISCOVERY_ENTAILMENT_CANDIDATES = 3
MAX_ATOMIC_SEMANTIC_BUDGET = 3

MIN_WINDOW_OVERLAP = 1
FALLBACK_WINDOW_COUNT = 3


class SupportType(str, Enum):
    DIRECT = "DIRECT"
    INFERRED = "INFERRED"
    CONTRADICTED = "CONTRADICTED"
    UNSUPPORTED = "UNSUPPORTED"


class AtomicClaim(BaseModel):
    claim_id: str
    text: str
    citation_ids: list[str] = Field(default_factory=list)


class AtomicClaimsResponse(BaseModel):
    claims: list[AtomicClaim]


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _safe_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    try:
        return str(value).strip()
    except Exception:
        return ""


def _safe_list(value: Any) -> List[Any]:
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    return [value]


def _dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    output: List[str] = []

    for value in values:
        value = _safe_text(value)

        if not value:
            continue

        key = value

        if key not in seen:
            seen.add(key)
            output.append(value)

    return output


# ---------------------------------------------------------------------------
# Markdown / claim extraction
# ---------------------------------------------------------------------------


CITATION_PATTERN = re.compile(r"\[([A-Za-z0-9_.:/-]+)\]")

BULLET_PATTERN = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")

HEADING_PATTERN = re.compile(r"^\s{0,3}#{1,6}\s+")

TABLE_SEPARATOR_PATTERN = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
)


def _strip_citations(text: str) -> str:
    return CITATION_PATTERN.sub("", text)


def _strip_markdown_prefix(text: str) -> str:
    text = HEADING_PATTERN.sub("", text)
    text = BULLET_PATTERN.sub("", text)
    return text.strip()


def _is_heading(text: str) -> bool:
    return bool(HEADING_PATTERN.match(text))


def _is_table_separator(text: str) -> bool:
    return bool(TABLE_SEPARATOR_PATTERN.match(text))


def _is_non_claim_line(text: str) -> bool:
    stripped = text.strip()

    if not stripped:
        return True

    if _is_heading(stripped):
        return True

    if _is_table_separator(stripped):
        return True

    if stripped in {"---", "***", "___"}:
        return True

    if (
        len(stripped) >= 2
        and stripped.startswith("**")
        and stripped.endswith("**")
        and stripped.count("**") == 2
    ):
        return True

    return False


def _extract_citations(text: str) -> List[str]:
    return _dedupe_preserve_order(
        match.group(1) for match in CITATION_PATTERN.finditer(text)
    )


def _clean_claim_text(text: str) -> str:
    text = _strip_citations(text)
    text = _strip_markdown_prefix(text)

    text = text.strip(" `*_")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def _split_compound_claim(text: str) -> List[str]:
    """
    Conservative atomic-claim splitter.

    Avoids aggressive splitting because technical statements frequently
    contain several dependent clauses that should be evaluated together.
    """

    text = _clean_claim_text(text)

    if not text:
        return []

    pieces = re.split(
        r";\s+",
        text,
    )

    cleaned: List[str] = []

    for piece in pieces:
        piece = piece.strip(" -•")

        if not piece:
            continue

        words = piece.split()

        if len(words) < 18:
            cleaned.append(piece)
            continue

        if " and " in piece.lower():
            candidates = re.split(
                r"\s+\band\b\s+",
                piece,
                flags=re.IGNORECASE,
            )

            if (
                len(candidates) == 2
                and len(candidates[0].split()) >= 6
                and len(candidates[1].split()) >= 6
            ):
                cleaned.extend(
                    candidate.strip(" -•")
                    for candidate in candidates
                    if candidate.strip()
                )
                continue

        cleaned.append(piece)

    return _dedupe_preserve_order(cleaned)


def _extract_claim_records(answer: str) -> List[Dict[str, Any]]:
    lines = answer.splitlines()

    records: List[Dict[str, Any]] = []

    for raw_line in lines:
        line = raw_line.strip()

        if _is_non_claim_line(line):
            continue

        claim = _clean_claim_text(line)

        if not claim:
            continue

        citations = _extract_citations(line)

        atomic_claims = _split_compound_claim(claim)

        if not atomic_claims:
            atomic_claims = [claim]

        records.append(
            {
                "text": claim,
                "raw_text": line,
                "citations": citations,
                "atomic_claims": atomic_claims,
            }
        )

    return records


# ---------------------------------------------------------------------------
# Evidence extraction
# ---------------------------------------------------------------------------


def _extract_text_from_document(document: Any) -> str:
    if document is None:
        return ""

    if isinstance(document, str):
        return document.strip()

    if isinstance(document, dict):
        for key in (
            "text",
            "page_content",
            "content",
            "evidence",
            "snippet",
            "document",
            "body",
        ):
            value = document.get(key)

            if isinstance(value, str) and value.strip():
                return value.strip()

            if isinstance(value, dict):
                nested = _extract_text_from_document(value)

                if nested:
                    return nested

        return ""

    for attr in (
        "text",
        "page_content",
        "content",
        "evidence",
        "snippet",
    ):
        value = getattr(document, attr, None)

        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def _extract_citation_id(document: Any) -> str:
    if isinstance(document, dict):
        for key in (
            "citation_id",
            "citation",
            "chunk_id",
            "id",
        ):
            value = document.get(key)

            if value:
                return _safe_text(value)

    for attr in (
        "citation_id",
        "citation",
        "chunk_id",
        "id",
    ):
        value = getattr(document, attr, None)

        if value:
            return _safe_text(value)

    return ""


# ---------------------------------------------------------------------------
# PDF / extraction normalization
# ---------------------------------------------------------------------------


def _normalize_extracted_text(text: str) -> str:
    """
    Normalize common PDF extraction artifacts.

    This is deliberately conservative.

    It fixes:
      - broken whitespace
      - repeated whitespace
      - soft line breaks
      - Unicode punctuation
      - common hyphenation across line breaks
      - obvious page-artifact spacing

    It does NOT attempt aggressive linguistic rewriting.
    """

    text = _safe_text(text)

    if not text:
        return ""

    replacements = {
        "\u00a0": " ",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00ad": "",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Normalize CRLF/CR.
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Repair words broken by PDF line wrapping:
    #
    #   retriev-
    #   al
    #
    # becomes:
    #
    #   retrieval
    #
    text = re.sub(
        r"([A-Za-z]{2,})-\s*\n\s*([a-z]{2,})",
        r"\1\2",
        text,
    )

    # Replace line breaks between ordinary prose with spaces.
    #
    # Keep blank lines because they often represent paragraph boundaries.
    text = re.sub(
        r"(?<!\n)\n(?!\n)",
        " ",
        text,
    )

    # Collapse excessive whitespace.
    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def _clean_evidence_line(line: str) -> str:
    line = _safe_text(line)

    if not line:
        return ""

    line = re.sub(
        r"\s+",
        " ",
        line,
    )

    return line.strip()


def _looks_like_structural_fragment(text: str) -> bool:
    """
    Detect tiny extraction fragments that should not become independent
    evidence units.

    Examples:
        lly,
        er,
        3.
        -
    """

    text = _safe_text(text)

    if not text:
        return True

    cleaned = text.strip(" -–—•:;,.()[]{}")

    if not cleaned:
        return True

    # Very short alphabetic fragments are usually extraction artifacts.
    if len(cleaned) <= 3 and cleaned.isalpha():
        return True

    return False


def _merge_short_fragments(units: Sequence[str]) -> List[str]:
    """
    Merge short PDF fragments into neighboring prose.

    This prevents artifacts such as:

        'lly,'
        'DPR follows...'

    from becoming independent evidence candidates.
    """

    merged: List[str] = []

    for unit in units:
        unit = _clean_evidence_line(unit)

        if not unit:
            continue

        if not merged:
            merged.append(unit)
            continue

        previous = merged[-1]

        # Tiny fragment: attach to previous unit.
        if len(unit) <= 8 and not re.search(r"[.!?]$", previous):
            merged[-1] = f"{previous} {unit}".strip()
            continue

        # Previous tiny fragment: merge into current unit.
        if len(previous) <= 8:
            merged[-1] = f"{previous} {unit}".strip()
            continue

        merged.append(unit)

    return merged


# ---------------------------------------------------------------------------
# Evidence sentence / unit construction
# ---------------------------------------------------------------------------


def _split_evidence_sentences(text: str) -> List[str]:
    """
    Build clean sentence-like evidence units.

    Important difference from the old implementation:

    We normalize PDF extraction BEFORE sentence construction.

    We also preserve paragraph boundaries and avoid treating every raw
    line as a standalone sentence.
    """

    text = _normalize_extracted_text(text)

    if not text:
        return []

    paragraphs = re.split(
        r"\n{2,}",
        text,
    )

    units: List[str] = []

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        # Tables should remain intact.
        if "|" in paragraph and paragraph.count("|") >= 2:
            cleaned = _clean_evidence_line(paragraph)

            if cleaned and not _looks_like_structural_fragment(cleaned):
                units.append(cleaned)

            continue

        # Markdown bullets can contain complete evidence statements.
        bullet_lines = re.split(
            r"(?=\n\s*(?:[-*•]|\d+[.)])\s+)",
            paragraph,
        )

        for block in bullet_lines:
            block = block.strip()

            if not block:
                continue

            block = BULLET_PATTERN.sub(
                "",
                block,
            ).strip()

            if not block:
                continue

            # Sentence splitting is intentionally conservative.
            #
            # Protect common abbreviations such as "vs.", "e.g.", "i.e.",
            # "Fig.", "Dr.", "Mr.", "Ms.", and "etc." so their periods
            # are not mistaken for sentence boundaries.
            abbreviation_placeholders = {
                "vs.": "__KM_ABBR_VS__",
                "e.g.": "__KM_ABBR_EG__",
                "i.e.": "__KM_ABBR_IE__",
                "fig.": "__KM_ABBR_FIG__",
                "dr.": "__KM_ABBR_DR__",
                "mr.": "__KM_ABBR_MR__",
                "ms.": "__KM_ABBR_MS__",
                "etc.": "__KM_ABBR_ETC__",
            }

            protected_block = block

            for abbreviation, placeholder in abbreviation_placeholders.items():
                protected_block = re.sub(
                    re.escape(abbreviation),
                    placeholder,
                    protected_block,
                    flags=re.IGNORECASE,
                )

            parts = re.split(
                r"(?<=[.!?])\s+(?=[A-Z0-9`\"'(\[])",
                protected_block,
            )

            for index, part in enumerate(parts):
                for abbreviation, placeholder in abbreviation_placeholders.items():
                    part = re.sub(
                        re.escape(placeholder),
                        abbreviation,
                        part,
                        flags=re.IGNORECASE,
                    )
                parts[index] = part

            for part in parts:
                part = _clean_evidence_line(part)

                if not part:
                    continue

                if _looks_like_structural_fragment(part):
                    continue

                units.append(part)

    units = _merge_short_fragments(units)

    return _dedupe_preserve_order(units)


def _split_evidence_units(text: str) -> List[str]:
    """
    Backward-compatible helper.

    Returns bounded evidence windows.
    """

    return _build_evidence_windows(text)


def _build_evidence_windows(text: str) -> List[str]:
    """
    Build bounded contextual windows from the complete cited chunk.

    We deliberately keep several granularities:

        sentence
        sentence +/- 1
        sentence +/- 2
        paragraph

    This gives HHEMv2 a local semantic context without passing an entire
    potentially noisy PDF chunk as one enormous premise.
    """

    normalized = _normalize_extracted_text(text)

    if not normalized:
        return []

    sentences = _split_evidence_sentences(normalized)

    if not sentences:
        return []

    windows: List[str] = []

    # ---------------------------------------------------------------
    # Single evidence units
    # ---------------------------------------------------------------

    for sentence in sentences:
        windows.append(sentence)

    # ---------------------------------------------------------------
    # Local contextual windows
    # ---------------------------------------------------------------

    for radius in (
        EVIDENCE_WINDOW_RADIUS,
        EVIDENCE_LARGE_WINDOW_RADIUS,
    ):
        width = (radius * 2) + 1

        if len(sentences) <= width:
            windows.append(" ".join(sentences))
            continue

        for center in range(len(sentences)):
            start = max(
                0,
                center - radius,
            )

            end = min(
                len(sentences),
                center + radius + 1,
            )

            window = " ".join(sentences[start:end]).strip()

            if window:
                windows.append(window)

    # ---------------------------------------------------------------
    # Paragraph-level fallback
    # ---------------------------------------------------------------

    for block in re.split(
        r"\n{2,}",
        normalized,
    ):
        block = re.sub(
            r"\s+",
            " ",
            block,
        ).strip()

        if not block:
            continue

        if len(block) >= 20:
            windows.append(block)

    return _dedupe_preserve_order(windows)


# ---------------------------------------------------------------------------
# Citation provenance
# ---------------------------------------------------------------------------


def _resolve_citation_provenance(
    citation_provenance: Any,
) -> Dict[str, List[str]]:

    result: Dict[str, List[str]] = {}

    if not citation_provenance:
        return result

    if isinstance(citation_provenance, dict):
        iterable = []

        for key, value in citation_provenance.items():
            if isinstance(value, dict):
                item = dict(value)

                if not item.get("citation_id"):
                    item["citation_id"] = key

                iterable.append(item)

            else:
                iterable.append(
                    {
                        "citation_id": key,
                        "evidence": value,
                    }
                )

    else:
        iterable = _safe_list(citation_provenance)

    for item in iterable:
        if not isinstance(item, dict):
            continue

        citation_id = _safe_text(
            item.get("citation_id") or item.get("citation") or item.get("id")
        )

        if not citation_id:
            continue

        evidence_text = ""

        for key in (
            "text",
            "page_content",
            "content",
            "evidence",
            "snippet",
        ):
            value = item.get(key)

            if isinstance(value, str) and value.strip():
                evidence_text = value.strip()
                break

        if not evidence_text:
            nested = item.get("document")

            if nested is not None:
                evidence_text = _extract_text_from_document(nested)

        if not evidence_text:
            continue

        windows = _build_evidence_windows(evidence_text)

        if windows:
            result.setdefault(
                citation_id,
                [],
            ).extend(windows)

    return {
        citation_id: _dedupe_preserve_order(windows)
        for citation_id, windows in result.items()
    }


def _resolve_citation_metadata(
    citation_provenance: Any,
) -> Dict[str, Dict[str, Any]]:
    """Resolve citation IDs to canonical corpus provenance."""

    result: Dict[str, Dict[str, Any]] = {}

    if not citation_provenance:
        return result

    if isinstance(citation_provenance, dict):
        iterable: List[Any] = []

        for key, value in citation_provenance.items():
            if isinstance(value, dict):
                item = dict(value)

                if not item.get("citation_id"):
                    item["citation_id"] = key

                iterable.append(item)
    elif isinstance(citation_provenance, (list, tuple)):
        iterable = list(citation_provenance)
    else:
        iterable = []

    for item in iterable:
        if not isinstance(item, dict):
            continue

        citation_id = _safe_text(
            item.get("citation_id")
            or item.get("citation")
            or item.get("id")
        )

        if not citation_id:
            continue

        document_id = item.get("document_id")
        chunk_id = item.get("chunk_id")
        point_id = item.get("point_id")

        nested = item.get("document")

        if isinstance(nested, dict):
            if document_id is None:
                document_id = nested.get("document_id")

            if chunk_id is None:
                chunk_id = nested.get("chunk_id")

            if point_id is None:
                point_id = (
                    nested.get("point_id")
                    or nested.get("id")
                )

        document_id = _safe_text(document_id) or None
        point_id = _safe_text(point_id) or None

        if chunk_id is not None:
            try:
                chunk_id = int(chunk_id)
            except (TypeError, ValueError):
                chunk_id = _safe_text(chunk_id) or None

        canonical_evidence_id = None

        if document_id is not None and chunk_id is not None:
            canonical_evidence_id = f"{document_id}::{chunk_id}"

        result[citation_id] = {
            "citation_id": citation_id,
            "document_id": document_id,
            "chunk_id": chunk_id,
            "point_id": point_id,
            "canonical_evidence_id": canonical_evidence_id,
        }

    return result

def _resolve_documents_by_citation(
    documents: Sequence[Any],
) -> Dict[str, List[str]]:

    result: Dict[str, List[str]] = {}

    for index, document in enumerate(
        documents,
        start=1,
    ):
        text = _extract_text_from_document(document)

        if not text:
            continue

        citation_id = _extract_citation_id(document)

        if not citation_id:
            citation_id = f"chunk_{index}"

        windows = _build_evidence_windows(text)

        if windows:
            result.setdefault(
                citation_id,
                [],
            ).extend(windows)

    return {
        citation_id: _dedupe_preserve_order(windows)
        for citation_id, windows in result.items()
    }


def _build_evidence_map(
    state: Dict[str, Any],
) -> Dict[str, List[str]]:
    """
    Resolve exact generation evidence.

    Priority:

        1. citation_provenance
        2. generation_documents
        3. web_documents
        4. documents
    """

    evidence_map: Dict[str, List[str]] = {}

    provenance = state.get("citation_provenance")

    provenance_map = _resolve_citation_provenance(provenance)

    for citation_id, windows in provenance_map.items():
        evidence_map.setdefault(
            citation_id,
            [],
        ).extend(windows)

    generation_documents = _safe_list(state.get("generation_documents"))

    web_documents = _safe_list(state.get("web_documents"))

    documents = _safe_list(state.get("documents"))

    fallback_documents: List[Any] = []

    if generation_documents:
        fallback_documents.extend(generation_documents)

    if web_documents:
        fallback_documents.extend(web_documents)

    if documents:
        fallback_documents.extend(documents)

    document_map = _resolve_documents_by_citation(fallback_documents)

    for citation_id, windows in document_map.items():
        if citation_id not in evidence_map:
            evidence_map[citation_id] = list(windows)

    return {
        citation_id: _dedupe_preserve_order(windows)
        for citation_id, windows in evidence_map.items()
    }


# ---------------------------------------------------------------------------
# Evidence relevance ranking
# ---------------------------------------------------------------------------


_WINDOW_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "then",
    "than",
    "that",
    "this",
    "these",
    "those",
    "with",
    "from",
    "into",
    "onto",
    "for",
    "of",
    "to",
    "in",
    "on",
    "by",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "it",
    "its",
    "they",
    "their",
    "them",
    "we",
    "our",
    "you",
    "your",
    "can",
    "may",
    "might",
    "must",
    "will",
    "would",
    "should",
    "do",
    "does",
    "did",
    "has",
    "have",
    "had",
    "also",
    "very",
    "more",
    "most",
    "some",
    "such",
    "not",
    "only",
    "when",
    "where",
    "which",
    "who",
    "how",
    "what",
    "why",
    "via",
    "per",
    "within",
    "through",
    "over",
    "under",
    "during",
    "while",
    "both",
    "each",
    "other",
    "another",
    "there",
    "here",
}


def _normalize_window_token(
    token: str,
) -> str:

    token = str(token).lower().strip()

    if not token:
        return ""

    token = re.sub(
        r"[^a-z0-9_-]",
        "",
        token,
    )

    if len(token) < 3:
        return ""

    return token


def _window_tokens(
    text: str,
) -> set[str]:

    raw_tokens = re.findall(
        r"[A-Za-z0-9][A-Za-z0-9_-]*",
        _strip_citations(text).lower(),
    )

    tokens = set()

    for raw_token in raw_tokens:
        token = _normalize_window_token(raw_token)

        if not token:
            continue

        if token in _WINDOW_STOPWORDS:
            continue

        tokens.add(token)

    return tokens


def _window_relevance_score(
    claim: str,
    evidence_window: str,
) -> Tuple[float, int, float]:

    claim_tokens = _window_tokens(claim)

    evidence_tokens = _window_tokens(evidence_window)

    if not claim_tokens or not evidence_tokens:
        return 0.0, 0, 0.0

    overlap = claim_tokens.intersection(evidence_tokens)

    overlap_count = len(overlap)

    density = overlap_count / max(
        len(claim_tokens),
        1,
    )

    claim_words = [
        token
        for token in re.findall(
            r"[A-Za-z0-9][A-Za-z0-9_-]*",
            claim.lower(),
        )
        if token not in _WINDOW_STOPWORDS
    ]

    evidence_words = [
        token
        for token in re.findall(
            r"[A-Za-z0-9][A-Za-z0-9_-]*",
            evidence_window.lower(),
        )
        if token not in _WINDOW_STOPWORDS
    ]

    claim_bigrams = {
        tuple(claim_words[i : i + 2])
        for i in range(
            max(
                0,
                len(claim_words) - 1,
            )
        )
    }

    evidence_bigrams = {
        tuple(evidence_words[i : i + 2])
        for i in range(
            max(
                0,
                len(evidence_words) - 1,
            )
        )
    }

    phrase_bonus = len(claim_bigrams.intersection(evidence_bigrams))

    score = overlap_count * 2.0 + density * 4.0 + phrase_bonus * 1.5

    return (
        score,
        overlap_count,
        density,
    )


def _select_entailment_candidates(
    claim: str,
    evidence_windows: Sequence[str],
) -> List[str]:

    unique_windows = _dedupe_preserve_order(evidence_windows)

    if not unique_windows:
        return []

    scored: List[
        Tuple[
            str,
            float,
            int,
            float,
            int,
        ]
    ] = []

    for index, window in enumerate(unique_windows):
        (
            relevance,
            overlap,
            density,
        ) = _window_relevance_score(
            claim,
            window,
        )

        scored.append(
            (
                window,
                relevance,
                overlap,
                density,
                index,
            )
        )

    relevant = [item for item in scored if item[2] >= MIN_WINDOW_OVERLAP]

    if relevant:
        relevant.sort(
            key=lambda item: (
                item[1],
                item[2],
                item[3],
                -item[4],
            ),
            reverse=True,
        )

        return [item[0] for item in relevant[:MAX_ENTAILMENT_CANDIDATES]]

    fallback = sorted(
        scored,
        key=lambda item: item[4],
    )

    return [item[0] for item in fallback[:FALLBACK_WINDOW_COUNT]]


# ---------------------------------------------------------------------------
# Conservative deterministic textual support
# ---------------------------------------------------------------------------


def _normalize_for_exact_match(
    text: str,
) -> str:

    text = _safe_text(text)

    if not text:
        return ""

    text = text.lower()

    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00a0": " ",
    }

    for old, new in replacements.items():
        text = text.replace(
            old,
            new,
        )

    text = _strip_citations(text)
    text = _strip_markdown_prefix(text)

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def _content_tokens(
    text: str,
) -> List[str]:

    normalized = _normalize_for_exact_match(text)

    if not normalized:
        return []

    return normalized.split()



def _deterministic_numeric_detail_support(
    claim: str,
    evidence: str,
) -> Tuple[bool, float]:
    """
    Detect conservative support for claims whose distinctive quantitative
    details are preserved but whose wording/order differs.

    Example:
        claim:    hallucinates at a 21% rate versus 41% for GPT-3
        evidence: ... GPT-3 ... (a 21% vs. 41% hallucination rate ...)

    This deliberately requires both:
      - multiple matching numeric details, and
      - meaningful non-numeric lexical overlap.
    """

    claim_tokens = _content_tokens(claim)
    evidence_tokens = _content_tokens(evidence)

    if not claim_tokens or not evidence_tokens:
        return False, 0.0

    claim_numeric = [
        token
        for token in claim_tokens
        if token.isdigit()
    ]

    if len(claim_numeric) < 2:
        return False, 0.0

    evidence_set = set(evidence_tokens)

    numeric_coverage = sum(
        1
        for token in claim_numeric
        if token in evidence_set
    ) / max(len(claim_numeric), 1)

    if numeric_coverage < 1.0:
        return False, 0.0

    claim_non_numeric = {
        token
        for token in claim_tokens
        if not token.isdigit()
    }

    if not claim_non_numeric:
        return False, 0.0

    non_numeric_overlap = (
        len(claim_non_numeric.intersection(evidence_set))
        / max(len(claim_non_numeric), 1)
    )

    if non_numeric_overlap < 0.30:
        return False, 0.0

    score = min(
        1.0,
        0.80
        + (numeric_coverage * 0.10)
        + (min(non_numeric_overlap, 1.0) * 0.10),
    )

    return True, score


def _deterministic_exact_support(
    claim: str,
    evidence: str,
) -> Tuple[bool, float]:

    claim_normalized = _normalize_for_exact_match(claim)

    evidence_normalized = _normalize_for_exact_match(evidence)

    if not claim_normalized or not evidence_normalized:
        return False, 0.0

    claim_tokens = claim_normalized.split()

    # An exact whole-evidence match is safe even for short claims.
    if claim_normalized == evidence_normalized:
        return True, 1.0

    # Keep the minimum-token guard for short claims that only appear
    # inside a larger evidence unit.
    if len(claim_tokens) < EXACT_MATCH_MIN_TOKENS:
        return False, 0.0

    # Longer literal claims can be safely accepted when contained
    # within a larger evidence unit.
    if claim_normalized in evidence_normalized:
        return True, 1.0

    numeric_supported, numeric_score = (
        _deterministic_numeric_detail_support(
            claim,
            evidence,
        )
    )

    if numeric_supported:
        return True, numeric_score

    evidence_tokens = evidence_normalized.split()

    if not evidence_tokens:
        return False, 0.0

    evidence_set = set(evidence_tokens)

    overlap = sum(1 for token in claim_tokens if token in evidence_set) / max(
        len(claim_tokens),
        1,
    )

    if overlap < EXACT_MATCH_TOKEN_COVERAGE:
        return False, 0.0

    ratio = SequenceMatcher(
        None,
        claim_normalized,
        evidence_normalized,
    ).ratio()

    if ratio >= EXACT_MATCH_SEQUENCE_RATIO:
        return (
            True,
            min(
                1.0,
                max(
                    overlap,
                    ratio,
                ),
            ),
        )

    return False, 0.0


# ---------------------------------------------------------------------------
# Entailment
# ---------------------------------------------------------------------------


def _get_entailment_scorer():
    try:
        from app.tools.entailment_tool import score_entailment, get_hhem_profile

        return score_entailment

    except ImportError as exc:
        raise ImportError(
            "Could not import KnowledgeMesh entailment scorer from "
            "app.tools.entailment_tool."
        ) from exc


def _score_claim_against_evidence(
    claim: str,
    evidence_units: Sequence[str],
    entailment_cache: Dict[Tuple[str, str], float] | None = None,
    max_candidates: int | None = None,
) -> Tuple[float, str]:

    claim = _clean_claim_text(claim)

    if not claim:
        return 0.0, ""

    candidates = _select_entailment_candidates(
        claim,
        evidence_units,
    )
    if max_candidates is not None:
        candidates = candidates[:max_candidates]

    if not candidates:
        return 0.0, ""

    scorer = _get_entailment_scorer()

    best_score = 0.0
    best_evidence_unit = ""

    for evidence_unit in candidates:
        evidence_unit = _safe_text(evidence_unit)

        if not evidence_unit:
            continue

        cache_key = (claim, evidence_unit)

        if entailment_cache is not None and cache_key in entailment_cache:
            score = entailment_cache[cache_key]
        else:
            try:
                score = scorer(
                    premise=evidence_unit,
                    hypothesis=claim,
                )

                score = max(
                    0.0,
                    min(
                        1.0,
                        float(score),
                    ),
                )

            except Exception:
                continue

            if entailment_cache is not None:
                entailment_cache[cache_key] = score

        if score > best_score:
            best_score = score
            best_evidence_unit = evidence_unit

            if best_score >= ENTAILMENT_THRESHOLD:
                break

    return (
        best_score,
        best_evidence_unit,
    )


# Markdown / claim extraction
async def extract_atomic_claims(
    answer: str,
    llm_client,
) -> list[AtomicClaim]:
    prompt = f"""
Split the answer below into atomic factual claims.

Rules:
- One claim must contain one independently verifiable assertion.
- Keep numbers, dates, names, conditions, and qualifications.
- Do not create claims for headings, questions, or opinions.
- Preserve citation markers attached to each claim.
- Do not add facts.
- Return an empty list if there are no factual claims.

Return JSON matching this schema:
{{
  "claims": [
    {{
      "claim_id": "c-001",
      "text": "...",
      "citation_ids": []
    }}
  ]
}}

Answer:
{answer}
"""

    result = await llm_client.generate_structured(
        prompt=prompt,
        schema=AtomicClaimsResponse,
    )

    if isinstance(result, AtomicClaimsResponse):
        return result.claims

    if isinstance(result, dict):
        return AtomicClaimsResponse.model_validate(result).claims

    raise TypeError(f"Unexpected structured output: {type(result).__name__}")


# ---------------------------------------------------------------------------
# Atomic claim validation
# ---------------------------------------------------------------------------



def _select_global_entailment_candidates(
    claim: str,
    evidence_map: Dict[str, List[str]],
    excluded_citations: Sequence[str] = (),
    included_citations: Sequence[str] = (),
    limit: int = MAX_DISCOVERY_ENTAILMENT_CANDIDATES,
) -> List[Tuple[str, str]]:
    """
    Select a globally bounded semantic-entailment candidate pool for one claim.

    All eligible evidence windows compete in one relevance-ranked pool.
    The caller controls the global HHEMv2 budget through ``limit`` so the
    same helper can be used for cited verification and independent discovery
    without multiplying calls by citation count.

    Returns:
        List of (citation_id, evidence_window) pairs.
    """

    claim = _clean_claim_text(claim)

    if not claim:
        return []

    excluded = set(
        _dedupe_preserve_order(excluded_citations)
    )

    included = set(
        _dedupe_preserve_order(included_citations)
    )

    scored: List[
        Tuple[
            str,
            str,
            float,
            int,
            float,
            int,
        ]
    ] = []

    stable_index = 0

    for citation_id, evidence_windows in evidence_map.items():
        if citation_id in excluded:
            continue

        if included and citation_id not in included:
            continue

        if not evidence_windows:
            continue

        for evidence_window in _dedupe_preserve_order(evidence_windows):
            evidence_window = _safe_text(evidence_window)

            if not evidence_window:
                continue

            (
                relevance,
                overlap,
                density,
            ) = _window_relevance_score(
                claim,
                evidence_window,
            )

            if overlap >= MIN_WINDOW_OVERLAP:
                scored.append(
                    (
                        citation_id,
                        evidence_window,
                        relevance,
                        overlap,
                        density,
                        stable_index,
                    )
                )

            stable_index += 1

    scored.sort(
        key=lambda item: (
            item[2],
            item[3],
            item[4],
            -item[5],
        ),
        reverse=True,
    )

    limit = max(1, int(limit))

    return [
        (citation_id, evidence_window)
        for (
            citation_id,
            evidence_window,
            _relevance,
            _overlap,
            _density,
            _stable_index,
        ) in scored[:limit]
    ]

def _discover_supporting_evidence(
    claim: str,
    evidence_map: Dict[str, List[str]],
    excluded_citations: Sequence[str] = (),
    entailment_cache: Dict[Tuple[str, str], float] | None = None,
    semantic_budget: int = MAX_DISCOVERY_ENTAILMENT_CANDIDATES,
) -> Dict[str, Any]:
    """
    Search all available evidence citations for independent support.

    Deterministic exact support scans all eligible evidence first.

    Semantic discovery receives the REMAINING shared per-claim semantic
    budget after cited verification. Deterministic exact support remains
    unlimited and does not consume the HHEMv2 budget.
    """

    claim = _clean_claim_text(claim)

    if not claim:
        return {
            "score": 0.0,
            "supported": False,
            "best_evidence_unit": "",
            "best_citation": "",
            "support_method": "empty_claim",
        }

    excluded = set(
        _dedupe_preserve_order(excluded_citations)
    )

    best_score = 0.0
    best_evidence_unit = ""
    best_citation = ""
    best_method = "entailment"

    # -----------------------------------------------------------
    # Phase 1: deterministic exact support across ALL evidence.
    # This remains cheap and citation-aware.
    # -----------------------------------------------------------
    for citation_id, evidence_windows in evidence_map.items():
        if citation_id in excluded:
            continue

        if not evidence_windows:
            continue

        for evidence_window in evidence_windows:
            (
                exact_supported,
                exact_score,
            ) = _deterministic_exact_support(
                claim,
                evidence_window,
            )

            if exact_supported and exact_score > best_score:
                best_score = exact_score
                best_evidence_unit = evidence_window
                best_citation = citation_id
                best_method = "deterministic_exact_match"

                if best_score >= ENTAILMENT_THRESHOLD:
                    break

        if best_score >= ENTAILMENT_THRESHOLD:
            break

    # -----------------------------------------------------------
    # Phase 2: semantic discovery with ONE global budget.
    # -----------------------------------------------------------
    if best_score < ENTAILMENT_THRESHOLD and semantic_budget > 0:
        candidates = _select_global_entailment_candidates(
            claim,
            evidence_map,
            excluded_citations=excluded,
            limit=min(
                MAX_DISCOVERY_ENTAILMENT_CANDIDATES,
                semantic_budget,
            ),
        )

        for citation_id, evidence_window in candidates:
            if citation_id in excluded:
                continue

            if semantic_budget <= 0:
                break

            try:
                scorer = _get_entailment_scorer()

                cache_key = (
                    claim,
                    _safe_text(evidence_window),
                )

                if (
                    entailment_cache is not None
                    and cache_key in entailment_cache
                ):
                    score = entailment_cache[cache_key]
                else:
                    score = scorer(
                        premise=evidence_window,
                        hypothesis=claim,
                    )
                    semantic_budget -= 1
                    score = max(
                        0.0,
                        min(1.0, float(score)),
                    )

                    if entailment_cache is not None:
                        entailment_cache[cache_key] = score

            except Exception:
                continue

            if score > best_score:
                best_score = score
                best_evidence_unit = evidence_window
                best_citation = citation_id
                best_method = "entailment"

            # Stop immediately once the grounding threshold is met.
            if best_score >= ENTAILMENT_THRESHOLD:
                break

    supported = bool(
        best_score >= ENTAILMENT_THRESHOLD
        or best_method == "deterministic_exact_match"
    )

    return {
        "score": round(best_score, 4),
        "supported": supported,
        "best_evidence_unit": best_evidence_unit,
        "best_citation": best_citation,
        "support_method": best_method,
    }

def _validate_atomic_claim(
    claim: str,
    citations: Sequence[str],
    evidence_map: Dict[str, List[str]],
    entailment_cache: Dict[Tuple[str, str], float] | None = None,
) -> Dict[str, Any]:
    """
    Validate one atomic claim against its cited evidence.

    Deterministic exact support scans all cited evidence windows.

    Semantic verification uses ONE shared HHEMv2 budget per atomic claim
    across both cited verification and independent discovery.

    Deterministic exact support is performed first and does not consume
    the semantic budget.
    """

    claim = _clean_claim_text(claim)
    citations = _dedupe_preserve_order(citations)

    if not claim:
        return {
            "claim": claim,
            "score": 0.0,
            "supported": False,
            "best_evidence_unit": "",
            "best_citation": "",
            "citations": citations,
            "support_method": "empty_claim",
        }

    # ---------------------------------------------------------------
    # First: validate the citations already attached to the claim.
    # ---------------------------------------------------------------
    if citations:
        best_score = 0.0
        best_evidence_unit = ""
        best_citation = ""
        best_method = "entailment"

        # -----------------------------------------------------------
        # Phase 1A: deterministic exact support across ALL cited
        # evidence. This is cheap and remains citation-aware.
        # -----------------------------------------------------------
        for citation_id in citations:
            evidence_windows = evidence_map.get(citation_id, [])

            if not evidence_windows:
                continue

            for evidence_window in evidence_windows:
                (
                    exact_supported,
                    exact_score,
                ) = _deterministic_exact_support(
                    claim,
                    evidence_window,
                )

                if exact_supported and exact_score > best_score:
                    best_score = exact_score
                    best_evidence_unit = evidence_window
                    best_citation = citation_id
                    best_method = "deterministic_exact_match"

        if best_score >= 1.0:
            return {
                "claim": claim,
                "score": round(best_score, 4),
                "supported": True,
                "best_evidence_unit": best_evidence_unit,
                "best_citation": best_citation,
                "citations": [best_citation] if best_citation else [],
                "support_method": best_method,
            }

        # -----------------------------------------------------------
        # Phase 1B: semantic cited verification.
        #
        # IMPORTANT: all cited evidence competes in ONE global pool.
        # MAX_CITED_ENTAILMENT_CANDIDATES therefore means exactly
        # what it says: at most N HHEMv2 candidates per claim,
        # regardless of how many citations the claim has.
        # -----------------------------------------------------------
        semantic_budget_remaining = MAX_ATOMIC_SEMANTIC_BUDGET

        cited_candidate_limit = min(
            MAX_CITED_ENTAILMENT_CANDIDATES,
            semantic_budget_remaining,
        )

        candidates = _select_global_entailment_candidates(
            claim=claim,
            evidence_map=evidence_map,
            included_citations=citations,
            limit=cited_candidate_limit,
        )

        for citation_id, evidence_window in candidates:
            if semantic_budget_remaining <= 0:
                break

            try:
                scorer = _get_entailment_scorer()

                cache_key = (
                    claim,
                    _safe_text(evidence_window),
                )

                if (
                    entailment_cache is not None
                    and cache_key in entailment_cache
                ):
                    score = entailment_cache[cache_key]
                else:
                    score = scorer(
                        premise=evidence_window,
                        hypothesis=claim,
                    )
                    semantic_budget_remaining -= 1
                    score = max(
                        0.0,
                        min(1.0, float(score)),
                    )

                    if entailment_cache is not None:
                        entailment_cache[cache_key] = score

            except Exception:
                continue

            if score > best_score:
                best_score = score
                best_evidence_unit = evidence_window
                best_citation = citation_id
                best_method = "entailment"

            if best_score >= ENTAILMENT_THRESHOLD:
                break

        if (
            best_score >= ENTAILMENT_THRESHOLD
            or best_method == "deterministic_exact_match"
        ):
            return {
                "claim": claim,
                "score": round(best_score, 4),
                "supported": True,
                "best_evidence_unit": best_evidence_unit,
                "best_citation": best_citation,
                "citations": [best_citation] if best_citation else [],
                "support_method": best_method,
            }

    # ---------------------------------------------------------------
    # Second: cited evidence failed, or the claim was uncited.
    #
    # Search other available citations independently. Exclude the
    # original citations so a bad citation cannot simply validate itself.
    # ---------------------------------------------------------------
    discovery_budget = min(
        MAX_DISCOVERY_ENTAILMENT_CANDIDATES,
        semantic_budget_remaining
        if citations
        else MAX_ATOMIC_SEMANTIC_BUDGET,
    )

    if discovery_budget > 0:
        # Temporarily apply the remaining shared semantic budget to
        # independent discovery. The discovery helper already performs
        # deterministic exact support before semantic entailment.
        discovery_limit = discovery_budget
    else:
        discovery_limit = 0

    if discovery_limit > 0:
        discovered = _discover_supporting_evidence(
            claim=claim,
            evidence_map=evidence_map,
            excluded_citations=citations,
            entailment_cache=entailment_cache,
            semantic_budget=discovery_limit,
        )
    else:
        discovered = {
            "score": best_score if citations else 0.0,
            "supported": False,
            "best_evidence_unit": best_evidence_unit if citations else "",
            "best_citation": best_citation if citations else "",
            "support_method": "semantic_budget_exhausted",
        }

    if discovered.get("supported"):
        discovered_citation = _safe_text(
            discovered.get("best_citation")
        )

        if discovered_citation:
            repaired_citations = [discovered_citation]
        else:
            repaired_citations = list(citations)

        return {
            "claim": claim,
            "score": discovered.get("score", 0.0),
            "supported": True,
            "best_evidence_unit": discovered.get(
                "best_evidence_unit",
                "",
            ),
            "best_citation": discovered_citation,
            "citations": repaired_citations,
            "support_method": discovered.get(
                "support_method",
                "discovered_support",
            ),
        }

    return {
        "claim": claim,
        "score": discovered.get("score", 0.0),
        "supported": False,
        "best_evidence_unit": discovered.get(
            "best_evidence_unit",
            "",
        ),
        "best_citation": discovered.get(
            "best_citation",
            "",
        ),
        "citations": citations,
        "support_method": discovered.get(
            "support_method",
            "unsupported",
        ),
    }


def _audit_citation_consistency(
    atomic_results: Sequence[Dict[str, Any]],
    evidence_map: Dict[str, List[str]],
    entailment_cache: Dict[Tuple[str, str], float] | None = None,
) -> Dict[str, Any]:
    """
    Verify that every effective citation attached to every atomic claim
    independently corresponds to the support result already established
    during `_validate_atomic_claim()`.

    The atomic validation pass is authoritative for entailment. This audit
    deliberately performs no additional HHEMv2 inference so that citation
    diagnostics do not duplicate expensive grounding work.
    """

    citation_results: List[Dict[str, Any]] = []
    violations: List[Dict[str, Any]] = []

    claims_checked = 0
    citations_checked = 0

    for atomic_result in atomic_results:
        claim = _clean_claim_text(
            _safe_text(atomic_result.get("claim"))
        )

        if not claim:
            continue

        citations = _dedupe_preserve_order(
            atomic_result.get("citations") or []
        )

        if not citations:
            continue

        claims_checked += 1

        best_citation = _safe_text(
            atomic_result.get("best_citation")
        )
        best_score = float(
            atomic_result.get("score", 0.0) or 0.0
        )
        best_evidence_unit = _safe_text(
            atomic_result.get("best_evidence_unit")
        )
        best_method = _safe_text(
            atomic_result.get("support_method")
        ) or "entailment"

        for citation_id in citations:
            citations_checked += 1

            evidence_windows = evidence_map.get(
                citation_id,
                [],
            )

            if not evidence_windows:
                result = {
                    "claim": claim,
                    "citation": citation_id,
                    "score": 0.0,
                    "supported": False,
                    "support_method": "missing_evidence",
                    "best_evidence_unit": "",
                }

                citation_results.append(result)
                violations.append(result)
                continue

            if citation_id == best_citation:
                score = best_score
                supported = bool(
                    (
                        atomic_result.get("supported")
                        and score >= ENTAILMENT_THRESHOLD
                    )
                    or best_method == "deterministic_exact_match"
                )
                evidence_unit = best_evidence_unit
                support_method = best_method

            else:
                score = 0.0
                supported = False
                evidence_unit = ""
                support_method = "not_selected"

            result = {
                "claim": claim,
                "citation": citation_id,
                "score": round(score, 4),
                "supported": supported,
                "support_method": support_method,
                "best_evidence_unit": evidence_unit,
            }

            citation_results.append(result)

            if not supported:
                violations.append(result)

    return {
        "passed": not violations,
        "claims_checked": claims_checked,
        "citations_checked": citations_checked,
        "violation_count": len(violations),
        "violations": violations,
        "results": citation_results,
    }

def _build_revision_prompt(
    unsupported_atomic_claims: Sequence[Dict[str, Any]],
) -> str:

    if not unsupported_atomic_claims:
        return ""

    lines = [
        "STRICT GROUNDING REVISION REQUIRED.",
        "",
        "Rewrite the answer using ONLY claims that are directly supported "
        "by the supplied generation evidence.",
        "",
        "Rules:",
        "1. Remove unsupported claims rather than guessing.",
        "2. Do not introduce new facts.",
        "3. Preserve supported claims when possible.",
        "4. Keep citations attached to the claims they support.",
        "5. Do not merge unsupported details into otherwise supported claims.",
        "6. If a claim cannot be supported, omit it.",
        "",
        "Unsupported atomic claims:",
    ]

    for index, item in enumerate(
        unsupported_atomic_claims,
        start=1,
    ):
        claim = _safe_text(item.get("claim"))

        score = item.get(
            "score",
            0.0,
        )

        evidence = _safe_text(item.get("best_evidence_unit"))

        citation = _safe_text(item.get("best_citation"))

        lines.append(f"{index}. {claim}")

        lines.append(f"   Entailment score: {score}")

        if citation:
            lines.append(f"   Citation: [{citation}]")

        if evidence:
            lines.append("   Best available evidence: " + evidence)

    lines.extend(
        [
            "",
            "Return ONLY the revised answer.",
        ]
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Grounding critic node
# ---------------------------------------------------------------------------


def grounding_critic_node(
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Strict post-generation grounding validator.
    """

    grounding_started = perf_counter()

    # Per-pass HHEMv2 profiling baseline.
    from app.tools.entailment_tool import get_hhem_profile
    hhem_profile_start = get_hhem_profile()

    final_answer = _safe_text(
        state.get("final_answer")
        or state.get("candidate_answer")
        or state.get("answer")
    )

    # ------------------------------------------------------------------
    # Empty answer
    # ------------------------------------------------------------------

    if not final_answer:
        grounding_scores = {
            "atomic_claims": [],
            "claims": [],
            "unsupported_claims": [],
            "uncited_claims": [],
            "invalid_citations": [],
            "entailment_threshold": (ENTAILMENT_THRESHOLD),
            "claim_count": 0,
            "atomic_claim_count": 0,
            "unsupported_atomic_count": 0,
            "available_citation_count": 0,
            "evidence_window": {
                "radius": EVIDENCE_WINDOW_RADIUS,
                "large_radius": (EVIDENCE_LARGE_WINDOW_RADIUS),
                "max_candidates": (MAX_ENTAILMENT_CANDIDATES),
                "fallback_candidates": (FALLBACK_WINDOW_COUNT),
                "minimum_lexical_overlap": (MIN_WINDOW_OVERLAP),
            },
        }

        trace_update = append_trace_event(
            state,
            step="grounding_critic",
            status="skipped",
            is_grounded=False,
            answer_supported=False,
            support_score=0.0,
            claim_count=0,
            atomic_claim_count=0,
            unsupported_atomic_count=0,
            reason="empty_answer",
        )

        return {
            **trace_update,
            "is_grounded": False,
            "answer_supported": False,
            "support_score": 0.0,
            "grounding_scores": grounding_scores,
            "unsupported_atomic_claims": [],
            "revision_prompt": "",
            "status": ("Grounding validation failed: empty answer."),
        }

    # ------------------------------------------------------------------
    # Extract claims/evidence
    # ------------------------------------------------------------------

    claim_records = _extract_claim_records(final_answer)

    # Reuse repeated HHEMv2 claim/evidence scores within this request only.
    entailment_cache: Dict[Tuple[str, str], float] = {}

    evidence_map = _build_evidence_map(state)

    citation_metadata = _resolve_citation_metadata(
        state.get("citation_provenance")
    )

    available_citations = {
        citation_id for citation_id, windows in evidence_map.items() if windows
    }

    atomic_results: List[Dict[str, Any]] = []
    claim_results: List[Dict[str, Any]] = []

    unsupported_claims: List[Dict[str, Any]] = []
    unsupported_atomic_claims: List[Dict[str, Any]] = []

    uncited_claims: List[Dict[str, Any]] = []
    invalid_citations: List[Dict[str, Any]] = []

    substantive_results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Validate every claim
    # ------------------------------------------------------------------

    for record in claim_records:
        claim_text = _safe_text(record.get("text"))

        citations = _dedupe_preserve_order(
            record.get(
                "citations",
                [],
            )
        )

        atomic_claims = record.get("atomic_claims") or [claim_text]

        # --------------------------------------------------------------
        # Citation validation
        # --------------------------------------------------------------

        missing_citations = [
            citation for citation in citations if citation not in available_citations
        ]

        if missing_citations:
            invalid_citations.append(
                {
                    "claim": claim_text,
                    "citations": missing_citations,
                }
            )


        # --------------------------------------------------------------
        # Atomic validation
        # --------------------------------------------------------------

        record_atomic_results: List[Dict[str, Any]] = []

        for atomic_claim in atomic_claims:
            result = _validate_atomic_claim(
                atomic_claim,
                citations,
                evidence_map,
                entailment_cache,
            )

            best_citation = _safe_text(
                result.get("best_citation")
            )

            if result.get("supported") and best_citation:
                result["citations"] = [best_citation]
            result["evidence_provenance"] = citation_metadata.get(
                best_citation,
                {
                    "citation_id": best_citation or None,
                    "document_id": None,
                    "chunk_id": None,
                    "point_id": None,
                    "canonical_evidence_id": None,
                },
            )
            result["parent_claim"] = claim_text

            record_atomic_results.append(result)

            atomic_results.append(result)

            if _safe_text(result.get("claim")):
                substantive_results.append(result)

            if not result["supported"]:
                unsupported_atomic_claims.append(
                    {
                        "claim": result["claim"],
                        "score": result["score"],
                        "supported": False,
                        "best_evidence_unit": (result["best_evidence_unit"]),
                        "best_citation": (result["best_citation"]),
                        "citations": result["citations"],
                        "support_method": result["support_method"],
                        "parent_claim": claim_text,
                    }
                )

        # --------------------------------------------------------------
        # Effective citations after atomic validation / repair
        # --------------------------------------------------------------

        effective_citations = _dedupe_preserve_order(
            citation
            for result in record_atomic_results
            for citation in result.get("citations", [])
            if _safe_text(citation)
        )

        # If no atomic result repaired or supplied a citation, preserve
        # the original responder citations.
        if not effective_citations:
            effective_citations = list(citations)

        if not effective_citations:
            uncited_claims.append(
                {
                    "claim": claim_text,
                }
            )

        # Top-level claim status
        # --------------------------------------------------------------

        if record_atomic_results:
            claim_supported = all(
                bool(result["supported"]) for result in record_atomic_results
            )

            claim_score = min(
                float(result["score"]) for result in record_atomic_results
            )

        else:
            claim_supported = False
            claim_score = 0.0

        claim_result = {
            "claim": claim_text,
            "score": round(
                claim_score,
                4,
            ),
            "supported": claim_supported,
            "citations": effective_citations,
            "atomic_claims": record_atomic_results,
        }

        claim_results.append(claim_result)

        if not claim_supported:
            unsupported_claims.append(claim_result)

    # ------------------------------------------------------------------
    # Citation consistency audit
    # ------------------------------------------------------------------

    citation_consistency = _audit_citation_consistency(
        atomic_results,
        evidence_map,
        entailment_cache,
    )

    # ------------------------------------------------------------------
    # Deduplicate unsupported atomic claims
    # ------------------------------------------------------------------

    deduped_unsupported_atomic_claims: List[Dict[str, Any]] = []

    seen_atomic = set()

    for item in unsupported_atomic_claims:
        key = (
            _safe_text(item.get("claim")),
            _safe_text(item.get("best_citation")),
        )

        if key in seen_atomic:
            continue

        seen_atomic.add(key)

        deduped_unsupported_atomic_claims.append(item)

    unsupported_atomic_claims = deduped_unsupported_atomic_claims

    # ------------------------------------------------------------------
    # Overall support score
    # ------------------------------------------------------------------

    if substantive_results:
        support_score = min(float(result["score"]) for result in substantive_results)
    else:
        support_score = 0.0

    # ------------------------------------------------------------------
    # Strict grounding decision
    # ------------------------------------------------------------------

    semantic_support_ok = bool(substantive_results and not unsupported_claims)

    citation_support_ok = bool(not uncited_claims and not invalid_citations)

    answer_supported = bool(semantic_support_ok and citation_support_ok)

    is_grounded = bool(answer_supported)

    # ------------------------------------------------------------------
    # Stable grounding schema
    # ------------------------------------------------------------------

    grounding_scores = {
        "atomic_claims": atomic_results,
        "claims": claim_results,
        "unsupported_claims": unsupported_claims,
        "uncited_claims": uncited_claims,
        "invalid_citations": invalid_citations,
        "entailment_threshold": (ENTAILMENT_THRESHOLD),
        "claim_count": len(claim_results),
        "atomic_claim_count": len(atomic_results),
        "unsupported_atomic_count": len(unsupported_atomic_claims),
        "available_citation_count": len(available_citations),
        "citation_consistency": citation_consistency,
        "evidence_window": {
            "radius": EVIDENCE_WINDOW_RADIUS,
            "large_radius": (EVIDENCE_LARGE_WINDOW_RADIUS),
            "max_candidates": (MAX_ENTAILMENT_CANDIDATES),
            "fallback_candidates": (FALLBACK_WINDOW_COUNT),
            "minimum_lexical_overlap": (MIN_WINDOW_OVERLAP),
        },
    }

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    feedback_parts: List[str] = []

    if unsupported_atomic_claims:
        feedback_parts.append(
            f"{len(unsupported_atomic_claims)} unsupported atomic claim(s)"
        )

    if uncited_claims:
        feedback_parts.append(f"{len(uncited_claims)} uncited claim(s)")

    if invalid_citations:
        feedback_parts.append(f"{len(invalid_citations)} invalid citation reference(s)")

    if not feedback_parts:
        feedback = "All substantive claims passed strict grounding validation."

    else:
        feedback = "Grounding validation failed: " + ", ".join(feedback_parts) + "."

    # ------------------------------------------------------------------
    # Revision prompt
    # ------------------------------------------------------------------

    revision_prompt = _build_revision_prompt(unsupported_atomic_claims)

    existing_revision_prompt = _safe_text(state.get("revision_prompt"))

    if not revision_prompt and existing_revision_prompt:
        revision_prompt = existing_revision_prompt

    # ------------------------------------------------------------------
    # Final state update
    # ------------------------------------------------------------------

    hhem_profile_end = get_hhem_profile()
    hhem_call_count = (
        int(hhem_profile_end.get("call_count", 0))
        - int(hhem_profile_start.get("call_count", 0))
    )
    hhem_total_ms = round(
        float(hhem_profile_end.get("total_ms", 0.0))
        - float(hhem_profile_start.get("total_ms", 0.0)),
        2,
    )

    grounding_ms = round(
        (perf_counter() - grounding_started) * 1000,
        2,
    )

    trace_update = append_trace_event(
        state,
        step="grounding_critic",
        status="passed" if is_grounded else "failed",
        is_grounded=is_grounded,
        answer_supported=answer_supported,
        support_score=round(float(support_score), 4),
        claim_count=len(claim_results),
        atomic_claim_count=len(atomic_results),
        unsupported_atomic_count=len(unsupported_atomic_claims),
        uncited_claim_count=len(uncited_claims),
        invalid_citation_count=len(invalid_citations),
        grounding_ms=grounding_ms,
        hhem_call_count=hhem_call_count,
        hhem_total_ms=hhem_total_ms,
    )

    return {
        **trace_update,
        "is_grounded": is_grounded,
        "answer_supported": answer_supported,
        "support_score": round(
            float(support_score),
            4,
        ),
        "grounding_scores": grounding_scores,
        "grounding_ms": grounding_ms,
        "hhem_call_count": hhem_call_count,
        "hhem_total_ms": hhem_total_ms,
        "grounding_feedback": feedback,
        "unsupported_atomic_claims": (unsupported_atomic_claims),
        "citation_consistency": citation_consistency,
        "revision_prompt": revision_prompt,
        "status": (
            "Grounding validation passed."
            if is_grounded
            else (
                "Grounding validation failed: "
                f"{len(unsupported_atomic_claims)} "
                "unsupported atomic claim(s)."
            )
        ),
    }

