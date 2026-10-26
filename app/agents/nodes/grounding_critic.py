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

Citation consistency policy
---------------------------

A claim must be supported (that is checked separately, per claim). The
citation audit then asks whether the citations ATTACHED to that claim are
honest:

    HARD violation (fails grounding, triggers a revision):
        * a cited chunk has no evidence at all, or
        * NONE of the cited chunks supports the claim (the claim was only
          supported by a chunk it did not cite).

    SOFT warning (reported, never fails grounding):
        * at least one cited chunk supports the claim, and an additional
          co-cited chunk was not verified as supporting it.

A redundant co-citation is a style problem, not a hallucination.
"""

from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field

import logging
import math
import re
from difflib import SequenceMatcher
from contextvars import ContextVar

from time import perf_counter
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from app.evaluation.trace import append_trace_event


logger = logging.getLogger(__name__)


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
MAX_COMBINED_PREMISE_CHARS = 3000

MIN_WINDOW_OVERLAP = 1
FALLBACK_WINDOW_COUNT = 3

# Atomic claims shorter than this are merged into a neighbouring claim
# instead of being evaluated as an orphan fragment without its subject.
MIN_ATOMIC_WORDS = 6


class _CountingEntailmentCache(dict):
    """Dictionary that counts cache membership hits and misses."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hits = 0
        self.misses = 0

    def __contains__(self, key):
        found = super().__contains__(key)
        if found:
            self.hits += 1
        else:
            self.misses += 1
        return found


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


def _prefetch_entailment_scores(
    claim_records: Sequence[Dict[str, Any]],
    evidence_map: Dict[str, List[str]],
    entailment_cache: Dict[Tuple[str, str], float],
) -> Dict[Tuple[str, str], float]:
    """
    Batch-score the pairs the per-claim logic is about to need.

    Mirrors the candidate selection of `_validate_atomic_claim`:
      * cited claim  -> top candidates from the cited chunks only
      * uncited claim -> top candidates across all evidence (discovery)
      * single-citation claim already resolved by a literal match -> skipped

    Scoring runs in waves by candidate rank so early exits are preserved:
    claims that already meet the threshold (and stop at the first hit in
    the sequential code) are dropped from later waves.

    Never raises. On any failure the caller simply falls back to the
    original one-pair-at-a-time scoring.
    """

    prefetched: Dict[Tuple[str, str], float] = {}

    try:
        from app.tools.entailment_tool import score_entailment_batch
    except ImportError:
        return prefetched

    try:
        plans: List[Dict[str, Any]] = []
        seen = set()

        for record in claim_records:
            citations = _dedupe_preserve_order(record.get("citations", []))

            atomic_claims = record.get("atomic_claims") or [record.get("text")]

            for atomic in atomic_claims:
                claim = _clean_claim_text(_safe_text(atomic))

                if not claim:
                    continue

                plan_key = (claim, tuple(citations))

                if plan_key in seen:
                    continue

                seen.add(plan_key)

                if citations:
                    # Same early return as _validate_atomic_claim: one
                    # citation + literal match scores 1.0 without HHEM.
                    if len(citations) <= 1 and any(
                        _deterministic_exact_support(claim, window)[1] >= 1.0
                        for citation_id in citations
                        for window in evidence_map.get(citation_id, [])
                    ):
                        continue

                    reserve = 1 if len(citations) > 1 else 0

                    limit = max(
                        1,
                        min(
                            MAX_CITED_ENTAILMENT_CANDIDATES,
                            MAX_ATOMIC_SEMANTIC_BUDGET - reserve,
                        ),
                    )

                    candidates = _select_global_entailment_candidates(
                        claim=claim,
                        evidence_map=evidence_map,
                        included_citations=citations,
                        limit=limit,
                    )

                else:
                    # Uncited: discovery path. A literal match anywhere
                    # resolves it without HHEM.
                    if any(
                        _deterministic_exact_support(claim, window)[0]
                        for windows in evidence_map.values()
                        for window in windows
                    ):
                        continue

                    candidates = _select_global_entailment_candidates(
                        claim=claim,
                        evidence_map=evidence_map,
                        limit=min(
                            MAX_DISCOVERY_ENTAILMENT_CANDIDATES,
                            MAX_ATOMIC_SEMANTIC_BUDGET,
                        ),
                    )

                if not candidates:
                    continue

                plans.append(
                    {
                        "claim": claim,
                        "windows": [_safe_text(w) for _cid, w in candidates],
                        # The sequential code stops at the first candidate
                        # meeting the threshold when there is <=1 citation.
                        "early_stop": len(citations) <= 1,
                    }
                )

        if not plans:
            return prefetched

        resolved: set = set()
        waves = 0
        pairs_scored = 0
        max_rank = max(len(plan["windows"]) for plan in plans)

        for rank in range(max_rank):
            wave_keys: List[Tuple[int, Tuple[str, str]]] = []
            wave_pairs: List[Tuple[str, str]] = []
            queued = set()

            for index, plan in enumerate(plans):
                if index in resolved or rank >= len(plan["windows"]):
                    continue

                window = plan["windows"][rank]
                key = (plan["claim"], window)

                # Scores restored from earlier passes: the lazy path will
                # hit them. Just use them for the early-stop decision.
                if dict.__contains__(entailment_cache, key):
                    if plan["early_stop"] and (
                        entailment_cache[key] >= ENTAILMENT_THRESHOLD
                    ):
                        resolved.add(index)
                    continue

                if key in prefetched:
                    continue

                wave_keys.append((index, key))

                if key not in queued:
                    queued.add(key)
                    wave_pairs.append((window, plan["claim"]))

            if not wave_pairs:
                continue

            scores = score_entailment_batch(wave_pairs)

            waves += 1
            pairs_scored += len(wave_pairs)

            score_by_key = {
                (claim, window): score
                for (window, claim), score in zip(wave_pairs, scores)
            }

            for index, key in wave_keys:
                score = score_by_key[key]
                prefetched[key] = score

                if plans[index]["early_stop"] and score >= ENTAILMENT_THRESHOLD:
                    resolved.add(index)

        logger.info(
            "GROUNDING_PREFETCH claims=%s waves=%s pairs=%s",
            len(plans),
            waves,
            pairs_scored,
        )

    except Exception:
        logger.warning(
            "Grounding prefetch failed; falling back to sequential scoring.",
            exc_info=True,
        )

    return prefetched


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


_ABSENCE_RE = re.compile(
    r"\b(?:evidence|sources?|documents?|context|passages?|paper|survey|report)\b"
    r"[^.;]{0,40}\b(?:does|do|did)\s+not\s+"
    r"(?:contain|provide|include|mention|state|specify|establish|report|cover|address|give)\b"
    r"|\b(?:there\s+is|there's|is)\s+no\s+(?:evidence|information|data|mention)\b"
    r"|\bcould(?:n't|\s+not)\s+find\b",
    re.IGNORECASE,
)


def _is_pure_absence(text: str) -> bool:
    """A short single-clause statement about what the sources lack.

    Mixed sentences (an absence clause joined to a factual clause) are NOT
    matched, so the factual part is still verified.
    """
    text = _strip_citations(text or "").strip()

    return bool(
        _ABSENCE_RE.search(text)
        and len(text.split()) <= 28
        and not re.search(r";|,\s*(?:while|whereas|but|and)\b", text, re.IGNORECASE)
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

    # Remove interior emphasis/code markers so the entailment scorer sees
    # clean prose ("**Multi-agent** - ..." must not become "Multi-agent** - ...").
    # "__" is left alone on purpose so identifiers like __init__ survive.
    text = text.replace("**", "").replace("`", "")

    text = text.strip(" `*_")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


_prefetched_scores: ContextVar["Dict[Tuple[str, str], float] | None"] = ContextVar(
    "grounding_prefetched_scores",
    default=None,
)


def _merge_short_claim_pieces(pieces: Sequence[str]) -> List[str]:
    """
    Merge pieces shorter than MIN_ATOMIC_WORDS into a neighbour.

    A fragment such as "the agent runs proven skills" has lost its
    subject and is evaluated out of context, where it can pass or fail
    by lexical accident. A short piece is appended to the previous
    piece; a short FIRST piece is carried forward into the next one.
    """

    merged: List[str] = []
    carry = ""

    for raw_piece in pieces:
        piece = raw_piece.strip(" -•")

        if not piece:
            continue

        if carry:
            piece = f"{carry}; {piece}"
            carry = ""

        if len(piece.split()) < MIN_ATOMIC_WORDS:
            if merged:
                merged[-1] = f"{merged[-1]}; {piece}"
            else:
                carry = piece

            continue

        merged.append(piece)

    if carry:
        merged.append(carry)

    return merged


def _split_compound_claim(text: str) -> List[str]:
    """
    Conservative atomic-claim splitter.

    Avoids aggressive splitting because technical statements frequently
    contain several dependent clauses that should be evaluated together.

    Two guards keep it from producing orphan fragments:

      * ';' pieces shorter than MIN_ATOMIC_WORDS are merged back into a
        neighbouring piece.
      * An " and " that closes an enumeration ("a trigger, a goal, ...,
        and an external memory ...") is never used as a split point.
    """

    text = _clean_claim_text(text)

    if not text:
        return []

    pieces = _merge_short_claim_pieces(
        re.split(
            r";\s+",
            text,
        )
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

            is_enumeration = (
                candidates[0].rstrip().endswith(",") or candidates[0].count(",") >= 2
            )

            if (
                not is_enumeration
                and len(candidates) == 2
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

    cleaned = [piece for piece in cleaned if not _is_pure_absence(piece)]

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
            if _is_pure_absence(claim):
                continue  # a limitation statement; nothing to verify

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
    """Return an explicit answer-facing citation label, not a chunk/point ID."""
    if isinstance(document, dict):
        for key in ("citation_id", "citation"):
            value = document.get(key)

            if value:
                return _safe_text(value)

    for attr in ("citation_id", "citation"):
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


def _explicit_query_source_entity_mismatch(
    query: Any,
    source: Any,
) -> bool:
    """Flag clear company-name mismatches in identifiable source titles."""
    query_text = _safe_text(query)
    source_text = _safe_text(source)

    if not query_text or not source_text:
        return False

    query_match = re.search(
        r"\bcompany\s+(?:named\s+)?([A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*)*)",
        query_text,
    )
    if not query_match:
        return False

    expected = re.sub(r"[^a-z0-9]", "", query_match.group(1).lower())
    if not expected:
        return False

    # Only infer identity from a recognizable company prefix before
    # common report/document markers. Generic titles are inconclusive.
    title = source_text.strip()
    if title.lower().startswith(("http://", "https://")):
        return False

    marker = re.search(
        r"\b(annual report|financial statements|financial results|"
        r"investor relations|quarterly report|company profile)\b",
        title,
        flags=re.IGNORECASE,
    )
    if not marker:
        return False

    source_entity = title[: marker.start()].strip(" \t-–—:|,")
    source_entity_normalized = re.sub(r"[^a-z0-9]", "", source_entity.lower())

    if not source_entity_normalized:
        return False

    # The source prefix must identify an entity; generic prefixes are
    # not reliable evidence of a different company.
    generic_prefixes = {
        "the",
        "company",
        "annual",
        "financial",
        "report",
        "official",
        "investor",
        "consolidated",
    }
    if source_entity.lower() in generic_prefixes:
        return False

    return expected not in source_entity_normalized


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
            item.get("citation_id") or item.get("citation") or item.get("id")
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
                point_id = nested.get("point_id") or nested.get("id")

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
            "source": _safe_text(
                item.get("source")
                or item.get("title")
                or (nested.get("source") if isinstance(nested, dict) else None)
                or (nested.get("title") if isinstance(nested, dict) else None)
            ),
            "url": _safe_text(
                item.get("url")
                or (nested.get("url") if isinstance(nested, dict) else None)
            ),
            "source_type": _safe_text(
                item.get("source_type")
                or (nested.get("source_type") if isinstance(nested, dict) else None)
            ),
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
    Resolve answer evidence without promoting ungraded documents.

    Citation provenance is authoritative. Approved generation documents
    may fill missing citation labels. Raw web/document fallbacks are
    permitted only for legacy states without explicit grader status.
    """

    evidence_map: Dict[str, List[str]] = {}
    provenance = state.get("citation_provenance")
    provenance_map = _resolve_citation_provenance(provenance)

    for citation_id, windows in provenance_map.items():
        evidence_map.setdefault(citation_id, []).extend(windows)

    generation_documents = _safe_list(state.get("generation_documents"))
    grader_status = str(state.get("grader_status") or "").strip().lower()
    grading_mode = str(state.get("grading_mode") or "").strip().lower()

    grading_not_approved = grader_status in {"failed", "skipped"} or grading_mode in {
        "unavailable",
        "no_documents",
    }

    if grading_not_approved:
        # Never promote stale generation documents after grading fails
        # or is skipped. Trusted citation provenance remains preserved.
        fallback_documents: List[Any] = []
    elif generation_documents:
        fallback_documents = list(generation_documents)
    elif not grader_status and not grading_mode:
        # Compatibility for legacy states that predate explicit grader
        # status fields.
        fallback_documents = _safe_list(state.get("web_documents")) + _safe_list(
            state.get("documents")
        )
    else:
        fallback_documents = []

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

    claim_numeric = [token for token in claim_tokens if token.isdigit()]

    if len(claim_numeric) < 2:
        return False, 0.0

    evidence_set = set(evidence_tokens)

    numeric_coverage = sum(1 for token in claim_numeric if token in evidence_set) / max(
        len(claim_numeric), 1
    )

    if numeric_coverage < 1.0:
        return False, 0.0

    claim_non_numeric = {token for token in claim_tokens if not token.isdigit()}

    if not claim_non_numeric:
        return False, 0.0

    non_numeric_overlap = len(claim_non_numeric.intersection(evidence_set)) / max(
        len(claim_non_numeric), 1
    )

    if non_numeric_overlap < 0.30:
        return False, 0.0

    score = min(
        1.0,
        0.80 + (numeric_coverage * 0.10) + (min(non_numeric_overlap, 1.0) * 0.10),
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

    numeric_supported, numeric_score = _deterministic_numeric_detail_support(
        claim,
        evidence,
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

_prefetched_scores: ContextVar["Dict[Tuple[str, str], float] | None"] = ContextVar(
    "grounding_prefetched_scores",
    default=None,
)


def _get_entailment_scorer():
    try:
        from app.tools.entailment_tool import score_entailment
    except ImportError as exc:
        raise ImportError(
            "Could not import KnowledgeMesh entailment scorer from "
            "app.tools.entailment_tool."
        ) from exc

    prefetched = _prefetched_scores.get()

    if not prefetched:
        return score_entailment

    def scorer(premise: str, hypothesis: str) -> float:
        hit = prefetched.get((hypothesis, _safe_text(premise)))

        if hit is not None:
            return hit

        return score_entailment(premise=premise, hypothesis=hypothesis)

    return scorer


def warm_up_entailment_model() -> bool:
    """
    Load HHEMv2 and run one throwaway inference.

    Call this ONCE at application startup (next to the NeMo Guardrails
    initialisation). Without it the first grounding request pays for the
    model load plus a cold inference (~7s in the logs).

    Never raises: a failed warm-up only means the first real request is
    slow, not broken. Returns True when the model answered.
    """

    try:
        scorer = _get_entailment_scorer()

        scorer(
            premise="Warm-up premise.",
            hypothesis="Warm-up hypothesis.",
        )

        logger.info("HHEMv2 grounding model warmed up.")

        return True

    except Exception:
        logger.warning(
            "HHEMv2 warm-up failed; the first grounding request will "
            "load the model lazily.",
            exc_info=True,
        )

        return False


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

    excluded = set(_dedupe_preserve_order(excluded_citations))

    included = set(_dedupe_preserve_order(included_citations))

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

    if len(included) > 1:
        selected = []
        selected_pairs = set()
        selected_citations = set()

        for item in scored:
            pair = (item[0], item[1])
            if item[0] in selected_citations:
                continue
            selected.append(item)
            selected_pairs.add(pair)
            selected_citations.add(item[0])
            if len(selected) >= limit:
                break

        for item in scored:
            pair = (item[0], item[1])
            if len(selected) >= limit:
                break
            if pair not in selected_pairs:
                selected.append(item)
                selected_pairs.add(pair)

        selected = [item for item in scored if (item[0], item[1]) in selected_pairs]
    else:
        selected = scored[:limit]

    return [
        (citation_id, evidence_window)
        for (
            citation_id,
            evidence_window,
            _relevance,
            _overlap,
            _density,
            _stable_index,
        ) in selected
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

    excluded = set(_dedupe_preserve_order(excluded_citations))

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

                if entailment_cache is not None and cache_key in entailment_cache:
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
        best_score >= ENTAILMENT_THRESHOLD or best_method == "deterministic_exact_match"
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

        if best_score >= 1.0 and len(citations) <= 1:
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

        # Multi-citation claims reserve one HHEM call for a combined-premise
        # check (the claim may need facts from several cited chunks).
        reserve_for_combined = 1 if len(citations) > 1 else 0

        cited_candidate_limit = max(
            1,
            min(
                MAX_CITED_ENTAILMENT_CANDIDATES,
                semantic_budget_remaining - reserve_for_combined,
            ),
        )

        per_citation_best: Dict[str, Tuple[float, str]] = {}

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

                if entailment_cache is not None and cache_key in entailment_cache:
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

            previous = per_citation_best.get(citation_id)
            if previous is None or score > previous[0]:
                per_citation_best[citation_id] = (score, evidence_window)

            if score > best_score:
                best_score = score
                best_evidence_unit = evidence_window
                best_citation = citation_id
                best_method = "entailment"

            if best_score >= ENTAILMENT_THRESHOLD and len(citations) <= 1:
                break

        # -----------------------------------------------------------
        # Phase 1C: combined-premise check for multi-citation claims.
        #
        # No single chunk entailed the claim, but the claim cites several
        # chunks and may be a fair synthesis of them. Score it once
        # against the best window from each cited chunk joined together.
        # best_citation stays the strongest single contributor so the
        # citation-consistency audit keeps working unchanged.
        # -----------------------------------------------------------
        if (
            best_score < ENTAILMENT_THRESHOLD
            and len(per_citation_best) > 1
            and semantic_budget_remaining > 0
        ):
            ordered = sorted(
                per_citation_best.values(),
                key=lambda item: item[0],
                reverse=True,
            )
            combined_premise = " ".join(
                _safe_text(window) for _score, window in ordered
            )[:MAX_COMBINED_PREMISE_CHARS]

            try:
                cache_key = (claim, combined_premise)

                if entailment_cache is not None and cache_key in entailment_cache:
                    combined_score = entailment_cache[cache_key]
                else:
                    combined_score = max(
                        0.0,
                        min(
                            1.0,
                            float(
                                _get_entailment_scorer()(
                                    premise=combined_premise,
                                    hypothesis=claim,
                                )
                            ),
                        ),
                    )
                    semantic_budget_remaining -= 1

                    if entailment_cache is not None:
                        entailment_cache[cache_key] = combined_score

                if combined_score >= ENTAILMENT_THRESHOLD:
                    best_score = combined_score
                    best_evidence_unit = combined_premise
                    best_method = "entailment_multi_chunk"
            except Exception:
                pass

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
        semantic_budget_remaining if citations else MAX_ATOMIC_SEMANTIC_BUDGET,
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

    discovered_citation = _safe_text(discovered.get("best_citation"))
    discovered_windows = evidence_map.get(
        discovered_citation,
        [],
    )

    if discovered.get("supported") and discovered_citation and discovered_windows:
        return {
            "claim": claim,
            "score": discovered.get("score", 0.0),
            "supported": True,
            "best_evidence_unit": discovered.get(
                "best_evidence_unit",
                "",
            ),
            "best_citation": discovered_citation,
            "citations": [discovered_citation],
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


# ---------------------------------------------------------------------------
# Citation consistency audit
# ---------------------------------------------------------------------------


def _check_citation_against_claim(
    claim: str,
    citation_id: str,
    atomic_result: Dict[str, Any],
    evidence_map: Dict[str, List[str]],
    entailment_cache: Dict[Tuple[str, str], float] | None,
) -> Dict[str, Any]:
    """
    Check ONE citation against ONE claim.

    Performs no new HHEMv2 inference. The atomic validation pass is
    authoritative for entailment; this only reuses its result, plus
    deterministic exact matches and already-cached qualifying scores.
    """

    windows = evidence_map.get(citation_id, [])

    def _result(
        score: float,
        supported: bool,
        method: str,
        unit: str = "",
    ) -> Dict[str, Any]:
        return {
            "claim": claim,
            "citation": citation_id,
            "score": round(float(score), 4),
            "supported": supported,
            "support_method": method,
            "best_evidence_unit": unit,
        }

    if not windows:
        return _result(0.0, False, "missing_evidence")

    if citation_id == _safe_text(atomic_result.get("best_citation")):
        score = float(atomic_result.get("score", 0.0) or 0.0)
        method = _safe_text(atomic_result.get("support_method")) or "entailment"
        supported = bool(
            atomic_result.get("supported")
            and (score >= ENTAILMENT_THRESHOLD or method == "deterministic_exact_match")
        )

        return _result(
            score,
            supported,
            method,
            _safe_text(atomic_result.get("best_evidence_unit")),
        )

    for window in windows:
        exact_supported, exact_score = _deterministic_exact_support(
            claim,
            window,
        )

        if exact_supported:
            return _result(
                exact_score,
                True,
                "deterministic_exact_match",
                window,
            )

    if entailment_cache is not None:
        for window in windows:
            cache_key = (
                claim,
                _safe_text(window),
            )

            if cache_key not in entailment_cache:
                continue

            cached_score = entailment_cache[cache_key]

            if (
                isinstance(cached_score, (int, float))
                and not isinstance(cached_score, bool)
                and math.isfinite(cached_score)
                and cached_score >= ENTAILMENT_THRESHOLD
            ):
                return _result(
                    float(cached_score),
                    True,
                    "entailment",
                    window,
                )

    return _result(0.0, False, "not_selected")


# Strict: every cited chunk must support its claim (matches main).
STRICT_CITATION_AUDIT = True


def _audit_citation_consistency(
    atomic_results: Sequence[Dict[str, Any]],
    evidence_map: Dict[str, List[str]],
    entailment_cache: Dict[Tuple[str, str], float] | None = None,
) -> Dict[str, Any]:
    """
    Verify that the citations attached to each SUPPORTED atomic claim are
    honest, using the support results already established during
    `_validate_atomic_claim()`. No additional HHEMv2 inference runs here.

    Hard violation (fails grounding, drives a revision):
        * a cited chunk has no evidence at all, or
        * none of the cited chunks supports the claim (the claim was only
          supported by a chunk it did not cite).

    Soft warning (reported only, never fails grounding):
        * at least one cited chunk supports the claim, and an additional
          co-cited chunk was not verified as supporting it.

    Atomic claims that are themselves unsupported are skipped: they are
    already reported through `unsupported_atomic_claims` and fail
    grounding on their own, so auditing their citations would only
    double-report the same problem.
    """

    citation_results: List[Dict[str, Any]] = []
    violations: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    claims_checked = 0
    citations_checked = 0

    for atomic_result in atomic_results:
        claim = _clean_claim_text(_safe_text(atomic_result.get("claim")))

        if not claim:
            continue

        citations = _dedupe_preserve_order(
            atomic_result.get("source_citations")
            or atomic_result.get("citations")
            or []
        )

        if not citations:
            continue

        claims_checked += 1

        per_claim = [
            _check_citation_against_claim(
                claim,
                citation_id,
                atomic_result,
                evidence_map,
                entailment_cache,
            )
            for citation_id in citations
        ]

        citations_checked += len(per_claim)

        any_supported = any(item["supported"] for item in per_claim)

        best_citation = _safe_text(atomic_result.get("best_citation"))

        for item in per_claim:
            citation_results.append(item)

            if item["supported"]:
                continue

            if (
                STRICT_CITATION_AUDIT
                or item["support_method"] == "missing_evidence"
                or not any_supported
            ):
                item["severity"] = "hard"

                if best_citation and best_citation not in citations:
                    item["suggested_citation"] = best_citation

                violations.append(item)

            else:
                item["severity"] = "soft"
                warnings.append(item)

    return {
        # Hard violations only.
        "passed": not violations,
        "claims_checked": claims_checked,
        "citations_checked": citations_checked,
        "violation_count": len(violations),
        "violations": violations,
        "warning_count": len(warnings),
        "warnings": warnings,
        "results": citation_results,
    }


# Sentence-initial words that point back at an earlier sentence. A line that
# starts with one, right after a deleted line, has lost its subject.
_ANAPHORIC_OPENERS = {
    "it",
    "its",
    "they",
    "their",
    "them",
    "this",
    "these",
    "those",
    "he",
    "she",
    "his",
    "her",
    "such",
}


def _starts_with_anaphor(line: str) -> bool:
    text = _strip_citations(_strip_markdown_prefix(line.strip()))
    text = text.lstrip(" *_`\"'(")

    if not text:
        return False

    first_word = re.split(r"\W+", text.lower(), maxsplit=1)[0]

    return first_word in _ANAPHORIC_OPENERS


def _prune_unsupported_lines(
    final_answer: str,
    unsupported_raw_lines: Sequence[str],
) -> str:
    """
    Deterministically remove the answer lines whose claims failed
    grounding, so the supported remainder can be salvaged instead of
    abstaining on the whole answer.

    Nothing is rewritten or invented: lines are only deleted (plus any
    heading left with no content under it). Returns "" when nothing was
    removed, no substantive claim line would remain, or a remaining line
    would be left dangling.

    Dangling: a kept line that opens with a pronoun or demonstrative
    ("It uses...", "These...") directly after a deleted line has lost its
    subject. Salvaging it would present a misleading statement, so the
    whole salvage is refused and the caller abstains instead.
    """

    drop = {line.strip() for line in unsupported_raw_lines if line.strip()}

    if not drop:
        return ""

    kept: List[str] = []

    previous_claim_dropped = False

    for line in _safe_text(final_answer).splitlines():
        stripped = line.strip()

        if not stripped:
            kept.append(line)
            continue

        if stripped in drop:
            previous_claim_dropped = True
            continue

        if not _is_non_claim_line(stripped):
            if previous_claim_dropped and _starts_with_anaphor(stripped):
                return ""

            previous_claim_dropped = False

        kept.append(line)

    cleaned: List[str] = []

    for index, line in enumerate(kept):
        if _is_heading(line.strip()):
            following = next(
                (item for item in kept[index + 1 :] if item.strip()),
                "",
            )

            if not following or _is_heading(following.strip()):
                continue

        cleaned.append(line)

    if not any(not _is_non_claim_line(line.strip()) for line in cleaned):
        return ""

    return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned)).strip()


def _build_revision_prompt(
    unsupported_atomic_claims: Sequence[Dict[str, Any]] = (),
    citation_violations: Sequence[Dict[str, Any]] = (),
    uncited_claims: Sequence[Dict[str, Any]] = (),
    invalid_citations: Sequence[Dict[str, Any]] = (),
) -> str:
    """
    Build an actionable revision prompt covering EVERY failure type the
    critic can produce, so the revision responder is never regenerating
    blind.
    """

    if not (
        unsupported_atomic_claims
        or citation_violations
        or uncited_claims
        or invalid_citations
    ):
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
        "7. Cite only chunk IDs that appear in the supplied evidence, and "
        "prefer citing just the chunk(s) that directly support each claim.",
        "8. Stay close to the evidence's own wording. Do not paraphrase "
        "with added detail (extra adjectives, purposes or explanations "
        "that the evidence does not state).",
    ]

    if unsupported_atomic_claims:
        lines.extend(
            [
                "",
                "Unsupported atomic claims:",
            ]
        )

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

            lines.append(
                "   Fix: restate it using the evidence's wording, or remove it."
            )

    if citation_violations:
        lines.extend(
            [
                "",
                "Citation corrections:",
            ]
        )

        for violation in citation_violations:
            line = (
                f"- Claim: {_safe_text(violation.get('claim'))}\n"
                f"  Citation [{_safe_text(violation.get('citation'))}] "
                "does not support it"
            )

            suggested = _safe_text(violation.get("suggested_citation"))

            if suggested:
                line += f"; cite [{suggested}] instead"
            else:
                line += "; remove the claim or the citation"

            lines.append(line)

    if uncited_claims:
        lines.extend(
            [
                "",
                "Claims missing a citation (add a valid chunk ID or remove the claim):",
            ]
        )

        for item in uncited_claims:
            lines.append(f"- {_safe_text(item.get('claim'))}")

    if invalid_citations:
        lines.extend(
            [
                "",
                "Citations not present in the evidence (use only supplied chunk IDs):",
            ]
        )

        for item in invalid_citations:
            cited = ", ".join(
                _safe_text(citation) for citation in item.get("citations", [])
            )

            lines.append(f"- {_safe_text(item.get('claim'))} -> {cited}")

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


def _grounding_critic_node_with_profile(
    state: Dict[str, Any],
    hhem_profile: Dict[str, float],
) -> Dict[str, Any]:
    """
    Strict post-generation grounding validator.
    """

    grounding_started = perf_counter()

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
            "pruned_answer": "",
            "prunable_claims": [],
            "status": ("Grounding validation failed: empty answer."),
        }

    # ------------------------------------------------------------------
    # Extract claims/evidence
    # ------------------------------------------------------------------

    claim_records = _extract_claim_records(final_answer)

    # Restore cached scores from earlier passes in this request.
    cached_entailment_entries = state.get("entailment_cache", [])
    entailment_cache = _CountingEntailmentCache()
    if isinstance(cached_entailment_entries, list):
        for entry in cached_entailment_entries:
            if isinstance(entry, dict):
                claim = entry.get("claim")
                evidence = entry.get("evidence")
                score = entry.get("score")
                if (
                    isinstance(claim, str)
                    and isinstance(evidence, str)
                    and isinstance(score, (int, float))
                    and not isinstance(score, bool)
                    and math.isfinite(score)
                    and 0.0 <= score <= 1.0
                ):
                    normalized_claim = _clean_claim_text(claim)
                    if normalized_claim:
                        entailment_cache[(normalized_claim, evidence)] = float(score)

    cache_entries_received = len(entailment_cache)

    evidence_map = _build_evidence_map(state)

    citation_metadata = _resolve_citation_metadata(state.get("citation_provenance"))

    available_citations = {
        citation_id for citation_id, windows in evidence_map.items() if windows
    }
    _prefetched_scores.set(
        _prefetch_entailment_scores(claim_records, evidence_map, entailment_cache)
    )

    atomic_results: List[Dict[str, Any]] = []
    claim_results: List[Dict[str, Any]] = []

    unsupported_claims: List[Dict[str, Any]] = []
    unsupported_atomic_claims: List[Dict[str, Any]] = []

    uncited_claims: List[Dict[str, Any]] = []
    invalid_citations: List[Dict[str, Any]] = []

    substantive_results: List[Dict[str, Any]] = []

    # Raw answer lines whose claim failed grounding (for salvage pruning).
    unsupported_raw_lines: List[str] = []

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

            result["source_citations"] = list(citations)

            best_citation = _safe_text(result.get("best_citation"))

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
            original_query = state.get("original_query") or state.get("current_query")
            provenance = result["evidence_provenance"]

            if _explicit_query_source_entity_mismatch(
                original_query,
                provenance.get("source"),
            ):
                result["supported"] = False
                result["entity_consistency"] = "mismatch"
                result["entity_consistency_reason"] = (
                    "The cited source label does not match the explicitly "
                    "named company in the original query."
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
            unsupported_raw_lines.append(_safe_text(record.get("raw_text")))

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
    #
    # citation_consistency["passed"] reflects HARD violations only;
    # redundant co-citations are warnings and never fail grounding.
    # ------------------------------------------------------------------

    semantic_support_ok = bool(substantive_results and not unsupported_claims)

    citation_support_ok = bool(
        not uncited_claims and not invalid_citations and citation_consistency["passed"]
    )

    answer_supported = bool(semantic_support_ok and citation_support_ok)

    is_grounded = bool(answer_supported)

    # ------------------------------------------------------------------
    # Salvage candidate
    #
    # When grounding fails ONLY because some lines are unsupported, the
    # supported remainder is still a valid, honest answer. Offer it to the
    # graph; the graph decides whether to use it (only after the revision
    # budget is spent) and re-validates it before returning it.
    # ------------------------------------------------------------------

    pruned_answer = (
        _prune_unsupported_lines(final_answer, unsupported_raw_lines)
        if (not is_grounded and unsupported_raw_lines)
        else ""
    )

    prunable_claims = (
        [_safe_text(item.get("claim")) for item in unsupported_claims]
        if pruned_answer
        else []
    )

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

    if citation_consistency["violation_count"]:
        feedback_parts.append(
            f"{citation_consistency['violation_count']} citation violation(s)"
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
    #
    # Built fresh from THIS pass's failures only. A prompt carried over
    # from an earlier pass would describe claims that may no longer
    # exist, and would linger in state after a passing pass.
    # ------------------------------------------------------------------

    revision_prompt = _build_revision_prompt(
        unsupported_atomic_claims,
        citation_consistency["violations"],
        uncited_claims,
        invalid_citations,
    )

    # ------------------------------------------------------------------
    # Final state update
    # ------------------------------------------------------------------

    hhem_call_count = int(hhem_profile["call_count"])
    hhem_total_ms = round(float(hhem_profile["total_ms"]), 2)

    grounding_ms = round(
        (perf_counter() - grounding_started) * 1000,
        2,
    )

    # Request-scoped grounding / HHEMv2 diagnostics.
    # Per-pass metrics above remain unchanged; these accumulate across
    # every grounding pass executed during this request.
    request_hhem_call_count = (
        int(state.get("request_hhem_call_count", 0)) + hhem_call_count
    )

    request_hhem_total_ms = round(
        float(state.get("request_hhem_total_ms", 0.0)) + hhem_total_ms,
        2,
    )

    grounding_pass_count = int(state.get("grounding_pass_count", 0)) + 1

    logging.getLogger(__name__).debug(
        "GROUNDING_CACHE_METRICS pass=%s received=%s hits=%s misses=%s returned=%s hhem_calls=%s hhem_ms=%s",
        grounding_pass_count,
        cache_entries_received,
        entailment_cache.hits,
        entailment_cache.misses,
        len(entailment_cache),
        hhem_call_count,
        hhem_total_ms,
    )

    if citation_consistency.get("warning_count"):
        logging.getLogger(__name__).info(
            "CITATION_AUDIT_WARNINGS pass=%s warnings=%s",
            grounding_pass_count,
            citation_consistency["warning_count"],
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
        substantive_result_count=len(substantive_results),
        unsupported_claim_count=len(unsupported_claims),
        semantic_support_ok=semantic_support_ok,
        citation_consistency_passed=citation_consistency["passed"],
        citation_support_ok=citation_support_ok,
        citation_violation_count=citation_consistency["violation_count"],
        citation_warning_count=citation_consistency.get("warning_count", 0),
        prunable=bool(pruned_answer),
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
        "request_hhem_call_count": request_hhem_call_count,
        "request_hhem_total_ms": request_hhem_total_ms,
        "grounding_pass_count": grounding_pass_count,
        "entailment_cache_hits": entailment_cache.hits,
        "entailment_cache_misses": entailment_cache.misses,
        "cache_entries_received": cache_entries_received,
        "cache_entries_returned": len(entailment_cache),
        "entailment_cache": [
            {
                "claim": claim,
                "evidence": evidence,
                "score": score,
            }
            for (claim, evidence), score in entailment_cache.items()
        ],
        "grounding_feedback": feedback,
        "unsupported_atomic_claims": (unsupported_atomic_claims),
        "citation_consistency": citation_consistency,
        "revision_prompt": revision_prompt,
        "pruned_answer": pruned_answer,
        "prunable_claims": prunable_claims,
        "status": ("Grounding validation passed." if is_grounded else feedback),
    }


_prefetched_scores: ContextVar["Dict[Tuple[str, str], float] | None"] = ContextVar(
    "grounding_prefetched_scores",
    default=None,
)


def grounding_critic_node(
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """Run grounding validation with exception-safe HHEM profiling."""
    from app.tools.entailment_tool import (
        start_hhem_profile,
        stop_hhem_profile,
    )

    profile_token, profile = start_hhem_profile()
    try:
        return _grounding_critic_node_with_profile(state, profile)
    finally:
        _prefetched_scores.set(None)
        stop_hhem_profile(profile_token)
