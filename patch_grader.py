"""Grader fixes for KnowledgeMesh (app/agents/nodes/grader.py).

1. Table-of-contents chunks (dot-leader rows like ". . . . . 17") are
   reclassified as NAVIGATION so they cannot be primary evidence.
2. A "strong" context is downgraded to "weak" when the selected evidence
   covers too few of the query's key terms, so the graph can retry / use web
   search instead of answering from off-topic chunks.

Run from the project root:   python patch_grader.py
Backs up to app/agents/nodes/grader.py.bak. Safe to run twice.
"""

import pathlib
import shutil
import sys

path = pathlib.Path("app/agents/nodes/grader.py")

if not path.exists():
    sys.exit("app/agents/nodes/grader.py not found. Run from the project root.")

text = path.read_text(encoding="utf-8")

if "_looks_like_table_of_contents" in text:
    sys.exit("Already patched. Nothing to do.")

HELPERS = '''# Minimum share of the query's key terms that the selected evidence must
# contain for the context to count as "strong".
MIN_QUERY_TERM_COVERAGE = 0.5
MIN_QUERY_TERMS_FOR_COVERAGE = 3

_DOT_LEADER_RE = re.compile(r"(?:\\.\\s?){6,}")

_QUERY_STOPWORDS = {
    "what", "which", "when", "where", "does", "about", "explain", "describe",
    "tell", "give", "show", "list", "main", "this", "that", "these", "those",
    "with", "from", "into", "have", "has", "how", "why", "are", "all", "the",
    "and", "for", "can", "use", "used", "using", "between", "their", "there",
}


def _looks_like_table_of_contents(text: Any) -> bool:
    """True for table-of-contents chunks (rows ending in dot leaders)."""

    return len(_DOT_LEADER_RE.findall(str(text or ""))) >= 3


def _reclassify_table_of_contents(documents):
    """Mark table-of-contents chunks as NAVIGATION; return how many changed."""

    changed = 0

    for doc in documents:
        if doc.get("evidence_type") == "NAVIGATION":
            continue

        if _looks_like_table_of_contents(doc.get("content")):
            doc["evidence_type"] = "NAVIGATION"
            changed += 1

    return changed


def _query_term_coverage(query: str, documents):
    """Share of the query's key terms found in the documents (None if n/a)."""

    words = re.findall(r"[a-z0-9]+", str(query or "").lower())
    terms = [
        word
        for word in dict.fromkeys(words)
        if len(word) > 3 and word not in _QUERY_STOPWORDS
    ]

    if len(terms) < MIN_QUERY_TERMS_FOR_COVERAGE or not documents:
        return None

    haystack = " ".join(str(doc.get("content", "")) for doc in documents).lower()

    # 5-letter prefixes so "patterns"/"pattern" and "agentic"/"agent" match.
    hits = sum(1 for term in terms if term[:5] in haystack)

    return hits / len(terms)


'''

ANCHOR_HELPERS = "def _safe_bool(value: Any) -> bool:"
ANCHOR_RANK = "        graded_documents = rank_evidence(graded_documents)\n"
ANCHOR_EMPTY = '            context_quality = "empty"\n'

for name, anchor in (
    ("helpers", ANCHOR_HELPERS),
    ("rank_evidence", ANCHOR_RANK),
    ("context_quality", ANCHOR_EMPTY),
):
    if text.count(anchor) != 1:
        sys.exit(
            f"Could not find a unique anchor for '{name}' "
            f"(found {text.count(anchor)}). No changes made."
        )

RECLASSIFY = """
        toc_reclassified = _reclassify_table_of_contents(graded_documents)

        if toc_reclassified:
            print(
                "🧪 TOC FILTER | reclassified_as_navigation="
                f"{toc_reclassified}"
            )
"""

COVERAGE = """
        term_coverage = _query_term_coverage(query, relevant_documents)

        if (
            context_quality == "strong"
            and term_coverage is not None
            and term_coverage < MIN_QUERY_TERM_COVERAGE
        ):
            print(
                "🧪 COVERAGE CHECK | strong -> weak | "
                f"query_term_coverage={term_coverage:.2f}"
            )
            context_quality = "weak"
"""

text = text.replace(ANCHOR_HELPERS, HELPERS + ANCHOR_HELPERS)
text = text.replace(ANCHOR_RANK, ANCHOR_RANK + RECLASSIFY)
text = text.replace(ANCHOR_EMPTY, ANCHOR_EMPTY + COVERAGE)

shutil.copy(path, path.with_suffix(".py.bak"))
path.write_text(text, encoding="utf-8")
print("Patched grader.py (backup: grader.py.bak)")
