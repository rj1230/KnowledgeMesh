from app.agents.nodes.responder import _build_citation_provenance


def test_web_citation_preserves_url_and_source_identity():
    url = "https://docs.example.com/api/overview"

    provenance = _build_citation_provenance(
        {
            "generation_documents": [
                {
                    "id": "web-1",
                    "source": "API Overview",
                    "source_type": "web",
                    "url": url,
                    "content": "The API provides endpoints for retrieving records.",
                    "metadata": {},
                }
            ],
            "grader_status": "passed",
            "grading_mode": "normal",
        }
    )

    citation = provenance["chunk_1"]
    assert citation["url"] == url
    assert citation["source_type"] == "web"
    assert citation["point_id"] == "web-1"
    assert citation["source"] == "API Overview"
    assert citation["text"] == "The API provides endpoints for retrieving records."


def test_web_citation_preserves_url_from_metadata():
    url = "https://docs.example.com/api/authentication"

    provenance = _build_citation_provenance(
        {
            "generation_documents": [
                {
                    "id": "web-2",
                    "content": "Authentication uses access tokens.",
                    "metadata": {
                        "url": url,
                        "source_type": "web",
                        "title": "Authentication",
                    },
                }
            ],
            "grader_status": "passed",
            "grading_mode": "normal",
        }
    )

    citation = provenance["chunk_1"]
    assert citation["url"] == url
    assert citation["source_type"] == "web"
    assert citation["point_id"] == "web-2"


def test_private_citation_preserves_document_and_chunk_identity():
    provenance = _build_citation_provenance(
        {
            "generation_documents": [
                {
                    "id": "point-private-7",
                    "document_id": "document_A",
                    "chunk_id": 7,
                    "total_chunks": 12,
                    "source": "internal-guide.pdf",
                    "source_type": "private",
                    "content": "The internal guide describes the retrieval workflow.",
                    "metadata": {},
                }
            ],
            "grader_status": "passed",
            "grading_mode": "normal",
        }
    )

    citation = provenance["chunk_1"]
    assert citation["document_id"] == "document_A"
    assert citation["chunk_id"] == 7
    assert citation["total_chunks"] == 12
    assert citation["point_id"] == "point-private-7"
    assert citation["source_type"] == "private"


def test_api_normalizer_preserves_web_url_and_provenance_fields():
    from app.main import _normalize_citation_provenance

    url = "https://docs.example.com/api/overview"
    input_provenance = {
        "chunk_1": {
            "citation_id": "chunk_1",
            "point_id": "web-1",
            "document_id": None,
            "chunk_id": None,
            "source": "API Overview",
            "source_type": "web",
            "url": url,
            "canonical_evidence_id": None,
        }
    }

    normalized = _normalize_citation_provenance(input_provenance)
    citation = normalized["chunk_1"]

    assert citation["url"] == url
    assert citation["source_type"] == "web"
    assert citation["point_id"] == "web-1"
    assert citation["document_id"] is None
    assert citation["chunk_id"] is None
    assert citation["canonical_evidence_id"] is None

    # Normalization must not mutate the original record.
    assert input_provenance["chunk_1"]["url"] == url


def test_fallback_evidence_uses_positional_citation_not_chunk_identity():
    from app.agents.nodes.grounding_critic import (
        _resolve_documents_by_citation,
    )

    documents = [
        {
            "chunk_id": 7,
            "document_id": "document_A",
            "id": "point-private-7",
            "page_content": (
                "The retrieval pipeline validates evidence before "
                "producing an answer."
            ),
            "metadata": {
                "document_id": "document_A",
                "chunk_id": 7,
            },
        }
    ]

    evidence = _resolve_documents_by_citation(documents)

    assert "chunk_1" in evidence
    assert "7" not in evidence
    assert "point-private-7" not in evidence
    assert evidence["chunk_1"]

def test_evidence_map_prefers_provenance_over_conflicting_fallback():
    from app.agents.nodes.grounding_critic import _build_evidence_map

    trusted_text = (
        "The retrieval pipeline validates approved evidence before "
        "generating a grounded response."
    )
    conflicting_text = (
        "The unrelated billing service calculates invoices and "
        "processes customer payments."
    )

    evidence_map = _build_evidence_map(
        {
            "citation_provenance": {
                "chunk_1": {
                    "citation_id": "chunk_1",
                    "text": trusted_text,
                }
            },
            "generation_documents": [
                {
                    "page_content": conflicting_text,
                    "metadata": {},
                }
            ],
        }
    )

    assert "chunk_1" in evidence_map
    assert any(
        trusted_text in window
        for window in evidence_map["chunk_1"]
    )
    assert not any(
        conflicting_text in window
        for window in evidence_map["chunk_1"]
    )


def test_evidence_map_uses_generation_documents_when_provenance_is_absent():
    from app.agents.nodes.grounding_critic import _build_evidence_map

    generation_text = (
        "The retrieval pipeline ranks candidate passages before "
        "answer generation."
    )

    evidence_map = _build_evidence_map(
        {
            "citation_provenance": {},
            "generation_documents": [
                {
                    "page_content": generation_text,
                    "metadata": {},
                }
            ],
        }
    )

    assert "chunk_1" in evidence_map
    assert any(
        generation_text in window
        for window in evidence_map["chunk_1"]
    )

def test_duplicate_provenance_label_merges_distinct_evidence_windows():
    from app.agents.nodes.grounding_critic import _resolve_citation_provenance

    first_text = (
        "The retrieval service ranks passages before generating "
        "a response to the user's query."
    )
    second_text = (
        "The billing service calculates invoice totals and records "
        "payments for customer accounts."
    )

    result = _resolve_citation_provenance(
        [
            {"citation_id": "chunk_1", "text": first_text},
            {"citation_id": "chunk_1", "text": second_text},
        ]
    )

    assert "chunk_1" in result
    assert any(first_text in window for window in result["chunk_1"])
    assert any(second_text in window for window in result["chunk_1"])


def test_evidence_map_does_not_add_unapproved_web_evidence():
    from app.agents.nodes.grounding_critic import _build_evidence_map

    approved_text = (
        "The approved retrieval pipeline ranks passages before "
        "generating a grounded answer."
    )
    unapproved_text = (
        "The unapproved web source claims the billing system "
        "automatically refunds every customer."
    )

    evidence_map = _build_evidence_map(
        {
            "citation_provenance": {
                "chunk_1": {
                    "citation_id": "chunk_1",
                    "text": approved_text,
                }
            },
            "generation_documents": [
                {
                    "page_content": approved_text,
                    "metadata": {},
                }
            ],
            "web_documents": [
                {
                    "citation_id": "web_unapproved",
                    "page_content": unapproved_text,
                    "metadata": {},
                }
            ],
        }
    )

    assert "chunk_1" in evidence_map
    assert "web_unapproved" not in evidence_map


def test_evidence_map_rejects_raw_web_documents_when_grading_was_skipped():
    from app.agents.nodes.grounding_critic import _build_evidence_map

    evidence_map = _build_evidence_map(
        {
            "grader_status": "skipped",
            "grading_mode": "no_documents",
            "generation_documents": [],
            "web_documents": [
                {
                    "citation_id": "raw_web",
                    "page_content": "Raw ungraded web content.",
                    "metadata": {},
                }
            ],
        }
    )

    assert "raw_web" not in evidence_map


def test_evidence_map_rejects_raw_documents_when_grading_failed():
    from app.agents.nodes.grounding_critic import _build_evidence_map

    evidence_map = _build_evidence_map(
        {
            "grader_status": "failed",
            "grading_mode": "unavailable",
            "generation_documents": [],
            "documents": [
                {
                    "citation_id": "ungraded_private",
                    "page_content": "Private content whose grading failed.",
                    "metadata": {},
                }
            ],
        }
    )

    assert "ungraded_private" not in evidence_map



def test_evidence_map_rejects_stale_generation_documents_after_grader_failure():
    from app.agents.nodes.grounding_critic import _build_evidence_map

    evidence_map = _build_evidence_map(
        {
            "grader_status": "failed",
            "grading_mode": "unavailable",
            "generation_documents": [
                {
                    "page_content": (
                        "Stale evidence that must not be trusted "
                        "after grading fails."
                    ),
                    "metadata": {"citation_id": "stale_private"},
                }
            ],
            "citation_provenance": {},
            "documents": [],
            "web_documents": [],
        }
    )

    assert evidence_map == {}


def test_evidence_map_rejects_stale_generation_documents_when_grading_is_skipped():
    from app.agents.nodes.grounding_critic import _build_evidence_map

    evidence_map = _build_evidence_map(
        {
            "grader_status": "skipped",
            "grading_mode": "no_documents",
            "generation_documents": [
                {
                    "page_content": (
                        "Stale evidence that must not be trusted "
                        "when grading is skipped."
                    ),
                    "metadata": {"citation_id": "stale_web"},
                }
            ],
            "citation_provenance": {},
            "documents": [],
            "web_documents": [],
        }
    )

    assert evidence_map == {}