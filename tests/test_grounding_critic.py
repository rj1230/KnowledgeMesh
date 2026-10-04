from app.agents.nodes.grounding_critic import _resolve_citation_metadata


def test_resolve_citation_metadata_preserves_document_scoped_identity():
    provenance = {
        "chunk_1": {
            "citation_id": "chunk_1",
            "document_id": "document_A",
            "chunk_id": 7,
            "point_id": "point-A-7",
        },
        "chunk_2": {
            "citation_id": "chunk_2",
            "document_id": "document_B",
            "chunk_id": 7,
            "point_id": "point-B-7",
        },
    }

    result = _resolve_citation_metadata(provenance)

    assert result["chunk_1"]["document_id"] == "document_A"
    assert result["chunk_1"]["chunk_id"] == 7
    assert result["chunk_1"]["point_id"] == "point-A-7"
    assert result["chunk_1"]["canonical_evidence_id"] == "document_A::7"

    assert result["chunk_2"]["document_id"] == "document_B"
    assert result["chunk_2"]["chunk_id"] == 7
    assert result["chunk_2"]["point_id"] == "point-B-7"
    assert result["chunk_2"]["canonical_evidence_id"] == "document_B::7"

    assert (
        result["chunk_1"]["canonical_evidence_id"]
        != result["chunk_2"]["canonical_evidence_id"]
    )
