import app.agents.nodes.retriever as retriever


def make_document(
    *,
    document_id: str,
    chunk_id: str,
    dense_score: float,
    rerank_score: float,
    content: str,
) -> dict:
    return {
        "document_id": document_id,
        "chunk_id": chunk_id,
        "score": dense_score,
        "rerank_score": rerank_score,
        "content": content,
    }


def test_selects_strong_semantic_candidate():
    document = make_document(
        document_id="doc-a",
        chunk_id="chunk-a",
        dense_score=0.60,
        rerank_score=0.60,
        content="Retrieval threshold calibration improves evidence selection.",
    )

    selected = retriever._select_complementary_evidence(
        query="retrieval threshold calibration",
        reranked_documents=[document],
        final_k=5,
    )

    assert len(selected) == 1
    assert selected[0]["document_id"] == "doc-a"


def test_recovers_low_rerank_candidate_when_dense_and_coverage_qualify():
    document = make_document(
        document_id="doc-b",
        chunk_id="chunk-b",
        dense_score=0.70,
        rerank_score=0.20,
        content="Retrieval threshold calibration requires relevant evidence.",
    )

    selected = retriever._select_complementary_evidence(
        query="retrieval threshold calibration",
        reranked_documents=[document],
        final_k=5,
    )

    assert len(selected) == 1
    assert selected[0]["document_id"] == "doc-b"
    assert "_dense_recovery" not in selected[0]


def test_rejects_candidate_below_both_eligibility_gates():
    document = make_document(
        document_id="doc-c",
        chunk_id="chunk-c",
        dense_score=0.63,
        rerank_score=0.20,
        content="Retrieval threshold calibration is discussed here.",
    )

    selected = retriever._select_complementary_evidence(
        query="retrieval threshold calibration",
        reranked_documents=[document],
        final_k=5,
    )

    assert selected == []


def test_rejects_dense_recovery_without_query_coverage():
    document = make_document(
        document_id="doc-d",
        chunk_id="chunk-d",
        dense_score=0.70,
        rerank_score=0.20,
        content="Database maintenance and system administration procedures.",
    )

    selected = retriever._select_complementary_evidence(
        query="retrieval threshold calibration",
        reranked_documents=[document],
        final_k=5,
    )

    assert selected == []


def test_empty_candidates_return_empty_selection():
    selected = retriever._select_complementary_evidence(
        query="retrieval threshold calibration",
        reranked_documents=[],
        final_k=5,
    )

    assert selected == []
def test_recovers_relevant_candidate_at_calibrated_dense_threshold():
    document = make_document(
        document_id="doc-e",
        chunk_id="chunk-e",
        dense_score=0.65,
        rerank_score=0.20,
        content="Retrieval threshold calibration improves relevant evidence selection.",
    )

    selected = retriever._select_complementary_evidence(
        query="retrieval threshold calibration",
        reranked_documents=[document],
        final_k=5,
    )

    assert len(selected) == 1
    assert selected[0]["document_id"] == "doc-e"

